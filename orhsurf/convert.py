"""Turn a published HuggingFace clip archive into a manifest this pipeline can run.

The published archives are HEVC video plus calibration; this pipeline wants per-frame image files
and a manifest in its own schema (orhsurf/_vendor/colmap_dataset.py::frame_image_path).  Without
this step `fetch --clip` produces a directory nothing can read, which is where the package stood.

THE ONE THING THAT SILENTLY CORRUPTS A RUN
------------------------------------------
The dataset README is explicit: decoded MP4 frames are indexed by `encoded_frame_index` (0..224).
`video_frame_index` refers to the ORIGINAL recording and must not be used as the MP4 index.  Confuse
them and every camera is offset by an arbitrary amount -- the reconstruction still completes, it
just fuses different time instants.  So this module NEVER derives an MP4 index from anything but
`encoded_frame_index`, and asserts the decoded frame count rather than trusting it.

MASKS
-----
The archives ship no foreground masks, and three places want them:
  prep.py:72          asserts a mask_path per view, and >= 8 non-empty ones
  build_scene         writes the mask into each image's alpha channel
  visual_hull_box     derives the per-frame scene centre that orders views into DA3 groups
Training uses unmasked RGB. Conversion uses all-foreground RGBA masks by default, with
--masks as an explicit override. This changes the visual-hull centre used for DA3 grouping;
it is not a claim of equivalent output to subject-mask reconstruction.
"""
from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
from . import cpubudget

EXPECT_W, EXPECT_H = 2048, 1536


def _ffprobe_frames(mp4: Path, exact: bool = False) -> int:
    """Frame count. `exact=False` reads the container header; `exact=True` decodes to count.

    -count_frames DECODES THE WHOLE FILE.  Calling it per view to validate a 5-frame decode cost
    ~50 s per view here -- 225 frames decoded to justify keeping 5.  The header's nb_frames is
    free and is what we check per view; the expensive exact count is done once, on the first
    video only, to confirm the header is not lying about this encoder.
    """
    if exact:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-threads", str(cpubudget.resolve()), "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, check=True).stdout.strip()
        return int(out)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def parse_frames(spec: str | None, n_total: int) -> list[int]:
    """'0-149' / '0,5,9' / None -> a sorted list of encoded_frame_index values.

    A published clip is 225 frames (15 s at 15 fps); the window this project measured on is the
    first 150 (10 s).  Decoding only what will be reconstructed saves 47 x 75 PNGs per clip, so the
    choice belongs at convert time as well as at run time.
    """
    if not spec:
        return list(range(n_total))
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if int(a) > int(b):
                raise ValueError(f"descending frame range: {part}")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("frame selection is empty")
    bad = [i for i in out if not 0 <= i < n_total]
    assert not bad, (f"frames {sorted(bad)[:5]} are outside this clip's 0..{n_total - 1} "
                     f"(encoded_frame_index). The clip has {n_total} frames.")
    return sorted(out)


def decode_views(clip_dir: Path, out_dir: Path, serials, n_expect: int, frames=None, log=print):
    """videos/<serial>.mp4 -> out_dir/rgb/<serial>/<encoded_frame_index:05d>.png

    ffmpeg emits frames in decode order starting at 1, so image N is encoded_frame_index N-1.
    That is the ONLY mapping used anywhere in this file.
    Resumable: a serial whose directory already holds n_expect files is skipped.
    """
    out_dir = Path(out_dir)
    want = list(range(n_expect)) if frames is None else list(frames)
    contiguous_from_zero = want == list(range(len(want)))
    done, todo = [], []
    for s in serials:
        d = out_dir / "rgb" / s
        if d.is_dir() and all((d / f"{i:05d}.png").exists() for i in want):
            done.append(s)
        else:
            todo.append(s)
    if done:
        log(f"[convert] {len(done)} view(s) already decoded, skipping")
    for i, s in enumerate(todo, 1):
        mp4 = clip_dir / "videos" / f"{s}.mp4"
        assert mp4.exists(), f"missing video for {s}: {mp4}"
        got = _ffprobe_frames(mp4, exact=(i == 1))     # exact decode-count on the first view only
        assert got == n_expect, (
            f"{s}: video holds {got} frames, manifest says {n_expect}. Refusing to decode -- a "
            f"frame-count mismatch means the index mapping is not what this code assumes."
            + ("" if got >= 0 else " (container header carries no nb_frames; rerun with exact=True)"))
        d = out_dir / "rgb" / s
        tmp = d.with_name(d.name + ".part")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        threads = str(cpubudget.resolve())
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", threads,
               "-filter_threads", threads, "-i", str(mp4)]
        if contiguous_from_zero and len(want) < n_expect:
            # decode-order prefix: stop early rather than writing 225 frames to keep 150
            cmd += ["-frames:v", str(len(want))]
        if not contiguous_from_zero:
            selection = "+".join(f"eq(n\\,{i})" for i in want)
            cmd += ["-vf", f"select={selection}", "-vsync", "0", "-frames:v", str(len(want))]
        cmd += ["-threads", threads, "-start_number", "0", str(tmp / "%05d.png")]
        subprocess.run(cmd, check=True)
        if not contiguous_from_zero:
            # Reverse order avoids collisions when a target is another temporary index.
            for j, frame in reversed(list(enumerate(want))):
                (tmp / f"{j:05d}.png").rename(tmp / f"{frame:05d}.png")
        n = len(list(tmp.glob("*.png")))
        assert n == len(want), f"{s}: ffmpeg left {n} frames, expected {len(want)}"
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)                                  # atomic: a half-decoded view is never visible
        log(f"[convert] decoded {s}  ({i}/{len(todo)})")
    return out_dir / "rgb"


def _solid_rgba_png(w: int, h: int, v: int = 255) -> bytes:
    """A w x h RGBA PNG, every channel `v`. Pure stdlib: cv2/PIL may not be in the caller's env,
    and one 15 KB buffer is reused for every file rather than re-encoding 235 times."""
    import struct, zlib
    raw = b"".join(b"\x00" + bytes([v, v, v, v]) * w for _ in range(h))   # filter byte 0 per row
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def write_all_foreground_masks(clip_dir: Path, rgb_root: Path, out_dir: Path, log=print, frames=None) -> Path:
    """Write an all-foreground (alpha = 255 everywhere) RGBA mask beside every decoded frame.

    Training uses full RGB, while the visual-hull centre affects DA3 camera grouping.
    This is the default policy, not a segmentation result or a claim of equivalent output.
    Supply subject masks with --masks to use subject-centred grouping.
    """
    import json
    vm = json.loads((Path(clip_dir) / "video_manifest.json").read_text())
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache, n = {}, 0
    for s in vm["valid_serials"]:
        d = Path(rgb_root) / s
        if not d.is_dir():
            continue
        c = vm["cameras"][s]; wh = (int(c["width"]), int(c["height"]))
        if wh not in cache:
            cache[wh] = _solid_rgba_png(*wh)
        (out_dir / s).mkdir(parents=True, exist_ok=True)
        files = sorted(d.glob("*.png")) if frames is None else [d / f"{i:05d}.png" for i in frames]
        for f in files:
            if not f.is_file():
                raise FileNotFoundError(f"missing decoded RGB: {f}")
            (out_dir / s / f.name).write_bytes(cache[wh]); n += 1
    log(f"[convert] wrote {n} all-foreground masks -> {out_dir}")
    log(f"[convert]   AmbiSuR ignores the alpha; these exist because prep asserts a mask per view.")
    log(f"[convert]   They DO change the DA3 azimuth grouping (see write_all_foreground_masks).")
    return out_dir


def build_manifest(clip_dir: Path, rgb_root: Path, mask_root: Path | None, out_json: Path,
                   frames=None, log=print, mask_mode=None):
    """video_manifest.json -> this pipeline's manifest schema, with resolved local paths."""
    clip_dir, rgb_root = Path(clip_dir).resolve(), Path(rgb_root).resolve()
    mask_root = Path(mask_root).resolve() if mask_root is not None else None
    vm = json.loads((clip_dir / "video_manifest.json").read_text())
    n = int(vm["window"]["n_timestamps"])
    valid = list(vm["valid_serials"])
    cams = {}
    missing_masks = []
    for s, c in vm["cameras"].items():
        c = dict(c)
        assert int(c["width"]) == EXPECT_W and int(c["height"]) == EXPECT_H, (
            f"{s}: {c['width']}x{c['height']}, expected {EXPECT_W}x{EXPECT_H}")
        if c.get("valid", True) and s in valid:
            want = list(range(n)) if frames is None else list(frames)
            frames_out = []
            for i in want:
                mp = None
                if mask_root is not None:
                    cand = Path(mask_root) / s / f"{i:05d}.png"
                    if cand.exists():
                        mp = str(cand)
                if mp is None:
                    missing_masks.append((s, i))
                # `index` is the encoded_frame_index -- the MP4 index, never video_frame_index
                frames_out.append(dict(index=i,
                                        frame_path=str(Path(rgb_root) / s / f"{i:05d}.png"),
                                        mask_path=mp))
            # frames is a DICT keyed by encoded_frame_index, not a list, so a converted subset
            # keeps its true indices -- `run --frames 40-41` means frames 40 and 41 of the CLIP,
            # never "the 40th thing that happened to be decoded".
            c["frames"] = {str(f["index"]): f for f in frames_out}
            c["n_frames_in_window"] = len(frames_out)
        c.pop("video_path", None)
        c.pop("timestamps_path", None)
        cams[s] = c
    man = dict(sequence_id=vm.get("clip_id", clip_dir.name),
               source_dir=str(clip_dir), read_only=True,
               generator="orhsurf.convert (HuggingFace HEVC archive)",
               conventions=vm.get("conventions"), window=vm["window"],
               timestamps=vm.get("timestamps"),
               n_cameras_total=vm.get("n_cameras_total"),
               n_cameras_valid=vm.get("n_cameras_valid"),
               calibrated_serials=vm.get("calibrated_serials"),
               valid_serials=valid, excluded_serials=vm.get("excluded_serials"),
               extrinsics_direction=vm.get("extrinsics_direction"),
               world_frame=vm.get("world_frame"),
               frame_index_basis="encoded_frame_index (0-based MP4 decode order)",
               decoded_frames=(list(range(n)) if frames is None else list(frames)),
               cameras=cams)
    if mask_mode is not None:
        man["mask_policy"] = dict(mode=mask_mode, root=str(mask_root))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_json) + ".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    tmp.rename(out_json)
    log(f"[convert] wrote {out_json}  ({len(valid)} valid views x {len(man['decoded_frames'])} frames)")
    return man, missing_masks
