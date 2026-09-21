"""Download the model weights and the clip data.

CODE lives on GitHub; DATA lives on HuggingFace.  Both fetches are separable from the run on
purpose: cluster compute nodes are frequently offline, so you pre-stage on a login node and then
run with HF_HUB_OFFLINE=1.  See docs/INSTALL_SLURM.md.
"""
from __future__ import annotations

import os
from pathlib import Path

# DA3 checkpoint: 6,759,558,100 B (6.76 GB), sha256
# 8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c
DA3_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"

#: The published clip dataset. A DATASET repo (repo_type="dataset"), pinned by revision.
CLIP_DATASET_REPO = "Sulwon/anonymous_dataset_lih_orh_0920"
CLIP_DATASET_REVISION = "535c72f84dc124cfed76fb1be2c17e136fb331c3"

#: Published layout: one tar per clip at BOTH of these prefixes. `data/` holds the JPEG frames,
#: `hevc/` the re-muxed video. A clip id is like "C001".
CLIP_ARCHIVE_TEMPLATES = ("data/{clip}.tar", "hevc/{clip}.tar")


def _repo() -> str:
    return os.environ.get("ORHSURF_CLIP_REPO") or CLIP_DATASET_REPO


def _revision() -> str:
    return os.environ.get("ORHSURF_CLIP_REVISION") or CLIP_DATASET_REVISION


def fetch_weights(cache: Path) -> int:
    """Download the pinned DA3 snapshot into OUR cache.

    `cache_dir=` is passed explicitly rather than relying on HF_HOME: huggingface_hub reads its
    environment at IMPORT time, so setting HF_HOME after the import had no effect and the download
    could land in $HOME/.cache (or fail outright on a read-only home) while the pipeline's own
    cache stayed empty.
    """
    hf = Path(cache) / "hf"
    hub = hf / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf))
    from huggingface_hub import snapshot_download
    print(f"[fetch] {DA3_REPO}@{DA3_REVISION[:8]} (6.76 GB) -> {hub}")
    p = snapshot_download(repo_id=DA3_REPO, revision=DA3_REVISION, cache_dir=str(hub))
    print(f"[fetch] weights at {p}")
    print("[fetch] NOTE: inference loads this same revision explicitly "
          "(orhsurf/stages/da3_prior.py::DA3_REVISION), so an offline snapshot works.")
    return 0


def fetch_clip(clip: str, data_root: Path, convert: bool = False,
               masks: str | None = None, frames: str | None = None) -> int:
    """Download and extract one clip archive.

    Raw archives contain RGB data and calibration, but no foreground masks. With
    convert=True, fetch HEVC and prepare a manifest with all-foreground masks by
    default; explicit masks override that policy. Without conversion, report
    missing pipeline inputs after extraction.
    """
    from huggingface_hub import hf_hub_download

    repo, rev = _repo(), _revision()
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    # Never mix HEVC camera metadata into an existing JPEG clip directory.
    extraction_root = data_root / "_hevc" if convert else data_root
    extraction_root.mkdir(parents=True, exist_ok=True)
    dest = extraction_root / clip
    templates = ("hevc/{clip}.tar",) if convert else CLIP_ARCHIVE_TEMPLATES

    last_err = None
    for tmpl in templates:
        rel = tmpl.format(clip=clip)
        try:
            print(f"[fetch] {repo}@{rev[:8]} :: {rel} -> {dest}")
            tar_path = hf_hub_download(repo_id=repo, filename=rel, repo_type="dataset",
                                       revision=rev, cache_dir=str(data_root / "_hf"))
        except Exception as e:
            last_err = e
            print(f"[fetch]   {rel} not available ({type(e).__name__})")
            continue
        import tarfile
        print(f"[fetch] extracting {tar_path}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(extraction_root, filter="data")
        print(f"[fetch] extracted to {dest}")
        if convert:
            return convert_clip(dest, data_root / f"{clip}_prepared", masks=masks,
                                frames=frames)
        _report_missing(dest, clip)
        return 0

    print(f"[fetch] ERROR: no archive for clip '{clip}' in {repo}@{rev[:8]}.\n"
          f"  Tried: {[t.format(clip=clip) for t in templates]}\n"
          f"  Clip ids in this dataset look like C001..C100.\n"
          f"  Last error: {type(last_err).__name__}: {last_err}\n"
          f"  Override the repo with ORHSURF_CLIP_REPO / ORHSURF_CLIP_REVISION.")
    return 2


def _report_missing(dest: Path, clip: str) -> None:
    """Say exactly what `orhsurf run` will still refuse to do, and why."""
    problems = []
    if not (dest / "manifest.json").is_file():
        problems.append(
            "no manifest.json in the pipeline's schema (the archive ships video_manifest.json, "
            "cameras.json and cam_param/{intrinsics,extrinsics}.json instead)")
    if not any((dest / d).is_dir() for d in ("masks", "masks_all")):
        problems.append(
            "no foreground masks -- required by prep and by the DA3 visual hull")
    if problems:
        print(f"\n[fetch] {clip} is downloaded, but `orhsurf run` CANNOT use it yet:")
        for p in problems:
            print(f"  - {p}")
        print("  This is a known gap between the published dataset and the input contract.\n"
              "  HEVC conversion: orhsurf fetch --clip <id> --convert --frames 0-4.\n"
              "  Conversion generates all-foreground masks by default. See docs/DATA_CONTRACT.md.")


def convert_clip(clip_dir: Path, out_dir: Path, masks: str | None = None,
                 frames: str | None = None) -> int:
    """Decode the archive's videos and emit a manifest this pipeline can load.

    Existing extracted inputs can be reused without downloading. Omitting --masks selects
    all-foreground RGBA masks (alpha=255). Explicit masks override this default and are
    never silently filled in when incomplete. The manifest records the chosen policy.
    """
    import json
    from . import convert as C

    clip_dir, out_dir = Path(clip_dir).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    masks = str(Path(masks).expanduser().resolve()) if masks else None
    vmj = clip_dir / "video_manifest.json"
    if not vmj.exists():
        print(f"[convert] ERROR: {vmj} not found -- is {clip_dir} an extracted clip archive?")
        return 2
    vm = json.loads(vmj.read_text())
    n = int(vm["window"]["n_timestamps"])
    serials = list(vm["valid_serials"])
    fps = n / max(float(vm["window"].get("duration_s") or 1), 1e-9)
    want = C.parse_frames(frames, n)
    print(f"[convert] {clip_dir.name}: {len(serials)} valid views, clip has {n} frames "
          f"({vm['window'].get('duration_s')} s at {fps:.0f} fps)")
    if len(want) != n:
        print(f"[convert] decoding {len(want)} of them "
              f"({want[0]}..{want[-1]}, {len(want) / fps:.1f} s) -- --frames {frames}")

    rgb_root = C.decode_views(clip_dir, out_dir, serials, n, frames=want)
    mask_mode = "all_foreground" if masks is None else "provided"
    if masks is None:
        # This default changes the visual-hull centre and potentially the DA3 groups.
        masks = C.write_all_foreground_masks(clip_dir, rgb_root, out_dir / "masks_all_foreground", frames=want)
    man_path = out_dir / "manifest.json"
    _man, missing = C.build_manifest(clip_dir, rgb_root, masks, man_path, frames=want, mask_mode=mask_mode)

    if missing:
        views = len({s for s, _ in missing})
        print(f"[convert] NO MASKS for {views} view(s) ({len(missing)} view-frames).")
        print(f"[convert] The manifest is written and the frames are decoded, but `orhsurf run`")
        print(f"[convert] will stop in prep: it asserts a foreground mask per view and needs >= 8")
        print(f"[convert] non-empty ones. Supply them with --masks <dir>, laid out as")
        print(f"[convert]   <dir>/<serial>/<frame:05d>.png   RGBA, foreground in the ALPHA channel")
        print("[convert] Explicit masks were supplied; missing masks are not replaced by defaults.")
        return 3
    print(f"[convert] ready: orhsurf run --clip {man_path}")
    return 0
