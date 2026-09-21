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

    WHAT THIS DOES NOT DO, stated up front because the docs used to imply otherwise: the published
    archive does NOT contain a `manifest.json` in the schema this pipeline's loader needs, and it
    ships NO foreground masks -- which `orhsurf run` requires in two places. So this fetches and
    extracts the raw capture, and then tells you plainly what is still missing rather than letting
    you discover it three stages into a run. See docs/DATA_CONTRACT.md.
    """
    from huggingface_hub import hf_hub_download

    repo, rev = _repo(), _revision()
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    dest = data_root / clip

    last_err = None
    for tmpl in CLIP_ARCHIVE_TEMPLATES:
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
            tf.extractall(data_root)
        print(f"[fetch] extracted to {dest}")
        if convert:
            return convert_clip(dest, data_root / f"{clip}_prepared", masks=masks,
                                frames=frames)
        _report_missing(dest, clip)
        return 0

    print(f"[fetch] ERROR: no archive for clip '{clip}' in {repo}@{rev[:8]}.\n"
          f"  Tried: {[t.format(clip=clip) for t in CLIP_ARCHIVE_TEMPLATES]}\n"
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
              "  See docs/DATA_CONTRACT.md. A converter is not part of this package yet.")


def convert_clip(clip_dir: Path, out_dir: Path, masks: str | None = None,
                 frames: str | None = None) -> int:
    """Decode the archive's videos and emit a manifest this pipeline can load.

    Separated from fetch_clip so an already-extracted archive can be converted without
    re-downloading 1.5 GB.  Masks are NOT invented here: --masks points at a directory laid out as
    <masks>/<serial>/<encoded_frame_index:05d>.png (RGBA, foreground in ALPHA).  Without it the
    manifest is written with mask_path=None and this function says exactly which step is still
    missing, because prep.py asserts a mask per view and >= 8 non-empty ones.
    """
    import json
    from . import convert as C

    clip_dir, out_dir = Path(clip_dir), Path(out_dir)
    vmj = clip_dir / "video_manifest.json"
    if not vmj.exists():
        print(f"[convert] ERROR: {vmj} not found -- is {clip_dir} an extracted clip archive?")
        return 2
    vm = json.load(open(vmj))
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
    man_path = out_dir / "manifest.json"
    _man, missing = C.build_manifest(clip_dir, rgb_root, masks, man_path, frames=want)

    if missing:
        views = len({s for s, _ in missing})
        print(f"[convert] NO MASKS for {views} view(s) ({len(missing)} view-frames).")
        print(f"[convert] The manifest is written and the frames are decoded, but `orhsurf run`")
        print(f"[convert] will stop in prep: it asserts a foreground mask per view and needs >= 8")
        print(f"[convert] non-empty ones. Supply them with --masks <dir>, laid out as")
        print(f"[convert]   <dir>/<serial>/<frame:05d>.png   RGBA, foreground in the ALPHA channel")
        print(f"[convert] The published archives do not carry masks; generating them is not part")
        print(f"[convert] of this package. See docs/DATA_CONTRACT.md.")
        return 3
    print(f"[convert] ready: orhsurf run --clip {man_path}")
    return 0
