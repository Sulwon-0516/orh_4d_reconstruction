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

# TODO(user): the HuggingFace dataset repo id for the clips is not known yet.  install.sh and
# `orhsurf fetch --clip` refuse to run while this is the placeholder, rather than guessing an id
# and failing with a confusing 404 from the Hub.
CLIP_DATASET_REPO = "PLACEHOLDER/ORHSURF-DATASET-REPO-ID-NOT-SET"


def _placeholder(repo: str) -> bool:
    return repo.startswith("PLACEHOLDER/")


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


def fetch_clip(clip: str, data_root: Path) -> int:
    if _placeholder(CLIP_DATASET_REPO):
        print("[fetch] ERROR: the HuggingFace dataset repo id is not configured.\n"
              "        Set CLIP_DATASET_REPO in orhsurf/fetch.py (or the ORHSURF_CLIP_REPO env\n"
              "        var) to the dataset repo that hosts the clips, then re-run.")
        return 2
    from huggingface_hub import snapshot_download
    repo = os.environ.get("ORHSURF_CLIP_REPO", CLIP_DATASET_REPO)
    dest = Path(data_root) / clip
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] clip {clip} from {repo} -> {dest}")
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[f"{clip}/*"],
                      local_dir=str(Path(data_root)))
    return 0
