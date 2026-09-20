"""Where things live, resolved at run time. No path in this package points at a specific machine.

Everything is resolved in the same order:  explicit env var  ->  inside this checkout  ->  $PATH.

Resolution is deliberately NON-FATAL: `colmap_bin()` returns its best guess even when nothing
exists, because `_vendor/colmap_dataset.py` binds it as a default argument at import time and we
do not want an import to explode on a machine that is merely missing COLMAP.  Validation is a
separate, explicit step -- `orhsurf doctor` -- which reports every missing piece at once instead of
failing one at a time, three minutes into a job.

Env vars, all optional:
  ORHSURF_COLMAP_BIN    colmap executable
  ORHSURF_AMBISUR_REPO  AmbiSuR checkout (default: third_party/AmbiSuR in this repo)
  ORHSURF_PYTHON        interpreter for the AmbiSuR subprocesses (default: the current one)
  ORHSURF_DATA_ROOT     where clips live
  ORHSURF_OUT_ROOT      where outputs go
  ORHSURF_CACHE_DIR     HF/torch caches (default: <repo>/cache, override on a cluster to scratch)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

#: repo root = the directory containing the `orhsurf` package
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(var: str) -> Path | None:
    v = os.environ.get(var)
    return Path(v).expanduser().resolve() if v else None


def colmap_bin() -> Path:
    """COLMAP executable. Needed for `database_creator`, `image_undistorter`, `model_converter`."""
    p = _env_path("ORHSURF_COLMAP_BIN")
    if p:
        return p
    local = REPO_ROOT / "env/bin/colmap"          # what install.sh creates
    if local.is_file():
        return local
    found = shutil.which("colmap")
    if found:
        return Path(found)
    return local                                   # best guess; `doctor` will report it missing


def ambisur_repo() -> Path:
    return _env_path("ORHSURF_AMBISUR_REPO") or (REPO_ROOT / "third_party/AmbiSuR")


def python_bin() -> Path:
    """Interpreter for AmbiSuR subprocesses. Defaults to the one running us, which is correct
    inside an activated env and inside a Slurm job that activated it in the prologue."""
    return _env_path("ORHSURF_PYTHON") or Path(sys.executable)


def cache_dir() -> Path:
    """HF / torch / triton caches.

    On a cluster point this at scratch: the DA3 checkpoint alone is 6.76 GB and $HOME quotas are
    routinely 10-50 GB.  See docs/INSTALL_SLURM.md.
    """
    return _env_path("ORHSURF_CACHE_DIR") or (REPO_ROOT / "cache")


def data_root() -> Path:
    return _env_path("ORHSURF_DATA_ROOT") or (REPO_ROOT / "data")


def out_root() -> Path:
    return _env_path("ORHSURF_OUT_ROOT") or (REPO_ROOT / "out")


def vendor_dir() -> Path:
    return Path(__file__).resolve().parent / "_vendor"


def subprocess_env(gpu: int | None = None, cpus: int | None = None) -> dict:
    """Environment for the AmbiSuR train/export subprocesses.

    Sets every cache under `cache_dir()` so a cluster run never writes to $HOME by accident, and
    sizes the thread pools from our allocation rather than from the node.
    """
    from . import alloc
    c = cache_dir()
    n = cpus or alloc.cpu_count()
    env = dict(os.environ)
    env.update(
        HF_HOME=str(c / "hf"),
        TORCH_HOME=str(c / "torch"),
        TORCH_EXTENSIONS_DIR=str(c / "torch_ext"),
        TRITON_CACHE_DIR=str(c / "triton"),
        PYTHONPATH=os.pathsep.join(filter(None, [
            str(ambisur_repo()), str(vendor_dir()), str(REPO_ROOT),
            os.environ.get("PYTHONPATH", "")])),
        OMP_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n),
        NUMEXPR_NUM_THREADS=str(n), VECLIB_MAXIMUM_THREADS=str(n),
        OPENCV_FOR_THREADS_NUM=str(n),
        # The source project gates an expensive full-resolution NCC debug dump behind this; it must
        # stay off or -r 1 exports blow past 24 GB.
        AMBISUR_FULL_NCC_DEBUG="0",
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for sub in ("hf", "torch", "torch_ext", "triton"):
        (c / sub).mkdir(parents=True, exist_ok=True)
    return env


def da3_python() -> Path:
    """Interpreter for the DA3 stage.

    IT IS A SEPARATE ENV ON PURPOSE, and this is measured, not stylistic.  Running the identical
    1008 pose-conditioned inference (same checkpoint, same 8-view group, same images) under
    torch 2.6.0+cu124 and under torch 2.7.1+cu128 gives materially different depths:

        same env, run twice    mean |d| 1.3e-06 m   p99.9 4.8e-07 m   median rel 0.0
        2.6.0+cu124 vs 2.7.1   mean |d| 1.2e-03 m   p99.9 3.5e-02 m   max 3.43 m

    i.e. the cross-env difference is ~900x the run-to-run nondeterminism at the mean and ~7e4x at
    p99.9.  The reference reconstruction was produced under torch 2.6.0+cu124, so that is what the
    DA3 stage pins.  Collapsing to one env would silently change the prior.
    (tools/da3_env_compare.py reproduces this.)
    """
    return _env_path("ORHSURF_DA3_PYTHON") or (REPO_ROOT / "env-da3/bin/python")
