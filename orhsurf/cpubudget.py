"""ONE CPU budget, resolved once, applied before any numeric import, threaded through every stage.

WHY THIS MODULE EXISTS.  The budget used to leak in five independent places, so a user had no
single knob and the Slurm allocation was routinely oversubscribed:

  * third_party/AmbiSuR/train.py:582   hardcoded torch.set_num_threads(8)
  * orhsurf/_vendor/export_surface.py  hardcoded cKDTree(...).query(workers=8)
  * orhsurf/cli.py                     every worker re-applied the WHOLE allocation in main()
                                       before parsing its own, smaller --cpus
  * orhsurf/atomicio.py                imported numpy at module scope, so OMP_NUM_THREADS was set
                                       after the library had already bound its pool
  * orhsurf/_vendor/colmap_dataset.py  COLMAP got no limit at all

THE "BEFORE IMPORTS" PART IS THE WHOLE POINT.  OpenMP, MKL and OpenBLAS read their environment
once, when the library first initialises.  Setting OMP_NUM_THREADS after `import numpy` changes a
string in os.environ and nothing else.  A budget that arrives after the library has bound is
exactly as useless as no budget -- the same class of bug cost this project 42 idle cores for 86% of
a filtering job's runtime.  So `apply()` must run before numpy/torch/cv2 are imported anywhere in
the process, and this module imports nothing numeric itself.

Resolution order for the per-job budget:
    explicit --cpus-per-job
    -> ORHSURF_CPUS_PER_JOB
    -> allocation // concurrent_jobs        (allocation from Slurm, never nproc)
and it is asserted >= 1 rather than silently floored.
"""
from __future__ import annotations

import os

#: Every threading knob that matters for this pipeline, set together so none is left unbounded.
THREAD_VARS = (
    "OMP_NUM_THREADS",          # AmbiSuR / torch CPU ops, scipy
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",   # cv2.setNumThreads equivalent
    "RAYON_NUM_THREADS",        # safetensors / tokenizers
)

#: The one resolved value, so every stage can read back what actually applied.
_RESOLVED: int | None = None

ENV_KEY = "ORHSURF_CPUS_PER_JOB"


def allocation_cpus() -> int:
    """CPUs this PROCESS may use. Never nproc: under Slurm that is the node, not the allocation."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    try:
        n = len(os.sched_getaffinity(0))     # respects a cpuset cgroup, unlike os.cpu_count()
        if n > 0:
            return n
    except AttributeError:
        pass
    return os.cpu_count() or 1


def resolve(explicit: int | None = None, concurrent_jobs: int = 1) -> int:
    """Compute the per-job budget. Asserts >= 1 and stops rather than silently falling back."""
    if explicit is not None:
        n = int(explicit)
        src = "--cpus-per-job"
    elif os.environ.get(ENV_KEY, "").isdigit():
        n = int(os.environ[ENV_KEY])
        src = ENV_KEY
    else:
        total = allocation_cpus()
        n = total // max(1, concurrent_jobs)
        src = f"allocation({total}) // jobs({concurrent_jobs})"
    if n < 1:
        raise SystemExit(
            f"CPU budget resolved to {n} from {src}, which is not runnable.\n"
            f"  Give fewer concurrent jobs (--gpus), or set --cpus-per-job / {ENV_KEY} explicitly.\n"
            f"  Refusing to silently round up to 1 and oversubscribe the allocation.")
    return n


def apply(n: int) -> int:
    """Export the budget into every threading variable. MUST run before numpy/torch/cv2 import."""
    global _RESOLVED
    if n < 1:
        raise SystemExit(f"refusing to apply a CPU budget of {n}")
    for var in THREAD_VARS:
        os.environ[var] = str(n)
    os.environ[ENV_KEY] = str(n)        # so child processes inherit the same single value
    _RESOLVED = n
    return n


def resolved() -> int:
    """The budget that actually applied in this process."""
    if _RESOLVED is not None:
        return _RESOLVED
    if os.environ.get(ENV_KEY, "").isdigit():
        return int(os.environ[ENV_KEY])
    return allocation_cpus()


def bind_torch(n: int | None = None) -> int:
    """Apply the budget to torch explicitly. Call AFTER torch is imported; env alone is advisory."""
    n = n or resolved()
    try:
        import torch
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, min(n, 4)))
    except Exception:
        pass            # interop threads can only be set once per process; not fatal
    try:
        import cv2
        cv2.setNumThreads(n)
    except Exception:
        pass
    return n


def describe() -> dict:
    """Recorded into doctor output and every frame's provenance, so the user can see what applied."""
    return dict(cpus_per_job=resolved(), allocation_cpus=allocation_cpus(),
                source=("explicit/env" if os.environ.get(ENV_KEY) else "allocation"),
                thread_vars={v: os.environ.get(v) for v in THREAD_VARS},
                slurm_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
