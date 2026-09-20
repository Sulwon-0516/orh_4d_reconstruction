"""Resource allocation that believes the scheduler, not the machine.

THE RULE: under Slurm, never ask the machine what it has.  `nproc`, `os.cpu_count()`,
`getconf _NPROCESSORS_ONLN` and `nvidia-smi -L` all report the whole NODE, not this job's
allocation.  A 4-CPU task on a 64-core node that reads `nproc` will start 64 BLAS threads inside a
4-CPU cgroup and run ~8x slower than if it had started 4.  That is exactly the defect in the
reference implementation this package replaces (handoff_ambisur/bin/cpu_pin.sh:17 reads
`getconf _NPROCESSORS_ONLN`, and bin/lane_exec:6 asserts the node has exactly 64 CPUs).

Precedence, highest first:
  CPUs  SLURM_CPUS_PER_TASK -> SLURM_CPUS_ON_NODE -> len(os.sched_getaffinity(0)) -> os.cpu_count()
  GPUs  CUDA_VISIBLE_DEVICES -> SLURM_GPUS_ON_NODE -> SLURM_GPUS -> torch/nvidia-smi probe

`os.sched_getaffinity(0)` is the correct non-Slurm fallback: unlike `os.cpu_count()` it *does*
respect a cpuset cgroup, so it is also right inside a plain Docker container with `--cpuset-cpus`.

CPU PINNING: we do not call taskset.  Inside a Slurm cpuset cgroup taskset can only ever narrow an
already-narrow mask, and if the mask it is given does not intersect the cgroup it fails outright.
Binding is Slurm's job (`srun --cpu-bind=cores`); ours is only to size the thread pools so we do
not oversubscribe whatever we were given.
"""
from __future__ import annotations

import os
import subprocess

# Every threading knob that matters for this pipeline.  numpy/scipy pick up the BLAS ones, OpenCV
# needs its own, and our k-NN gate passes `workers=` explicitly from cpu_count().
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENCV_FOR_THREADS_NUM")


def under_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


def cpu_count() -> int:
    """CPUs this process may actually use."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n
    except AttributeError:          # not Linux
        pass
    return os.cpu_count() or 1


def apply_thread_limits(n: int | None = None) -> int:
    """Pin every threading library to `n` (default: our CPU allocation).

    Must run BEFORE numpy/torch are imported to be fully effective -- OpenMP reads its environment
    once, at first use.  cli.py calls this at the top of main() for that reason.
    """
    n = n or cpu_count()
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    try:                            # harmless if torch is not imported yet
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    return n


def visible_gpus() -> list[int]:
    """GPU indices usable by this process, as indices into CUDA's own view.

    CUDA_VISIBLE_DEVICES remaps devices: with CUDA_VISIBLE_DEVICES=3,5 the process sees exactly two
    devices, addressed as cuda:0 and cuda:1.  So when it is set we return range(len(entries)), NOT
    the physical ids -- handing a physical id to torch would address the wrong device or crash.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        entries = [e for e in cvd.split(",") if e.strip() != ""]
        if entries and all(e.strip() == "-1" for e in entries):
            return []
        return list(range(len(entries)))
    # SLURM_GPUS_ON_NODE is a COUNT; SLURM_JOB_GPUS is a device-ID LIST. Conflating them made
    # SLURM_JOB_GPUS="3" mean "three GPUs" instead of "device 3".
    v = os.environ.get("SLURM_GPUS_ON_NODE")
    if v and v.isdigit() and int(v) > 0:
        return list(range(int(v)))
    for var in ("SLURM_JOB_GPUS", "SLURM_GPUS"):
        v = os.environ.get(var)
        if v and v.strip():
            return list(range(len([e for e in v.split(",") if e.strip()])))
    if under_slurm():
        # Under Slurm with no GPU variable set at all, we were probably given no GPUs.  Probing
        # nvidia-smi here would report the whole node and oversubscribe someone else's job.
        return []
    return _probe_gpus()


def _probe_gpus() -> list[int]:
    """Last resort, NON-SLURM ONLY: ask the machine.  See the module docstring for why."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return list(range(len([l for l in out.stdout.splitlines() if l.startswith("GPU ")])))
    except Exception:
        pass
    return []


def resolve_gpus(requested: int | None) -> list[int]:
    """Final GPU list for this run.

    `requested` is the user's --gpus count.  We never hand back more than the allocation exposes;
    asking for 8 inside a 2-GPU allocation is a user error we correct downward loudly rather than
    honour, because honouring it would mean addressing devices we do not own.
    """
    avail = visible_gpus()
    if not avail:
        raise SystemExit(
            "no GPUs available.\n"
            "  Under Slurm: request them, e.g. `--gres=gpu:1`, and check CUDA_VISIBLE_DEVICES "
            "is set inside the job.\n"
            "  Outside Slurm: check `nvidia-smi` works and CUDA_VISIBLE_DEVICES is not set to ''.")
    if requested is None:
        return avail
    if requested < 1:
        raise SystemExit(f"--gpus must be >= 1, got {requested}")
    if requested > len(avail):
        print(f"[alloc] WARNING: --gpus {requested} requested but the allocation exposes "
              f"{len(avail)} ({avail}); using {len(avail)}.", flush=True)
        return avail
    return avail[:requested]


def physical_gpu_ids() -> list[int]:
    """The ABSOLUTE device ids behind our logical indices.

    CUDA_VISIBLE_DEVICES is interpreted absolutely by every process independently, so a child that
    should use our logical GPU 1 must be given the PHYSICAL id, not "1".  With
    CUDA_VISIBLE_DEVICES=6,7 our logical [0,1] are physical [6,7]; handing a child "1" would point
    it at physical GPU 1 -- someone else's job.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        out = []
        for e in cvd.split(","):
            e = e.strip()
            if e == "":
                continue
            out.append(e)          # keep as a STRING: GPU-<uuid> is legal and must not be int()ed
        return out
    return [str(i) for i in visible_gpus()]


def contiguous_slices(items: list, n_parts: int) -> list[list]:
    """Partition `items` into `n_parts` CONTIGUOUS slices, preserving order.

    Contiguous, not round-robin: the spec is that each GPU works an in-order slice of the frame
    list.  Leftovers go to the earliest parts, so sizes differ by at most one.
    """
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1, got {n_parts}")
    n, out, start = len(items), [], 0
    base, extra = divmod(n, n_parts)
    for i in range(n_parts):
        size = base + (1 if i < extra else 0)
        out.append(items[start:start + size])
        start += size
    assert start == n and sum(len(p) for p in out) == n, (start, n)
    return out


def describe() -> dict:
    """What we decided and why -- recorded into every run's provenance."""
    return dict(
        under_slurm=under_slurm(),
        cpu_count=cpu_count(),
        cpu_source=("SLURM_CPUS_PER_TASK" if os.environ.get("SLURM_CPUS_PER_TASK")
                    else "SLURM_CPUS_ON_NODE" if os.environ.get("SLURM_CPUS_ON_NODE")
                    else "sched_getaffinity"),
        visible_gpus=visible_gpus(),
        gpu_source=("CUDA_VISIBLE_DEVICES" if os.environ.get("CUDA_VISIBLE_DEVICES")
                    else "SLURM_GPUS_ON_NODE" if os.environ.get("SLURM_GPUS_ON_NODE")
                    else "probe"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        slurm_array_task_count=os.environ.get("SLURM_ARRAY_TASK_COUNT"),
        nodelist=os.environ.get("SLURM_JOB_NODELIST"),
    )
