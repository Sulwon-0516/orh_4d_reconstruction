# Compute-node installation and Slurm deployment

Start with the [README TL;DR](../README.md#tldr--process-complete-clips-with-one-command).
An existing one-GPU Slurm allocation has been exercised end to end on A100 80 GB PCIe.
The array and multi-GPU `slurm/*.sbatch` templates remain **unvalidated on this cluster**.

## Allocation

If already inside an allocated compute shell, reuse it. Otherwise, on the tested cluster:

```bash
srun --partition=debug --gres=gpu:1 --time=03:00:00 --pty bash
```

Use your site's actual partition, account/project and time limit. Do not copy historical job IDs.
Do not invent CPU/memory requests: the program derives its thread budget from the actual
allocation. The tested debug limit is three hours; use a permitted longer allocation or resume
manually later for a full clip. Do not loop job submissions to circumvent limits.

Heavy installation, extraction, preprocessing and reconstruction belong on compute nodes.
Data can be downloaded in an active GPU allocation or pre-staged according to local policy;
no separate CPU-partition submission is required by this package.

## Install once

Prerequisites: conda/mamba/micromamba on PATH, CUDA Toolkit (`nvcc`) and a supported C++ compiler.
Use your site's actual module names; no universal `module load cuda/...` command is assumed.

```bash
./install.sh --no-weights
source env.sh
orhsurf process --clips C001 --gpus 1  # weights, full clip, reconstruction and verification
```

The installer creates `env/` (torch 2.7.1+cu128) and `env-da3/` (torch 2.6.0+cu124), builds the
extensions, then imports `scene.cameras`. It no longer requires that import before the extensions
exist. CUDA Toolkit 12.4 and GCC 11.4 were exercised successfully with the tested driver/build;
use actual CUDA smoke tests to establish compatibility on another machine.

`source env.sh` exports absolute interpreter, COLMAP and cache paths, and adds `bin/orhsurf` to
PATH. It does **not** activate bare `python`/`pip`; use `"$ORHSURF_PYTHON"` for auxiliary scripts.
If conda lives outside PATH, add your existing conda installation's `bin` directory first.
Environment creation, wheel downloads and extension builds can take substantial time and disk.

## Memory and threads

C001 with default all-foreground masks, DA3 1008 and groups of 18 peaked at **24,745 MiB**.
This is an A100 measurement, not a newly imposed 40 GB minimum. Earlier RTX 4090 runs recorded
23,353 MiB allocated and succeeded with the expandable allocator. The free-memory guard remains
23,450 MiB. Current C001/RTX 3090 compatibility is **unverified**; memory can differ across inputs
and backends, and the cause of the measured increase is not established. 40 GB+ provides headroom;
it is not required by a new code check.
Reducing group size changes the depth prior and has not been established as equivalent output.

`--cpus-per-job N` is a program thread limit, not a Slurm allocation request. If running other
CPU work alongside reconstruction, keep the combined budgets within the allocation. If omitted,
the pipeline derives a per-worker budget from Slurm. Do not use node-wide `nproc` for sizing jobs.

## Run, resume and batch use

```bash
source env.sh
orhsurf process --clips C001 C002 --gpus 1  # sequential whole clips; reuse completed results
```

The same command reuses completed matching frames. Keep the same manifest, recipe and output
root. Changing the manifest path/size/mtime changes the resume fingerprint. Failed frame scratch
is retained for diagnosis; do not delete active work directories.

For a batch job, place those commands after `set -euo pipefail` in a shell script that first
changes into the checkout. Submit with your site's verified partition/account/time options and
one GPU. `slurm/recon_array.sbatch` and `slurm/recon_singlenode.sbatch` contain explicit multi-GPU,
CPU, memory, partition and module assumptions; they are examples, not the tested quick-start.
Create their `slurm_logs` directory before submission if using those templates.
For one different clip per array task, use `slurm/process_clips.sbatch` as shown in README.
That script has no CPU/memory, module or project defaults; it uses `MODEL_OUTPUT_DIR` as its output
root when set. This cluster enforces WCKey and MODEL_OUTPUT_DIR at submission; supply actual
approved values. `--test-only` does not create a job but still checks those site requirements.

## Paths and shared storage

Defaults are checkout-local `data/`, `cache/` and `out/`. Choose `ORHSURF_CACHE_DIR` before
installation; data/output roots can be overridden before sourcing `env.sh`. Explicit `--out`
and `--work` make output and scratch placement clear. Do not move an environment or rewrite an
active manifest. Relative image/mask paths resolve against the manifest directory.

Use archive/package sizes and known output-file metadata for estimates. Avoid broad `find`,
`du`, `ls -R`, repeated `df` and automatic monitoring over shared storage. A full clip's estimated
capacity budget is in [README.md](../README.md#time-and-storage-budget); it is not a quota check.

## If the torch download fails with a DNS error

`download.pytorch.org` may redirect to a CDN inaccessible through the site's network route.
Inspect the failing URL once; use an approved proxy/mirror or pre-downloaded pinned wheels.
Do not silently change torch versions: the DA3 environment version affects predicted depths.
See [detail.md](../detail.md) for the numerical comparison and the known numpy metadata conflict.

Other common build failures:

| Failure | Check |
|---|---|
| Undefined symbols importing extension | Match the build and runtime torch environment |
| Unsupported GNU compiler | Use a compiler supported by the actual CUDA Toolkit |
| No kernel image for device | Rebuild for the allocated GPU architecture |
| Compiler killed | Check allocated memory and reduce build parallelism |
| Frame not decoded | Use a run range contained in manifest `decoded_frames` |

For interactive viewing, bind to `127.0.0.1` and use the README's SSH jump-host tunnel.
