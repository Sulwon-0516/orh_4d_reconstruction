# Installing and running orhsurf on a Slurm cluster (no root)

> ## ⚠️ The Slurm-specific parts of this document are UNTESTED
>
> There is no Slurm on the machine this package was developed on. Everything about the *pipeline*
> below was executed and measured; everything about `sbatch`, `srun`, `module load` and cgroup
> behaviour is written from documented behaviour and is **not verified here**.
>
> **Check these before your first real submission:**
> 1. The four `<<< CHECK` lines in `slurm/recon_array.sbatch` (`--gres`, `--cpus-per-task`,
>    `--mem`, `--time`, `--partition`) against `sinfo -o "%P %G %c %m %l"`.
> 2. Your CUDA module name/version (`module avail cuda`). The pinned torch is `cu128`.
> 3. Whether your site wants `--gres=gpu:1` or `--gpus-per-task=1`.
> 4. That `echo $CUDA_VISIBLE_DEVICES` inside a 1-GPU job prints a single index.
>
> Run `orhsurf doctor` inside an interactive allocation first. It prints what it actually found.

---

## 0. The one hard requirement: a GPU with ≥ 24 GB

The DA3 prior stage runs 18 views at 1008 px in one batch and **peaks at 24,090 MiB**.

This is tight enough to matter: on an RTX 4090 (23.53 GiB usable) it OOMs with the default CUDA
allocator even on a completely idle card, because fragmentation strands ~1.9 GiB. The package sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for that stage, which is what makes it fit.

| GPU | Works? |
|---|---|
| A100 80 GB / H100 80 GB / A6000 48 GB | comfortable |
| RTX 4090, L40S, A100 40 GB (24–40 GB) | fits, with the allocator setting above |
| 16 GB or less (V100-16, T4, RTX 4080) | **does not fit at the default `--group-size 18`** |

On a smaller GPU, lower `--group-size`. **This changes the output**: DA3 predicts jointly over the
whole group, so fewer views means less cross-view context and a different prior. It is a trade, not
a free knob. Measured: 8 views ≈ 21.0 GB, 4 views ≈ 14.4 GB.

Training and export are not the constraint (~10.3 GB and ~7.8 GB at `-r 2`).

---

## 1. Environment setup without root

**Use micromamba.** Not `uv`, not a bare `venv` — and the reason is specific to this dependency
set: it needs a pinned Python (3.10) *and* a COLMAP binary (`database_creator`,
`image_undistorter`, `model_converter` are all used by the dataset stage). conda-forge ships
COLMAP; PyPI does not. Without conda your only no-root route to COLMAP is building it from source
with Ceres, Eigen, FreeImage and Qt, which is a far worse afternoon than this one.

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C "$HOME" bin/micromamba
export PATH="$HOME/bin:$PATH"          # add to ~/.bashrc
```

Then:

```bash
git clone <this repo> orhsurf && cd orhsurf
./install.sh --no-weights              # on the login node: envs + deps, no compile, no download
```

`install.sh` is non-interactive and re-runnable; it skips anything already present.

### Why two environments

Not a style choice. Running the *identical* DA3 1008 inference under the two torch builds gives
materially different depths:

| | mean \|Δdepth\| | p99.9 | max |
|---|---|---|---|
| same env, run twice (GPU nondeterminism) | 1.3e-06 m | 4.8e-07 m | 0.019 m |
| torch 2.6.0+cu124 vs 2.7.1+cu128 | **1.2e-03 m** | **3.5e-02 m** | 3.43 m |

That is ~900× the run-to-run noise at the mean. The reference reconstruction was produced on
2.6.0+cu124, so the DA3 stage is pinned there (`env-da3/`) and AmbiSuR keeps 2.7.1+cu128 (`env/`).
Collapsing them would silently change the prior. Reproduce with `tools/da3_env_compare.py`.

---

## 2. CUDA extensions — build on a COMPUTE node

Two extensions compile at install time: `diff-plane-rasterization-ambisur` and `simple-knn`. These
are the most likely thing to fail on an unfamiliar cluster.

```bash
module load cuda/12.8        # must match the torch build (cu128)
srun --gres=gpu:1 --cpus-per-task=8 --time=1:00:00 --pty \
     ./install.sh --extensions-only
```

**Build on a compute node, not the login node**, because:
- `torch.utils.cpp_extension` compiles for the **visible GPU's** architecture. A login node with no
  GPU produces binaries the compute nodes may refuse to load. If you must build on the login node,
  set `TORCH_CUDA_ARCH_LIST` explicitly (`8.9` = RTX 4090/L40S, `8.0` = A100, `9.0` = H100).
- Login nodes have strict CPU/RAM limits; `nvcc` gets killed mid-compile and leaves a confusing
  half-built package.

Requirements: nvcc 12.x matching your torch CUDA, and a host compiler nvcc accepts
(**gcc ≤ 12**; gcc 13+ is commonly rejected). If your default gcc is too new:

```bash
module load gcc/11.4.0
export CC=$(which gcc) CXX=$(which g++)
```

Build parallelism comes from `SLURM_CPUS_PER_TASK`, never `nproc` — see §5.

Symptom → cause:

| Symptom | Cause |
|---|---|
| `undefined symbol: _ZN3c10...` on import | extension built against a different torch/CUDA than the one loaded |
| `unsupported GNU version` | gcc too new; `module load gcc/11` |
| `no kernel image is available for execution` | built for the wrong arch; rebuild on a compute node |
| compile killed, no error | login-node memory limit; build under `srun` |

**PyTorch3D is not required.** Upstream AmbiSuR imported it for exactly one function;
`orhsurf/quat.py` replaces it, proven equal to 1.1e-15 in `tests/test_quat.py`. This removes the
single worst build from the install. `ORHSURF_USE_PYTORCH3D=1` restores the original if needed.

---

## 3. Weights and data — pre-stage on the login node

Compute nodes are frequently offline. Fetch everything first, then run with `HF_HUB_OFFLINE=1`.

```bash
source env.sh
orhsurf fetch --weights                # DA3 checkpoint, 6.76 GB
orhsurf fetch --clip C001              # one clip
```

| What | Size |
|---|---|
| DA3NESTED-GIANT-LARGE-1.1 (`model.safetensors`) | **6,759,558,100 B ≈ 6.76 GB** |
| one clip archive (`data/C0NN.tar`) | ≈ 4.3 GB |
| whole published dataset | ≈ 77 GiB (17 JPEG clips + 4 HEVC) |

Pin the revision, do not track `main`:
`Sulwon/anonymous_dataset_lih_orh_0920`, `repo_type="dataset"`,
revision `535c72f84dc124cfed76fb1be2c17e136fb331c3`.

Verify offline mode actually works before you submit:

```bash
HF_HUB_OFFLINE=1 orhsurf doctor        # must still report the checkpoint as cached
```

---

## 4. Scratch vs home — where to put things

$HOME quotas (10–50 GB) cannot hold this. Put the working directory and every cache on scratch:

```bash
export ORHSURF_CACHE_DIR=$SCRATCH/orhsurf/cache    # HF + torch + triton + extension build
export ORHSURF_DATA_ROOT=$SCRATCH/orhsurf/data
export ORHSURF_OUT_ROOT=$SCRATCH/orhsurf/out
```

Disk per clip:

| | per frame | 150 frames |
|---|---|---|
| `surface.npz` (filtered, shipped) | ~620 MB | **~93 GB** |
| `surface.ply` (opt-in, `--write-ply`) | ~688 MB | ~103 GB |
| per-frame scratch (`_work`, auto-deleted) | ~6 GB peak | — |

Leave the `.ply` off unless you need it in MeshLab; nothing in the pipeline reads it back.

If your scratch is purged on a timer, note that `_work` is deleted per frame automatically but the
outputs are not — copy them off before the purge window.

---

## 5. Allocation awareness — the failure mode to avoid

**Never use `nproc`, `os.cpu_count()`, `getconf _NPROCESSORS_ONLN` or `nvidia-smi` to size anything
inside a Slurm job.** They report the whole node, not your cgroup. An 8-CPU task on a 64-core node
that reads `nproc` starts 64 BLAS threads inside an 8-CPU cgroup and runs several times slower.

`orhsurf/alloc.py` reads, in order: `SLURM_CPUS_PER_TASK` → `SLURM_CPUS_ON_NODE` →
`os.sched_getaffinity(0)` (which *does* respect a cgroup) → `os.cpu_count()`. GPUs come from
`CUDA_VISIBLE_DEVICES` → `SLURM_GPUS_ON_NODE` → `SLURM_GPUS`, and under Slurm it will **not** fall
back to probing.

**No `taskset`.** Inside a cpuset cgroup `taskset` can only narrow an already-narrow mask, and
fails outright if the mask does not intersect the cgroup. Let `srun --cpu-bind=cores` bind; the
package only sizes thread pools. (The reference implementation this replaces used
`getconf _NPROCESSORS_ONLN` plus `taskset`, and a sibling script hard-asserted a 64-CPU node —
both are exactly the bug described here.)

---

## 6. Submitting

**Job array (recommended):**

```bash
sbatch --array=0-7 slurm/recon_array.sbatch C001 0-149
```

Each task takes a contiguous in-order slice of the frame list and one GPU. Rationale in the header
of that file. Re-submitting the identical command after a preemption or timeout resumes: completed
frames are skipped via their `_DONE.json` marker.

**Single node, N GPUs** (for a node you own outright):

```bash
sbatch slurm/recon_singlenode.sbatch C001 0-149 8
```

**One job per GPU, always.** Packing two AmbiSuR trainings onto one GPU was measured twice and lost
both times (451 s solo vs 1705 s each at 7k), and the DA3 stage OOMs at two per GPU regardless of
`-r`.

Timing: ~11 min/frame at `-r 2` × 7k iterations. 150 frames ÷ 8 tasks ≈ **3.5 h**.

---

## 7. Smoke test — the smallest thing that proves the install

```bash
source env.sh
orhsurf doctor                                    # every component, one report
orhsurf run --clip C001 --frames 0-0 --gpus 1     # ONE frame, end to end (~11 min)
orhsurf verify --clip C001                        # opens every array
```

Success looks like:
- `out/C001/00000/surface.npz` — tens of millions of points
- `out/C001/00000/_DONE.json` — written last; its absence means the frame is not finished
- `out/C001/_check/check_00000.png` — a 2×3 contact sheet (two viewpoints × shaded/normal/rgb),
  produced on **every** run

Look at the PNG. Two viewpoints with a real baseline, because a single view hides error along its
own optical axis — the cheapest way to be fooled by a broken reconstruction.

Under Slurm, run the smoke test inside an allocation:

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=1:00:00 --pty \
     bash -c 'source env.sh && orhsurf run --clip C001 --frames 0-0 --gpus 1'
```

---

## 8. Interruptions

Preemption and timeouts are normal; the package is built for them.

- Outputs are written to a temp name, fsynced, then renamed. A killed job cannot leave a
  half-written `surface.npz` in place.
- A frame is built in a staging directory and swapped in only when complete, so a crashed **re-run
  cannot destroy a good frame**. (This is not hypothetical: the source project has two truncated
  494 MB `surface.npz` files whose zip magic bytes are intact, where a re-run overwrote a good
  output and died. A cheap header check passes; the failure surfaces as `BadZipFile` inside a
  render, hours later.)
- Resume checks `_DONE.json`, not file existence.
- **After any interruption, run `orhsurf verify`.** It opens and decompresses every array and
  checks dtypes, ranks and mutual lengths. `--shallow` checks markers only.

```bash
orhsurf verify --clip C001            # exit 0 = all good; non-zero lists each problem
```
