# Implementation details and historical measurements

Start with [README.md](README.md) for the supported first-run commands and result format.
This page holds background, prior experiments and troubleshooting; historical numbers below
are **not** the current C001/A100 deployment benchmark.

## Current validation (2026-09-21)

The compute-node installation and an existing Slurm allocation were exercised with one A100
80 GB PCIe, 20 allocated CPUs, and 150,794 MiB allocated host memory. Reconstruction used eight
CPU threads; that is a program limit, not a new Slurm resource request. Both Python 3.10
environments, CUDA extension operations, full `doctor`, DA3 inference, 7,000-iteration training,
export, deep verification and the first static preview passed for C001 frame 0.
Frame 1 also completed (25,458,442 points); subsequent frames are being processed; a complete 225-frame run is **not yet validated**.
The generic job-array/multi-GPU templates have not been validated on this cluster.

Main env: torch 2.7.1+cu128; DA3 env: torch 2.6.0+cu124 with numpy 2.2.6.
CUDA extensions built with Toolkit 12.4 and GCC 11.4; actual GPU operations passed.
DA3's upstream numpy<2 package metadata conflicts with this numpy pin: runtime tests passed,
but this does not mean `pip check` is clean.

Measured frame-0 stage seconds: prep 3.03, dataset build 31.80, scene build 14.86,
DA3 147.75, rewarp 14.06, training 517.18, export 125.28. End-to-end: approximately 14.3 minutes.
DA3 used 1008 px, groups of 18 with overlap 6, four groups, peak 24,745 MiB allocated.
This A100 allocated-memory peak exceeds 24 GiB; the earlier 23,353 MiB result below belongs to
a successful RTX 4090 run. This is not evidence of a new 40 GB algorithmic minimum or a direct
RTX 3090 test. The runtime guard remains 23,450 MiB free and uses expandable allocator segments.
The cause of the difference is unresolved; mask/group ordering, input shape and backend differences
must be compared rather than assuming a cause. 40 GB+ is only a headroom recommendation.

## Installation and paths

Use conda/mamba/micromamba on PATH, CUDA nvcc and a supported host compiler on a compute node.
`./install.sh --no-weights` creates `env/`, `env-da3/`, compiles extensions, then checks imports;
`orhsurf fetch --weights` downloads the 6.76 GB checkpoint. `source env.sh` selects the absolute
interpreters and cache paths; it does not activate bare `python` or `pip`.
Use `"$ORHSURF_PYTHON"` for auxiliary tools and optional packages.

The fresh-install bug was an early `scene.cameras` import before CUDA extensions existed.
The check now runs after extension installation, followed by real CUDA operations.
Build, decoder, filter and encoder thread counts are bounded by the allocation.

Choose `ORHSURF_CACHE_DIR` before installation and `ORHSURF_DATA_ROOT` / `ORHSURF_OUT_ROOT`
before sourcing `env.sh` when overriding defaults. Prefer explicit `--out` and `--work`.
Relative image and mask paths resolve against the manifest directory. Old absolute paths
must be remapped explicitly into a new manifest; originals are preserved.
Do not expand or rewrite a prepared manifest while a run uses it: resume fingerprints include
its path, size and modification time. Prepare a new directory and use disjoint frame ranges
for continuation if an active subset must stay untouched.

## Input conversion and masks

`fetch --convert --clip C001 --frames 0-4` selects `hevc/C001.tar`, extracts into
`data/_hevc/C001`, decodes selected frames into `data/C001_prepared`, and writes a local manifest.
The plain `fetch --clip C001` path prefers the JPEG archive; it is not the documented HEVC
conversion workflow. The original JPEG directory is not overwritten by HEVC extraction.

The published clip has 47 valid cameras and 225 encoded frames (15 seconds at 15 fps).
Decode indices are `encoded_frame_index`, never the source video's `video_frame_index`.
The converter checks frame counts and preserves nonzero/sparse frame indices.
The CLI rejects requests outside `decoded_frames` before GPU dispatch.

Without `--masks`, the converter generates all-foreground RGBA masks and records that policy.
With `--masks`, missing supplied masks are errors. Masks affect visual-hull camera grouping;
they do not make the reconstructed point cloud foreground-only. See [the input contract](docs/DATA_CONTRACT.md).

An extracted clip can be converted without downloading again:

```python
from pathlib import Path
from orhsurf.fetch import convert_clip
assert convert_clip(Path("data/_hevc/C001"), Path("data/C001_full_prepared"), frames="0-224") == 0
```

## Recipe and numerical background

### Default recipe

| | value | note |
|---|---|---|
| resolution | `-r 2` | half res |
| DA3 prior | **1008**, our calibrated poses passed in | groups of 18, overlap 6 |
| training | AmbiSuR, **7000** iterations, warmups at 0.4× | |
| export filter | `--min-views 2 --nn-k 5 --nn-max-mm 5.0` | a point is dropped if its 5th-nearest neighbour is > 5 mm away |

Every one of these is passed **explicitly at every call site**, never inherited from an argparse
default. That is deliberate: in the source project, changing the exporter's defaults silently
altered the output of an unrelated analysis script.

Measured on frame 37: the filter takes **31,184,126 → 25,475,445** points (4,145,365 dropped for
support < 2, a further 1,563,316 as isolated).

### An honest note on 1008 vs 504

The 1008 prior is the default because the user judged the 3D reconstructions better by eye. The
aggregate metrics do **not** separate the two: on frame 37, the only frame where both were run,
PSNR was 23.492 (1008) vs 23.538 (504), with point counts and hole fractions within noise. PSNR is
a poor instrument for the defect that matters here — floaters are too few points to move it — so
this is recorded as unresolved rather than as evidence either way. The 504 path remains available
as a documented fallback for GPUs under 24 GB.

### Why two Python environments

Measured, not stylistic. The identical DA3 inference under the two torch builds differs by ~900×
the run-to-run nondeterminism:

| | mean \|Δdepth\| | p99.9 |
|---|---|---|
| same env, twice | 1.3e-06 m | 4.8e-07 m |
| torch 2.6.0+cu124 vs 2.7.1+cu128 | 1.2e-03 m | 3.5e-02 m |

The reference reconstruction was produced on 2.6.0+cu124, so the DA3 stage is pinned there
(`env-da3/`) while AmbiSuR keeps 2.7.1+cu128 (`env/`). Reproduce with `tools/da3_env_compare.py`.

**PyTorch3D is not needed.** AmbiSuR imported it for exactly one function; `orhsurf/quat.py`
replaces it. `tests/test_quat.py` always checks orthonormality and known rotations, and compares
against the real PyTorch3D **only when it is importable** — on a machine where it was, the max
absolute difference was 1.1e-15. `install.sh` does not install PyTorch3D, so on a clean install
that comparison reports **SKIP**, not PASS: the figure above comes from a development machine, not
from your install.

## Durability

Built for preemptible clusters, because interrupted runs have already cost this project real data:

- Outputs are written to a temp name, fsynced, then renamed — never written in place.
- Each frame is built in a staging directory and swapped in only when complete, so **a crashed
  re-run cannot destroy a good frame**.
- `_DONE.json` is written last and checked first; resume never trusts file existence.
- `orhsurf verify` opens every array. Run it after any interruption.

The failure this defends against is real: the source project holds two truncated 494 MB
`surface.npz` files (against a healthy ~757 MB) whose zip magic bytes are intact, so a cheap header
check passes and the failure only appears as `BadZipFile` inside a render, hours later. One of them
is *newer* than its own sidecar files — a re-run overwrote a good output and then died.

## Historical performance experiments

These are earlier development-host observations, not current deployment guarantees.

### Earlier two-GPU run

This exact command was run on the development machine (2 frames, 2 GPUs) and produced filtered
clouds:

```bash
orhsurf run --clip <clip-id-or-path-to-manifest.json> --frames 40-41 --gpus 2
```

| frame | points (filtered) | dropped: support<2 | dropped: isolated |
|---|---|---|---|
| 00040 | **25.4 M** (±0.5%) | ~4.11–4.18 M | ~1.54–1.56 M |
| 00041 | **25.47 M** (±0.05%) | ~4.10–4.11 M | ~1.53–1.57 M |

**These counts are not deterministic.** Three independent runs of frame 40 with identical code and
inputs gave 25,493,841 / 25,426,861 / 25,366,727 — a **0.50% spread**. AmbiSuR's densification
samples views in a random order, so the fitted Gaussian count varies run to run (measured 648,689
to 667,291, a 2.8% spread) and the exported point count follows it. Quote these with a tolerance;
a sub-percent difference between two runs is not a regression.

`orhsurf verify` → `2/2 frames ok, 50,962,741 points total`. Wall clock 19.5 min for both frames in
parallel, on a heavily contended box (load average ~65–98 from unrelated jobs).

Per-frame stage timings (frame 00040): prep 2 s, dataset_build 35 s, scene_build 16 s,
**da3_1008 88 s**, rewarp 19 s, **train 706 s**, **export 296 s**.

The DA3 stage reproduced the reference recipe exactly: torch peak **23,353 MiB** (reference:
23,353 MiB), worst scene-frame coverage **1.00000**, 50,000-point initial cloud. The k-NN gate
reported `k=5 median 2.419 mm`; its drop counts differ from the reference frame's by **+1.25%**
and **+2.33%** — a different frame, so they are not expected to match exactly.

### Wall time — two numbers that do not agree

Per-stage, frame 00040, one RTX 4090, 8 cores, `-r 2` × 7k:

```
prep 2 s · dataset_build 35 s · scene_build 16 s · da3_1008 88 s · rewarp 19 s
train 706 s · export 296 s                    = 1,162 s = 19.4 min
```

End-to-end:

| run | measured |
|---|---|
| 1 frame, 1 GPU | **11.1 min** |
| 2 frames, 2 GPUs | **19.5 min** and **19.1 min** (two independent runs) |
| 8 frames, 8 GPUs | 74.1 min — **invalid**, that run was CPU-capped to 8 cores *total* |

**The stage sum (19.4 min) and the single-frame end-to-end (11.1 min) do not agree, and we do not
know why** — the load each was taken under was not recorded. Both are reported rather than
averaged. Do not size a job from 11 min alone.

**Per-frame time is not independent of concurrency.** One frame alone is 11.1 min; two frames on
two GPUs take ~19 min *each*, measured twice. That is a **~1.7× slowdown at only 2-way
concurrency**, and it is the number that matters for `--time`. The 8-GPU figure is genuinely
unmeasured: the only 8-way run was CPU-starved and proves nothing about throughput. So:

- optimistic (frames independent): 150 frames ÷ 8 GPUs × 11.1 min ≈ **3.5 h**
- with the measured 2-way penalty applied: 150 ÷ 8 × 19 min ≈ **5.9 h**

Request time against the second, and note that 8-way contention may be worse than 2-way.

**`export` roughly doubled — 296 s → 781 s on frame 40 — after the durability change**, because
every ~620 MB payload is now fsynced before the completion marker. That is the cost of the
guarantee actually holding. The 19.4 min stage sum above is **pre-fsync**; a post-fsync frame is
roughly 8 min longer.

### Disk warnings, learned the hard way

- **`--write-ply` nearly doubles a clip to ~198 GB.** On the development box accumulated PLYs
  reached ~110 GB and filled the filesystem to **99.9%, 0 bytes available**. Throughput collapsed
  to a few MB/min, `rm` took ~18 s *per file*, and five processes wedged in uninterruptible `D`
  state. On a shared cluster scratch this affects other people. **Check your quota before
  launching a full clip**, and leave PLY off unless you need it.
- **Scratch is kept on failure, by design** — it holds the trained model and DA3 depths, ~13 min of
  GPU work. A run with many failures therefore accumulates **~6 GB per failed frame**. Clean up
  `<out>/_work/` after a bad run.

### Point counts are not reproducible to the digit

Three independent runs of frame 40, identical code and inputs:

```
25,493,841   25,426,861   25,366,727      -> 0.50% spread
```

while the **pre-filter** cloud reproduced to ~1e-5. **The cause is not established.** Each of the
three runs trained separately, so the filter never saw the same input twice — nothing in that data
separates AmbiSuR's training variance from sensitivity in the filter itself. Treat a sub-percent
difference between two runs as expected, and do not compare point counts to the digit.

## Further references

- [Slurm setup and troubleshooting](docs/INSTALL_SLURM.md)
- [Manifest and camera conventions](docs/DATA_CONTRACT.md)
- `tools/check_cuda_env.py`: actual CUDA smoke tests in the selected environment.
- `tests/test_convert.py`: sparse conversion, manifest paths and mask policy regressions.
- `tools/render_sequence.py`: fixed-camera sequence contact sheet and decodable MP4.

AmbiSuR is vendored with its upstream `third_party/AmbiSuR/LICENSE.md`.
DA3 code and checkpoint retain their upstream terms; weights, data and generated environments
are not committed. Review upstream terms before redistributing them.


## Point-cloud simplification alternatives

The exported product is an oriented point cloud, without triangle connectivity.

| Library | Relevant operation | Fit for this output |
|---|---|---|
| [trimesh](https://trimesh.org/trimesh.html#trimesh.Trimesh.simplify_quadric_decimation) | Quadric mesh decimation | Requires a mesh; not a direct point-cloud simplifier |
| [Blender](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/geometry/operations/merge_by_distance.html) | Merge by Distance for points | Distance-based; normal-aware separation requires additional logic |
| [Open3D](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html) | `voxel_down_sample` | Averages normals/colours; averaging normals does not enforce a normal-angle merge gate |
| [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/filter_list.html#generate-simplified-point-cloud) | `generate_simplified_point_cloud` | Direct Poisson-disk point sampling; target count with tolerance, no documented normal-angle gate |

PyMeshLab is the closest ready-made baseline. `samplenum` sets the desired count;
`exactnumflag=True` searches for a radius to meet a tolerance (default 0.5%), rather than
promising an exact count. It has not been benchmarked on this 25.5M-point output. Meshing just
for decimation would introduce another reconstruction step and change the product.

Our dependency-free simplifier uses conservative bins of normal vector components and a spatial
voxel grid. Same-bin normalized normals have a bounded angular difference; normals across bin
boundaries may stay separate even when close. Voxel tuning is heuristic because moving the grid
scale does not give strictly monotonic occupancy. Final seeded trimming enforces exact counts
but removes some occupied groups, so the voxel diagonal is **not** a global coverage bound after
trimming. All six NPZ attributes are copied from original representatives and checked after save.


## Development-only frame subset checks

The README's `process` command handles complete clips. These lower-level commands are only
for debugging a first frame or five-frame subset; they are not the production entry point.

```bash
source env.sh
orhsurf fetch --weights
orhsurf fetch --clip C001 --convert --frames 0-4
orhsurf doctor
orhsurf run --clip data/C001_prepared/manifest.json --gpus 1 --frames 0-0 --out out/C001_test5
orhsurf verify --out out/C001_test5
# Inspect out/C001_test5/_check/check_00000.png, then reuse frame 0:
orhsurf run --clip data/C001_prepared/manifest.json --gpus 1 --frames 0-4 --out out/C001_test5
orhsurf verify --out out/C001_test5
```

### Measured simplification (C001 frame 0)

Source: 25,475,015 points, 617,821,162 bytes; float32 positions/normals.
All derived NPZ arrays were decompressed, schema-checked and compared against the selected
original attributes. These results describe one frame, not a full-clip benchmark.

| Target | NPZ size (decimal MB) | Reduction vs source | CPU wall time, including writes/checks |
|---|---:|---:|---:|
| 10 M | 247.4 | 60.0% | 60.3 s |
| 5 M | 124.4 | 79.9% | 44.0 s |
| 1 M | 25.1 | 95.9% | 32.7 s |

165,236 source points (0.65%) have zero-length normals. They are kept in a separate
unknown-direction bin and never merged with points having a valid normal. The 30° bound applies
to valid normal bins; it is not a bound on reconstruction error. Exact-budget trimming removed
roughly 1.5–2.7% of the voxel representatives in this experiment.


Shared-camera comparison: two views, shaded/normal/RGB, with at most 1.5M displayed points per
cloud. The PNG was visually inspected; 1M shows more normal variation than the larger versions.
For 50,000 seeded original-to-simplified nearest-neighbour samples:

| Budget | Distance median / p95 / p99 (mm) | Nearest-normal angle median / p95 (degrees) |
|---|---|---|
| 10M | 1.10 / 2.96 / 3.70 | 0.97 / 29.05 |
| 5M | 2.17 / 5.43 / 7.11 | 4.98 / 75.13 |
| 1M | 5.55 / 17.20 / 24.66 | 29.71 / 114.63 |

Nearest spatial neighbours are not necessarily members of the same normal bin or the same
oriented surface, so these angle statistics are not a violation of the within-bin 30° gate.
They also do not demonstrate orientation fidelity of the simplified cloud. Zero-normal pairs
are excluded from angle statistics. Full-resolution/thin-feature quality remains application-dependent.

The new `process` wrapper's orchestration tests check full-range discovery, clip ordering,
manifest-preserving resume and stopping on reconstruction/verification failure. Heavy stages
are replaced with test doubles in those tests. Existing real single-frame pipeline execution
has passed; the new multi-clip command has not yet completed a real multi-clip GPU run.


## Scaling deployment to many clips

Install once on a compute node into a shared checkout/environment and cache, then reuse those
paths from batch jobs. CUDA extension builds must cover the target GPU architectures and the
nodes must have compatible drivers/libraries; an A100-only build is not automatically portable
to RTX 3090. Do not run installation concurrently inside every array task.

The clip-array script requests one node and one GPU per task by default. `%10` limits running
array tasks, not node count. For a homogeneous eight-GPU node pool, this submission shape requests
eight GPUs for each of up to ten concurrent clip tasks (supply real site/project/time options):

```bash
sbatch --array=0-99%10 --nodes=1 --gres=gpu:8 \
  slurm/process_clips.sbatch --gpus 8 C{001..100}
```

This is a Bash brace-expansion example and was not submitted. Node packing, availability and
account limits remain scheduler decisions. The inspected pinned archive inventory contains
C001 through C100; the historical full-clip estimate below assumes explicit `--all-frames` and 225 frames per clip.
The current default is the first 150 frames (10 seconds at 15 fps), with separate
`*_first150_prepared` manifests; neither frame rate nor reconstruction quality is reduced.
At 14.3 minutes per frame, 100 x 225 frames = 5,362.5 GPU-worker hours. Eighty equally fast GPUs
would take about 67 hours under ideal scaling; 12 hours would require about 447 such GPUs before
extra overhead. A 3-hour debug limit cannot fit the estimated 6.7-hour eight-GPU 225-frame task.
Use a permitted production partition or separately designed frame-chunk tasks; do not bypass
limits with resubmission loops. Chunked multi-clip scheduling is not implemented by this script.

Recommended order: install shared environments/weights; validate one clip; validate multi-GPU
throughput and CPU/RAM budgets on one node; prepare the intended inputs; submit a bounded array;
verify all expected outputs and keep the selected point budget. Input preparation can remain
inside each allocation or be staged first. For many jobs, pre-staging avoids simultaneous
model downloads and makes failures easier to diagnose. Decoding shared inputs must happen once
per clip, before any frame-sharded consumers use its manifest.

The single-frame estimates imply roughly 13.9 TB of full point clouds and 5.3 TB of decoded
inputs across 100 C001-sized clips. A 5M retained point budget is roughly 2.8 TB, 1M roughly
0.56 TB, excluding inputs/environment/scratch. These are extrapolations, not storage scans.
The current simplifier creates additional derived files and never deletes full outputs. Storage
savings therefore require a deliberate retention decision after quality checks. Postprocessing
point count does not reduce the 7,000-iteration training work.


Spatial stratified sampling was exercised on C001 frame 0 with 50 mm cells: 47,981 populated
cells. The 10M / 5M / 1M products were 245.9 / 123.7 / 25.0 MB and took 39.9 / 24.6 / 11.7 seconds
respectively after the shared source load, including write/readback validation. Integer quotas
left 1,938 / 3,083 / 7,699 cells empty respectively; these are primarily low-population cells and
are reported explicitly rather than promising minimum coverage. The maximum deviation from a
cell's ideal fractional quota stayed below one point. Attributes remain exact source values.
