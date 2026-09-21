# orhsurf

One multi-view clip → one filtered surface point cloud per timestamp, in calibrated world coordinates.

## TL;DR — process complete clips with one command

```bash
orhsurf process --clips C001 C002 C003 C004 --gpus 1
```

For **each clip in order**, this prepares the downloaded videos, reconstructs **every encoded
frame**, verifies the outputs, then starts the next clip. Weights are fetched once and reused.
C001 means all 225 frames, not a five-frame test. Results go to `out/C001/00000/`, etc.
The command stops on a failed stage. Re-run it to resume completed matching frames.

**First-time setup:** `orhsurf` is this repository's CLI. The installer creates `bin/orhsurf`;
`source env.sh` makes it available in the current shell. Use a CUDA GPU (A100 80 GB tested; **24 GB compatibility is configuration-dependent**),
conda/mamba, CUDA Toolkit and a supported compiler. Run setup on a compute node.

```bash
# Only if you do not already have a compute allocation; adapt to your site's actual settings:
srun --partition=debug --gres=gpu:1 --time=03:00:00 --pty bash

git clone https://github.com/Sulwon-0516/orh_4d_reconstruction.git
cd orh_4d_reconstruction
./install.sh --no-weights
source env.sh
orhsurf process --clips C001 --gpus 1
```

In every new terminal or batch script, change into the checkout and `source env.sh` again.
No separate `fetch`, `doctor`, `run` or `verify` commands are needed for this workflow.
`orhsurf process --help` lists options, including `--out-root` and a CPU thread limit.
Without PATH setup, `./bin/orhsurf process --clips C001 --gpus 1` also works after installation.

The command uses your existing compute allocation; it does not acquire or extend one. A full
C001 is estimated at **~54 hours on one GPU**: the tested debug partition's three-hour limit
cannot finish it. Use a permitted longer allocation or resume manually later. Downloads require
network unless inputs and weights are already present. Clip preparation happens one clip at a
time, not by pre-downloading the entire list. No CPU/memory resource request is invented.

### Separate clips as separate Slurm jobs

Submit from the installed checkout, adding your site's required account/project options:

```bash
sbatch --array=0-3%2 slurm/process_clips.sbatch C001 C002 C003 C004
```

Array tasks 0/1/2/3 handle C001/C002/C003/C004 respectively; `%2` allows at most two clips
concurrently. Each task gets one GPU by default and performs preparation, full reconstruction
and verification. Set an approved partition and wall time via sbatch options. Different tasks
can run on different nodes. Unlike `process --clips ...` in one shell, these clips are independent.
To use two GPUs **per clip**, specify both `sbatch --gres=gpu:2` and script argument `--gpus 2`.

Optional first-frame smoke run (same quality, separate inputs and outputs):

```bash
sbatch --partition=debug --time=00:30:00 --array=0-1%1 \
  slurm/process_clips.sbatch --smoke C001 C002
```

`--smoke` runs frame 0 at the unchanged 7,000-iteration/DA3-1008 recipe and verifies it; allow
roughly 14 minutes plus input setup per clip based on the A100 measurement. Outputs go to
`out/_smoke/<clip>/00000/`; full runs use `out/<clip>/`. When `MODEL_OUTPUT_DIR` is exported,
the script uses that directory instead of `out/`. Logs are `slurm-process-<job>_<task>.out`.
The array range must match the list; to retry selected indices, preserve the **original clip list**.

On the tested cluster, sbatch additionally requires an approved `--account`,
`--wckey=project-short-name:...` and `MODEL_OUTPUT_DIR`. Set your actual project values;
the script does not invent them. Add `--test-only` before the script name to validate submission
without creating jobs. Local array mapping/smoke isolation tests passed; the scheduler check
currently awaits the project's approved `MODEL_OUTPUT_DIR`, so actual array execution is not
claimed as validated. Existing `slurm/recon_array.sbatch` partitions frames of **one** clip;
this new script partitions **different clips**.

### More GPUs on one node

After obtaining an allocation with four GPUs visible to the same process:

```bash
orhsurf process --clips C001 C002 C003 C004 --gpus 4
```

Clips stay sequential; each clip's remaining frames are divided into contiguous ranges, with
one worker per GPU. For C001 on two GPUs, an untouched clip divides into frames 0–112 and 113–224.
This increases frame throughput; it does not split a single frame's model across GPUs.
Completed matching frames can be resumed with a different GPU count.

`--gpus` selects from GPUs already allocated on this node; it does not request more from Slurm
or span multiple nodes. If fewer GPUs are visible, `process` stops before downloads. CPU threads
default to allocated CPUs divided by the requested GPU count: 20 CPUs / 4 GPUs → 5 threads each.
Explicit `--cpus-per-job` or `ORHSURF_CPUS_PER_JOB` overrides are **per worker**; their total must
fit the allocation or the command stops. Each GPU needs its own VRAM; memory is not pooled across GPUs.
Host RAM and concurrent scratch requirements grow with the number of workers; plan roughly
6 GB scratch per worker based on historical measurements. Busy GPUs may be skipped by the
runtime VRAM check. Speedup is not guaranteed to be linear because CPU and storage are shared.

GPU mapping, frame partitioning, CPU budgets and allocation-error handling are covered by tests.
Actual multi-GPU performance on this cluster has **not** been measured; the current live run
continues on its existing single GPU.

<details>
<summary>Resume, Slurm batch setup and development details</summary>

`process` uses `data/C001_full_prepared/manifest.json`, keeping advanced `*_prepared` subsets
separate. It reuses an existing complete manifest without rewriting it, preserving resume
fingerprints. Existing subset outputs are not automatically migrated to the full-clip workflow.
Do not run two processes on the same prepared/output directories concurrently.

For batch use, put `source env.sh` and the single `process` command in a shell script; submit it
from the checkout with your site's actual partition/account/time options. `slurm/*.sbatch` are
multi-GPU/array templates with resource assumptions, not the one-GPU quick start.
See [detail.md](detail.md) and [Slurm setup](docs/INSTALL_SLURM.md) for background and troubleshooting.

</details>

## What you get

**One cloud per frame**, usually around **25.4–25.5 million points** in the measured C001/development
examples. This is a sample range, not a measured average over a complete clip. Current A100 frame 0:
**25,475,015 points**, **617.8 MB** compressed. Frames 0 and 1 now completed with a mean of
**25,466,729 points** across those two frames. There is no `.pcd` file by default: the product is
`surface.npz`. Optional `--write-ply` adds a PLY for external viewers.

The output includes the room/background. Default all-foreground masks are not segmentation.
Frames share calibrated world coordinates, but points have **no persistent identities** across time;
point index `i` in consecutive files is not a trajectory or correspondence.

With `--out out/C001`, paths are exactly:

```text
out/C001/
  _EXPECTED.json            requested frame set (verify checks missing frames too)
  _logs/                    worker logs
  _check/check_00000.png    automatic two-view static preview
  00000/
    surface.npz            positions, normals, RGB and support
    surface_export.json    point counts and filter parameters
    provenance.json        recipe, stage timings, DA3 revision, CPU budget
    metadata.json          timestamp and source manifest
    _DONE.json             completion marker, sizes and resume fingerprint
  00001/ ...
```

Wait for `_DONE.json` and run `orhsurf verify --out out/C001` before consuming results.
A partially written file is not a completed frame. Metadata's `pose_version`/`config_hash` are
currently placeholders; use the manifest and `provenance.json` recipe hash instead.

### Load a cloud

Run with the installed interpreter (`source env.sh`, then `"$ORHSURF_PYTHON" your_script.py`):

```python
from pathlib import Path
import numpy as np

frame = Path("out/C001/00000")
assert (frame / "_DONE.json").is_file()  # run orhsurf verify first
with np.load(frame / "surface.npz", allow_pickle=False) as z:
    xyz = z["xyz"]          # float32 [N,3], world metres
    normal = z["normal"]    # float32 [N,3], unit normals
    rgb = z["rgb"]          # uint8   [N,3], original image colour
    support = z["support"]  # int16   [N], agreeing camera count
# Optional stricter read-time filtering (changes the consumed cloud):
keep = support >= 3
xyz, normal, rgb = xyz[keep], normal[keep], rgb[keep]
```

Also stored: `confidence` (float32 [N], support-derived [0,1]) and `observed` (bool [N], currently
all true). All six arrays total approximately **866 MB uncompressed** at 25.5 M points; allow extra
RAM for loading, masks and copies. Compressed NPZ arrays do not provide memory-mapped access.
PLY contains only positions/normals/RGB and loses support/confidence.

### Smaller point clouds — point count per frame, not fewer frames

10M / 5M / 1M means the **number of 3D points inside each frame**. It does not mean fewer
timestamps. The complete-clip reconstruction command above still processes every frame.

Measured on C001 frame 0; all three derived files passed deep verification and an exact comparison
of saved attributes against the selected source points:

| Version | Points per frame | NPZ size | Saved vs original | CPU processing time* |
|---|---:|---:|---:|---:|
| Original | 25,475,015 | 617.8 MB | — | — |
| 10M | 10,000,000 | 247.4 MB | 60.0% | 60 s |
| 5M | 5,000,000 | 124.4 MB | 79.9% | 44 s |
| 1M | 1,000,000 | 25.1 MB | 95.9% | 33 s |

*CPU thread limit 2; includes clustering, writing and verification, excludes initial source load.
These are one-frame measurements; full-clip quality and storage averages are not yet measured.
On 50,000 sampled source points, the 95th-percentile distance to the nearest retained point was
**2.96 mm / 5.43 mm / 17.20 mm** for 10M / 5M / 1M. These are sample statistics, not worst-case
bounds. The shared-camera preview shows more normal variation at 1M; choose a point budget after
checking the detail you need, rather than treating storage saving as proof of equivalent quality.

The optional simplifier operates on a completed frame (it is not automatically enabled by `process`):

```bash
"$ORHSURF_PYTHON" -m orhsurf.simplify --source out/C001/00000 --out out/C001_simplified \
  --targets 10000000,5000000,1000000 --cpus 2
```

Outputs: `out/C001_simplified/{10M,5M,1M}/00000/surface.npz`, with completion markers and
provenance. The original is preserved. The same NPZ loader and viewer work on these files.
Use a thread count within your allocation. Float attributes remain **float32**.

The method groups by spatial voxel **and normal direction** (30° maximum within a normal bin),
retains the highest-support original point per group, then trims surplus representatives with
a fixed random seed to hit the exact budget. Each target is computed from the original; versions
are not nested. Voxel size and trim counts are recorded. This is a practical approximation,
not an optimal nearest-neighbour/mesh decimator or a guarantee against losing thin structures.
No coordinates, colours or support values are averaged. Zero-length source normals use a separate
unknown-direction bin. See [library alternatives](detail.md#point-cloud-simplification-alternatives).
Keeping the original and all three versions adds storage; savings require choosing which derived
version to retain later. Nothing deletes the original automatically.

For a simple random baseline, use the same exporter with `--method random`:

```bash
"$ORHSURF_PYTHON" -m orhsurf.simplify --source out/C001/00000 --out out/C001_random \
  --method random --targets 10000000,5000000,1000000 --cpus 2
```

This selects original points uniformly without replacement with a fixed seed. It does not enforce
uniform spatial spacing or use neighbours/normals. All attributes remain paired with their point.
Viser's existing `points drawn` also uses a fixed random permutation; it only changes what is drawn,
not file size. Creating all variants for evaluation adds storage until a retention choice is made.

Compare saved versions in the **same viser session and camera**:

```bash
orhsurf view --npz out/C001/00000/surface.npz --host 127.0.0.1 --port 8080 --budget '1 M' \
  --variant 'Normal 1M=out/C001_simplified/1M/00000/surface.npz' \
  --variant 'Random 1M=out/C001_random/1M/00000/surface.npz'
```

Repeat `--variant LABEL=PATH` for 5M/10M files too. The **version** dropdown preserves the camera,
colour mode and point size. **Stored** is the file's full point count; **drawn** is the display budget
(after support filtering). For a full 1M-vs-1M comparison, select 1M or ALL. A 1M display cap on a
10M file shows only a random preview of it. Larger budgets increase browser memory/network use.
Variants load on demand and the server keeps only two loaded selections in memory.

## Cameras and matching RGB frames

For reconstruction, keep **all 47 valid C001 cameras** and the default DA3 groups (18 views,
overlap 6, 1008 px). Reducing camera count or group size changes the prior and is not a measured
speed/quality equivalent. Invalid cameras are excluded by the manifest.

For inspection, start with two oblique viewpoints about 50° apart; keep identical camera poses
and world scale across timestamps. The static preview already uses two viewpoints. For comparing
against captured RGB, use the camera's calibrated pose and intrinsics, not a guessed orbit camera.

```python
from pathlib import Path
from orhsurf.paths import read_manifest

m = read_manifest(Path("data/C001_full_prepared/manifest.json"))
serial = m["valid_serials"][0]  # choose a valid camera with a clear view of your region
camera = m["cameras"][serial]
frame_index = 0                # encoded MP4 index, NOT source video_frame_index
entry = camera["frames"][str(frame_index)]
print(entry["frame_path"], entry["mask_path"])
K = camera["K_original"]
distortion = camera["dist_params"]
world_to_camera = camera["T_cam_from_world"]
```

Converted full-clip inputs are `data/C001_full_prepared/rgb/<serial>/<index:05d>.png`, at 2048×1536 for C001.
They retain lens distortion: pair them with `K_original` and distortion coefficients. DA3's
undistorted images and the cropped/recentred training images use different intrinsics; see
[the camera contract](docs/DATA_CONTRACT.md). C001 spans 15 seconds at 15 fps; `0-4` are its first
five encoded timestamps. Actual timing metadata is in the manifest.

## Preview and sequence export

```bash
source env.sh
"$ORHSURF_PYTHON" -m pip install viser  # optional, once, on a compute node
orhsurf view --npz out/C001/00000/surface.npz \
  --manifest data/C001_full_prepared/manifest.json --host 127.0.0.1 --port 8080 --budget '1.2 M'
```

From your own computer, using your SSH login alias and the actual allocated compute hostname:

```bash
ssh -J YOUR_LOGIN_ALIAS -N -L 8080:127.0.0.1:8080 YOUR_USER@YOUR_COMPUTE_NODE
# Open http://localhost:8080; keep this SSH session open.
```

The viewer displays one frame; start at 1.2 M displayed points for responsiveness. It supports
RGB, shaded, normal and support modes. For five completed frames with the **same two cameras**:

```bash
"$ORHSURF_PYTHON" tools/render_sequence.py --out out/C001 --frames 0,1,2,3,4 \
  --dest out/C001/visualization
```

This writes `contact_sheet.png`, `sequence.mp4` (one second per reconstruction frame),
`time_00000.png` etc. and `cameras.json`. The tool also decodes the entire MP4 to check it.
The one-second hold is for inspection, not playback at the capture's original 15 fps.

## Time and storage budget

Current measurement: C001 frame 0, A100 80 GB PCIe, one GPU, eight CPU threads, default 7,000
iterations / resolution 2 / DA3 1008. **14.3 minutes/frame**, including CPU stages and export;
DA3 148 s, training 517 s, export 125 s. First-frame deep verification passed **1/1** and its
static preview was inspected. Full-clip stability and an average over all 225 frames remain unverified.

| Item | Five frames | Full C001, 225 frames |
|---|---:|---:|
| Reconstruction time, 1 GPU (extrapolated) | ~72 min | ~54 h |
| Final NPZ outputs (extrapolated) | ~3.1 GB | ~139 GB |
| Decoded RGB + masks | **1.19 GB measured** | ~53.4 GB estimated |
| Working scratch, one sequential worker | ~6 GB historical estimate | ~6 GB reused; failures retain scratch |

One-off inputs: **1.56 GB HEVC archive**, plus its extracted videos (roughly another archive's
size), and **6.76 GB DA3 weights**. The archive download contains the whole clip even when only
five frames are decoded. Environments/build/download caches need tens of GB: an earlier clean
installation was roughly 32 GB including weights/caches, **not a fresh measurement here**.
Allow roughly **230 GB plus headroom** for the full workflow using these estimates, without PLY.
Optional PLY adds substantial storage; leave it off unless needed. GB here is decimal.
Conversion/download time is additional and depends on CPU, network and shared storage.

DA3 peaked at **24,745 MiB torch-allocated on A100** in this run. Earlier RTX 4090 runs recorded
23,353 MiB allocated / 24,090 MiB device usage, so **40 GB is not a code-enforced minimum**.
The current preflight guard checks for 23,450 MiB free; expandable allocator segments remain
enabled. 24 GB cards can be tight, and this exact C001 workflow has not been tested on RTX 3090.
An A100 measurement alone does not establish the peak on another GPU/backend. A 40 GB+ card
provides headroom, not a new algorithmic requirement. The cause of the difference from the earlier
measurement is unresolved. Host-memory peak has not been measured in this run.
Inspect quota using site-approved tools; avoid recursive shared-storage scans or continuous polling.

<details>
<summary>Background, numerical reproducibility, installation troubleshooting and licensing</summary>

See [detail.md](detail.md). It contains the two-environment rationale, default recipe,
mask semantics, atomic output/resume behavior, historical benchmarks, known limitations and
upstream license references. [docs/INSTALL_SLURM.md](docs/INSTALL_SLURM.md) covers cluster setup;
[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) specifies calibration and manifest fields.

</details>
