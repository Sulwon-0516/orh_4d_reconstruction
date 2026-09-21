# orhsurf

One multi-view clip → one filtered surface point cloud per timestamp, in calibrated world coordinates.

## TL;DR — install once, submit all clips with one command

Each clip defaults to **the first 150 frames (0–149)**: **10 seconds at 15 fps**, not
150 fps. Every selected frame uses the unchanged reconstruction recipe. The workflow downloads
inputs as needed, prepares them, reconstructs them, and verifies all expected outputs.

### Arguments — choose these before running

| Argument / setting | Meaning and default |
|---|---|
| `--gpus 8` | GPUs used **per clip on one node**; default `1`. For sbatch, also request `--gres=gpu:8`. |
| `--durations 10` / `15` / `10,15` | Save a 10-second version, a 15-second version, or both at 15 fps. Explicit durations use separate `10s/` and `15s/` output roots. Without this argument: first 150 frames directly under the output root. |
| `--simplify` | Add **1M and 5M** point clouds for every frame; **keep originals**. Random subsampling is the default. |
| `--simplify-only` | Save and verify **1M and 5M**, then **delete original cloud payloads**. Works alone; no `--simplify` argument is required. |
| `--simplify 5M --simplify-only` | Override the point budgets: retain only 5M. Comma-separated budgets also work, e.g. `--simplify 1M,5M`. |
| `--simplify-method random` | Default sampling method. Alternatives: `stratified` or `normal-voxel`. Requires `--simplify` or `--simplify-only` to produce derivatives. |
| `--cleanup-decoded` | After successful clip verification, delete generated RGB and automatic-mask PNGs. Preserve source videos, calibration, manifests and supplied masks. Default: keep decoded inputs. |
| `--smoke` | Test only frame 0 with unchanged reconstruction quality, under separate `_smoke/` outputs. |
| `--all-frames` | Process every encoded frame; cannot be combined with `--durations`. |
| `--array=0-99%10` | **sbatch option**: 100 clip tasks, at most 10 running concurrently. Match the index range to the number of clip IDs. |
| `MODEL_OUTPUT_DIR` | Export the approved output root for sbatch. For direct `orhsurf process`, use `--out-root /your/output/root` (default: `out/`). |

**Argument placement matters:** Slurm options go **before** the script path; processing options
go **after** it and **before** the clip list. In direct CLI commands, pass clips with `--clips`.
Bash expands `C{001..100}` into C001 through C100.

After completing installation and the cluster settings below, this command saves **both durations**,
keeps only **1M + 5M**, and cleans decoded inputs, using eight GPUs per clip:

```bash
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --durations 10,15 --simplify-only --cleanup-decoded C{001..100}
```

Inside an existing one-GPU allocation, the equivalent processing options are:

```bash
orhsurf process --clips C001 C002 --gpus 1 --durations 10,15 --simplify-only --cleanup-decoded
```

Omit `--durations 10,15` for just the default first 150 frames. Omit `--simplify-only` to retain
full point clouds, or replace it with `--simplify` to keep originals **and** the two reduced versions.
Both durations currently reconstruct overlapping frames independently. Deletion happens only after
all requested outputs for that clip/version verify; temporary original and decoded storage is still
needed while it runs. See [storage estimates](#smaller-point-clouds--point-count-per-frame-not-fewer-frames)
for retained and temporary capacity.

### 1. Install once in a shared checkout

Run on an **allocated compute node**, using an existing allocation if available. Prerequisites:
conda/mamba, CUDA Toolkit and a supported compiler. All worker nodes must see the same checkout,
environments, cache and data paths; CUDA extensions must support their GPU architectures.

```bash
git clone https://github.com/Sulwon-0516/orh_4d_reconstruction.git
cd orh_4d_reconstruction
./install.sh --no-weights
source env.sh
orhsurf fetch --weights
orhsurf doctor
```

`orhsurf` is this repository's command, created as `bin/orhsurf` by the installer.
`source env.sh` sets its PATH and the environment/data/cache paths. Install and fetch shared
weights once, **not inside every array task**. Existing installations can skip this step.
A100 80 GB is tested; 24 GB compatibility depends on the configuration.

### 2. Set your cluster options

In Bash, from the installed checkout, replace the marked values with your **actual approved**
settings. Use a partition and time limit suitable for a full task; the debug partition has a
three-hour maximum on the tested cluster. No CPU or memory allocation is hard-coded.

```bash
export MODEL_OUTPUT_DIR="/REPLACE_WITH_APPROVED_SHARED_OUTPUT_ROOT"
SBATCH_SITE=(--partition="REPLACE_WITH_PARTITION" --account="REPLACE_WITH_ACCOUNT"
             --wckey="REPLACE_WITH_PROJECT_WCKEY" --time="REPLACE_WITH_TIME_LIMIT")
```

The tested cluster requires account, project WCKey and `MODEL_OUTPUT_DIR`; other sites may omit
unused options. Keep these settings in your own shell configuration if desired. The output root
must be writable and shared across nodes. Submit from this checkout in each new terminal.

### 3. Submit C001–C100

For separate 10-second and 15-second versions, add `--durations 10,15` **after the script path**.
Use `--durations 10` or `--durations 15` for just one version. At the dataset's verified 15 fps,
these select frames 0–149 or 0–224 respectively (shorter clips stop at their last frame).
Outputs are isolated under `<output-root>/10s/` and `<output-root>/15s/`.
Prepared inputs use `*_first150_prepared` and `*_first225_prepared`; raw videos are shared.
Both versions currently run independently, including reconstruction of overlapping frames;
requesting both costs more compute and storage. Without `--durations`, the default remains
150 frames under the original output-root layout.


For **8 GPUs per clip**, with **at most 10 clips running concurrently**:

```bash
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 C{001..100}
```

This is one submission. Each array task selects a different clip, prepares its first 150 frames,
processes those frames across its allocated GPUs, then verifies the result. The script sources
`env.sh` automatically. There is no separate manual fetch/run/verify sequence per clip.

`%10` limits **tasks, not nodes**. Each task requests one node with eight GPUs; on eight-GPU nodes
this can occupy up to ten nodes, subject to scheduler availability. Slurm places the tasks;
the wrapper does not discover spare GPUs or spread one clip across nodes. Change **both**
`--gres=gpu:8` and `--gpus 8` to the desired GPUs per task. With one GPU per task, several tasks
may share a node. CPU thread counts stay within the actual allocation.

Results: `$MODEL_OUTPUT_DIR/C001/00000/surface.npz` through `00149/`, then the corresponding
paths for each other clip. Clips shorter than 150 frames use all available frames.
Logs: `slurm-process-<job>_<task>.out` in the submission directory.
Downloaded videos remain complete archives; the 150-frame default reduces decoded inputs and
reconstruction work, **not the archive download size**.

<details>
<summary>Short validation, retries, full 15-second clips, and running inside an existing allocation</summary>

First validate one frame at unchanged quality (including preparation, reconstruction and verify):

```bash
sbatch "${SBATCH_SITE[@]}" --gres=gpu:1 slurm/process_clips.sbatch --smoke C001
```

Smoke inputs and outputs are separate; results go to `$MODEL_OUTPUT_DIR/_smoke/C001/00000/`.
Add `--test-only` before the script path to check scheduler submission without creating a job.
Local orchestration tests passed; actual array execution has not been validated on this cluster:
its submission check still needs the user's approved `MODEL_OUTPUT_DIR`.

Retry only failed task indices, preserving the **original complete clip list** and output root:

```bash
sbatch "${SBATCH_SITE[@]}" --array=2,7%2 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 C{001..100}
```

This retries C003 and C008. Matching completed frames are reused; failed stages are not counted
as successes. Reissuing the original submission resumes all clips. Never overlap jobs writing
the same clip's prepared/output directories.

To request every encoded frame (225 frames / 15 seconds for C001), explicitly add `--all-frames`.
Use a distinct output root when switching from an existing 150-frame run:

```bash
MODEL_OUTPUT_DIR="$MODEL_OUTPUT_DIR/all_frames" sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --all-frames C{001..100}
```

Default inputs are `data/C001_first150_prepared/manifest.json`; full inputs are
`data/C001_full_prepared/manifest.json`. Existing manifests and subset outputs are preserved;
they are not automatically migrated between selections. Reconstruction quality is unchanged.

Inside an existing compute allocation, run clips sequentially without submitting new jobs:

```bash
source env.sh
orhsurf process --clips C001 C002 C003 C004 --gpus 1
```

This uses `out/<clip>/` by default; pass `--out-root "$MODEL_OUTPUT_DIR"` for your chosen root.
`--gpus` uses already allocated GPUs; it does not acquire or extend an allocation. With multiple
GPUs, frames are partitioned within each clip; clips remain sequential in this shell command.
The Slurm array is what makes different clips run independently.

See [detail.md](detail.md) for timings, capacity estimates and implementation background, and
[Slurm setup](docs/INSTALL_SLURM.md) for installation troubleshooting. The older
`slurm/recon_array.sbatch` shards frames of one clip; use `process_clips.sbatch` for clip lists.

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

**Storage-saving batch command** (after the shared installation and `SBATCH_SITE` setup above):

```bash
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --simplify-only --cleanup-decoded C{001..100}
```

To save both durations with the same cleanup policy:

```bash
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --durations 10,15 --simplify-only --cleanup-decoded C{001..100}
```

For example, the 15-second 5M output is
`$MODEL_OUTPUT_DIR/15s/_simplified/C001/random/5M/00000/surface.npz`.
Verify that version with `orhsurf verify --out "$MODEL_OUTPUT_DIR/15s/C001"`.

`--simplify-only` alone retains **1M and 5M random-sampled points per frame** plus metadata,
after verifying both versions and deleting the original clouds. `--simplify` alone creates the
same two versions **while preserving originals**. 10M is no longer generated by default.
For a custom budget, pass e.g. `--simplify 5M --simplify-only` to keep only 5M.
The equivalent interactive command is:

```bash
orhsurf process --clips C001 C002 --gpus 1 --simplify-only --cleanup-decoded
```

- `--simplify-only` defaults to 1M and 5M without requiring `--simplify`. All requested derivatives must pass verification
  before original NPZ/PLY payloads are removed. Original provenance and export metadata remain;
  `_DONE.json` is renamed `_ORIGINAL_DONE.json` so it cannot masquerade as an available original.
- `--cleanup-decoded` removes only generated RGB PNGs and automatic foreground-mask PNGs
  named by the successfully processed manifest. Supplied masks, downloaded MP4/archive caches,
  calibration and manifests are preserved. No shared-directory scan or cache purge is performed.
- `_RETENTION.json` records the selection, recipe and cleanup state. Repeating the same command
  verifies retained outputs and finishes interrupted cleanup without reconstructing deleted originals.
  Changed settings require a new output root. `orhsurf verify --out <root>/C001` follows this record
  and checks every expected frame and every retained point budget.
- Cleanup occurs **after a complete clip**, not after each frame. Failed reconstruction or
  simplification leaves originals and decoded inputs intact. Once intentionally cleaned, RGBs must
  be explicitly re-decoded for new reconstruction or image-based inspection; manifests remain intact.

**Estimated retained cloud storage per clip** using the real C001 frame-0 random-sampling
measurements (decimal GB; original cloud payloads removed with `--simplify-only`):

| Retained version | 10 seconds / 150 frames | 15 seconds / 225 frames | Reduction vs original clouds |
|---|---:|---:|---:|
| Original | 92.67 GB | 139.01 GB | — |
| 10M | 36.88 GB | 55.32 GB | 60.2% |
| 5M | 18.55 GB | 27.83 GB | 80.0% |
| 1M | 3.75 GB | 5.62 GB | 96.0% |
| **Default 1M + 5M** | **22.30 GB** | **33.45 GB** | **75.9%** |
| All three budgets | 59.18 GB | 88.78 GB | 36.1% |

For example, retaining only 5M saves about **74.12 GB per 10-second clip**, or **111.18 GB
per 15-second clip**, in cloud payloads. `--cleanup-decoded` additionally releases approximately
**35 GB / 53 GB** of generated inputs respectively, using the earlier decoded-input estimate.
These are extrapolations from one frame, not measured completed-clip totals. Metadata, retained
MP4/archive caches and working scratch are excluded. Retaining both durations adds their two
columns; the current independent-version workflow does not deduplicate overlapping frames.

At the measured random NPZ sizes, **100 clips × 150 frames** retain approximately **0.375 TB
at 1M**, **1.86 TB at 5M**, or **3.69 TB at 10M**. Keeping all three is approximately **5.92 TB**.
Saving both 10s and 15s versions multiplies these retained totals by 2.5: about **0.94 TB
at 1M**, **4.64 TB at 5M**, or **14.80 TB for all three point budgets** across 100 clips.
These are extrapolations, excluding videos/caches and temporary working data. While processing,
each active 150-frame C001-sized clip still needs roughly **93 GB original clouds + 35 GB decoded
inputs**, plus its derivatives and worker scratch. Ten concurrent clips multiply that temporary
requirement; reduce array concurrency if necessary. This bounds accumulating decoded/original
storage to active or failed clips, rather than keeping them for every successfully completed clip.


10M / 5M / 1M means the **number of 3D points inside each frame**. It does not mean fewer
timestamps. The default reconstruction processes all first 150 frames; simplification changes points per frame.

The **normal-voxel** comparison below was measured on C001 frame 0; all three derived files passed deep verification and an exact comparison
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

Enable saved subsampled versions in the same processing command (default method: **random**,
seed 0, sampling without replacement):

```bash
orhsurf process --clips C001 C002 --gpus 1 --simplify
# Or submit all clips; choose just --simplify 5M to save only the 5M version:
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --simplify C{001..100}
```

After each clip's reconstruction and verification, CPU postprocessing saves every selected frame
under `<output-root>/_simplified/<clip>/random/{1M,5M}/<frame>/surface.npz`.
Each derived frame is verified, records provenance, and preserves float32 attributes. Matching
completed derivatives are verified and reused on rerun; incompatible outputs are not overwritten.
Original clouds remain in `<output-root>/<clip>/<frame>/` unless `--simplify-only` is supplied. Without `--simplify` or `--simplify-only`, no derivatives
are generated. A requested point count greater than the source count fails explicitly.
Use `--simplify-method stratified` or `--simplify-method normal-voxel` for alternatives.
This postprocessing does not reduce GPU reconstruction time. Keeping originals plus derivatives
increases storage; choose a retention policy after inspecting quality.

For a previously completed frame, use the standalone simplifier (also random by default):

```bash
"$ORHSURF_PYTHON" -m orhsurf.simplify --source out/C001/00000 --out out/C001_simplified \
  --targets 1M,5M --cpus 2
```

Outputs: `out/C001_simplified/{1M,5M}/00000/surface.npz`, with completion markers and
provenance. The original is preserved. The same NPZ loader and viewer work on these files.
Use a thread count within your allocation. Float attributes remain **float32**.

With `--method normal-voxel`, the method groups by spatial voxel **and normal direction** (30° maximum within a normal bin),
retains the highest-support original point per group, then trims surplus representatives with
a fixed random seed to hit the exact budget. Each target is computed from the original; versions
are not nested. Voxel size and trim counts are recorded. This is a practical approximation,
not an optimal nearest-neighbour/mesh decimator or a guarantee against losing thin structures.
No coordinates, colours or support values are averaged. Zero-length source normals use a separate
unknown-direction bin. See [library alternatives](detail.md#point-cloud-simplification-alternatives).
Keeping the original and derived versions adds storage; savings require choosing which derived
version to retain later. Only the explicit `--simplify-only` process option deletes original cloud payloads after verification.

For density-preserving **spatial stratified sampling**, assign every 5 cm cell a point quota
proportional to its original population, then randomly choose within each cell:

```bash
"$ORHSURF_PYTHON" -m orhsurf.simplify --source out/C001/00000 --out out/C001_stratified \
  --method stratified --voxel-mm 50 --targets 1M,5M --cpus 2
```

This retains the original dense/sparse distribution to within one point of each cell's ideal
quota while hitting the exact total. It does **not** collapse each cell to one point or normalize
spatial density. Largest-remainder rounding distributes the integer quotas; cells with an ideal
quota below one may receive zero. There is no minimum-one guarantee. Normals are retained as
attributes but do not influence this sampling method. The 5 cm region size is configurable and
is an initial comparison setting, not a measured optimum.

For a simple random baseline, use the same exporter with `--method random`:

```bash
"$ORHSURF_PYTHON" -m orhsurf.simplify --source out/C001/00000 --out out/C001_random \
  --method random --targets 1M,5M --cpus 2
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

m = read_manifest(Path("data/C001_first150_prepared/manifest.json"))
serial = m["valid_serials"][0]  # choose a valid camera with a clear view of your region
camera = m["cameras"][serial]
frame_index = 0                # encoded MP4 index, NOT source video_frame_index
entry = camera["frames"][str(frame_index)]
print(entry["frame_path"], entry["mask_path"])
K = camera["K_original"]
distortion = camera["dist_params"]
world_to_camera = camera["T_cam_from_world"]
```

Converted default inputs are `data/C001_first150_prepared/rgb/<serial>/<index:05d>.png`, at 2048×1536 for C001.
They retain lens distortion: pair them with `K_original` and distortion coefficients. DA3's
undistorted images and the cropped/recentred training images use different intrinsics; see
[the camera contract](docs/DATA_CONTRACT.md). C001 spans 15 seconds at 15 fps; `0-4` are its first
five encoded timestamps. Actual timing metadata is in the manifest.

## Preview and sequence export

```bash
source env.sh
"$ORHSURF_PYTHON" -m pip install viser  # optional, once, on a compute node
orhsurf view --npz out/C001/00000/surface.npz \
  --manifest data/C001_first150_prepared/manifest.json --host 127.0.0.1 --port 8080 --budget '1.2 M'
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

## Speed / quality presets

`--preset` picks a point on the iteration curve. Measured on **frame 40 of one clip** (the
development ORH clip, not C001), same scene and same export filter, so only training changed:

**`fast` is the default on both paths** — `orhsurf run` and `orhsurf process` (and so
`slurm/process_clips.sbatch`). `--smoke` always runs `quality`: a smoke test on a cheaper recipe
would not tell you anything about a quality run. Pass `--preset quality` for the full recipe.

```bash
sbatch --array=0-3%2 slurm/process_clips.sbatch C001 C002 C003 C004              # economy
sbatch --array=0-3%2 slurm/process_clips.sbatch --preset quality C001 C002       # full
orhsurf process --clips C001 --preset balanced --gpus 8
```

| preset | iterations | densify | gaussians | support | wall/frame |
|---|---|---|---|---|---|
| `quality` | 7000 | 500/100 | 645,249 | **10.21** | 692 s |
| `balanced` | 3000 | 500/100 | 728,730 | 9.38 | 343 s |
| `economy` | 2000 | 500/**80** | 676,926 | 9.07 | **261 s** |
| `draft` | 1000 | 500/100 | 148,377 | 7.90 | 198 s |
| **`fast`** (default) | 2000 @ **`-r 4`** | 500/80 | 615,437 | 7.50 | **151 s**\* |

\* Training + export only. **End to end, with DA3, `fast` measures 307 s/frame** — see below.

`fast` is the only preset that changes the **raster**, and it is the only one whose isolation
threshold moves with it: it sets `--resolution 4` **and** `--nn-max-mm 10`. Do not set one without
the other — see below.

`support` is the mean number of cameras whose rendered depth agrees within `--consistency-mm`. It
is the pipeline's own quality signal and it is a **proxy**, so these were also compared in 3D
before being written down. They have **not** been re-checked across frames or clips.

An explicit flag overrides the preset, so `--preset economy --iterations 2500` is legal;
`--densify-from-iter` and `--densification-interval` are exposed for the same reason.

### What `fast` actually costs per frame, end to end

The preset table's times are **training + export only**, measured by reusing an already-built
scene, so they exclude the DA3 prior and the COLMAP preparation. Those cost the same whichever
preset you pick, and they dominate once you stop reusing a scene.

Five consecutive frames (40–44) through `orhsurf run --preset fast`, one GPU, sequential:

```
verify: 5/5 frames ok (expected 5), 32,550,219 points
wall:   1533 s  ->  307 s/frame, DA3 included
peak:   23,961 MiB of a 24,564 MiB card (97.5%)
```

| frame | points | support | gaussians |
|---|---|---|---|
| 00040 | 6,517,462 | 7.44 | 619,521 |
| 00041 | 6,516,671 | 7.38 | 622,720 |
| 00042 | 6,507,767 | 7.37 | 625,242 |
| 00043 | 6,502,430 | 7.46 | 628,527 |
| 00044 | 6,505,889 | 7.48 | 616,665 |

Point count varies by **0.23%** across the five, half the run-to-run noise floor, so the preset is
stable frame to frame — every earlier number in this section came from frame 40 alone.

**Size a job from 307 s/frame, not from 151 s.** And note the peak: even at group size 17 the DA3
stage leaves about **600 MB** of a 24 GB card unused, so a card sharing work with anything else
will still fail. `run` checks free VRAM and simply skips such GPUs, which is why asking for
`--gpus 5` on a busy box may dispatch onto two.

### DA3 view grouping, and why the default is 17 rather than 18

DA3 processes the 47 views in overlapping azimuth groups. The reference reconstruction used
**18** views per group with overlap 6 — four groups — and on an 80 GB card that is fine. On a
**24 GB card it is not**: DA3 at 1008 px peaks around **23.35 GB** against ~**23.70 GB** free on a
*completely idle* card, and ~350 MB of headroom is thinner than allocator fragmentation.

The failure does not announce itself as OOM. It arrives as

```
RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`
  at depth_anything_3/utils/ray_utils.py:229  torch.linalg.svd(A)
```

which sends you reading about linear-algebra backends instead of about memory. Five consecutive
frames failed this way on an otherwise empty 24 GB card at group size 18.

**17 is the right step down**, not 12 or 8: with overlap 6 it still cuts 47 views into **four**
groups, so DA3 sees the same view neighbourhoods with one fewer view each. Measured on frame 45:

| group size | groups | DA3 | points | support |
|---|---|---|---|---|
| **17** | **4** | 98.3 s | 6,553,117 | **7.55** |
| 12 | 7 | 91.7 s | 6,535,663 | 7.52 |
| 8 | 8 | **131.3 s** | 6,523,671 | 7.44 |

The point-count spread is 0.45%, inside the 0.50% run-to-run noise floor — headroom bought without
meaningfully changing the output. Going further is counterproductive: at 8 the per-group overhead
makes DA3 *slower* than at 17. Pass `--group-size 18` to reproduce the reference exactly on a card
with room for it.

### `fast`: a quarter of the points, and why the threshold has to move with the raster

A point is one **pixel** of one camera's rendered depth, so `-r 4` produces a cloud roughly 4x
sparser — **6.5 M points instead of 24-25 M** — and the k=5 nearest-neighbour distance roughly
doubles: measured **median 2.42 mm at `-r 2` against 4.985 mm at `-r 4`**. Leaving `--nn-max-mm` at
5.0 there puts the threshold on top of the median, and the export refuses outright:

```
AssertionError: the isolation gate would drop 51.2% of the cloud; nn-k=5 nn-max-mm=5.0
is wrong for this point density (k=5 distance median 4.985 mm)
```

**The two signals disagree here, and the disagreement is left visible on purpose.** By `support`
alone, `-r 4` with a 5 mm threshold scores *higher* than with 10 mm:

| `-r 4`, 3000 it | points | dropped as isolated | support |
|---|---|---|---|
| `--nn-max-mm 5` | 4,340,511 | 2,582,417 | **9.39** |
| `--nn-max-mm 10` | 6,636,905 | 286,023 | 7.70 |

So the 2.58 M points that 5 mm removes really are the poorer ones — that is not an artefact.
`fast` uses 10 mm anyway, because the clouds were compared in 3D and 10 mm was judged the better
surface. `support` is a proxy and the 3D reading decides; both numbers are here so a later reader
can revisit the call rather than inherit it.

Speed, for the same models, `-r 4`: 2000 iterations trains in 105 s, 3000 in 187 s, and export is
~45 s either way (export scales with points, not with training length). So `fast` at 151 s is
**4.6x** faster end-to-end than `quality` at 692 s — for a quarter of the points and support 7.50
against 10.21.

**What the speedup actually is.** `economy` is **2.65x** faster than `quality` on the part it
changes (training + export, 692 s -> 261 s). End to end it is less, because the DA3 prior and the
COLMAP preparation cost the same either way: with DA3 at ~148 s the frame goes from ~840 s to
~409 s, i.e. **about 2x**. Quote the second number when sizing a Slurm `--time`.

### Two things the sweep settled, against expectation

**Densifying harder is not better.** Starting earlier and running more often (`300/50`) multiplies
the Gaussian count and makes the result *worse*:

| iterations | densify | gaussians | support |
|---|---|---|---|
| 3000 | 500/100 (25x) | 728,730 | **9.38** |
| 3000 | 300/50 (54x) | 1,684,432 | 8.64 |
| 2000 | 500/100 (15x) | 504,506 | 9.02 |
| 2000 | 300/50 (34x) | 1,414,340 | 8.76 |

Over-split Gaussians stay small and unconverged, so the cameras agree with each other less. Note
too that `quality` reaches the best support with **fewer** Gaussians than `balanced` does — a good
reconstruction is well-placed Gaussians, not many of them. Only a mild increase (`500/80`, 18
passes at 2000) came out marginally ahead, which is what `economy` uses.

**Lower resolution buys speed without costing support, but costs points.** `-r 4` at 3000
iterations: support 9.39 (vs 9.38 at `-r 2`), 223 s, but **4.34 M points instead of 24.75 M** — a
point is one pixel of one camera's rendered depth, so the cloud shrinks with the raster, not with
the Gaussians. Export falls 107 s → 36 s; training only 236 s → 187 s, because roughly half of a
training step scales with Gaussian count rather than with pixels.

> `-r 3` **crashes** AmbiSuR around 40% of training (`utils/loss_utils.py:102`,
> `get_img_grad_weight` on an empty tensor). Use even divisors.

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
