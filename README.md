# orhsurf

One multi-view clip → one filtered surface point cloud per timestamp, in calibrated world coordinates.

## TL;DR — install once, submit all clips with one command

Each clip defaults to **the first 150 frames (0–149)**: **10 seconds at 15 fps**, not
150 fps. Every selected frame uses the unchanged reconstruction recipe. The workflow downloads
inputs as needed, prepares them, reconstructs them, and verifies all expected outputs.

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
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --simplify 5M --simplify-only --cleanup-decoded C{001..100}
```

To save both durations with the same cleanup policy:

```bash
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --durations 10,15 --simplify 5M --simplify-only --cleanup-decoded C{001..100}
```

For example, the 15-second 5M output is
`$MODEL_OUTPUT_DIR/15s/_simplified/C001/random/5M/00000/surface.npz`.
Verify that version with `orhsurf verify --out "$MODEL_OUTPUT_DIR/15s/C001"`.

This retains only **5M random-sampled points per frame** plus metadata. Choose `10M`, `5M`,
`1M`, or a comma-separated list such as `10M,5M,1M`. The equivalent interactive command is:

```bash
orhsurf process --clips C001 C002 --gpus 1 --simplify 5M --simplify-only --cleanup-decoded
```

- `--simplify-only` requires `--simplify`. All requested derivatives must pass verification
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
orhsurf process --clips C001 C002 --gpus 1 --simplify 10M,5M,1M
# Or submit all clips; choose just --simplify 5M to save only the 5M version:
sbatch "${SBATCH_SITE[@]}" --array=0-99%10 --gres=gpu:8 slurm/process_clips.sbatch --gpus 8 --simplify 10M,5M,1M C{001..100}
```

After each clip's reconstruction and verification, CPU postprocessing saves every selected frame
under `<output-root>/_simplified/<clip>/random/{10M,5M,1M}/<frame>/surface.npz`.
Each derived frame is verified, records provenance, and preserves float32 attributes. Matching
completed derivatives are verified and reused on rerun; incompatible outputs are not overwritten.
Original clouds remain in `<output-root>/<clip>/<frame>/` unless `--simplify-only` is supplied. Without `--simplify`, no derivatives
are generated. A requested point count greater than the source count fails explicitly.
Use `--simplify-method stratified` or `--simplify-method normal-voxel` for alternatives.
This postprocessing does not reduce GPU reconstruction time. Keeping originals plus derivatives
increases storage; choose a retention policy after inspecting quality.

For a previously completed frame, use the standalone simplifier (also random by default):

```bash
"$ORHSURF_PYTHON" -m orhsurf.simplify --source out/C001/00000 --out out/C001_simplified \
  --targets 10000000,5000000,1000000 --cpus 2
```

Outputs: `out/C001_simplified/{10M,5M,1M}/00000/surface.npz`, with completion markers and
provenance. The original is preserved. The same NPZ loader and viewer work on these files.
Use a thread count within your allocation. Float attributes remain **float32**.

With `--method normal-voxel`, the method groups by spatial voxel **and normal direction** (30° maximum within a normal bin),
retains the highest-support original point per group, then trims surplus representatives with
a fixed random seed to hit the exact budget. Each target is computed from the original; versions
are not nested. Voxel size and trim counts are recorded. This is a practical approximation,
not an optimal nearest-neighbour/mesh decimator or a guarantee against losing thin structures.
No coordinates, colours or support values are averaged. Zero-length source normals use a separate
unknown-direction bin. See [library alternatives](detail.md#point-cloud-simplification-alternatives).
Keeping the original and all three versions adds storage; savings require choosing which derived
version to retain later. Only the explicit `--simplify-only` process option deletes original cloud payloads after verification.

For density-preserving **spatial stratified sampling**, assign every 5 cm cell a point quota
proportional to its original population, then randomly choose within each cell:

```bash
"$ORHSURF_PYTHON" -m orhsurf.simplify --source out/C001/00000 --out out/C001_stratified \
  --method stratified --voxel-mm 50 --targets 10000000,5000000,1000000 --cpus 2
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
