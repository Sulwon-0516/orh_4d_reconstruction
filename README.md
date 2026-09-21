# orhsurf

One multi-view clip → one filtered surface point cloud per timestamp, in calibrated world coordinates.

## TL;DR — one GPU, first frame, then five

Use a **40 GB+ GPU** for the default recipe (tested on A100 80 GB). Have conda/mamba,
CUDA Toolkit and a supported compiler on PATH. Run installation and processing on a compute node.

```bash
# Only if you do not already have a compute allocation (site-specific partition/account):
srun --partition=debug --gres=gpu:1 --time=03:00:00 --pty bash

# Inside the allocated compute shell:
git clone https://github.com/Sulwon-0516/orh_4d_reconstruction.git
cd orh_4d_reconstruction
./install.sh --no-weights
source env.sh
orhsurf fetch --weights
orhsurf fetch --clip C001 --convert --frames 0-4
orhsurf doctor

orhsurf run --clip data/C001_prepared/manifest.json --gpus 1 --frames 0-0 --out out/C001
orhsurf verify --out out/C001
# Inspect out/C001/_check/check_00000.png, then reuse completed frame 0:
orhsurf run --clip data/C001_prepared/manifest.json --gpus 1 --frames 0-4 --out out/C001
orhsurf verify --out out/C001                   # expect 5/5
```

For an existing checkout/environment, start at `source env.sh`; reuse prepared data and weights.
`debug` and its three-hour limit are from the tested cluster: use your site's actual partition
and required account. No CPU/memory allocation is assumed in these commands; program threads
are derived from the allocation. Downloads can happen inside that allocation.
The install, weights and clip downloads need network access; reconstruction can run offline.

<details>
<summary>Full clip, batch submission, installation and debugging details</summary>

Choose the full range **before conversion**: replace both conversion and run selections with
`--frames 0-224`. C001 contains 225 frames. Five decoded frames cannot serve a full-clip run.
Do not overwrite the input manifest of an active run. Full-clip processing on one GPU is estimated
at about 54 hours, so a three-hour debug allocation cannot finish it.
Re-run the same reconstruction command with the same manifest, recipe and output root to resume
completed frames in a later allocation; interrupted frames may require reconstruction again.
There is no automatic resubmission loop.

For batch use, put `source env.sh`, `run` and `verify` in a shell script and submit it from the
checkout with your site's real partition/account/time options. Existing `slurm/*.sbatch` files
are multi-GPU/array templates with resource assumptions; review them before use.
See [detail.md](detail.md) for installation, conversion, resume semantics, experiments and
[Slurm notes](docs/INSTALL_SLURM.md) for deployment setup.

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

m = read_manifest(Path("data/C001_prepared/manifest.json"))
serial = m["valid_serials"][0]  # choose a valid camera with a clear view of your region
camera = m["cameras"][serial]
frame_index = 0                # encoded MP4 index, NOT source video_frame_index
entry = camera["frames"][str(frame_index)]
print(entry["frame_path"], entry["mask_path"])
K = camera["K_original"]
distortion = camera["dist_params"]
world_to_camera = camera["T_cam_from_world"]
```

Converted inputs are `data/C001_prepared/rgb/<serial>/<index:05d>.png`, at 2048×1536 for C001.
They retain lens distortion: pair them with `K_original` and distortion coefficients. DA3's
undistorted images and the cropped/recentred training images use different intrinsics; see
[the camera contract](docs/DATA_CONTRACT.md). C001 spans 15 seconds at 15 fps; `0-4` are its first
five encoded timestamps. Actual timing metadata is in the manifest.

## Preview and sequence export

```bash
source env.sh
"$ORHSURF_PYTHON" -m pip install viser  # optional, once, on a compute node
orhsurf view --npz out/C001/00000/surface.npz \
  --manifest data/C001_prepared/manifest.json --host 127.0.0.1 --port 8080 --budget '1.2 M'
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

DA3 peaked at **24,745 MiB** in this run; a nominal 24 GB card is insufficient for this observed
case. **40 GB+ is recommended**, A100 80 GB is tested. The runtime's free-memory guard is only
a preliminary check, not a guarantee. Host-memory peak has not been measured in this run.
Inspect quota using site-approved tools; avoid recursive shared-storage scans or continuous polling.

<details>
<summary>Background, numerical reproducibility, installation troubleshooting and licensing</summary>

See [detail.md](detail.md). It contains the two-environment rationale, default recipe,
mask semantics, atomic output/resume behavior, historical benchmarks, known limitations and
upstream license references. [docs/INSTALL_SLURM.md](docs/INSTALL_SLURM.md) covers cluster setup;
[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) specifies calibration and manifest fields.

</details>
