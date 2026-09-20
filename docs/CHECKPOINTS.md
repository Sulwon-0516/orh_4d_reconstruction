# Checkpoints

**Only one model weight is downloaded: the Depth Anything 3 checkpoint.** Verified against the
code, not assumed:

- `third_party/AmbiSuR/train.py` and `orhsurf/_vendor/export_surface.py` load no pretrained weights.
  The only `load_state_dict` calls are `scene/app_model.py:30` and `scene/gaussian_model.py:128`,
  and both restore state the run itself produced.
- LPIPS/VGG weights appear only in `third_party/AmbiSuR/metrics.py`, `lpipsPyTorch/` and
  `scripts/stat/` — none of which this pipeline invokes. If you run upstream `metrics.py` yourself,
  torchvision will fetch VGG weights at that point; that is outside `orhsurf run`.
- COLMAP is a **binary**, not a weight; `install.sh` gets it from conda-forge.

## The one checkpoint

| | |
|---|---|
| model | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` |
| type | HuggingFace **model** repo |
| revision | `b2359bdf726fb44ef62acca04d629dcf158053e7` (pinned; do not track `main`) |
| file | `model.safetensors` + `config.json` |
| size | **6,759,558,100 B ≈ 6.76 GB** |
| sha256 | `8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c` |
| install path | `$ORHSURF_CACHE_DIR/hf/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf.../` |
| loaded by | `orhsurf/stages/da3_prior.py::run_da3_1008`, via `DepthAnything3.from_pretrained` |

### A second, different DA3 checkpoint exists — do not confuse them

The 504-resolution fallback path uses `DA3NESTED-GIANT-LARGE` (no `-1.1`), sha256
`8899faf998dedbc230261ab736fa57015280727399429122d44d4f9e7aac2ddd`. Same size, **different blob**.
`orhsurf run` does not use it. It is only relevant if you deliberately run the 504 prior.

## Fetching

```bash
./fetch_ckpts.sh                        # non-interactive; skips if already present
# or
source env.sh && orhsurf fetch --weights
```

Both honour `ORHSURF_CACHE_DIR`. On a cluster set that to scratch before fetching — 6.76 GB will
not fit in a typical $HOME quota:

```bash
export ORHSURF_CACHE_DIR=$SCRATCH/orhsurf/cache
```

## Offline / air-gapped compute nodes

Fetch on the login node, then run with the Hub disabled:

```bash
./fetch_ckpts.sh                        # login node, has network
export HF_HUB_OFFLINE=1                 # compute node
orhsurf doctor                          # must still report the checkpoint as cached
```

To move a checkpoint between machines by hand, copy the whole
`$ORHSURF_CACHE_DIR/hf/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1` directory, including
`blobs/`, `refs/` and `snapshots/`. Copying only the snapshot breaks the symlinks it contains.

## Dataset (not a checkpoint)

Clip data is a separate HuggingFace **dataset** repo — see `docs/DATA_CONTRACT.md`, which also
records where the published layout currently disagrees with what the pipeline needs.
