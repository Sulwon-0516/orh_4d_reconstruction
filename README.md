# orhsurf

Multi-view surface reconstruction for room-scale capture rigs: one clip in, a filtered per-frame
surface point cloud out.

```
prep  →  DA3 depth prior @1008 (our poses passed in)  →  AmbiSuR 7k @ -r 2  →  export + filter
```

**One command does all of it.** There is no separate filtering pass and nothing to remember
afterwards:

```bash
orhsurf run --clip C001 --gpus 8 --frames 0-224
```

When it returns, every frame has a filtered `surface.npz` on disk with a `_DONE.json` marker beside
it, plus a sanity render of the first frame.

---

## Validated end-to-end run

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

## Quick start

**Prerequisites:** `micromamba`, `mamba` or `conda` on `$PATH`, and a network that can reach
`download.pytorch.org` over **IPv4 or IPv6**. Its CDN resolves IPv6-only on some networks, and an
IPv4-only host then fails the torch install with a confusing name-resolution error — see
[docs/INSTALL_SLURM.md](docs/INSTALL_SLURM.md#if-the-torch-download-fails-with-a-dns-error). `install.sh` needs it for a pinned
Python 3.10 *and* for COLMAP (conda-forge ships it; PyPI does not). No root required:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$HOME" bin/micromamba
export PATH="$HOME/bin:$PATH"
```

```bash
git clone <this repo> orhsurf && cd orhsurf
./install.sh --no-weights        # two Python envs + CUDA extensions; use a compute node
source env.sh
orhsurf doctor --phase noweights

# One published HEVC clip, only the five frames needed for this test:
orhsurf fetch --clip C001 --convert --frames 0-4
```

The conversion command downloads only `hevc/C001.tar`, extracts into `data/_hevc/C001`, and
writes decoded images plus `data/C001_prepared/manifest.json`. An existing JPEG clip at
`data/C001` is preserved. `fetch --clip C001` without `--convert` still prefers the JPEG archive.
Conversion decodes on CPU, with thread counts limited by the current allocation. Downloads and
conversion may run inside an existing GPU compute allocation; no separate Slurm submission is
inherently required. Follow your site's rules and do not run heavy work on a login node.

**The published clip has no masks.** Without masks, conversion writes the images and manifest but
returns **exit code 3**, explicitly reporting missing inputs. This is not reconstruction success.
Do not substitute empty or all-white masks. Provide validated RGBA foreground masks at
`<mask-dir>/<camera-serial>/<encoded-frame-index:05d>.png`. Then:

```bash
orhsurf fetch --clip C001 --convert --frames 0-4 --masks /absolute/path/to/validated/masks
orhsurf fetch --weights
orhsurf doctor
orhsurf run --clip data/C001_prepared/manifest.json --gpus 1 --frames 0-0
orhsurf verify --clip data/C001_prepared/manifest.json
# Inspect the first frame and its preview before expanding; completed frame 0 is reused.
orhsurf run --clip data/C001_prepared/manifest.json --gpus 1 --frames 0-4
orhsurf verify --clip data/C001_prepared/manifest.json
```

`--clip` accepts a clip id under `ORHSURF_DATA_ROOT`, a directory, or a manifest path.
Relative image/mask paths are resolved against the manifest directory, without rewriting the
original JSON. Converted frame subsets retain their true encoded indices, including nonzero or
noncontiguous selections. The converter does not generate masks; see [the input contract](docs/DATA_CONTRACT.md).

For an already-extracted HEVC archive, `orhsurf.fetch.convert_clip(clip_dir, out_dir, masks=..., frames="0-4")`
can be called directly without downloading or extracting again.

On a Slurm cluster read **[docs/INSTALL_SLURM.md](docs/INSTALL_SLURM.md)** first — it covers
no-root setup, `module load`, building the CUDA extensions on a compute node, offline pre-staging,
scratch quotas and submission. Its Slurm-specific parts are marked **UNTESTED**; they were written
from documented behaviour, not verified on a real cluster.

## Requirements

| | |
|---|---|
| **GPU** | **≥ 24 GB.** Set by the DA3 stage (peaks **23,353 MiB** torch-allocated, measured), not by training. The margin on a 24 GB card is ~3%, so another user's process on the same GPU will OOM it; `run` checks free VRAM before dispatching. |
| CPU | 8 cores per concurrent job is enough (one job uses ~280%) |
| Disk | ~620 MB per frame (~93 GB per 150-frame clip), plus ~6 GB transient scratch per frame |
| Time | 11–19 min/frame at `-r 2` × 7k, depending on concurrency. See **Capacity planning** — do not size a job from the 11 min figure alone. |

A 16 GB GPU does not fit at the default `--group-size 18`. Lowering it works but **changes the
output** — DA3 predicts jointly over the group. See INSTALL_SLURM.md §0.

## Commands

| | |
|---|---|
| `orhsurf run` | the whole pipeline; `--frames`, `--gpus`, `--cpus-per-job`, `--shard/--shards` for job arrays |
| `orhsurf verify` | opens and decompresses every array, checks dtypes/shapes/lengths |
| `orhsurf doctor` | env, allocation, CUDA extensions, weights — one report |
| `orhsurf fetch` | DA3 weights or one clip; `--convert --frames 0-4` decodes HEVC inputs |
| `orhsurf render` | headless debug renders (PNG / mp4) from finished frames |
| `orhsurf view` | **interactive viewer** (viser). Optional: `pip install viser` |

Staged subcommands exist for debugging, but `run` is the path.

## The recipe, and what is settled

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

## Interactive viewer

```bash
pip install viser                      # optional extra; install.sh does not add it
orhsurf view --clip C001               # defaults to the first COMPLETED frame
orhsurf view --npz <path>/surface.npz --manifest <path>/manifest.json
```

Colour modes **rgb / shaded / normal / facing / support**, a points-drawn budget (a 25 M-point
cloud will not stream at full density), and a **`support >=` slider** that re-filters at read time
without re-exporting.

**`facing` is the one that earns its place.** Green where a normal points at the nearest camera,
red where it points away. `shaded` uses `|n·L|` and is blind to normal sign *by construction* —
which is precisely why this project's always-on sanity render could not see that every exported
normal was back-facing. **~25% red is expected** ("nearest" is a proxy for "producing" camera); a
**mostly red** cloud means the sign bug is back.

It binds `0.0.0.0`, so on a cluster forward the port — a compute node is not reachable directly:

```bash
ssh -L 8080:<compute-node>:8080 <user>@<login-node>
# then open http://localhost:8080
```

## Debug renders

Headless, offline, ffmpeg-based — no viser, no display. House convention: **shading → normal →
rgb**.

- **Always produced by every run:** one static contact sheet of the first frame, from **two**
  viewpoints 50° apart. Two, because a single view hides error along its own optical axis entirely.
- **Opt-in, individually flagged:** `--render-orbit` (static frame, moving camera), `--render-time`
  (all frames, static camera), `--render-both`.

## Output format

One directory per frame:

```
<out>/<clip-id>/<frame:05d>/
    surface.npz           the product
    surface_export.json   point counts + every filter parameter actually used
    provenance.json       full recipe, stage timings, CPU budget, DA3 revision
    metadata.json         timestamp, method, source manifest
    _DONE.json            written LAST, after every payload is fsynced
    surface.ply           only with --write-ply
```

**What a reader should check, in order:**
1. `_DONE.json` — **its absence means the frame is not finished.** It records every file's size, so
   a truncated payload is caught without decompressing anything. Never infer completion from
   `surface.npz` existing.
2. `surface_export.json` — `n_points` should equal the NPZ's array length (`orhsurf verify` checks
   this), and the filter parameters record what was actually applied rather than what a default
   would have given.
3. `provenance.json` — the recipe hash, per-stage wall times, the CPU budget that applied, and the
   pinned DA3 revision.

### `surface.npz` schema

| array | dtype | shape | units / range | meaning |
|---|---|---|---|---|
| `xyz` | float32 | (N,3) | world metres | point position, manifest world frame |
| `normal` | float32 | (N,3) | unit vector | surface normal, **flipped to face the producing camera** |
| `rgb` | uint8 | (N,3) | 0–255 | the ground-truth photo's colour, not the model's |
| `confidence` | float32 | (N,) | 0–1 | `clip((support - min_views) / 6, 0, 1)` |
| `observed` | bool | (N,) | — | currently **all True**; reserved |
| `support` | int16 | (N,) | 1..n_cameras | how many cameras' rendered depth agreed within `--consistency-mm` |

**`support` is the post-hoc filter handle, and it is the most useful field after `xyz`.** A stricter
`support >= k` can be applied at read time without re-exporting and without retraining — that is
how the filter variants in this project were compared. `orhsurf view` exposes it as a slider.

`surface.ply` is a **strict subset**: `xyz`, `normal`, `rgb` only — no `confidence`, `observed` or
`support`, so it cannot be re-filtered. Nothing in the pipeline reads it back; it exists for
MeshLab. It is off by default because it is expensive (see below).

## Capacity planning

Size a Slurm allocation from this table rather than from prose. Every figure is measured; the
conditions are given because some of them disagree.

| | per frame | per 150-frame clip |
|---|---|---|
| output (`surface.npz`) | ~620 MB | **~93 GB** |
| + PLY (`--write-ply`) | ~1.3 GB | **~198 GB** |
| transient scratch | ~6 GB (peak, one frame at a time) | — |
| wall time, 1 GPU | **11–19 min** — see the note | — |
| wall time, 8 GPUs | **unmeasured** | ~3.5 h *if* frames were independent; they are not |

Plus a **one-off ~32 GB install**: `env` 8.7 GB + `env-da3` 7.3 GB + DA3 checkpoint 6.76 GB + HF
cache 9.7 GB (measured on a clean machine).

`surface.npz` measured at 616,913,984 B and 617,142,152 B for N ≈ 25.4 M points.

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

## Input data

See **[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)** for the `manifest.json` schema, camera
conventions and output format.

> **Note:** the published HuggingFace dataset does not yet match this contract — most importantly
> it does not ship foreground masks, which this pipeline requires. The mismatch is documented in
> full at the top of DATA_CONTRACT.md and needs a decision before a stranger can run this.
>
> **Masks do not make the output foreground-only.** Training uses the unmasked RGB and export
> validity uses opacity/depth; masks affect only the camera ordering for the DA3 prior. The
> exported cloud is the whole scene. See DATA_CONTRACT.md.

## Layout

```
install.sh              non-interactive, re-runnable, no root
env.sh                  written by install.sh; source it
orhsurf/
  cli.py                subcommands
  pipeline.py           one frame, end to end
  alloc.py              Slurm-aware CPU/GPU allocation (never probes the node)
  atomicio.py           atomic writes, DONE markers, verify
  render.py             headless debug renders
  quat.py               the PyTorch3D replacement
  stages/               prep, DA3 1008 + rewarp
  _vendor/              COLMAP dataset + scene build + surface export, de-hardcoded
third_party/AmbiSuR/    vendored, patched (no pytorch3d)
slurm/                  job array + single-node sbatch
docs/                   INSTALL_SLURM.md, DATA_CONTRACT.md
tools/                  verification scripts for the claims above
tests/
```

## Licensing

AmbiSuR is vendored under `third_party/AmbiSuR` with its upstream `LICENSE.md`; check its terms
before redistributing. Depth Anything 3 is fetched at install time, not vendored. The DA3
checkpoint is downloaded from HuggingFace and is **not** committed here.

---

## Status / TODO

This repo is deliberately pushed early: it runs end to end today, and the remaining items are
refinements rather than blockers. What is and is not proven:

**Proven on the development machine**
- End-to-end single command, 2 frames across 2 GPUs (exact command and counts above).
- DA3 stage reproduces the reference recipe's memory profile exactly (23,353 MiB peak, worst
  frame coverage 1.00000, 50,000-point initial cloud).
- Durability properties, against deliberate truncation and a simulated crashed re-run
  (`tests/test_atomicio.py`).
- PyTorch3D replacement equals upstream to 1.1e-15 (`tests/test_quat.py`).
- Two-env necessity (`tools/da3_env_compare.py`).

**Not yet proven — do not assume these work**
- [ ] **Everything Slurm.** No Slurm exists on the development machine. `slurm/*.sbatch` and the
      Slurm half of `docs/INSTALL_SLURM.md` are written from documented behaviour and are marked
      UNTESTED in place. Run `orhsurf doctor` inside an interactive allocation first.
- [x] Clean Python 3.10 environments and CUDA extensions installed and smoke-tested on A100.
      `doctor --phase noweights`, CUDA matmul/backward, simple-knn, rasterizer visibility, and
      xformers attention/backward pass. This does not substitute for full model inference.
      DA3 retains its documented numpy<2 metadata conflict with the pinned numpy 2.2.6.
- [x] C001 JPEG fetch and HEVC conversion are exercised on a compute node. Missing masks remain an explicit blocker; this does not establish end-to-end reconstruction.
- [ ] Full 150-frame clip, and 8-GPU scaling (only 2 GPUs were exercised).
- [ ] The debug video renders (`--render-orbit/--render-time/--render-both`). The always-on static
      check render is exercised; the video paths are not.

**Blocked on a decision**
- [ ] **The published dataset ships no foreground masks**, which this pipeline requires in two
      places (AmbiSuR's per-view alpha, and the visual hull that orders cameras for DA3). Either
      masks get published alongside the clips, or a mask-generation stage has to be added. See
      `docs/DATA_CONTRACT.md`.
- [ ] Our own `manifest.json` carries `mask_path: None`; the masks live in a second file,
      `manifest_fg.json`. A published manifest needs them in one place.
- [x] Relative manifest paths resolve against the manifest directory. Obsolete absolute paths still require explicit remapping.
