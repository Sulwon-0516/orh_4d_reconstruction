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

**Prerequisite:** `micromamba`, `mamba` or `conda` on `$PATH`. `install.sh` needs it for a pinned
Python 3.10 *and* for COLMAP (conda-forge ships it; PyPI does not). No root required:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$HOME" bin/micromamba
export PATH="$HOME/bin:$PATH"
```

```bash
git clone <this repo> orhsurf && cd orhsurf
./install.sh                     # two conda envs, CUDA extensions, DA3 weights (6.76 GB)
source env.sh

orhsurf doctor                   # check everything at once
orhsurf fetch --clip C001        # downloads + extracts the clip archive -- but see the warning below
orhsurf run --clip C001 --gpus 1 --frames 0-0     # smoke test: one frame, ~11 min
orhsurf verify --clip C001       # open every output and check it

> **`fetch --clip` does not yet give you a runnable clip.** It downloads and extracts the published
> archive, then tells you what is missing. The archive has no `manifest.json` in this pipeline's
> schema and **no foreground masks**, both of which `orhsurf run` requires. There is no converter in
> this package yet. See [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

`--clip` accepts a clip id (resolved under `ORHSURF_DATA_ROOT`), a clip directory, or a path to a
`manifest.json`. `run` and `verify` accept all three forms.
```

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
| Time | ~11 min/frame at `-r 2` × 7k. A 225-frame published clip on 8 GPUs ≈ 5 h |

A 16 GB GPU does not fit at the default `--group-size 18`. Lowering it works but **changes the
output** — DA3 predicts jointly over the group. See INSTALL_SLURM.md §0.

## Commands

| | |
|---|---|
| `orhsurf run` | the whole pipeline; `--frames`, `--gpus`, `--cpus-per-job`, `--shard/--shards` for job arrays |
| `orhsurf verify` | opens and decompresses every array, checks dtypes/shapes/lengths |
| `orhsurf doctor` | env, allocation, CUDA extensions, weights — one report |
| `orhsurf fetch` | DA3 weights and/or clip data |
| `orhsurf render` | debug renders from finished frames |

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

## Debug renders

Headless, offline, ffmpeg-based — no viser, no display. House convention: **shading → normal →
rgb**.

- **Always produced by every run:** one static contact sheet of the first frame, from **two**
  viewpoints 50° apart. Two, because a single view hides error along its own optical axis entirely.
- **Opt-in, individually flagged:** `--render-orbit` (static frame, moving camera), `--render-time`
  (all frames, static camera), `--render-both`.

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
- [ ] **`install.sh` on a clean machine.** It was written against the known-good dependency set but
      has not been run from scratch on a fresh host; the existing environments were reused for
      testing. The CUDA extension build in particular is untested here.
- [ ] **`orhsurf fetch --clip`.** The published dataset layout does not match the pipeline's input
      contract (see below), so the clip fetch path cannot be exercised yet.
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
- [ ] Manifests store absolute paths, which do not survive a move between machines.
