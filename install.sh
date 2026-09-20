#!/usr/bin/env bash
# orhsurf installer -- non-interactive, re-runnable, no root required.
#
#   ./install.sh                     # everything: both envs, CUDA extensions, weights
#   ./install.sh --no-weights        # skip the 6.76 GB DA3 download (pre-stage it separately)
#   ./install.sh --envs-only         # envs + deps, no CUDA extensions, no weights
#   ./install.sh --extensions-only   # just (re)build the CUDA extensions -- run this under srun
#
# WHY TWO ENVIRONMENTS.  Not a style choice, a measurement.  Running the identical DA3 1008
# pose-conditioned inference under torch 2.6.0+cu124 and torch 2.7.1+cu128 gives depths that differ
# by ~900x the run-to-run nondeterminism (mean |d| 1.2e-3 m vs 1.3e-6 m; p99.9 3.5e-2 m vs 4.8e-7 m).
# The reference reconstruction was produced on 2.6.0+cu124, so the DA3 stage is pinned there and
# AmbiSuR keeps 2.7.1+cu128.  See orhsurf/paths.py::da3_python and tools/da3_env_compare.py.
#
# ON A SLURM CLUSTER: run the CUDA-extension build on a COMPUTE node, not the login node --
#     srun --gres=gpu:1 --cpus-per-task=8 --time=1:00:00 --pty ./install.sh --extensions-only
# Login nodes usually have no GPU, a different CPU architecture, and hard CPU/RAM limits that kill
# nvcc. Full notes: docs/INSTALL_SLURM.md   (the Slurm parts of which are UNTESTED -- read them.)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DO_ENVS=1; DO_EXT=1; DO_WEIGHTS=1
for arg in "$@"; do
  case "$arg" in
    --no-weights)      DO_WEIGHTS=0 ;;
    --envs-only)       DO_EXT=0; DO_WEIGHTS=0 ;;
    --extensions-only) DO_ENVS=0; DO_WEIGHTS=0 ;;
    --no-extensions)   DO_EXT=0 ;;
    -h|--help)         sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "install.sh: unknown option '$arg'" >&2; exit 2 ;;
  esac
done

# --- knobs (all overridable from the environment) ---------------------------------------
PY_VERSION="${ORHSURF_PY_VERSION:-3.10}"
ENV_AMBISUR="$HERE/env"          # torch 2.7.1+cu128, AmbiSuR + export + render
ENV_DA3="$HERE/env-da3"          # torch 2.6.0+cu124, DA3 1008 prior ONLY
CACHE="${ORHSURF_CACHE_DIR:-$HERE/cache}"
# Build parallelism: respect the allocation. NEVER nproc here -- inside a Slurm cgroup it reports
# the whole node and nvcc will fork enough jobs to get the step OOM-killed.
JOBS="${ORHSURF_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

say "orhsurf install"
echo "  repo          $HERE"
echo "  ambisur env   $ENV_AMBISUR   (torch 2.7.1+cu128)"
echo "  da3 env       $ENV_DA3   (torch 2.6.0+cu124)"
echo "  cache         $CACHE"
echo "  build jobs    $JOBS  (from ${SLURM_CPUS_PER_TASK:+SLURM_CPUS_PER_TASK}${SLURM_CPUS_PER_TASK:-default})"
mkdir -p "$CACHE"/{hf,torch,torch_ext,triton,pip}
export PIP_CACHE_DIR="$CACHE/pip" HF_HOME="$CACHE/hf" TORCH_HOME="$CACHE/torch"
export MAX_JOBS="$JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

# --- environment creation ----------------------------------------------------------------
# micromamba/conda preferred over venv: this stack needs a specific Python (3.10) and, if COLMAP
# is not already installed site-wide, a COLMAP binary -- and conda-forge ships one, which is the
# only realistic no-root way to get COLMAP. uv/venv cannot provide either.
find_conda() {
  for c in micromamba mamba conda; do command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }; done
  echo ""
}
CONDA="$(find_conda)"

create_env() {  # <path> <extra conda pkgs...>
  local path="$1"; shift
  if [ -x "$path/bin/python" ]; then echo "  exists: $path (reusing)"; return; fi
  [ -n "$CONDA" ] || die "no micromamba/mamba/conda found.
  Install micromamba without root:
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C \"\$HOME\" bin/micromamba
    export PATH=\"\$HOME/bin:\$PATH\"
  then re-run this script."
  echo "  creating $path with $CONDA"
  "$CONDA" create -y -p "$path" --override-channels -c conda-forge \
      "python=$PY_VERSION" "$@" >/dev/null
}

if [ "$DO_ENVS" = 1 ]; then
  say "1/4  python environments"
  # COLMAP goes in the AmbiSuR env: it is needed for database_creator / image_undistorter /
  # model_converter in the dataset stage.
  # COLMAP >= 3.11. NOTE: image_undistorter does NOT expose --num_threads even in 3.13.0
  # (verified), so orhsurf bounds it by restricting the child's CPU affinity instead; the flag is
  # probed at run time and used if a future version adds it. See _vendor/colmap_dataset.py.
  create_env "$ENV_AMBISUR" "colmap>=3.11"
  create_env "$ENV_DA3"

  PA="$ENV_AMBISUR/bin/python"; PD="$ENV_DA3/bin/python"

  say "2/4  dependencies (AmbiSuR env: torch 2.7.1+cu128)"
  "$PA" -m pip install -q --upgrade pip
  "$PA" -m pip install -q torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
  "$PA" -m pip install -q "numpy<2.3" "opencv-python==4.11.0.86" "scipy>=1.11" \
        "plyfile" "tqdm" "huggingface_hub>=0.34" "safetensors" "einops" "e3nn" "Pillow" \
        "imageio" "matplotlib"
  # imageio is NOT optional: train.py -> utils/mono_utils.py imports it at module scope, so a
  # clean environment used to die at the first training step -- after paying for the 6.76 GB
  # checkpoint download. Smoke-import the training dependency graph so that can never recur.
  echo "  smoke-importing the training dependency graph"
  ( cd "$HERE/third_party/AmbiSuR" && PYTHONPATH="$HERE:$HERE/third_party/AmbiSuR" "$PA" - <<'SMOKE'
import importlib, sys
for m in ("imageio", "cv2", "scipy.spatial", "plyfile", "matplotlib",
          "utils.mono_utils", "utils.graphics_utils", "utils.general_utils",
          "scene.cameras", "arguments"):
    importlib.import_module(m)
print("  [ok ] training imports resolve")
SMOKE
  ) || { echo "training dependency graph is incomplete (see above)"; exit 1; }
  # NOTE: pytorch3d is deliberately NOT installed. AmbiSuR imported it for exactly one function;
  # orhsurf/quat.py replaces it and tests/test_quat.py proves the two agree to 1e-15.

  say "    dependencies (DA3 env: torch 2.6.0+cu124)"
  "$PD" -m pip install -q --upgrade pip
  "$PD" -m pip install -q torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124
  "$PD" -m pip install -q xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124
  "$PD" -m pip install -q "numpy==2.2.6" "opencv-python==4.11.0.86" "huggingface_hub>=0.34" \
        "hf-xet" "safetensors" "einops" "e3nn"
  if [ -d "$HERE/third_party/Depth-Anything-3" ]; then
    "$PD" -m pip install -q -e "$HERE/third_party/Depth-Anything-3"
  else
    "$PD" -m pip install -q \
      "git+https://github.com/ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
  fi
  # Upstream pins numpy<2; 2.2.6 is what the reference env has and DA3 imports and runs with it.
fi

# --- CUDA extensions ----------------------------------------------------------------------
if [ "$DO_EXT" = 1 ]; then
  say "3/4  CUDA extensions (diff-plane-rasterization-ambisur, simple-knn)"
  PA="$ENV_AMBISUR/bin/python"
  [ -x "$PA" ] || die "$PA missing; run without --extensions-only first"
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "  WARNING: nvcc not on PATH. On a cluster try:  module load cuda/12.8"
    echo "           The build below will fail without it."
  else
    echo "  nvcc: $(nvcc --version | tail -1)"
  fi
  if ! "$PA" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "  WARNING: no GPU visible to torch."
    echo "           These extensions compile for the CURRENT GPU's architecture. Building on a"
    echo "           login node usually produces binaries the compute nodes cannot load."
    echo "           Set TORCH_CUDA_ARCH_LIST (e.g. 8.9 for RTX 4090 / L40S, 8.0 for A100,"
    echo "           9.0 for H100) or build under: srun --gres=gpu:1 ./install.sh --extensions-only"
  fi
  for ext in submodules/diff-plane-rasterization-ambisur submodules/simple-knn; do
    d="$HERE/third_party/AmbiSuR/$ext"
    [ -d "$d" ] || die "missing extension source: $d"
    echo "  building $ext (MAX_JOBS=$MAX_JOBS)"
    "$PA" -m pip install -q --no-build-isolation "$d"
  done
fi

# --- weights ---------------------------------------------------------------------------------
if [ "$DO_WEIGHTS" = 1 ]; then
  say "4/4  model weights"
  PD="$ENV_DA3/bin/python"
  echo "  DA3NESTED-GIANT-LARGE-1.1, 6.76 GB -> $CACHE/hf"
  PYTHONPATH="$HERE" HF_HOME="$CACHE/hf" "$PD" -c "
from pathlib import Path
from orhsurf.fetch import fetch_weights
raise SystemExit(fetch_weights(Path('$CACHE')))"
fi

# --- verify -----------------------------------------------------------------------------------
say "verifying"
cat > "$HERE/env.sh" <<EOF
# Source this before running orhsurf. Generated by install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
export ORHSURF_PYTHON="$ENV_AMBISUR/bin/python"
export ORHSURF_DA3_PYTHON="$ENV_DA3/bin/python"
export ORHSURF_COLMAP_BIN="\${ORHSURF_COLMAP_BIN:-$ENV_AMBISUR/bin/colmap}"
export ORHSURF_CACHE_DIR="$CACHE"
export ORHSURF_DATA_ROOT="\${ORHSURF_DATA_ROOT:-$HERE/data}"
export ORHSURF_OUT_ROOT="\${ORHSURF_OUT_ROOT:-$HERE/out}"
export PYTHONPATH="$HERE:\${PYTHONPATH:-}"
export HF_HOME="$CACHE/hf"
export PATH="$HERE/bin:\${PATH:-}"
# `orhsurf` is a real executable in $HERE/bin, NOT an alias: aliases are not expanded by
# non-interactive shells, so the documented `bash -c 'source env.sh && orhsurf ...'` and every
# sbatch script would otherwise fail with "orhsurf: command not found".
EOF
mkdir -p "$HERE/bin"
cat > "$HERE/bin/orhsurf" <<'EOF'
#!/usr/bin/env bash
# orhsurf launcher. Resolves the interpreter from env.sh if it is not already exported.
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${ORHSURF_PYTHON:-}" ] && [ -f "$_here/env.sh" ]; then . "$_here/env.sh"; fi
PY="${ORHSURF_PYTHON:-$_here/env/bin/python}"
[ -x "$PY" ] || { echo "orhsurf: no interpreter at $PY; run install.sh" >&2; exit 1; }
export PYTHONPATH="$_here:${PYTHONPATH:-}"
exec "$PY" -m orhsurf.cli "$@"
EOF
chmod +x "$HERE/bin/orhsurf"
echo "  wrote $HERE/bin/orhsurf (executable)"
echo "  wrote $HERE/env.sh"

# shellcheck disable=SC1090
source "$HERE/env.sh"
"$ORHSURF_PYTHON" "$HERE/tests/test_quat.py" >/dev/null && echo "  [ok ] quaternion replacement"
# Phase-aware: an envs-only / login-node install has no extensions, no GPU and no weights yet, so
# the full check would always "fail" there. Each environment is checked with ITS OWN interpreter --
# DA3 lives only in env-da3, and asking the AmbiSuR interpreter for it always failed.
DOCTOR_PHASE="full"
[ "$DO_EXT" = 1 ]     || DOCTOR_PHASE="envs"
[ "$DO_WEIGHTS" = 1 ] || [ "$DOCTOR_PHASE" = "envs" ] || DOCTOR_PHASE="noweights"
echo "  doctor phase: $DOCTOR_PHASE"
"$ORHSURF_PYTHON" -m orhsurf.cli doctor --phase "$DOCTOR_PHASE" || {
  echo
  echo "install.sh finished but 'doctor' reports problems (above)."
  echo "Common fixes are in docs/INSTALL_SLURM.md."
  exit 1
}

cat <<EOF

Done. To use it:

    source $HERE/env.sh
    orhsurf fetch --clip <CLIP_ID>          # pull the data (login node; needs network)
    orhsurf run --clip <CLIP_ID> --gpus 1 --frames 0-0     # smoke test: one frame
    orhsurf verify --clip <CLIP_ID>

The DA3 stage needs a GPU with >= 24 GB. See docs/INSTALL_SLURM.md for smaller GPUs, Slurm
submission, offline pre-staging and scratch/quota notes.
EOF
