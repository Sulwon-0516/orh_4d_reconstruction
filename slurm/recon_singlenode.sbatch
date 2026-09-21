#!/bin/bash
# orhsurf on ONE node with N GPUs, alternative to the job array.
#
#   sbatch slurm/recon_singlenode.sbatch <CLIP_ID> [FRAMES] [NGPUS]
#
# Use this when you hold a whole node anyway (a reservation, or a single-node partition). The job
# array in recon_array.sbatch is the better default: it queues more easily and a preemption costs
# one frame rather than the whole allocation. This script's scheduler is orhsurf's own -- one
# worker PROCESS per GPU, each taking a contiguous in-order slice of the frame list.
#
# UNTESTED on a real Slurm cluster. Check the <<< CHECK lines. See docs/INSTALL_SLURM.md.

#SBATCH --job-name=orhsurf-node
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8                 # <<< CHECK  match [NGPUS] below and your node's GPU count
#SBATCH --cpus-per-task=64           # <<< CHECK  8 per GPU; orhsurf caps each worker at 8 anyway
#SBATCH --mem=256G                   # <<< CHECK  ~35 GB per concurrent worker at export
#SBATCH --time=24:00:00              # <<< CHECK  11-19 min/frame; budget ~6 h for 150 frames / 8 GPUs
#SBATCH --partition=gpu              # <<< CHECK

set -euo pipefail
CLIP="${1:?usage: sbatch slurm/recon_singlenode.sbatch <CLIP_ID> [FRAMES] [NGPUS]}"
FRAMES="${2:-0-149}"
NGPUS="${3:-${SLURM_GPUS_ON_NODE:-1}}"

# Slurm copies this file to its spool directory and runs it from there, so BASH_SOURCE points at
# the spool, NOT at the checkout. Resolve the repo explicitly, in this order:
#   $ORHSURF_REPO  ->  $SLURM_SUBMIT_DIR (if it looks like the checkout)  ->  hard failure.
if [ -n "${ORHSURF_REPO:-}" ]; then
  REPO="$ORHSURF_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/orhsurf/cli.py" ]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  echo "[orhsurf] cannot locate the checkout." >&2
  echo "  Submit FROM the repo root, or export ORHSURF_REPO=/path/to/orhsurf first:" >&2
  echo "    sbatch --export=ALL,ORHSURF_REPO=/path/to/orhsurf ..." >&2
  exit 2
fi
[ -f "$REPO/orhsurf/cli.py" ] || { echo "[orhsurf] $REPO is not an orhsurf checkout" >&2; exit 2; }
[ -f "$REPO/env.sh" ] || { echo "[orhsurf] $REPO/env.sh missing; run install.sh first" >&2; exit 2; }
mkdir -p "$REPO/slurm_logs"
cd "$REPO"
module load cuda/12.8 2>/dev/null || echo "[warn] 'module load cuda/12.8' failed; is nvcc on PATH?"
source "$REPO/env.sh"
export HF_HUB_OFFLINE=1
# Per-worker thread count is derived inside orhsurf from the allocation; we only set a sane global.
export OMP_NUM_THREADS=8

echo "[orhsurf] job ${SLURM_JOB_ID:-?} on $(hostname), ${NGPUS} GPU(s)"
echo "[orhsurf] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# No srun wrapper here: orhsurf forks its own per-GPU workers inside this single task. Wrapping it
# in `srun` as well would create a task that then forks, which some sites' cgroup settings dislike.
"$REPO/bin/orhsurf" run --clip "$CLIP" --frames "$FRAMES" --gpus "$NGPUS" --cpus-per-job "$((${SLURM_CPUS_PER_TASK:-8} / NGPUS))"
