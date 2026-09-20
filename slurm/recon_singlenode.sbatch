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
#SBATCH --time=24:00:00              # <<< CHECK  ~11 min/frame; 150 frames / 8 GPUs ~ 3.5 h
#SBATCH --partition=gpu              # <<< CHECK

set -euo pipefail
CLIP="${1:?usage: sbatch slurm/recon_singlenode.sbatch <CLIP_ID> [FRAMES] [NGPUS]}"
FRAMES="${2:-0-149}"
NGPUS="${3:-${SLURM_GPUS_ON_NODE:-1}}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO/slurm_logs"; cd "$REPO"
module load cuda/12.8 2>/dev/null || echo "[warn] 'module load cuda/12.8' failed; is nvcc on PATH?"
source "$REPO/env.sh"
export HF_HUB_OFFLINE=1
# Per-worker thread count is derived inside orhsurf from the allocation; we only set a sane global.
export OMP_NUM_THREADS=8

echo "[orhsurf] job ${SLURM_JOB_ID:-?} on $(hostname), ${NGPUS} GPU(s)"
echo "[orhsurf] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

# No srun wrapper here: orhsurf forks its own per-GPU workers inside this single task. Wrapping it
# in `srun` as well would create a task that then forks, which some sites' cgroup settings dislike.
"$ORHSURF_PYTHON" -m orhsurf.cli run --clip "$CLIP" --frames "$FRAMES" --gpus "$NGPUS"
