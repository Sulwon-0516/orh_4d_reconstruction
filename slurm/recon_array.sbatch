#!/bin/bash
# orhsurf reconstruction as a Slurm JOB ARRAY -- the recommended shape.
#
#   sbatch --array=0-7 slurm/recon_array.sbatch <CLIP_ID> [FRAMES]
#   sbatch --array=0-7 slurm/recon_array.sbatch C001 0-149
#
# WHY AN ARRAY OF SINGLE-GPU TASKS, rather than one job holding N GPUs:
#   1. The unit of work is one frame and frames are independent -- there is nothing to synchronise,
#      so a gang reservation buys nothing.
#   2. N x `--gres=gpu:1` queues far better than one `--gres=gpu:N` gang on a busy cluster.
#   3. Preemption or a timeout costs ONE task, not the whole allocation. With the DONE-marker
#      resume, re-submitting the identical array picks up exactly what is missing.
#   4. `--gres=gpu:1` makes Slurm itself enforce the one-job-per-GPU rule that was measured twice
#      to matter here (packing two AmbiSuR trainings on one GPU: 451 s solo vs 1705 s each).
# Cost: each task re-imports torch and reloads the 6.76 GB DA3 checkpoint (~17 s), which is noise
# against 11 min/frame. For a node you own outright, slurm/recon_singlenode.sbatch is the
# alternative.
#
# ############################################################################################
# # UNTESTED. There is no Slurm on the machine this was written on. Before the first real     #
# # submission, check the four lines marked  <<< CHECK  below against your site's docs and    #
# # `sinfo -o "%P %G %c %m %l"`. Everything else is ordinary shell that was exercised locally. #
# ############################################################################################

#SBATCH --job-name=orhsurf
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
#SBATCH --gres=gpu:1                 # <<< CHECK  some sites want --gpus-per-task=1, or a type:
                                     #            e.g. --gres=gpu:a100:1
#SBATCH --cpus-per-task=8            # <<< CHECK  8 is what one job actually uses (~280% CPU)
#SBATCH --mem=64G                    # <<< CHECK  export peaks ~35 GB RSS on a 47-camera clip
#SBATCH --time=24:00:00              # <<< CHECK  11-19 min/frame (concurrency-dependent); budget ~6 h for 150 frames / 8 tasks
#SBATCH --partition=gpu              # <<< CHECK  your GPU partition name

set -euo pipefail

CLIP="${1:?usage: sbatch --array=0-N slurm/recon_array.sbatch <CLIP_ID> [FRAMES]}"
FRAMES="${2:-0-149}"

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

# --- site modules. Comment out if your site does not use Lmod, or adjust the versions. -------
# The CUDA module must match the toolkit the extensions were BUILT against (12.8 for the pinned
# torch 2.7.1+cu128). A mismatch shows up as an undefined-symbol error importing the rasteriser.
module load cuda/12.8 2>/dev/null || echo "[warn] 'module load cuda/12.8' failed; is nvcc on PATH?"

source "$REPO/env.sh"          # written by install.sh

# --- offline: compute nodes usually have no network. Weights and data must be pre-staged. ----
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# --- allocation-aware. Do NOT use nproc / nvidia-smi here: inside the cgroup they report the ---
# --- whole node, and sizing thread pools from the node oversubscribes an 8-CPU task by 8x.  ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "[orhsurf] job ${SLURM_JOB_ID:-?} task ${TASK_ID}/${TASK_COUNT} on $(hostname)"
echo "[orhsurf] clip=$CLIP frames=$FRAMES cpus=${SLURM_CPUS_PER_TASK:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Each task takes a contiguous slice of the frame list. `orhsurf run` does the partitioning; we
# just tell it which shard this task is. One GPU per task, so --gpus 1.
# NOTE: no `taskset` anywhere. Inside a Slurm cpuset cgroup taskset can only narrow an already
# narrow mask and fails outright if the mask does not intersect the cgroup. Binding is srun's job.
srun --cpu-bind=cores \
  "$REPO/bin/orhsurf" run \
    --clip "$CLIP" \
    --frames "$FRAMES" \
    --gpus 1 \
    --cpus-per-job "${SLURM_CPUS_PER_TASK:-8}" \
    --shard "$TASK_ID" --shards "$TASK_COUNT"

echo "[orhsurf] task ${TASK_ID} finished rc=$?"
