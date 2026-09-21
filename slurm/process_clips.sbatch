#!/usr/bin/env bash
# One clip per array task; invoke from the checkout (or set ORHSURF_REPO).
# sbatch --array=0-3%2 [site options] slurm/process_clips.sbatch C001 C002 C003 C004
# Add --smoke AFTER the script name for frame 0 only, with unchanged reconstruction defaults.
#SBATCH --job-name=orhsurf-clips
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-process-%A_%a.out

set -euo pipefail
SMOKE=()
GPUS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE+=(--smoke); shift ;;
        --all-frames) SMOKE+=(--all-frames); shift ;;
        --gpus) GPUS="${2:?--gpus needs a count}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo '--gpus must be positive' >&2; exit 2; }
CLIPS=("$@")
(( ${#CLIPS[@]} > 0 )) || { echo 'give clip IDs, e.g. C001 C002' >&2; exit 2; }
declare -A SEEN=()
for clip in "${CLIPS[@]}"; do
    [[ "$clip" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "invalid clip ID: $clip" >&2; exit 2; }
    [[ ! ${SEEN[$clip]+present} ]] || { echo "duplicate clip: $clip" >&2; exit 2; }
    SEEN[$clip]=1
done
if [[ -z ${SLURM_ARRAY_TASK_ID+x} && ${#CLIPS[@]} -ne 1 ]]; then
    echo 'multiple clips require --array=0-(number of clips minus 1)' >&2; exit 2
fi
TASK="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$TASK" =~ ^[0-9]+$ ]] || { echo "invalid array index: $TASK" >&2; exit 2; }
TASK=$((10#$TASK))
(( TASK < ${#CLIPS[@]} )) || { echo "array index $TASK has no clip" >&2; exit 2; }
REPO="${ORHSURF_REPO:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$REPO" && -f "$REPO/orhsurf/cli.py" && -f "$REPO/env.sh" ]] || {
    echo 'submit from the installed checkout or set ORHSURF_REPO' >&2; exit 2;
}
cd "$REPO"
source "$REPO/env.sh"
printf '[clips] task=%s clip=%s gpus=%s smoke=%s\n' "$TASK" "${CLIPS[$TASK]}" "$GPUS" "${SMOKE[*]:-no}"
# process explicitly disables frame sharding: this array partitions CLIPS, not frames.
# No hard-coded partition/account, CPU/memory requests, module names or offline policy.
OUTPUT=()
if [[ -n ${MODEL_OUTPUT_DIR:-} ]]; then
    OUTPUT=(--out-root "$MODEL_OUTPUT_DIR")
fi
srun --ntasks=1 "$REPO/bin/orhsurf" process --clips "${CLIPS[$TASK]}" --gpus "$GPUS" "${SMOKE[@]}" "${OUTPUT[@]}"
