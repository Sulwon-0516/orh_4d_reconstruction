#!/usr/bin/env bash
# Download every model weight orhsurf needs. Non-interactive and re-runnable.
# There is exactly one: the Depth Anything 3 checkpoint (6.76 GB). See docs/CHECKPOINTS.md.
#
#   ./fetch_ckpts.sh                  # into $ORHSURF_CACHE_DIR (default <repo>/cache)
#   ORHSURF_CACHE_DIR=$SCRATCH/c ./fetch_ckpts.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${ORHSURF_CACHE_DIR:-$HERE/cache}"
REPO_ID="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
REVISION="b2359bdf726fb44ef62acca04d629dcf158053e7"

PY="${ORHSURF_DA3_PYTHON:-${ORHSURF_PYTHON:-$(command -v python3)}}"
[ -x "$PY" ] || { echo "no python found; run install.sh first, or set ORHSURF_PYTHON" >&2; exit 1; }

mkdir -p "$CACHE/hf"
echo "fetching $REPO_ID @ ${REVISION:0:8} (6.76 GB) -> $CACHE/hf"
HF_HOME="$CACHE/hf" "$PY" - "$REPO_ID" "$REVISION" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1], sys.argv[2]
p = snapshot_download(repo_id=repo, revision=rev)   # resumes; no-op if already complete
print(f"ok: {p}")
PYEOF
echo "done. Verify with:  HF_HUB_OFFLINE=1 orhsurf doctor"
