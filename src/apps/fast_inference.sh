#!/usr/bin/env bash
# Segment any folder of images with the study's models and fusion (src/apps/fast_inference.py; docs/apps.md).
#
# Usage: src/apps/fast_inference.sh <image folder> [--out <folder>] [--exp <experiment>] [--limit N]
#   src/apps/fast_inference.sh ~/photos --limit 3      a quick test: three images, into ~/photos-segmented/
#
# Runs from any directory. A CUDA GPU is meant for many images; Apple's GPU or the CPU serve for a few.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

[ $# -ge 1 ] && [ "$1" != -h ] && [ "$1" != --help ] || { sed -n 4,5p "$0" | sed 's/^# //'; exit 1; }
command -v uv >/dev/null || { echo "uv not found: install it (https://docs.astral.sh/uv/), then run uv sync in $REPO"; exit 1; }
# SAM 3 is gated on Hugging Face: a first run downloads it, which needs a login with access
uv run --project "$REPO" hf auth whoami >/dev/null 2>&1 ||
  echo "warning: not logged in to Hugging Face; unless SAM 3 is already downloaded, run: $REPO/.venv/bin/hf auth login"
exec uv run --project "$REPO" python "$REPO/src/apps/fast_inference.py" "$@"
