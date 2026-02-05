#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
PYTHON=""

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "Python not found on PATH" >&2
  exit 1
fi

BRANCH="$(git branch --show-current 2>/dev/null || echo "unknown")"
TOP_LEVEL="$(git rev-parse --show-toplevel 2>/dev/null || echo "unknown")"
STREAMLIT_VERSION="$($PYTHON -m streamlit --version 2>/dev/null || echo "streamlit not available")"

printf "Context\n"
printf "  pwd: %s\n" "$ROOT_DIR"
printf "  git branch: %s\n" "$BRANCH"
printf "  git top-level: %s\n" "$TOP_LEVEL"
printf "  python: %s\n" "$PYTHON"
printf "  streamlit: %s\n" "$STREAMLIT_VERSION"

make smoke

scripts/run_app.sh
