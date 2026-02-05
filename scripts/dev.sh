#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ROOT="/Users/anthansunder/Documents/Machine Learning/nba-edge"

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
BRANCH=$(git branch --show-current 2>/dev/null || true)

printf "Repo root: %s\n" "${REPO_ROOT:-unknown}"
printf "Branch: %s\n" "${BRANCH:-unknown}"

if [[ "$REPO_ROOT" != "$EXPECTED_ROOT" ]]; then
  echo "Refusing to run: unexpected repo root."
  echo "Expected: $EXPECTED_ROOT"
  exit 1
fi

make app-clean
