#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=".run"
PID_FILE="$RUN_DIR/streamlit.pid"
LOG_FILE="$RUN_DIR/streamlit.log"

mkdir -p "$RUN_DIR"

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

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" >/dev/null 2>&1; then
    echo "Streamlit already running with PID $EXISTING_PID"
    exit 0
  fi
fi

nohup "$PYTHON" -m streamlit run app/app.py \
  --server.port 8501 \
  --server.headless true \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

printf "Started Streamlit (PID %s). Logs: %s\n" "$(cat "$PID_FILE")" "$LOG_FILE"
