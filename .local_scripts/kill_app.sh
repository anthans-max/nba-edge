#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=".run"
PID_FILE="$RUN_DIR/streamlit.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE" || true)"
if [[ -z "$PID" ]]; then
  echo "PID file is empty"
  rm -f "$PID_FILE"
  exit 0
fi

if kill -0 "$PID" >/dev/null 2>&1; then
  kill "$PID"
  sleep 0.5
  if kill -0 "$PID" >/dev/null 2>&1; then
    echo "Process $PID did not exit, sending SIGKILL"
    kill -9 "$PID"
  fi
  echo "Stopped Streamlit (PID $PID)"
else
  echo "Process $PID not running"
fi

rm -f "$PID_FILE"
