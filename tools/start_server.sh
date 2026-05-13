#!/usr/bin/env bash
set -euo pipefail

# Niblit cloud runtime process controller
# usage: ./tools/start_server.sh {start|stop|status|restart|smoke}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${NIBLIT_RUNTIME_PID_FILE:-/tmp/niblit-cloud-runtime.pid}"
LOG_FILE="${NIBLIT_RUNTIME_LOG_FILE:-/tmp/niblit-cloud-runtime.log}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

start_server() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "runtime already running pid=$(cat "$PID_FILE")"
    return 0
  fi
  nohup python -m uvicorn app.main:app --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  echo "started runtime pid=$(cat "$PID_FILE") log=$LOG_FILE"
}

stop_server() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "runtime not running"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "stopped runtime pid=$pid"
  fi
  rm -f "$PID_FILE"
}

status_server() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running pid=$(cat "$PID_FILE")"
    return 0
  fi
  echo "stopped"
  return 1
}

smoke_server() {
  python "$ROOT_DIR/tools/cloud_runtime_ctl.py" --url "http://127.0.0.1:$PORT" health
  python "$ROOT_DIR/tools/cloud_runtime_ctl.py" --url "http://127.0.0.1:$PORT" diagnostics --json
}

cd "$ROOT_DIR"
case "${1:-status}" in
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status) status_server ;;
  smoke) smoke_server ;;
  *) echo "usage: $0 {start|stop|status|restart|smoke}"; exit 2 ;;
esac
