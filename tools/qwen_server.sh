#!/usr/bin/env bash
# Niblit Llama Server — Full Orchestrated Runtime (OpenAI /v1 Compatible)

set -euo pipefail

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_TMPDIR="${TMPDIR:-/tmp}"

_PID_FILE="${_TMPDIR}/niblit-llama-server.pid"
_TUNNEL_PID_FILE="${_TMPDIR}/niblit-llama-tunnel.pid"
_LOG_FILE="${_TMPDIR}/niblit-llama-server.log"
_GOVERNANCE_LOG="${_TMPDIR}/niblit_cloud_reflection.jsonl"

# ─────────────────────────────────────────────
# Default Windows paths
# ─────────────────────────────────────────────

DEFAULT_MODEL="C:/Users/Riyaad/llama_migration/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
DEFAULT_BACKEND="C:/Users/Riyaad/llama_migration/llama.cpp/build/bin/Release/llama-server.exe"
DEFAULT_MODEL_ID="qwen2.5-coder-3b"

# ─────────────────────────────────────────────
# Runtime config
# ─────────────────────────────────────────────

PORT="${NIBLIT_PORT:-8000}"
HOST="${NIBLIT_HOST:-127.0.0.1}"

MODEL_PATH="${NIBLIT_MODEL_PATH:-}"

N_CTX="${NIBLIT_N_CTX:-32768}"
N_THREADS="${NIBLIT_N_THREADS:-8}"
N_GPU_LAYERS="${NIBLIT_N_GPU_LAYERS:-35}"

BACKEND_BIN="${NIBLIT_LLAMA_SERVER_BIN:-}"

TUNNEL_PROVIDER="${NIBLIT_TUNNEL_PROVIDER:-none}"
PUBLIC_URL_OVERRIDE="${NIBLIT_TUNNEL_PUBLIC_URL:-}"

PROFILE="${NIBLIT_PROFILE:-}"
GOVERNANCE_ENABLED=1
DRY_RUN=0
READINESS_TIMEOUT=90

# ─────────────────────────────────────────────
# PATH FIX (IMPORTANT)
# ─────────────────────────────────────────────

_normalize_path() {
  local path="$1"

  # remove accidental brace / corruption artifacts
  path="${path//\}/}"
  path="${path%%\}*}"

  # Windows → Git Bash conversion
  if [[ "$path" =~ ^[A-Za-z]:\\ ]]; then
    local drive="${path:0:1}"
    local rest="${path:2}"
    rest="${rest//\\//}"
    path="/${drive,,}/${rest}"
  fi

  # Preserve a stable absolute path for logging and tests.
  if [[ "$path" =~ ^/[A-Za-z]/ ]]; then
    path="${path#/}"
  fi

  echo "$path"
}

# ─────────────────────────────────────────────
# Model resolution
# ─────────────────────────────────────────────

_find_model_in_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  find "$dir" -maxdepth 1 -type f -iname "*.gguf" | sort | head -n 1
}

_resolve_model() {
  if [[ -n "$MODEL_PATH" && "$MODEL_PATH" == ~* ]]; then
    MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
  fi
  MODEL_PATH="$(_normalize_path "$MODEL_PATH")"

  if [[ -z "$MODEL_PATH" ]]; then
    local candidate_dir
    for candidate_dir in "$HOME/models" "$TERMUX_HOME/models" "$PREFIX/models" "/data/data/com.termux/files/usr/models"; do
      [[ -d "$candidate_dir" ]] || continue
      MODEL_PATH="$(_find_model_in_dir "$candidate_dir")"
      [[ -n "$MODEL_PATH" ]] && break
    done
  fi

  if [[ -d "$MODEL_PATH" ]]; then
    MODEL_PATH="$(_find_model_in_dir "$MODEL_PATH")"
  fi

  MODEL_PATH="$(_normalize_path "$MODEL_PATH")"

  if [[ -z "$MODEL_PATH" ]]; then
    echo "[niblit] ERROR: model not found; set NIBLIT_MODEL_PATH or provide --model"
    exit 1
  fi

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[niblit] ERROR: model not found: $MODEL_PATH"
    exit 1
  fi
}

_resolve_backend() {
  if [[ -n "$BACKEND_BIN" && "$BACKEND_BIN" == ~* ]]; then
    BACKEND_BIN="${BACKEND_BIN/#\~/$HOME}"
  fi
  BACKEND_BIN="$(_normalize_path "$BACKEND_BIN")"

  if [[ -z "$BACKEND_BIN" ]]; then
    local candidate
    for candidate in \
      "$HOME/llama.cpp/build/bin/llama-server" \
      "$HOME/llama.cpp/build/bin/llama-server.exe" \
      "$HOME/llama.cpp/build/bin/Release/llama-server.exe" \
      "$TERMUX_HOME/llama.cpp/build/bin/llama-server" \
      "$PREFIX/llama.cpp/build/bin/llama-server"; do
      if [[ -f "$candidate" ]]; then
        BACKEND_BIN="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$BACKEND_BIN" ]]; then
    if command -v llama-server >/dev/null 2>&1; then
      BACKEND_BIN="$(command -v llama-server)"
    elif command -v llama-server.exe >/dev/null 2>&1; then
      BACKEND_BIN="$(command -v llama-server.exe)"
    fi
  fi

  if [[ ! -f "$BACKEND_BIN" ]] && ! command -v "$BACKEND_BIN" >/dev/null 2>&1; then
    echo "[niblit] ERROR: backend not found: $BACKEND_BIN"
    exit 1
  fi
}

# ─────────────────────────────────────────────
# Profiles
# ─────────────────────────────────────────────

_load_profile() {
  [[ -z "$PROFILE" ]] && return 0

  local file="$_SCRIPT_DIR/runtime_profiles/${PROFILE}.env"
  [[ -f "$file" ]] && source "$file"

  PORT="${NIBLIT_PORT:-$PORT}"
  HOST="${NIBLIT_HOST:-$HOST}"
  MODEL_PATH="${NIBLIT_MODEL_PATH:-$MODEL_PATH}"
  N_CTX="${NIBLIT_N_CTX:-$N_CTX}"
  N_THREADS="${NIBLIT_N_THREADS:-$N_THREADS}"
  N_GPU_LAYERS="${NIBLIT_N_GPU_LAYERS:-$N_GPU_LAYERS}"
  BACKEND_BIN="${NIBLIT_LLAMA_SERVER_BIN:-$BACKEND_BIN}"
}

_apply_defaults() {
  if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$DEFAULT_MODEL"
  fi
  if [[ -z "$BACKEND_BIN" ]]; then
    BACKEND_BIN="$DEFAULT_BACKEND"
  fi
}

# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

_parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile)
        PROFILE="$2"
        shift 2
        ;;
      --model)
        MODEL_PATH="$2"
        shift 2
        ;;
      --backend)
        BACKEND_BIN="$2"
        shift 2
        ;;
      --port)
        PORT="$2"
        shift 2
        ;;
      --host)
        HOST="$2"
        shift 2
        ;;
      --n-ctx)
        N_CTX="$2"
        shift 2
        ;;
      --n-threads)
        N_THREADS="$2"
        shift 2
        ;;
      --n-gpu-layers)
        N_GPU_LAYERS="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --help|-h)
        echo "Usage: $0 [--profile PROFILE] [--model PATH] [--backend PATH] [--dry-run]"
        exit 0
        ;;
      *)
        echo "[niblit] warning: ignoring unknown arg: $1"
        shift
        ;;
    esac
  done
}

# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

_validate() {
  _resolve_model
  _resolve_backend
}

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

_print_config() {
cat <<EOF
[niblit-llama] ==============================
model:   $MODEL_PATH
bin:     $BACKEND_BIN
host:    $HOST:$PORT
ctx:     $N_CTX
threads: $N_THREADS
gpu:     $N_GPU_LAYERS
============================================
EOF
}

_emit() {
  [[ "$GOVERNANCE_ENABLED" -eq 1 ]] || return 0
  echo "{\"event\":\"$1\",\"ts\":$(date +%s),\"port\":$PORT}" >> "$_GOVERNANCE_LOG"
}

# ─────────────────────────────────────────────
# Server start
# ─────────────────────────────────────────────

_start_server() {
  echo "[niblit] starting llama-server (OpenAI /v1 enabled automatically)"

  local cmd=(
    "$BACKEND_BIN"
    -m "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    -t "$N_THREADS"
    -c "$N_CTX"
    --n-predict 1024
    --cont-batching
    --batch-size 128
    --ubatch-size 64
    --temp 0.7
    --repeat-penalty 1.1
  )

  if [[ "$N_GPU_LAYERS" -gt 0 ]]; then
    cmd+=(--n-gpu-layers "$N_GPU_LAYERS")
  fi

  export GGML_KV_CACHE_TYPE="f16"

  echo "[niblit] command:"
  printf " %q" "${cmd[@]}"
  echo

  nohup "${cmd[@]}" >>"$_LOG_FILE" 2>&1 &
  echo $! >"$_PID_FILE"

  sleep 3

  if ! kill -0 "$(cat $_PID_FILE)" 2>/dev/null; then
    echo "[niblit] server crashed"
    tail -n 50 "$_LOG_FILE"
    exit 1
  fi

  echo "[niblit] running pid=$(cat $_PID_FILE)"
}

# ─────────────────────────────────────────────
# Readiness probe
# ─────────────────────────────────────────────

_probe() {
  local base="http://${HOST}:${PORT}"

  echo "[niblit] waiting for OpenAI endpoint..."

  for i in {1..60}; do
    if curl -sf "$base/v1/models" >/dev/null 2>&1; then
      echo "[niblit] ready: /v1/models available"
      return 0
    fi
    sleep 2
  done

  echo "[niblit] warning: readiness timeout"
}

# ─────────────────────────────────────────────
# Tunnel
# ─────────────────────────────────────────────

_start_tunnel() {
  [[ "$TUNNEL_PROVIDER" == "none" ]] && return 0
  [[ -n "$PUBLIC_URL_OVERRIDE" ]] && return 0

  if [[ "$TUNNEL_PROVIDER" == "cloudflared" ]]; then
    nohup cloudflared tunnel --url "http://$HOST:$PORT" >/dev/null 2>&1 &
    echo $! > "$_TUNNEL_PID_FILE"
  fi

  if [[ "$TUNNEL_PROVIDER" == "ngrok" ]]; then
    nohup ngrok http "$PORT" >/dev/null 2>&1 &
    echo $! > "$_TUNNEL_PID_FILE"
  fi
}

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────

_cleanup() {
  echo "[niblit] shutting down..."
  [[ -f "$_PID_FILE" ]] && kill "$(cat $_PID_FILE)" 2>/dev/null || true
  exit 0
}

trap _cleanup EXIT INT TERM

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

main() {
  _parse_args "$@"
  _load_profile
  _apply_defaults
  _validate
  _print_config

  [[ "$DRY_RUN" -eq 1 ]] && exit 0

  _emit "start"
  _start_server
  _probe
  _start_tunnel
  _emit "ready"

  wait "$(cat "$_PID_FILE")"
}

main "$@"
