#!/usr/bin/env bash
# tools/llama_server.sh — Niblit Llama3 model inference runtime launcher
#
# Launches a llama-server or llama-cli inference backend loaded with a Llama3 model
# with optional cloudflared or ngrok tunnel exposure, governance telemetry,
# runtime readiness probing, and process lifecycle management.
#
# Usage:
#   ./tools/llama_server.sh [options]
#
# Options:
#   --model PATH        Path to GGUF model file (or $NIBLIT_MODEL_PATH)
#   --port PORT         Port to bind (default: 8000 or $NIBLIT_PORT)
#   --host HOST         Host to bind (default: 127.0.0.1)
#   --n-ctx N           Context size (default: 16384 or $NIBLIT_N_CTX)
#   --n-threads N       Thread count (default: 4 or $NIBLIT_N_THREADS)
#   --n-gpu-layers N    GPU layers (default: 0 or $NIBLIT_N_GPU_LAYERS)
#   --backend CMD       Backend binary (default: llama-server)
#   --tunnel PROVIDER   Tunnel provider: cloudflared|ngrok|none (default: none)
#   --public-url URL    Override public URL instead of using tunnel
#   --profile PROFILE   Load a runtime profile (niblit|cloud-server|termux-local)
#   --no-governance     Disable governance telemetry logging
#   --dry-run           Print resolved configuration without starting processes
#
# Environment variables (all overridable on command line):
#   NIBLIT_MODEL_PATH       Path to GGUF model
#   NIBLIT_PORT             Server port
#   NIBLIT_N_CTX            Context window size
#   NIBLIT_N_THREADS        CPU threads
#   NIBLIT_N_GPU_LAYERS     GPU offload layers
#   NIBLIT_LLAMA_SERVER_BIN llama-server binary name/path
#   NIBLIT_TUNNEL_PROVIDER  cloudflared|ngrok|none
#   NIBLIT_TUNNEL_PUBLIC_URL Override tunnel URL
#   NIBLIT_PROFILE          Runtime profile to load

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(dirname "$_SCRIPT_DIR")"
_TMPDIR="${TMPDIR:-/tmp}"
_RUNTIME_TAG="niblit-llama-$$"
_PID_FILE="${_TMPDIR}/niblit-llama-server.pid"
_TUNNEL_PID_FILE="${_TMPDIR}/niblit-llama-tunnel.pid"
_LOG_FILE="${_TMPDIR}/niblit-llama-server.log"
_GOVERNANCE_LOG="${_TMPDIR}/niblit_cloud_reflection.jsonl"

PORT="${NIBLIT_PORT:-8000}"
HOST="${NIBLIT_HOST:-127.0.0.1}"
MODEL_PATH="${NIBLIT_MODEL_PATH:-}"
N_CTX="${NIBLIT_N_CTX:-8192}"
N_THREADS="${NIBLIT_N_THREADS:-4}"
N_GPU_LAYERS="${NIBLIT_N_GPU_LAYERS:-35}"
N_BATCH="${NIBLIT_N_BATCH:-128}"
N_UBATCH="${NIBLIT_N_UBATCH:-64}"
ROPE_SCALE="${NIBLIT_ROPE_SCALE:-1}"
BACKEND_BIN="${NIBLIT_LLAMA_SERVER_BIN:-llama-server}"
TUNNEL_PROVIDER="${NIBLIT_TUNNEL_PROVIDER:-none}"
PUBLIC_URL_OVERRIDE="${NIBLIT_TUNNEL_PUBLIC_URL:-}"
PROFILE="${NIBLIT_PROFILE:-}"
GOVERNANCE_ENABLED=1
DRY_RUN=0
READINESS_TIMEOUT=60

_find_model_in_dir() {
  local dir="$1"
  local pattern candidate candidate_name
  local -a patterns=(
    "*llama-3*.gguf"
    "*llama3*.gguf"
    "*llama*.gguf"
    "*mistral*.gguf"
    "*mixtral*.gguf"
    "*phi*.gguf"
    "*gemma*.gguf"
    "*deepseek*.gguf"
    "*qwen*.gguf"
    "*.gguf"
  )
  local -a candidates=()
  [[ -d "$dir" ]] || return 1

  mapfile -t candidates < <(find "$dir" -maxdepth 1 -type f -iname "*.gguf" | LC_ALL=C sort)

  for pattern in "${patterns[@]}"; do
    for candidate in "${candidates[@]}"; do
      candidate_name="${candidate##*/}"
      candidate_name="${candidate_name,,}"
      if [[ "$candidate_name" == $pattern ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  done

  return 1
}

_expand_path_tokens() {
  local value="$1"
  local home_dir="${HOME:-/home/riddo9906}"
  local termux_home="${TERMUX_HOME:-$home_dir}"

  value="${value//\$\{TERMUX_HOME\}/$termux_home}"
  value="${value//\$TERMUX_HOME/$termux_home}"
  value="${value//\$\{HOME\}/$home_dir}"
  value="${value//\$HOME/$home_dir}"

  printf '%s\n' "$value"
}

_resolve_model_path() {
  local original_model_path="${MODEL_PATH:-}"
  local candidate_dir resolved=""

  MODEL_PATH="$(_expand_path_tokens "$MODEL_PATH")"

  if [[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]]; then
    resolved="$(_find_model_in_dir "$MODEL_PATH")"
  elif [[ -z "$MODEL_PATH" ]]; then
    candidate_dir="${HOME:-/home/riddo9906}/models"
    resolved="$(_find_model_in_dir "$candidate_dir")"
  fi

  if [[ -n "$resolved" ]]; then
    MODEL_PATH="$resolved"
    if [[ "$MODEL_PATH" != "$original_model_path" ]]; then
      echo "[niblit-llama] resolved model path: $MODEL_PATH" >&2
    fi
  fi
}

_resolve_backend_bin() {
  local original_backend_bin="${BACKEND_BIN:-}"
  local resolved=""

  BACKEND_BIN="$(_expand_path_tokens "$BACKEND_BIN")"

  if [[ -n "$BACKEND_BIN" && "$BACKEND_BIN" == */* && -x "$BACKEND_BIN" ]]; then
    return 0
  fi

  if command -v "$BACKEND_BIN" >/dev/null 2>&1; then
    BACKEND_BIN="$(command -v "$BACKEND_BIN")"
    return 0
  fi

  local default_candidate="${HOME:-/home/riddo9906}/llama.cpp/build/bin/llama-server"
  if [[ -x "$default_candidate" ]]; then
    resolved="$default_candidate"
  fi

  if [[ -n "$resolved" ]]; then
    BACKEND_BIN="$resolved"
    if [[ "$BACKEND_BIN" != "$original_backend_bin" ]]; then
      echo "[niblit-llama] resolved backend binary: $BACKEND_BIN" >&2
    fi
  fi
}

_cleanup() {
  local exit_code=$?
  echo "[niblit-llama] shutdown signal received (exit=$exit_code)"
  _stop_processes
  exit "$exit_code"
}

trap '_cleanup' EXIT INT TERM HUP

_stop_processes() {
  if [[ -f "$_PID_FILE" ]]; then
    local pid
    pid="$(cat "$_PID_FILE" 2>/dev/null || echo "")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 0.5
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$_PID_FILE"
  fi

  if [[ -f "$_TUNNEL_PID_FILE" ]]; then
    local tpid
    tpid="$(cat "$_TUNNEL_PID_FILE" 2>/dev/null || echo "")"
    if [[ -n "$tpid" ]] && kill -0 "$tpid" 2>/dev/null; then
      kill "$tpid" 2>/dev/null || true
    fi
    rm -f "$_TUNNEL_PID_FILE"
  fi
}

_parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL_PATH="$2"; shift 2 ;;
      --port) PORT="$2"; shift 2 ;;
      --host) HOST="$2"; shift 2 ;;
      --n-ctx) N_CTX="$2"; shift 2 ;;
      --n-threads) N_THREADS="$2"; shift 2 ;;
      --n-gpu-layers) N_GPU_LAYERS="$2"; shift 2 ;;
      --n-batch) N_BATCH="$2"; shift 2 ;;
      --n-ubatch) N_UBATCH="$2"; shift 2 ;;
      --rope-scale) ROPE_SCALE="$2"; shift 2 ;;
      --backend) BACKEND_BIN="$2"; shift 2 ;;
      --tunnel) TUNNEL_PROVIDER="$2"; shift 2 ;;
      --public-url) PUBLIC_URL_OVERRIDE="$2"; shift 2 ;;
      --profile) PROFILE="$2"; shift 2 ;;
      --no-governance) GOVERNANCE_ENABLED=0; shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      *) shift ;;
    esac
  done
}

_load_profile() {
  [[ -z "$PROFILE" ]] && return 0

  local profile_dir="$_SCRIPT_DIR/runtime_profiles"
  local env_file="$profile_dir/${PROFILE}.env"

  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi

  PORT="${NIBLIT_PORT:-$PORT}"
  HOST="${NIBLIT_HOST:-$HOST}"
  MODEL_PATH="${NIBLIT_MODEL_PATH:-$MODEL_PATH}"
  N_CTX="${NIBLIT_N_CTX:-$N_CTX}"
  N_THREADS="${NIBLIT_N_THREADS:-$N_THREADS}"
  N_GPU_LAYERS="${NIBLIT_N_GPU_LAYERS:-$N_GPU_LAYERS}"
  N_BATCH="${NIBLIT_N_BATCH:-$N_BATCH}"
  N_UBATCH="${NIBLIT_N_UBATCH:-$N_UBATCH}"
  ROPE_SCALE="${NIBLIT_ROPE_SCALE:-$ROPE_SCALE}"
  BACKEND_BIN="${NIBLIT_LLAMA_SERVER_BIN:-$BACKEND_BIN}"
  TUNNEL_PROVIDER="${NIBLIT_TUNNEL_PROVIDER:-$TUNNEL_PROVIDER}"
  PUBLIC_URL_OVERRIDE="${NIBLIT_TUNNEL_PUBLIC_URL:-$PUBLIC_URL_OVERRIDE}"
}

_validate() {
  _resolve_model_path
  _resolve_backend_bin

  [[ -z "$MODEL_PATH" ]] && exit 1
  [[ ! -f "$MODEL_PATH" ]] && exit 1

  command -v "$BACKEND_BIN" >/dev/null 2>&1 || exit 1

  if [[ "$TUNNEL_PROVIDER" != "none" && -z "$PUBLIC_URL_OVERRIDE" ]]; then
    if [[ "$TUNNEL_PROVIDER" == "cloudflared" ]] && ! command -v cloudflared >/dev/null 2>&1; then
      TUNNEL_PROVIDER=none
    fi
    if [[ "$TUNNEL_PROVIDER" == "ngrok" ]] && ! command -v ngrok >/dev/null 2>&1; then
      TUNNEL_PROVIDER=none
    fi
  fi
}

_print_config() {
  cat <<EOF
[niblit-llama] model=$MODEL_PATH backend=$BACKEND_BIN host=$HOST:$PORT ctx=$N_CTX threads=$N_THREADS gpu=$N_GPU_LAYERS
EOF
}

_start_server() {
  echo "[niblit-llama] starting $BACKEND_BIN..."

  local -a cmd=(
    "$BACKEND_BIN"
    -m "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    -c "$N_CTX"
    -t "$N_THREADS"

    # IMPORTANT: prevent KV cache explosion
    --n-predict 512

    # stable batching (safe for CPU + low VRAM)
    --batch-size 128
    --ubatch-size 64
    --cont-batching

    # generation stability
    --temp 0.7
    --repeat-penalty 1.1
    --repeat-last-n 64
  )

  #
  # GPU (only if supported AND safe)
  #
  if "$BACKEND_BIN" --help 2>/dev/null | grep -qi "gpu"; then
    if [[ "$N_GPU_LAYERS" -gt 0 ]]; then
      cmd+=(--n-gpu-layers "$N_GPU_LAYERS")
    fi
  fi

  #
  # safe optional overrides
  #
  if [[ -n "$ROPE_SCALE" && "$ROPE_SCALE" != "1" ]]; then
    cmd+=(--rope-scale "$ROPE_SCALE")
  fi

  #
  # CRITICAL FIX: ensure server doesn't silently hang due to batching pressure
  #
  export GGML_KV_CACHE_TYPE="f16"

  echo "[niblit-llama] command:"
  printf ' %q' "${cmd[@]}"
  echo

  nohup "${cmd[@]}" >>"$_LOG_FILE" 2>&1 &
  echo $! >"$_PID_FILE"

  sleep 3

  if ! kill -0 "$(cat $_PID_FILE)" 2>/dev/null; then
    echo "[niblit-llama] ❌ server crashed"
    tail -n 50 "$_LOG_FILE"
    exit 1
  fi

  echo "[niblit-llama] llama-server running (pid=$(cat "$_PID_FILE"))"
}

_probe_fallback_endpoints() {
  local base="http://${HOST}:${PORT}"
  curl -sf "${base}/v1/models" >/dev/null 2>&1 && return 0
  curl -sf "${base}/props" >/dev/null 2>&1 && return 0
  return 0
}
_probe_readiness() {
  local health_url="http://${HOST}:${PORT}/health"
  local deadline=$(($(date +%s) + READINESS_TIMEOUT))
  echo "[niblit-llama] waiting for server readiness at $health_url..."
  while [[ $(date +%s) -lt $deadline ]]; do
    if curl -sf "$health_url" >/dev/null 2>&1; then
      echo "[niblit-llama] ✅ server is ready"
      return 0
    fi
    # Check process is still alive
    if [[ -f "$_PID_FILE" ]] && ! kill -0 "$(cat "$_PID_FILE")" 2>/dev/null; then
      echo "[niblit-llama] ❌ server process died — check $_LOG_FILE" >&2
      return 1
    fi
    sleep 2
  done
  echo "[niblit-llama] ⚠️ readiness timeout after ${READINESS_TIMEOUT}s" >&2
  # Non-fatal — server may still be loading model
  return 0
}

_probe_fallback_endpoints() {
  local base="http://${HOST}:${PORT}"
  for ep in "/v1/models" "/props"; do
    if curl -sf "${base}${ep}" >/dev/null 2>&1; then
      echo "[niblit-llama] fallback probe OK: ${base}${ep}"
      return 0
    fi
  done
  echo "[niblit-llama] WARN: all fallback probes failed" >&2
  return 0
}


_start_tunnel() {
  [[ -n "$PUBLIC_URL_OVERRIDE" ]] && return 0

  if [[ "$TUNNEL_PROVIDER" == "cloudflared" ]]; then
    nohup cloudflared tunnel --url "http://localhost:$PORT" \
      >>"${_TMPDIR}/tunnel.log" 2>&1 &
    echo $! > "$_TUNNEL_PID_FILE"
  fi

  if [[ "$TUNNEL_PROVIDER" == "ngrok" ]]; then
    nohup ngrok http "$PORT" >>"${_TMPDIR}/ngrok.log" 2>&1 &
    echo $! > "$_TUNNEL_PID_FILE"
  fi
}

_emit_governance_event() {
  [[ "$GOVERNANCE_ENABLED" -ne 1 ]] && return 0
  echo "{\"event\":\"$1\",\"ts\":$(date +%s),\"port\":$PORT}" >> "$_GOVERNANCE_LOG"
}

main() {
  _parse_args "$@"
  _load_profile
  _validate
  _print_config

  [[ "$DRY_RUN" -eq 1 ]] && exit 0

  _emit_governance_event "start"
  _start_server
  _probe_readiness
  _probe_fallback_endpoints
  _start_tunnel
  _emit_governance_event "ready"

  wait "$(cat "$_PID_FILE")"
}

main "$@"
