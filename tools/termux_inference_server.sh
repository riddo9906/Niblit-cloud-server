#!/usr/bin/env bash
# tools/termux_inference_server.sh — Niblit Termux local inference runtime launcher
#
# Launches a llama-server or llama-cli inference backend on Termux / local Linux
# with optional cloudflared or ngrok tunnel exposure, governance telemetry,
# runtime readiness probing, and process lifecycle management.
#
# Usage:
#   ./tools/termux_inference_server.sh [options]
#
# Options:
#   --model PATH        Path to GGUF model file (or $NIBLIT_MODEL_PATH)
#   --port PORT         Port to bind (default: 8000 or $NIBLIT_PORT)
#   --host HOST         Host to bind (default: 127.0.0.1)
#   --n-ctx N           Context size (default: 2048 or $NIBLIT_N_CTX)
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

# ── Defaults ──────────────────────────────────────────────────────────────────

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(dirname "$_SCRIPT_DIR")"
_TMPDIR="${TMPDIR:-/tmp}"
_RUNTIME_TAG="niblit-termux-$$"
_PID_FILE="${_TMPDIR}/niblit-llama-server.pid"
_TUNNEL_PID_FILE="${_TMPDIR}/niblit-tunnel.pid"
_LOG_FILE="${_TMPDIR}/niblit-llama-server.log"
_GOVERNANCE_LOG="${_TMPDIR}/niblit_cloud_reflection.jsonl"

PORT="${NIBLIT_PORT:-8000}"
HOST="${NIBLIT_HOST:-127.0.0.1}"
MODEL_PATH="${NIBLIT_MODEL_PATH:-}"
N_CTX="${NIBLIT_N_CTX:-2048}"
N_THREADS="${NIBLIT_N_THREADS:-4}"
N_GPU_LAYERS="${NIBLIT_N_GPU_LAYERS:-0}"
BACKEND_BIN="${NIBLIT_LLAMA_SERVER_BIN:-llama-server}"
TUNNEL_PROVIDER="${NIBLIT_TUNNEL_PROVIDER:-none}"
PUBLIC_URL_OVERRIDE="${NIBLIT_TUNNEL_PUBLIC_URL:-}"
PROFILE="${NIBLIT_PROFILE:-}"
GOVERNANCE_ENABLED=1
DRY_RUN=0
READINESS_TIMEOUT=60

# ── Path resolution ────────────────────────────────────────────────────────────

_find_model_in_dir() {
  local dir="$1"
  local pattern candidate candidate_name
  local -a patterns=(
    "*qwen*.gguf"
    "*llama*.gguf"
    "*mistral*.gguf"
    "*mixtral*.gguf"
    "*phi*.gguf"
    "*gemma*.gguf"
    "*deepseek*.gguf"
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
      echo "[niblit-termux] resolved model path: $MODEL_PATH" >&2
    fi
  fi
}

_resolve_backend_bin() {
  local original_backend_bin="${BACKEND_BIN:-}"
  local candidate resolved=""

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
      echo "[niblit-termux] resolved backend binary: $BACKEND_BIN" >&2
    fi
  fi
}

# ── Trap / cleanup ─────────────────────────────────────────────────────────────

_cleanup() {
  local exit_code=$?
  echo "[niblit-termux] shutdown signal received (exit=$exit_code)"
  _stop_processes
  exit "$exit_code"
}

trap '_cleanup' EXIT INT TERM HUP

_stop_processes() {
  if [[ -f "$_PID_FILE" ]]; then
    local pid
    pid="$(cat "$_PID_FILE" 2>/dev/null || echo "")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[niblit-termux] stopping llama-server (pid=$pid)"
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
      echo "[niblit-termux] stopping tunnel (pid=$tpid)"
      kill "$tpid" 2>/dev/null || true
    fi
    rm -f "$_TUNNEL_PID_FILE"
  fi
}

# ── Argument parsing ──────────────────────────────────────────────────────────

_parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)      MODEL_PATH="$2"; shift 2 ;;
      --port)       PORT="$2"; shift 2 ;;
      --host)       HOST="$2"; shift 2 ;;
      --n-ctx)      N_CTX="$2"; shift 2 ;;
      --n-threads)  N_THREADS="$2"; shift 2 ;;
      --n-gpu-layers) N_GPU_LAYERS="$2"; shift 2 ;;
      --backend)    BACKEND_BIN="$2"; shift 2 ;;
      --tunnel)     TUNNEL_PROVIDER="$2"; shift 2 ;;
      --public-url) PUBLIC_URL_OVERRIDE="$2"; shift 2 ;;
      --profile)    PROFILE="$2"; shift 2 ;;
      --no-governance) GOVERNANCE_ENABLED=0; shift ;;
      --dry-run)    DRY_RUN=1; shift ;;
      *)            echo "[niblit-termux] WARN: unknown arg '$1'" >&2; shift ;;
    esac
  done
}

# ── Profile loading ───────────────────────────────────────────────────────────

_load_profile() {
  if [[ -z "$PROFILE" ]]; then
    return 0
  fi
  local profile_dir="$_SCRIPT_DIR/runtime_profiles"
  local env_file="$profile_dir/${PROFILE}.env"
  if [[ ! -f "$env_file" ]]; then
    echo "[niblit-termux] WARN: profile '$PROFILE' not found at $env_file" >&2
    return 0
  fi
  echo "[niblit-termux] loading profile: $PROFILE"
  # shellcheck disable=SC1090
  source "$profile_dir/profile_loader.sh" "$PROFILE" || true
  # Re-apply overrides from loaded env
  PORT="${NIBLIT_PORT:-$PORT}"
  HOST="${NIBLIT_HOST:-$HOST}"
  MODEL_PATH="${NIBLIT_MODEL_PATH:-$MODEL_PATH}"
  N_CTX="${NIBLIT_N_CTX:-$N_CTX}"
  N_THREADS="${NIBLIT_N_THREADS:-$N_THREADS}"
  N_GPU_LAYERS="${NIBLIT_N_GPU_LAYERS:-$N_GPU_LAYERS}"
  BACKEND_BIN="${NIBLIT_LLAMA_SERVER_BIN:-$BACKEND_BIN}"
  TUNNEL_PROVIDER="${NIBLIT_TUNNEL_PROVIDER:-$TUNNEL_PROVIDER}"
  PUBLIC_URL_OVERRIDE="${NIBLIT_TUNNEL_PUBLIC_URL:-$PUBLIC_URL_OVERRIDE}"
}

# ── Validation ────────────────────────────────────────────────────────────────

_validate() {
  _resolve_model_path
  _resolve_backend_bin

  if [[ -z "$MODEL_PATH" ]]; then
    echo "[niblit-termux] ERROR: model path not set. Use --model or NIBLIT_MODEL_PATH." >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[niblit-termux] ERROR: model file not found: $MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! -x "$BACKEND_BIN" ]] && ! command -v "$BACKEND_BIN" >/dev/null 2>&1; then
    echo "[niblit-termux] ERROR: backend binary '$BACKEND_BIN' not found in PATH." >&2
    echo "[niblit-termux] Hint: run tools/install_llama_server.sh or install llama.cpp" >&2
    exit 1
  fi
  if [[ "$TUNNEL_PROVIDER" != "none" && -z "$PUBLIC_URL_OVERRIDE" ]]; then
    if [[ "$TUNNEL_PROVIDER" == "cloudflared" ]] && ! command -v cloudflared >/dev/null 2>&1; then
      echo "[niblit-termux] WARN: cloudflared not found; tunnel will be skipped" >&2
      TUNNEL_PROVIDER=none
    elif [[ "$TUNNEL_PROVIDER" == "ngrok" ]] && ! command -v ngrok >/dev/null 2>&1; then
      echo "[niblit-termux] WARN: ngrok not found; tunnel will be skipped" >&2
      TUNNEL_PROVIDER=none
    fi
  fi
}

# ── Config display ────────────────────────────────────────────────────────────

_print_config() {
  cat <<EOF
[niblit-termux] ─────────────────────────────────────────────
[niblit-termux]  Niblit Termux Inference Runtime
[niblit-termux]  model:      $MODEL_PATH
[niblit-termux]  backend:    $BACKEND_BIN
[niblit-termux]  host:       $HOST:$PORT
[niblit-termux]  n_ctx:      $N_CTX
[niblit-termux]  n_threads:  $N_THREADS
[niblit-termux]  n_gpu:      $N_GPU_LAYERS
[niblit-termux]  tunnel:     $TUNNEL_PROVIDER
[niblit-termux]  governance: $([ "$GOVERNANCE_ENABLED" -eq 1 ] && echo enabled || echo disabled)
[niblit-termux]  log:        $_LOG_FILE
[niblit-termux] ─────────────────────────────────────────────
EOF
}

# ── Server start ──────────────────────────────────────────────────────────────

_start_server() {
  echo "[niblit-termux] starting $BACKEND_BIN..."
  nohup "$BACKEND_BIN" \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --ctx-size "$N_CTX" \
    --threads "$N_THREADS" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    >>"$_LOG_FILE" 2>&1 &
  echo $! > "$_PID_FILE"
  echo "[niblit-termux] llama-server started (pid=$(cat "$_PID_FILE"))"
}

# ── Readiness probe ───────────────────────────────────────────────────────────

_probe_readiness() {
  local health_url="http://${HOST}:${PORT}/health"
  local deadline=$(($(date +%s) + READINESS_TIMEOUT))
  echo "[niblit-termux] waiting for server readiness at $health_url..."
  while [[ $(date +%s) -lt $deadline ]]; do
    if curl -sf "$health_url" >/dev/null 2>&1; then
      echo "[niblit-termux] ✅ server is ready"
      return 0
    fi
    # Check process is still alive
    if [[ -f "$_PID_FILE" ]] && ! kill -0 "$(cat "$_PID_FILE")" 2>/dev/null; then
      echo "[niblit-termux] ❌ server process died — check $_LOG_FILE" >&2
      return 1
    fi
    sleep 2
  done
  echo "[niblit-termux] ⚠️ readiness timeout after ${READINESS_TIMEOUT}s" >&2
  # Non-fatal — server may still be loading model
  return 0
}

_probe_fallback_endpoints() {
  local base="http://${HOST}:${PORT}"
  for ep in "/v1/models" "/props"; do
    if curl -sf "${base}${ep}" >/dev/null 2>&1; then
      echo "[niblit-termux] fallback probe OK: ${base}${ep}"
      return 0
    fi
  done
  echo "[niblit-termux] WARN: all fallback probes failed" >&2
  return 0
}

# ── Tunnel ────────────────────────────────────────────────────────────────────

_start_tunnel() {
  if [[ -n "$PUBLIC_URL_OVERRIDE" ]]; then
    echo "[niblit-termux] using manual public URL: $PUBLIC_URL_OVERRIDE"
    echo "[niblit-termux] set NIBLIT_LLAMA_SERVER_URL=$PUBLIC_URL_OVERRIDE in Niblit"
    return 0
  fi

  if [[ "$TUNNEL_PROVIDER" == "cloudflared" ]]; then
    echo "[niblit-termux] starting cloudflared tunnel on port $PORT..."
    nohup cloudflared tunnel --url "http://localhost:$PORT" \
      >>"${_TMPDIR}/niblit-tunnel.log" 2>&1 &
    echo $! > "$_TUNNEL_PID_FILE"
    sleep 3
    # Try to extract URL from log
    local tunnel_url
    tunnel_url=$(grep -o 'https://[a-z0-9.-]*\.trycloudflare\.com' \
      "${_TMPDIR}/niblit-tunnel.log" 2>/dev/null | head -1 || echo "")
    if [[ -n "$tunnel_url" ]]; then
      echo "[niblit-termux] ✅ cloudflared URL: $tunnel_url"
      echo "[niblit-termux] set NIBLIT_LLAMA_SERVER_URL=$tunnel_url in Niblit"
    else
      echo "[niblit-termux] WARN: could not detect cloudflared URL automatically"
      echo "[niblit-termux] check ${_TMPDIR}/niblit-tunnel.log for the URL"
    fi

  elif [[ "$TUNNEL_PROVIDER" == "ngrok" ]]; then
    echo "[niblit-termux] starting ngrok tunnel on port $PORT..."
    nohup ngrok http "$PORT" >>"${_TMPDIR}/niblit-ngrok.log" 2>&1 &
    echo $! > "$_TUNNEL_PID_FILE"
    sleep 3
    # Try to extract URL via ngrok API
    local ngrok_url
    ngrok_url=$(curl -sf "http://localhost:4040/api/tunnels" 2>/dev/null | \
      python3 -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print([x for x in t if x.get('proto')=='https'][0]['public_url'])" \
      2>/dev/null || echo "")
    if [[ -n "$ngrok_url" ]]; then
      echo "[niblit-termux] ✅ ngrok URL: $ngrok_url"
      echo "[niblit-termux] set NIBLIT_LLAMA_SERVER_URL=$ngrok_url in Niblit"
    else
      echo "[niblit-termux] WARN: could not detect ngrok URL; check ngrok dashboard"
    fi
  fi
}

# ── Governance telemetry ──────────────────────────────────────────────────────

_emit_governance_event() {
  local event_type="$1"
  local note="${2:-}"
  if [[ "$GOVERNANCE_ENABLED" -ne 1 ]]; then
    return 0
  fi
  local ts
  ts="$(date +%s)"
  local json_line
  json_line="{\"event\":\"${event_type}\",\"ts\":${ts},\"pid\":$$,\"runtime\":\"termux-local\",\"port\":${PORT},\"tag\":\"${_RUNTIME_TAG}\",\"note\":\"${note}\"}"
  echo "$json_line" >> "$_GOVERNANCE_LOG" 2>/dev/null || true
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
  _parse_args "$@"
  _load_profile
  _validate
  _print_config

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[niblit-termux] --dry-run: configuration resolved, not starting processes"
    exit 0
  fi

  _emit_governance_event "runtime_start" "Termux inference server starting"
  _start_server
  _probe_readiness
  _probe_fallback_endpoints
  _start_tunnel
  _emit_governance_event "runtime_ready" "Termux inference server ready"

  echo "[niblit-termux] ✅ inference runtime ready on http://${HOST}:${PORT}"
  echo "[niblit-termux] ctrl-c or SIGTERM to stop"

  # Wait for server process
  local server_pid
  server_pid="$(cat "$_PID_FILE" 2>/dev/null || echo "")"
  if [[ -n "$server_pid" ]]; then
    wait "$server_pid" 2>/dev/null || true
  fi
}

main "$@"
