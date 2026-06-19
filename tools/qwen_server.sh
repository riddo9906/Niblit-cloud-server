#!/usr/bin/env bash
# tools/qwen_server.sh — Niblit Coder Model Runtime (llama.cpp Vulkan CLI edition)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 🔥 FIXED: your actual working llama.cpp build path
_LLAMA_CLI_DEFAULT="C:/Users/Riyaad/llama_migration/llama.cpp/build/bin/Release/llama-server.exe"

# ── Defaults ──────────────────────────────────────────────────────────────────

HOST="${NIBLIT_HOST:-127.0.0.1}"
PORT="${NIBLIT_PORT:-8000}"   # (unused for CLI, kept for compatibility)

MODEL_PATH="${NIBLIT_MODEL_PATH:-C:/Users/Riyaad/llama_migration/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf}"

THREADS="${NIBLIT_N_THREADS:-8}"
CTX="${NIBLIT_N_CTX:-4096}"

# 🔥 GPU OFFLOAD (your RX 580 Vulkan backend)
N_GPU_LAYERS="${NIBLIT_N_GPU_LAYERS:-35}"

# backend
LLAMA_CLI="${NIBLIT_LLAMA_CLI_BIN:-$_LLAMA_CLI_DEFAULT}"

SYSTEM_PROMPT="You are a coding assistant. Be concise and technical."

DRY_RUN=0

# ── Helpers ───────────────────────────────────────────────────────────────────

_print_config() {
cat <<EOF
[niblit-coder] ─────────────────────────────
model:      $MODEL_PATH
binary:     $LLAMA_CLI
threads:    $THREADS
ctx:        $CTX
gpu layers: $N_GPU_LAYERS
────────────────────────────────────────────
EOF
}

_validate() {
  if [[ ! -e "$MODEL_PATH" ]]; then
    echo "[niblit-coder] ERROR: model not found: $MODEL_PATH" >&2
    exit 1
  fi

  if [[ ! -e "$LLAMA_CLI" ]]; then
    echo "[niblit-coder] ERROR: llama-cli not found: $LLAMA_CLI" >&2
    exit 1
  fi
}

# ── Run ───────────────────────────────────────────────────────────────────────

_run() {
  echo "[niblit-coder] starting inference runtime..."

  "$LLAMA_CLI" \
    -m "$MODEL_PATH" \
    -sys "$SYSTEM_PROMPT" \
    -cnv \
    -t "$THREADS" \
    -c "$CTX" \
    -fa auto \
    -ngl "$N_GPU_LAYERS"
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
  _print_config
  _validate

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[niblit-coder] dry-run complete"
    exit 0
  fi

  _run
}

main "$@"