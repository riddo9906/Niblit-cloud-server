#!/usr/bin/env bash
# tools/install_llama_server.sh — Niblit portable llama.cpp installer
#
# Downloads or builds a llama-server / llama-cli binary for the current
# platform and installs it to a configurable target directory.
#
# Supports:
#   - Direct binary download (when LLAMA_SERVER_URL / LLAMA_CLI_URL set)
#   - Build from source (cmake + make)
#   - Platform architecture matrix: x86_64, arm64, armv7
#   - Termux, Linux, Docker, Fly.io, VPS
#
# Usage:
#   ./tools/install_llama_server.sh [--version VERSION] [--target DIR] [options]
#
# Options:
#   --version VERSION   llama.cpp version/tag (default: b5380 or LLAMA_CPP_VERSION)
#   --target DIR        Install target directory (default: ~/.local/niblit-runtime/bin)
#   --backend BACKEND   Binary to install: llama-server|llama-cli|both (default: llama-server)
#   --url URL           Direct download URL (skips build from source)
#   --checksum SHA256   Expected SHA-256 checksum for downloaded binary
#   --skip-if-exists    Skip installation if binary already exists
#   --force             Overwrite existing binary
#   --dry-run           Print resolved config without installing
#
# Environment variables:
#   LLAMA_CPP_VERSION   Version tag (default: b5380)
#   LLAMA_SERVER_URL    Direct download URL for llama-server
#   LLAMA_CLI_URL       Direct download URL for llama-cli
#   LLAMA_SERVER_SHA256 SHA-256 checksum for llama-server binary
#   LLAMA_CLI_SHA256    SHA-256 checksum for llama-cli binary
#   INSTALL_DIR         Install directory (default: ~/.local/niblit-runtime)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

VERSION="${LLAMA_CPP_VERSION:-b5380}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/niblit-runtime}"
BIN_DIR="${INSTALL_DIR}/bin"
BACKEND="llama-server"
DIRECT_URL="${LLAMA_SERVER_URL:-}"
CHECKSUM="${LLAMA_SERVER_SHA256:-}"
SKIP_IF_EXISTS=0
FORCE=0
DRY_RUN=0
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

# ── Argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)       VERSION="$2"; shift 2 ;;
    --target)        BIN_DIR="$2"; shift 2 ;;
    --backend)       BACKEND="$2"; shift 2 ;;
    --url)           DIRECT_URL="$2"; shift 2 ;;
    --checksum)      CHECKSUM="$2"; shift 2 ;;
    --skip-if-exists) SKIP_IF_EXISTS=1; shift ;;
    --force)         FORCE=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    *) echo "WARN: unknown option '$1'" >&2; shift ;;
  esac
done

# When installing llama-cli, update URL/checksum defaults
if [[ "$BACKEND" == "llama-cli" ]]; then
  DIRECT_URL="${LLAMA_CLI_URL:-$DIRECT_URL}"
  CHECKSUM="${LLAMA_CLI_SHA256:-$CHECKSUM}"
fi

# ── Utility functions ─────────────────────────────────────────────────────────

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing required command: $1" >&2; exit 1; }
}

log() {
  echo "[install-llama] $*"
}

sha256_verify() {
  local file="$1"
  local expected="$2"
  if [[ -z "$expected" ]]; then
    log "WARN: no checksum provided for $file — skipping verification"
    return 0
  fi
  require_cmd sha256sum
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: checksum mismatch for $file" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 2
  fi
  log "checksum OK: $file"
}

arch_display() {
  case "$ARCH" in
    x86_64)  echo "x86_64" ;;
    aarch64) echo "arm64" ;;
    arm64)   echo "arm64" ;;
    armv7*)  echo "armv7" ;;
    *)       echo "$ARCH" ;;
  esac
}

# ── Config display ────────────────────────────────────────────────────────────

log "─────────────────────────────────────────────"
log " Niblit llama.cpp installer"
log " version:  $VERSION"
log " backend:  $BACKEND"
log " target:   $BIN_DIR"
log " os:       $OS"
log " arch:     $(arch_display)"
log " url:      ${DIRECT_URL:-<build from source>}"
log "─────────────────────────────────────────────"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "--dry-run: configuration resolved, nothing installed"
  exit 0
fi

# ── Skip check ────────────────────────────────────────────────────────────────

if [[ "$SKIP_IF_EXISTS" -eq 1 ]] && command -v "$BACKEND" >/dev/null 2>&1; then
  log "$BACKEND already in PATH — skipping (use --force to reinstall)"
  exit 0
fi

if [[ "$SKIP_IF_EXISTS" -eq 1 ]] && [[ -x "$BIN_DIR/$BACKEND" ]]; then
  log "$BACKEND already at $BIN_DIR/$BACKEND — skipping (use --force to reinstall)"
  exit 0
fi

if [[ "$FORCE" -eq 0 ]] && [[ -x "$BIN_DIR/$BACKEND" ]]; then
  log "$BACKEND already exists at $BIN_DIR/$BACKEND"
  log "Use --force to overwrite or --skip-if-exists to skip silently"
  exit 0
fi

mkdir -p "$BIN_DIR"

# ── Download path ─────────────────────────────────────────────────────────────

_install_from_url() {
  local artifact_name="$1"
  local url="$2"
  local checksum="$3"
  local tmp_path="${TMPDIR:-/tmp}/${artifact_name}-${VERSION}-${OS}-$(arch_display)"

  require_cmd curl
  log "downloading $url..."
  curl -fsSL --retry 3 --retry-delay 2 "$url" -o "$tmp_path"

  sha256_verify "$tmp_path" "$checksum"

  install -m 0755 "$tmp_path" "$BIN_DIR/$artifact_name"
  rm -f "$tmp_path"
  log "installed $artifact_name -> $BIN_DIR/$artifact_name"
}

# ── Build from source ─────────────────────────────────────────────────────────

_build_from_source() {
  require_cmd git
  require_cmd cmake
  require_cmd make

  local src="${TMPDIR:-/tmp}/llama.cpp-${VERSION}"
  log "cloning llama.cpp $VERSION..."
  rm -rf "$src"
  git clone --branch "$VERSION" --depth 1 https://github.com/ggml-org/llama.cpp "$src"

  log "configuring build..."
  cmake -S "$src" -B "$src/build" \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release

  local jobs
  jobs="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 2)"
  log "building with $jobs parallel jobs..."
  cmake --build "$src/build" -j"$jobs"

  if [[ "$BACKEND" == "llama-server" || "$BACKEND" == "both" ]]; then
    if [[ -x "$src/build/bin/llama-server" ]]; then
      install -m 0755 "$src/build/bin/llama-server" "$BIN_DIR/llama-server"
      log "installed llama-server -> $BIN_DIR/llama-server"
    fi
  fi
  if [[ "$BACKEND" == "llama-cli" || "$BACKEND" == "both" ]]; then
    if [[ -x "$src/build/bin/llama-cli" ]]; then
      install -m 0755 "$src/build/bin/llama-cli" "$BIN_DIR/llama-cli"
      log "installed llama-cli -> $BIN_DIR/llama-cli"
    fi
  fi
}

# ── Main install ──────────────────────────────────────────────────────────────

if [[ -n "$DIRECT_URL" ]]; then
  _install_from_url "$BACKEND" "$DIRECT_URL" "$CHECKSUM"
else
  log "LLAMA_SERVER_URL not set; building from source"
  _build_from_source
fi

# ── Completion summary ─────────────────────────────────────────────────────────

cat <<MSG
[install-llama] ─────────────────────────────────────────────
[install-llama] Installation complete
[install-llama]   backend: $BACKEND
[install-llama]   version: $VERSION
[install-llama]   bin_dir: $BIN_DIR

[install-llama] Next steps:
[install-llama]   export PATH="$BIN_DIR:\$PATH"
[install-llama]   $BACKEND --help
[install-llama]
[install-llama] To launch Niblit Termux runtime:
[install-llama]   NIBLIT_MODEL_PATH=/path/to/model.gguf \\
[install-llama]     tools/termux_inference_server.sh --profile termux-local
MSG
