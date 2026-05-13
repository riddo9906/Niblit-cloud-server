#!/usr/bin/env bash
set -euo pipefail

# Niblit Cloud Runtime portable backend installer
# Targets: flyio, docker, vps, arm-server, local-linux, edge-arm

TARGET="${1:-local-linux}"
BACKEND="${2:-llama-server}" # llama-server | llama-cli | future
VERSION="${LLAMA_CPP_VERSION:-b5380}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/niblit-runtime}"
BIN_DIR="$INSTALL_DIR/bin"
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

mkdir -p "$BIN_DIR"

echo "[runtime-installer] target=$TARGET backend=$BACKEND version=$VERSION os=$OS arch=$ARCH"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1" >&2; exit 1; }
}

sha256_verify() {
  local file="$1"
  local expected="$2"
  if [[ -z "$expected" ]]; then
    echo "[runtime-installer] WARN: no checksum provided for $file"
    return 0
  fi
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "checksum mismatch for $file" >&2
    echo "expected=$expected actual=$actual" >&2
    exit 2
  fi
}

install_llama_binary() {
  local artifact_name="$1"
  local url="$2"
  local checksum_var="$3"
  local out="/tmp/${artifact_name}-${VERSION}-${OS}-${ARCH}"

  require_cmd curl
  require_cmd sha256sum

  echo "[runtime-installer] downloading $url"
  curl -fsSL "$url" -o "$out"

  local expected="${!checksum_var:-}"
  sha256_verify "$out" "$expected"

  install -m 0755 "$out" "$BIN_DIR/$artifact_name"
  echo "[runtime-installer] installed $artifact_name -> $BIN_DIR/$artifact_name"
}

install_from_source() {
  require_cmd git
  require_cmd cmake
  require_cmd make
  require_cmd gcc

  local src="/tmp/llama.cpp-${VERSION}"
  rm -rf "$src"
  git clone --branch "$VERSION" --depth 1 https://github.com/ggml-org/llama.cpp "$src"
  cmake -S "$src" -B "$src/build" -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON
  cmake --build "$src/build" -j"$(nproc)"

  if [[ -x "$src/build/bin/llama-server" ]]; then
    install -m 0755 "$src/build/bin/llama-server" "$BIN_DIR/llama-server"
  fi
  if [[ -x "$src/build/bin/llama-cli" ]]; then
    install -m 0755 "$src/build/bin/llama-cli" "$BIN_DIR/llama-cli"
  fi
}

case "$TARGET" in
  flyio|docker|vps|arm-server|local-linux|edge-arm)
    ;;
  *)
    echo "unsupported target: $TARGET" >&2
    exit 3
    ;;
esac

if [[ "$BACKEND" == "future" ]]; then
  echo "[runtime-installer] backend=future is a reserved stub for upcoming runtime backends"
  exit 0
fi

if [[ "$BACKEND" == "llama-server" ]]; then
  if [[ -n "${LLAMA_SERVER_URL:-}" ]]; then
    install_llama_binary "llama-server" "$LLAMA_SERVER_URL" "LLAMA_SERVER_SHA256"
  else
    echo "[runtime-installer] LLAMA_SERVER_URL not set; building from source"
    install_from_source
  fi
elif [[ "$BACKEND" == "llama-cli" ]]; then
  if [[ -n "${LLAMA_CLI_URL:-}" ]]; then
    install_llama_binary "llama-cli" "$LLAMA_CLI_URL" "LLAMA_CLI_SHA256"
  else
    echo "[runtime-installer] LLAMA_CLI_URL not set; building from source"
    install_from_source
  fi
else
  echo "unsupported backend: $BACKEND" >&2
  exit 4
fi

cat <<MSG
[runtime-installer] complete
- target: $TARGET
- backend: $BACKEND
- version: $VERSION
- bin_dir: $BIN_DIR

Next:
  export PATH="$BIN_DIR:\$PATH"
  python /app/tools/cloud_runtime_ctl.py --url http://localhost:8000 health
MSG
