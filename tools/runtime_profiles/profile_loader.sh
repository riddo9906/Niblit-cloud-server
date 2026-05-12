#!/usr/bin/env bash
# tools/runtime_profiles/profile_loader.sh — Niblit runtime profile loader
#
# Sources a named runtime profile into the current shell environment.
# All profile variables are exported for use by subsequent processes.
#
# Usage:
#   source tools/runtime_profiles/profile_loader.sh [profile]
#   profile_loader.sh niblit
#   profile_loader.sh cloud-server
#   profile_loader.sh termux-local
#
# Or via environment variable:
#   NIBLIT_PROFILE=termux-local source tools/runtime_profiles/profile_loader.sh
#
# After sourcing, use:
#   niblit-cloud-server runtime tooling (cloud_runtime_ctl.py, start_server.sh)

set -euo pipefail

_PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_VALID_PROFILES=(niblit cloud-server termux-local)

_usage() {
  cat >&2 <<MSG
usage: source profile_loader.sh [profile]
       NIBLIT_PROFILE=<profile> source profile_loader.sh

Profiles: ${_VALID_PROFILES[*]}
MSG
}

_load_profile() {
  local profile="${1:-${NIBLIT_PROFILE:-cloud-server}}"
  local env_file="$_PROFILE_DIR/${profile}.env"

  # Validate profile name
  local valid=0
  for p in "${_VALID_PROFILES[@]}"; do
    [[ "$p" == "$profile" ]] && valid=1 && break
  done
  if [[ $valid -eq 0 ]]; then
    echo "[profile_loader] ERROR: unknown profile '$profile'. Valid: ${_VALID_PROFILES[*]}" >&2
    return 1
  fi

  if [[ ! -f "$env_file" ]]; then
    echo "[profile_loader] ERROR: profile file not found: $env_file" >&2
    return 1
  fi

  echo "[profile_loader] loading profile: $profile ($env_file)"

  # Parse and export each non-comment, non-blank line from the .env file.
  # Uses eval to allow variable interpolation (e.g., ${HOME}/models).
  while IFS= read -r line; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Skip lines without an = sign
    [[ "$line" != *=* ]] && continue

    local key="${line%%=*}"
    local val="${line#*=}"

    # Only process valid variable names
    if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      # Expand shell variables in value
      # shellcheck disable=SC2163
      eval "export ${key}=${val}" 2>/dev/null || true
    fi
  done < "$env_file"

  echo "[profile_loader] profile '$profile' loaded — NIBLIT_PROFILE=$profile"
}

# If executed directly (not sourced), just run and exit
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
  _load_profile "${1:-}"
else
  # Being sourced — run and return
  _load_profile "${1:-${NIBLIT_PROFILE:-cloud-server}}"
fi
