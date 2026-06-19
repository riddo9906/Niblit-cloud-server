#!/usr/bin/env bash
# tools/runtime_profiles/profile_loader.sh — deterministic runtime profile loader

set -euo pipefail

_PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_REPO_ROOT="$(cd "${_PROFILE_DIR}/../.." && pwd)"

_list_profiles() {
  find "$_PROFILE_DIR" -maxdepth 1 -type f -name '*.env' -printf '%f\n' \
    | sed 's/\.env$//' | sort | tr '\n' ' '
}

_usage() {
  cat >&2 <<MSG
usage: source profile_loader.sh [profile]
       NIBLIT_PROFILE=<profile> source profile_loader.sh

Profiles: $(_list_profiles)
MSG
}

_load_profile_exports() {
  local profile="$1"
  PYTHONPATH="$_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - "$profile" <<'PY'
import shlex
import sys

from tools.lib.runtime_profiles import ProfileNotFoundError, get_profile_env

profile = sys.argv[1]
try:
    env = get_profile_env(profile)
except ProfileNotFoundError as exc:
    print(f"ERROR:{exc}")
    raise SystemExit(2)

for key, value in env.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
}

_apply_profile() {
  local profile="${1:-${NIBLIT_PROFILE:-cloud-server}}"
  local env_file="$_PROFILE_DIR/${profile}.env"

  if [[ ! -f "$env_file" ]]; then
    echo "[profile_loader] ERROR: unknown profile '$profile'" >&2
    _usage
    return 1
  fi

  echo "[profile_loader] loading profile: $profile"
  while IFS= read -r export_cmd; do
    [[ -z "$export_cmd" ]] && continue
    if [[ "$export_cmd" == ERROR:* ]]; then
      echo "[profile_loader] ${export_cmd#ERROR:}" >&2
      return 1
    fi
    eval "$export_cmd"
  done < <(_load_profile_exports "$profile")

  export NIBLIT_PROFILE="$profile"
  echo "[profile_loader] profile '$profile' loaded"
}

if [[ "${BASH_SOURCE[0]:-}" == "$0" ]]; then
  _apply_profile "${1:-${NIBLIT_PROFILE:-cloud-server}}"
else
  _apply_profile "${1:-${NIBLIT_PROFILE:-cloud-server}}"
fi
