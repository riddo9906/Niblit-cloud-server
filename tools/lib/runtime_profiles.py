#!/usr/bin/env python3
"""tools/lib/runtime_profiles.py — Niblit portable runtime profile library.

Loads runtime profiles (niblit, cloud-server, termux-local) from
``tools/runtime_profiles/*.env`` files and makes them available as
normalized Python dicts.

Profiles define the full runtime environment for a Niblit deployment:
- app name and listen address
- IPC socket paths (UNIX / TCP)
- model paths and backend type
- governance defaults (mode, strictness, token limits)
- tunnel configuration
- node identity configuration
- reflection/telemetry paths

Usage
-----
    from tools.lib.runtime_profiles import load_profile, get_profile_env, list_profiles

    # Load a profile into the current process environment
    load_profile("cloud-server")

    # Get profile as a dict (without modifying os.environ)
    env = get_profile_env("termux-local")
    print(env["NIBLIT_PROFILE"])   # "termux-local"

    # List known profiles
    print(list_profiles())         # ["niblit", "cloud-server", "termux-local"]
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

# Canonical location of profile files
_PROFILES_DIR = Path(__file__).parent.parent / "runtime_profiles"

# Known profile names
KNOWN_PROFILES: tuple[str, ...] = ("niblit", "cloud-server", "termux-local")

# Mapping of profile name → governance mode semantics
GOVERNANCE_MODES: dict[str, dict[str, Any]] = {
    "niblit": {
        "runtime_mode": "normal",
        "strict": True,
        "resource_class": "balanced",
        "description": "Full Niblit main-app runtime — balanced governance",
    },
    "cloud-server": {
        "runtime_mode": "normal",
        "strict": True,
        "resource_class": "balanced",
        "description": "Cloud server runtime — strict governance, balanced throughput",
    },
    "termux-local": {
        "runtime_mode": "minimal",
        "strict": True,
        "resource_class": "minimal",
        "description": "Edge/Termux local runtime — minimal resources, conservative queuing",
    },
}


class ProfileNotFoundError(ValueError):
    """Raised when a requested profile name is not found."""


def _profile_path(profile: str) -> Path:
    """Return the path to the profile .env file."""
    return _PROFILES_DIR / f"{profile}.env"


def list_profiles() -> list[str]:
    """Return a list of all available profile names."""
    profiles = []
    if _PROFILES_DIR.exists():
        for p in sorted(_PROFILES_DIR.iterdir()):
            if p.suffix == ".env" and p.stem not in ("__pycache__",):
                profiles.append(p.stem)
    return profiles


def get_profile_env(profile: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Parse a profile .env file and return a dict of key → value.

    Values are shell-expanded relative to *base_env* (defaults to ``os.environ``).
    Variable references like ``${HOME}/models`` are resolved.

    Args:
        profile:   Profile name (e.g. ``"cloud-server"``).
        base_env:  Base environment for variable expansion.  Uses
                   ``os.environ`` when not provided.

    Returns:
        A dict mapping variable names to their expanded string values.

    Raises:
        ProfileNotFoundError: If the profile file is missing.
    """
    path = _profile_path(profile)
    if not path.exists():
        raise ProfileNotFoundError(
            f"Profile '{profile}' not found.  Expected: {path}.  "
            f"Available profiles: {list_profiles()}"
        )

    env = dict(base_env or os.environ)
    result: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Skip comments and blank lines
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, _, val = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue

        # Expand shell variables using current merged environment
        try:
            expanded = os.path.expandvars(val)
            # Also expand ${VAR:-default} patterns via shlex-safe substitution
            expanded = _expand_braces(expanded, env)
        except Exception:  # noqa: BLE001
            expanded = val

        result[key] = expanded
        # Make newly defined vars available for subsequent expansions
        env[key] = expanded

    return result


def load_profile(profile: str, override: bool = False) -> dict[str, str]:
    """Load a profile into ``os.environ``.

    Args:
        profile:  Profile name (e.g. ``"cloud-server"``).
        override: If ``True``, overwrite existing env vars.  If ``False``
                  (default), existing values take precedence (safe merge).

    Returns:
        The profile dict that was applied.
    """
    env = get_profile_env(profile)
    for key, val in env.items():
        if override or key not in os.environ:
            os.environ[key] = val
    return env


def active_profile() -> str:
    """Return the currently active profile name from ``NIBLIT_PROFILE``.

    Defaults to ``"cloud-server"`` if not set.
    """
    return os.environ.get("NIBLIT_PROFILE", "cloud-server")


def profile_summary(profile: str) -> dict[str, Any]:
    """Return a structured summary dict for a profile.

    Includes governance metadata and key configuration fields.
    """
    try:
        env = get_profile_env(profile)
    except ProfileNotFoundError:
        return {"profile": profile, "error": "not_found"}

    gov = GOVERNANCE_MODES.get(profile, {})
    return {
        "profile": profile,
        "app_name": env.get("NIBLIT_APP_NAME", ""),
        "host": env.get("NIBLIT_HOST", ""),
        "port": env.get("NIBLIT_PORT", ""),
        "runtime_mode": env.get("NIBLIT_RUNTIME_MODE", gov.get("runtime_mode", "normal")),
        "governance_strict": env.get("NIBLIT_GOVERNANCE_STRICT", "1") not in ("0", "false"),
        "resource_class": gov.get("resource_class", "balanced"),
        "gguf_backend": env.get("NIBLIT_GGUF_BACKEND", ""),
        "unix_socket": env.get("NIBLIT_UNIX_SOCKET", ""),
        "tunnel_provider": env.get("NIBLIT_TUNNEL_PROVIDER", "none"),
        "federation_enabled": env.get("NIBLIT_FEDERATION_ENABLED", "0") not in ("0", "false"),
        "description": gov.get("description", ""),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _expand_braces(val: str, env: dict[str, str]) -> str:
    """Expand ``${VAR:-default}`` and ``${VAR}`` patterns in *val*."""
    import re

    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        if ":-" in inner:
            var_name, _, default = inner.partition(":-")
            return env.get(var_name.strip(), default)
        return env.get(inner.strip(), m.group(0))

    return re.sub(r"\$\{([^}]+)\}", _replace, val)
