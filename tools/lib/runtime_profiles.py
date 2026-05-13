#!/usr/bin/env python3
"""Portable runtime profile utilities for Niblit cloud/local orchestration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_PROFILES_DIR = Path(__file__).parent.parent / "runtime_profiles"

CANONICAL_RUNTIME_MODES: tuple[str, ...] = ("normal", "cautious", "survival", "lockdown")
_MODE_ALIASES: dict[str, str] = {
    "constrained": "cautious",
    "minimal": "cautious",  # backward-compatible alias from Phase Ω.8
}

KNOWN_PROFILES: tuple[str, ...] = (
    "niblit",
    "cloud-server",
    "termux-local",
    "local-runtime",
    "edge-runtime",
    "degraded-runtime",
    "disconnected-runtime",
)

PROFILE_META: dict[str, dict[str, Any]] = {
    "niblit": {
        "runtime_mode": "normal",
        "resource_class": "balanced",
        "topology": "local",
        "connectivity": "connected",
        "description": "Niblit core runtime profile",
    },
    "cloud-server": {
        "runtime_mode": "normal",
        "resource_class": "balanced",
        "topology": "cloud",
        "connectivity": "connected",
        "description": "Canonical cloud runtime node",
    },
    "termux-local": {
        "runtime_mode": "cautious",
        "resource_class": "minimal",
        "topology": "edge",
        "connectivity": "connected",
        "description": "Termux edge runtime with constrained resources",
    },
    "local-runtime": {
        "runtime_mode": "normal",
        "resource_class": "balanced",
        "topology": "local",
        "connectivity": "connected",
        "description": "Portable local runtime profile",
    },
    "edge-runtime": {
        "runtime_mode": "cautious",
        "resource_class": "minimal",
        "topology": "edge",
        "connectivity": "connected",
        "description": "Portable edge runtime profile",
    },
    "degraded-runtime": {
        "runtime_mode": "survival",
        "resource_class": "degraded",
        "topology": "hybrid",
        "connectivity": "unstable",
        "description": "Degraded runtime under pressure",
    },
    "disconnected-runtime": {
        "runtime_mode": "lockdown",
        "resource_class": "isolated",
        "topology": "disconnected",
        "connectivity": "offline",
        "description": "Disconnected/offline runtime profile",
    },
}


class ProfileNotFoundError(ValueError):
    """Raised when a requested profile is not available."""


def normalize_runtime_mode(mode: object, default: str = "normal") -> str:
    """Normalize runtime mode to canonical Ω.7 four-mode contract."""
    candidate = str(mode or default).strip().lower()
    candidate = _MODE_ALIASES.get(candidate, candidate)
    if candidate not in CANONICAL_RUNTIME_MODES:
        return default
    return candidate


def list_profiles() -> list[str]:
    """List all available profile names from the runtime_profiles directory."""
    if not _PROFILES_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in _PROFILES_DIR.iterdir()
        if p.suffix == ".env" and p.is_file() and not p.name.startswith(".")
    )


def _profile_path(profile: str) -> Path:
    return _PROFILES_DIR / f"{profile}.env"


def _expand_braces(val: str, env: dict[str, str]) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` tokens."""

    def _replace(m: re.Match[str]) -> str:
        inner = m.group(1)
        if ":-" in inner:
            var_name, _, fallback = inner.partition(":-")
            return env.get(var_name.strip(), fallback)
        return env.get(inner.strip(), m.group(0))

    return re.sub(r"\$\{([^}]+)\}", _replace, val)


def get_profile_env(profile: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Parse a profile file and return expanded key/value pairs."""
    path = _profile_path(profile)
    if not path.exists():
        raise ProfileNotFoundError(
            f"Profile '{profile}' not found. Expected: {path}. Available: {list_profiles()}"
        )

    env = dict(base_env or os.environ)
    out: dict[str, str] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue

        expanded = _expand_braces(os.path.expandvars(val), env)
        out[key] = expanded
        env[key] = expanded

    if "NIBLIT_RUNTIME_MODE" in out:
        out["NIBLIT_RUNTIME_MODE"] = normalize_runtime_mode(out["NIBLIT_RUNTIME_MODE"])

    # Keep profile identity deterministic
    out.setdefault("NIBLIT_PROFILE", profile)
    return out


def load_profile(profile: str, override: bool = False) -> dict[str, str]:
    """Load profile variables into ``os.environ``."""
    env = get_profile_env(profile)
    for k, v in env.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return env


def active_profile() -> str:
    """Return active profile, defaulting to cloud-server."""
    return os.environ.get("NIBLIT_PROFILE", "cloud-server")


def resolve_profile(
    *,
    profile: str | None = None,
    topology: str | None = None,
    degraded: bool = False,
    disconnected: bool = False,
) -> str:
    """Resolve profile from explicit profile or topology/runtime conditions."""
    if profile:
        return profile
    if disconnected:
        return "disconnected-runtime"
    if degraded:
        return "degraded-runtime"

    topo = (topology or "cloud").strip().lower()
    if topo in {"edge", "termux"}:
        return "edge-runtime"
    if topo in {"local", "desktop"}:
        return "local-runtime"
    return "cloud-server"


def compatibility_summary() -> dict[str, str]:
    """Compatibility metadata aligned with Niblit PR #219 governance contract."""
    return {
        "schema_version": "2.x",
        "event_contract_version": "omega-7",
        "governance_contract_version": "1.x",
        "advisor_protocol_version": "2.x",
        "runtime_mode_contract": "2026.05",
    }


def profile_summary(profile: str) -> dict[str, Any]:
    """Return profile summary with topology and compatibility metadata."""
    try:
        env = get_profile_env(profile)
    except ProfileNotFoundError:
        return {"profile": profile, "error": "not_found"}

    meta = PROFILE_META.get(profile, {})
    mode = normalize_runtime_mode(env.get("NIBLIT_RUNTIME_MODE", meta.get("runtime_mode", "normal")))

    return {
        "profile": profile,
        "app_name": env.get("NIBLIT_APP_NAME", ""),
        "host": env.get("NIBLIT_HOST", ""),
        "port": env.get("NIBLIT_PORT", ""),
        "runtime_mode": mode,
        "governance_strict": env.get("NIBLIT_GOVERNANCE_STRICT", "1").lower() not in {"0", "false"},
        "resource_class": meta.get("resource_class", "balanced"),
        "topology": meta.get("topology", "unknown"),
        "connectivity": meta.get("connectivity", "unknown"),
        "gguf_backend": env.get("NIBLIT_GGUF_BACKEND", ""),
        "unix_socket": env.get("NIBLIT_UNIX_SOCKET", ""),
        "tunnel_provider": env.get("NIBLIT_TUNNEL_PROVIDER", "none"),
        "federation_enabled": env.get("NIBLIT_FEDERATION_ENABLED", "0").lower() not in {"0", "false"},
        "compatibility": compatibility_summary(),
        "description": meta.get("description", ""),
    }
