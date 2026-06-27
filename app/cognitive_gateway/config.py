"""Configuration for the Cognitive Gateway Layer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


REQUEST_TYPES: tuple[str, ...] = (
    "coding",
    "reasoning",
    "memory_lookup",
    "tool_usage",
    "general_chat",
)

COMPUTE_PRIORITIES: tuple[str, ...] = ("low", "medium", "high")

ROUTING_STRATEGIES: tuple[str, ...] = ("dynamic", "passthrough", "static")


@dataclass(frozen=True)
class CognitiveGatewayConfig:
    enabled: bool = True
    routing_strategy: str = "dynamic"
    memory_hook_enabled: bool = False
    tool_hook_enabled: bool = False
    routing_rules: dict[str, str] = field(default_factory=dict)
    fallback_model: str | None = None
    respect_explicit_model: bool = True


def _load_routing_rules() -> dict[str, str]:
    """Load routing rules from env JSON or config file path."""
    raw_json = os.getenv("COGNITIVE_GATEWAY_ROUTING_JSON", "").strip()
    if raw_json:
        try:
            loaded = json.loads(raw_json)
            if isinstance(loaded, dict):
                return {str(k): str(v) for k, v in loaded.items()}
        except json.JSONDecodeError:
            pass

    rules_path = os.getenv("COGNITIVE_GATEWAY_ROUTING_FILE", "").strip()
    if rules_path and os.path.isfile(rules_path):
        try:
            with open(rules_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                return {str(k): str(v) for k, v in loaded.items()}
        except (OSError, json.JSONDecodeError):
            pass

    return {}


def load_cognitive_gateway_config() -> CognitiveGatewayConfig:
    strategy = os.getenv("ROUTING_STRATEGY", "dynamic").strip().lower()
    if strategy not in ROUTING_STRATEGIES:
        strategy = "dynamic"

    rules = _load_routing_rules()
    fallback = rules.get("fallback") or os.getenv("COGNITIVE_GATEWAY_FALLBACK_MODEL")

    return CognitiveGatewayConfig(
        enabled=_env_bool("ENABLE_COGNITIVE_GATEWAY", True),
        routing_strategy=strategy,
        memory_hook_enabled=_env_bool("MEMORY_HOOK_ENABLED", False),
        tool_hook_enabled=_env_bool("TOOL_HOOK_ENABLED", False),
        routing_rules=rules,
        fallback_model=fallback,
        respect_explicit_model=_env_bool("COGNITIVE_GATEWAY_RESPECT_EXPLICIT_MODEL", True),
    )
