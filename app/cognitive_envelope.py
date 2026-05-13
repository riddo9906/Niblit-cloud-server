#!/usr/bin/env python3
"""app/cognitive_envelope.py — Niblit Cognitive Cloud Runtime Request Envelope.

The cloud runtime accepts an enriched *cognitive envelope* alongside standard
OpenAI/HF-compatible chat payloads.  Envelope fields are optional — plain
requests that omit them continue to work exactly as before.

Schema v2 structure is aligned with:
- ``freqtrade_strategies/cognitive_envelope.py`` in niblit-lean-algos PR#20
- Phase Ω.7 execution envelope spec from ``modules/lean_algo_manager.py``

Backward compatibility rule
----------------------------
Any request without envelope fields is treated as a ``schema_version="1.0"``
legacy request and receives sensible defaults.  This preserves compatibility
with ``QwenLocalBrain``, llama.cpp clients, and all HF-style API consumers.
"""

from __future__ import annotations

import time
from typing import Any

# ── Envelope defaults ──────────────────────────────────────────────────────────

_SCHEMA_VERSION = "2.0"

_DEFAULT_GOVERNANCE = {
    "constitution_passed": True,
    "risk_tier": "medium",
    "authority": "cloud_runtime",
    "survival_mode": False,
    "governance_mode": "normal",
    "governance_stability": 1.0,
    "current_drawdown_pct": 0.0,
    "max_drawdown_pct": 0.12,
}

_DEFAULT_EXECUTION = {
    "max_position_size": 0.02,
    "stoploss_override": None,
    "allow_scale_in": False,
    "hold_only": False,
    "runtime_stability": 1.0,
    "execution_priority": "normal",
}

_DEFAULT_TEMPORAL = {
    "epoch_id": 0,
    "coherence_score": 1.0,
    "epoch_alignment": "aligned",
}

_DEFAULT_RUNTIME = {
    "mode": "normal",
    "health": "ok",
    "instability": 0.0,
    "attention_pressure": 0.0,
    "runtime_health": 1.0,
}

_DEFAULT_RESOURCES = {
    "cognitive_budget": 1.0,
    "attention_available": 1.0,
}

_VALID_INTENTS = frozenset({
    "conversational", "analytical", "forecasting", "trading", "reasoning",
    "tool_use", "code_generation", "summarization", "creative", "unknown",
})

_VALID_EXECUTION_MODES = frozenset({
    "analytical", "generative", "fast", "deep", "balanced", "survival",
})

_VALID_RESOURCE_MODES = frozenset({
    "balanced", "conservative", "aggressive", "minimal", "unlimited",
})

_VALID_GOVERNANCE_MODES = frozenset({"normal", "cautious", "survival", "lockdown"})

_VALID_CONSTITUTIONAL_PRIORITIES = frozenset({"safety", "alignment", "efficiency", "exploration"})


def normalize_envelope(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a raw request dict into a stable cognitive envelope shape.

    Accepts either:
    - A plain OpenAI/HF request (``messages``, ``model``, etc. only) → defaults
    - A partial or full schema-v2 envelope → validated and normalized

    Never raises — returns a safe default envelope on any error.
    """
    if not isinstance(raw, dict):
        return _build_default_envelope()

    src = dict(raw)

    # ── Intent ────────────────────────────────────────────────────────────────
    intent = str(src.get("intent", "conversational")).lower().strip()
    if intent not in _VALID_INTENTS:
        intent = "unknown"

    # ── Execution mode ────────────────────────────────────────────────────────
    exec_mode = str(src.get("execution_mode", "balanced")).lower().strip()
    if exec_mode not in _VALID_EXECUTION_MODES:
        exec_mode = "balanced"

    # ── Coherence score ───────────────────────────────────────────────────────
    coherence = _clamp(float(src.get("coherence_score", 1.0)), 0.0, 1.0)

    # ── Constitutional priority ───────────────────────────────────────────────
    const_priority = str(src.get("constitutional_priority", "safety")).lower().strip()
    if const_priority not in _VALID_CONSTITUTIONAL_PRIORITIES:
        const_priority = "safety"

    # ── Attention budget ──────────────────────────────────────────────────────
    attention_budget = _clamp(float(src.get("attention_budget", 1.0)), 0.0, 1.0)

    # ── Resource mode ─────────────────────────────────────────────────────────
    resource_mode = str(src.get("resource_mode", "balanced")).lower().strip()
    if resource_mode not in _VALID_RESOURCE_MODES:
        resource_mode = "balanced"

    # ── Epoch tag ─────────────────────────────────────────────────────────────
    epoch_tag = str(src.get("epoch_tag", f"epoch_{int(time.time())}"))

    # ── Nested context fields (pass-through with type safety) ─────────────────
    forecast_context = dict(src.get("forecast_context") or {})
    governance_context = dict(src.get("governance_context") or {})
    tool_context = dict(src.get("tool_context") or {})
    reflection_context = dict(src.get("reflection_context") or {})
    identity_context = dict(src.get("identity_context") or {})

    # ── Governance ────────────────────────────────────────────────────────────
    raw_gov = src.get("governance") or {}
    if not isinstance(raw_gov, dict):
        raw_gov = {}
    governance = {**_DEFAULT_GOVERNANCE, **_coerce_governance(raw_gov)}
    gov_mode = str(governance.get("governance_mode", "normal")).lower()
    if gov_mode not in _VALID_GOVERNANCE_MODES:
        gov_mode = "normal"
    governance["governance_mode"] = gov_mode

    # ── Temporal ──────────────────────────────────────────────────────────────
    raw_temp = src.get("temporal") or {}
    if not isinstance(raw_temp, dict):
        raw_temp = {}
    temporal = {**_DEFAULT_TEMPORAL, **_coerce_temporal(raw_temp)}
    # coherence_score in envelope always wins over nested temporal coherence
    temporal["coherence_score"] = coherence

    # ── Runtime ───────────────────────────────────────────────────────────────
    raw_runtime = src.get("runtime") or {}
    if not isinstance(raw_runtime, dict):
        raw_runtime = {}
    runtime = {**_DEFAULT_RUNTIME, **_coerce_runtime(raw_runtime)}

    return {
        "schema_version": _SCHEMA_VERSION,
        "intent": intent,
        "execution_mode": exec_mode,
        "coherence_score": coherence,
        "constitutional_priority": const_priority,
        "attention_budget": attention_budget,
        "resource_mode": resource_mode,
        "epoch_tag": epoch_tag,
        "forecast_context": forecast_context,
        "governance_context": governance_context,
        "governance": governance,
        "tool_context": tool_context,
        "reflection_context": reflection_context,
        "identity_context": identity_context,
        "temporal": temporal,
        "runtime": runtime,
        "resources": {
            "cognitive_budget": attention_budget,
            "attention_available": attention_budget,
        },
    }


def _build_default_envelope() -> dict[str, Any]:
    ts = int(time.time())
    return {
        "schema_version": _SCHEMA_VERSION,
        "intent": "conversational",
        "execution_mode": "balanced",
        "coherence_score": 1.0,
        "constitutional_priority": "safety",
        "attention_budget": 1.0,
        "resource_mode": "balanced",
        "epoch_tag": f"epoch_{ts}",
        "forecast_context": {},
        "governance_context": {},
        "governance": dict(_DEFAULT_GOVERNANCE),
        "tool_context": {},
        "reflection_context": {},
        "identity_context": {},
        "temporal": {**_DEFAULT_TEMPORAL, "epoch_id": ts, "coherence_score": 1.0},
        "runtime": dict(_DEFAULT_RUNTIME),
        "resources": dict(_DEFAULT_RESOURCES),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _coerce_governance(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "constitution_passed" in raw:
        out["constitution_passed"] = bool(raw["constitution_passed"])
    if "risk_tier" in raw:
        out["risk_tier"] = str(raw["risk_tier"])
    if "authority" in raw:
        out["authority"] = str(raw["authority"])
    if "survival_mode" in raw:
        out["survival_mode"] = bool(raw["survival_mode"])
    if "governance_mode" in raw:
        out["governance_mode"] = str(raw["governance_mode"]).lower()
    if "governance_stability" in raw:
        out["governance_stability"] = _clamp(float(raw["governance_stability"]), 0.0, 1.0)
    if "current_drawdown_pct" in raw:
        out["current_drawdown_pct"] = max(0.0, float(raw["current_drawdown_pct"]))
    if "max_drawdown_pct" in raw:
        out["max_drawdown_pct"] = max(0.0, float(raw["max_drawdown_pct"]))
    return out


def _coerce_temporal(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "epoch_id" in raw:
        out["epoch_id"] = int(raw["epoch_id"])
    if "coherence_score" in raw:
        out["coherence_score"] = _clamp(float(raw["coherence_score"]), 0.0, 1.0)
    if "epoch_alignment" in raw:
        out["epoch_alignment"] = str(raw["epoch_alignment"])
    return out


def _coerce_runtime(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "mode" in raw:
        out["mode"] = str(raw["mode"]).lower()
    if "health" in raw:
        out["health"] = str(raw["health"])
    if "instability" in raw:
        out["instability"] = _clamp(float(raw["instability"]), 0.0, 1.0)
    if "attention_pressure" in raw:
        out["attention_pressure"] = _clamp(float(raw["attention_pressure"]), 0.0, 1.0)
    if "runtime_health" in raw:
        out["runtime_health"] = _clamp(float(raw["runtime_health"]), 0.0, 1.0)
    return out


def is_trading_intent(envelope: dict[str, Any]) -> bool:
    """Return True when the envelope signals a trading/forecasting intent."""
    intent = str(envelope.get("intent", "")).lower()
    return intent in ("forecasting", "trading")


def governance_mode(envelope: dict[str, Any]) -> str:
    """Extract the effective governance mode from a normalized envelope."""
    gov = envelope.get("governance") or {}
    runtime = envelope.get("runtime") or {}
    mode = str(
        gov.get("governance_mode") or runtime.get("mode") or "normal"
    ).lower()
    if mode not in _VALID_GOVERNANCE_MODES:
        return "normal"
    return mode
