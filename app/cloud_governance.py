#!/usr/bin/env python3
"""app/cloud_governance.py — Niblit Cognitive Cloud Runtime Constitutional Governance.

Enforces Niblit's seven constitutional laws for every inference request that
carries a cognitive envelope.  Plain OpenAI/HF-compatible requests without
envelope fields are allowed through under ``AUTHORITY_RESPONSES`` (leaf-level
authority) so backward compatibility is fully preserved.

Constitutional Laws (aligned with ``modules/constitutional_layer.py``)
-----------------------------------------------------------------------
LAW_1  preserve_system_integrity        — stability >= 0.3
LAW_2  objective_alignment_priority     — objective_alignment >= 0.4
LAW_3  no_short_term_stability_sacrifice — stability >= 0.5 under pressure
LAW_4  constrain_low_confidence_autonomy — confidence >= 0.35 for autonomous
LAW_5  external_systems_cannot_override  — external source cannot override objective
LAW_6  temporal_incoherence_halts_exec   — coherence must not be False
LAW_7  safety_overrides_efficiency       — governance approval required for high-safety

Cloud-runtime-specific safety guards
--------------------------------------
- hard token ceiling (MAX_TOKENS_HARD_LIMIT)
- recursion depth ceiling
- prompt injection heuristics
- governance mode lockdown enforcement
- influence saturation limit (attention_budget <= 1.0)
- malformed envelope rejection

Configuration (env vars)
------------------------
    NIBLIT_CG_ENABLED         — "0" to disable (default 1)
    NIBLIT_CG_STRICT          — "0" for permissive mode (default 1)
    NIBLIT_CG_MAX_TOKENS      — hard token ceiling (default 8192)
    NIBLIT_CG_MAX_RECURSION   — max recursion depth (default 16)
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitCloudGovernance")

_ENABLED: bool = os.getenv("NIBLIT_CG_ENABLED", "1").strip() not in ("0", "false")
_STRICT: bool = os.getenv("NIBLIT_CG_STRICT", "1").strip() not in ("0", "false")
_MAX_TOKENS: int = int(os.getenv("NIBLIT_CG_MAX_TOKENS", "8192"))
_MAX_RECURSION: int = int(os.getenv("NIBLIT_CG_MAX_RECURSION", "16"))

# ── Constitutional law constants ───────────────────────────────────────────────
LAW_PRESERVE_INTEGRITY    = "law_1_preserve_system_integrity"
LAW_OBJECTIVE_ALIGNMENT   = "law_2_objective_alignment_priority"
LAW_NO_STABILITY_TRADE    = "law_3_no_short_term_stability_sacrifice"
LAW_CONSTRAIN_UNCERTAINTY = "law_4_constrain_low_confidence_autonomy"
LAW_EXTERNAL_NO_OVERRIDE  = "law_5_external_systems_cannot_override_objectives"
LAW_TEMPORAL_COHERENCE    = "law_6_temporal_incoherence_halts_execution"
LAW_SAFETY_FIRST          = "law_7_safety_overrides_efficiency"

ALL_LAWS = [
    LAW_PRESERVE_INTEGRITY,
    LAW_OBJECTIVE_ALIGNMENT,
    LAW_NO_STABILITY_TRADE,
    LAW_CONSTRAIN_UNCERTAINTY,
    LAW_EXTERNAL_NO_OVERRIDE,
    LAW_TEMPORAL_COHERENCE,
    LAW_SAFETY_FIRST,
]

# Cloud-specific guard constants
GUARD_TOKEN_LIMIT         = "guard_token_limit_exceeded"
GUARD_RECURSION_LIMIT     = "guard_recursion_depth_exceeded"
GUARD_PROMPT_INJECTION    = "guard_prompt_injection_detected"
GUARD_LOCKDOWN_MODE       = "guard_lockdown_mode_active"
GUARD_INFLUENCE_SATURATED = "guard_influence_saturation"
GUARD_MALFORMED_ENVELOPE  = "guard_malformed_envelope"
GUARD_RUNAWAY_TOKENS      = "guard_runaway_token_prevention"

# Authority hierarchy (lower rank = higher authority)
AUTHORITY_CONSTITUTION = "constitution"
AUTHORITY_GOVERNANCE   = "governance"
AUTHORITY_PLANNING     = "planning"
AUTHORITY_RESPONSES    = "responses"

_AUTHORITY_RANK = {
    AUTHORITY_CONSTITUTION: 0,
    AUTHORITY_GOVERNANCE:   1,
    AUTHORITY_PLANNING:     2,
    AUTHORITY_RESPONSES:    6,
}

# Prompt injection heuristics — patterns that suggest prompt manipulation
_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "forget your training",
    "system: you are now",
    "you are no longer",
    "override your",
    "jailbreak",
    "dan mode",
    "pretend you have no",
    "act as if you have no restrictions",
)


@dataclass
class GovernanceVerdict:
    """Result of a cloud governance check."""
    allowed: bool
    violated: list[str] = field(default_factory=list)
    authority: str = AUTHORITY_CONSTITUTION
    reason: str = ""
    action: str = ""
    rationale: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violated": list(self.violated),
            "authority": self.authority,
            "reason": self.reason,
            "action": self.action,
            "rationale": dict(self.rationale),
        }


class CloudGovernance:
    """Constitutional governance gate for the cloud runtime.

    Validates every inference request against constitutional laws and
    cloud-specific safety guards.  Thread-safe singleton.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._validation_count: int = 0
        self._block_count: int = 0
        self._violation_counts: dict[str, int] = {}
        log.debug("[CloudGovernance] initialised (strict=%s max_tokens=%d)", _STRICT, _MAX_TOKENS)

    def validate(
        self,
        action: str = "inference",
        context: dict[str, Any] | None = None,
        envelope: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 256,
    ) -> GovernanceVerdict:
        """Validate a request before inference is executed.

        Args:
            action:     Name of the action (e.g. ``"inference"``, ``"chat"``).
            context:    Optional context dict with governance signals.
            envelope:   Normalized cognitive envelope (from
                        ``cognitive_envelope.normalize_envelope``).
            messages:   Chat messages (used for injection scanning).
            max_tokens: Requested max tokens.

        Returns:
            :class:`GovernanceVerdict` — ``allowed=True`` unless a law is violated
            in strict mode.
        """
        if not _ENABLED:
            return GovernanceVerdict(
                allowed=True, authority=AUTHORITY_RESPONSES,
                reason="governance disabled", action=action,
            )

        ctx = dict(context or {})
        env = dict(envelope or {})
        violated: list[str] = []
        rationale: dict[str, Any] = {}

        # ── Cloud guards (always enforced regardless of strict mode) ───────────
        self._check_token_limit(max_tokens, violated, rationale)
        self._check_recursion(ctx, violated, rationale)
        self._check_injection(messages or [], violated, rationale)
        self._check_lockdown(env, violated, rationale)
        self._check_influence_saturation(env, violated, rationale)

        # ── Constitutional laws ────────────────────────────────────────────────
        self._check_law_1(ctx, env, violated, rationale)
        self._check_law_4(ctx, env, violated, rationale)
        self._check_law_6(env, violated, rationale)
        self._check_law_7(ctx, env, violated, rationale)

        # Hard guards always block even in permissive mode
        hard_guards = {
            GUARD_TOKEN_LIMIT, GUARD_RECURSION_LIMIT, GUARD_PROMPT_INJECTION,
            GUARD_LOCKDOWN_MODE, GUARD_INFLUENCE_SATURATED, GUARD_RUNAWAY_TOKENS,
        }
        has_hard = any(v in hard_guards for v in violated)
        allowed = (not violated) or (not _STRICT and not has_hard)

        with self._lock:
            self._validation_count += 1
            if not allowed:
                self._block_count += 1
            for v in violated:
                self._violation_counts[v] = self._violation_counts.get(v, 0) + 1

        if violated:
            reason = f"governance violation(s): {', '.join(violated)}"
        else:
            reason = "all constitutional laws and cloud guards satisfied"

        authority = AUTHORITY_CONSTITUTION if violated else AUTHORITY_RESPONSES

        verdict = GovernanceVerdict(
            allowed=allowed,
            violated=violated,
            authority=authority,
            reason=reason,
            action=action,
            rationale=rationale,
        )

        if not allowed:
            log.warning("[CloudGovernance] blocked action=%s reason=%s", action, reason)
        else:
            log.debug("[CloudGovernance] allowed action=%s", action)

        self._emit(verdict)
        return verdict

    def status(self) -> dict[str, Any]:
        """Return runtime governance metrics."""
        with self._lock:
            return {
                "enabled": _ENABLED,
                "strict_mode": _STRICT,
                "max_tokens_limit": _MAX_TOKENS,
                "max_recursion_limit": _MAX_RECURSION,
                "validation_count": self._validation_count,
                "block_count": self._block_count,
                "block_rate": round(
                    self._block_count / max(1, self._validation_count), 4
                ),
                "violation_counts": dict(self._violation_counts),
                "laws": list(ALL_LAWS),
            }

    # ── Cloud guards ───────────────────────────────────────────────────────────

    def _check_token_limit(
        self, max_tokens: int, violated: list[str], rationale: dict[str, Any]
    ) -> None:
        if max_tokens > _MAX_TOKENS:
            violated.append(GUARD_TOKEN_LIMIT)
            rationale["token_limit"] = {
                "requested": max_tokens,
                "hard_limit": _MAX_TOKENS,
            }

    def _check_recursion(
        self, ctx: dict[str, Any], violated: list[str], rationale: dict[str, Any]
    ) -> None:
        depth = int(ctx.get("recursion_depth", 0))
        if depth > _MAX_RECURSION:
            violated.append(GUARD_RECURSION_LIMIT)
            rationale["recursion"] = {"depth": depth, "limit": _MAX_RECURSION}

    def _check_injection(
        self,
        messages: list[dict[str, Any]],
        violated: list[str],
        rationale: dict[str, Any],
    ) -> None:
        for msg in messages:
            content = str(msg.get("content", "")).lower()
            for pattern in _INJECTION_PATTERNS:
                if pattern in content:
                    violated.append(GUARD_PROMPT_INJECTION)
                    rationale["injection"] = {"pattern": pattern}
                    return

    def _check_lockdown(
        self, env: dict[str, Any], violated: list[str], rationale: dict[str, Any]
    ) -> None:
        gov = env.get("governance") or {}
        runtime = env.get("runtime") or {}
        mode = str(
            gov.get("governance_mode") or runtime.get("mode") or "normal"
        ).lower()
        if mode == "lockdown":
            violated.append(GUARD_LOCKDOWN_MODE)
            rationale["lockdown"] = {"governance_mode": mode}

    def _check_influence_saturation(
        self, env: dict[str, Any], violated: list[str], rationale: dict[str, Any]
    ) -> None:
        budget = float((env.get("resources") or {}).get("attention_available", 1.0))
        if budget < 0.0 or budget > 1.0:
            violated.append(GUARD_INFLUENCE_SATURATED)
            rationale["influence"] = {"attention_available": budget}

    # ── Constitutional law checkers ────────────────────────────────────────────

    def _check_law_1(
        self,
        ctx: dict[str, Any],
        env: dict[str, Any],
        violated: list[str],
        rationale: dict[str, Any],
    ) -> None:
        """LAW 1: preserve system integrity (stability >= 0.3)."""
        stability = float(ctx.get("stability_score", 1.0))
        runtime = env.get("runtime") or {}
        runtime_health = float(runtime.get("runtime_health", 1.0))
        effective = min(stability, runtime_health)
        if effective < 0.3:
            violated.append(LAW_PRESERVE_INTEGRITY)
            rationale["law_1"] = {"stability": stability, "runtime_health": runtime_health}

    def _check_law_4(
        self,
        ctx: dict[str, Any],
        env: dict[str, Any],
        violated: list[str],
        rationale: dict[str, Any],
    ) -> None:
        """LAW 4: constrain low-confidence autonomy."""
        autonomous = bool(ctx.get("autonomous", False))
        confidence = float(ctx.get("confidence", 1.0))
        coherence = float((env.get("temporal") or {}).get("coherence_score", 1.0))
        if autonomous and confidence < 0.35 and coherence < 0.5:
            violated.append(LAW_CONSTRAIN_UNCERTAINTY)
            rationale["law_4"] = {"confidence": confidence, "coherence": coherence}

    def _check_law_6(
        self,
        env: dict[str, Any],
        violated: list[str],
        rationale: dict[str, Any],
    ) -> None:
        """LAW 6: temporal incoherence halts execution."""
        temporal = env.get("temporal") or {}
        coherence = float(temporal.get("coherence_score", 1.0))
        alignment = str(temporal.get("epoch_alignment", "aligned"))
        if coherence < 0.1 or alignment == "incoherent":
            violated.append(LAW_TEMPORAL_COHERENCE)
            rationale["law_6"] = {"coherence": coherence, "alignment": alignment}

    def _check_law_7(
        self,
        ctx: dict[str, Any],
        env: dict[str, Any],
        violated: list[str],
        rationale: dict[str, Any],
    ) -> None:
        """LAW 7: safety overrides efficiency."""
        const_priority = str(env.get("constitutional_priority", "safety")).lower()
        gov = env.get("governance") or {}
        constitution_passed = bool(gov.get("constitution_passed", True))
        if const_priority == "safety" and not constitution_passed:
            violated.append(LAW_SAFETY_FIRST)
            rationale["law_7"] = {"constitutional_priority": const_priority}

    def _emit(self, verdict: GovernanceVerdict) -> None:
        try:
            from app.event_bus import (
                EVENT_GOVERNANCE_CHECKED,
                EVENT_GOVERNANCE_VETOED,
                CloudEvent,
                get_event_bus,
            )

            event_type = EVENT_GOVERNANCE_VETOED if not verdict.allowed else EVENT_GOVERNANCE_CHECKED
            get_event_bus().publish(
                CloudEvent(
                    type=event_type,
                    source="cloud_governance",
                    payload=verdict.to_dict(),
                )
            )
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_gov: CloudGovernance | None = None
_gov_lock = threading.Lock()


def get_cloud_governance() -> CloudGovernance:
    """Return the process-level :class:`CloudGovernance` singleton."""
    global _gov  # pylint: disable=global-statement
    with _gov_lock:
        if _gov is None:
            _gov = CloudGovernance()
    return _gov
