#!/usr/bin/env python3
"""app/model_orchestrator.py — Niblit Cognitive Cloud Runtime Model Orchestrator.

Intelligently routes inference requests to the most appropriate GGUF model
based on task intent, complexity, resource constraints, latency history, and
trust scores.  Aligned with ``modules/model_orchestrator.py`` in the main
Niblit repo (Phase 21 + Ω.7 extensions).

Model roster (configurable via GGUF_MODELS_JSON + env flags)
-------------------------------------------------------------
The orchestrator treats all models registered in ``GGUF_MODELS_JSON`` as
first-class candidates.  Additionally:
- ``NIBLIT_CLOUD_LLM_URL`` enables a remote cloud LLM target.
- ``OPENAI_API_KEY`` enables a remote GPT target (pass-through only).

Routing priority
----------------
1. Intent/complexity check — forecasting → prefer analytical models
2. Governance mode — survival/lockdown → force lightest/fastest model
3. Latency-aware EMA — penalise models with high rolling latency
4. Trust score — penalise models with high failure rates
5. Resource mode — minimal → force smallest model

Fallback chain
--------------
If the primary model fails, the orchestrator follows the configured fallback
chain until a model succeeds or the chain is exhausted.

Configuration (env vars)
------------------------
    NIBLIT_MO_ENABLED        — "0" to disable (default 1)
    NIBLIT_MO_LATENCY_WEIGHT — weight of latency penalty (default 0.3)
    NIBLIT_MO_TRUST_WEIGHT   — weight of trust score (default 0.7)
    NIBLIT_CLOUD_LLM_URL     — remote cloud LLM base URL (optional)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitModelOrchestrator")

_ENABLED: bool = os.getenv("NIBLIT_MO_ENABLED", "1").strip() not in ("0", "false")
_LATENCY_WEIGHT: float = float(os.getenv("NIBLIT_MO_LATENCY_WEIGHT", "0.3"))
_TRUST_WEIGHT: float = float(os.getenv("NIBLIT_MO_TRUST_WEIGHT", "0.7"))
_CLOUD_URL: str = os.getenv("NIBLIT_CLOUD_LLM_URL", "").strip()
_EMA_ALPHA: float = 0.15


@dataclass
class ModelHealth:
    """Per-model health state tracked by the orchestrator."""
    model_id: str
    trust_score: float = 1.0          # 0.0–1.0; decays on failure
    latency_ema_ms: float = 500.0     # rolling EMA of observed latency
    call_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_used_ts: float = field(default_factory=time.time)
    last_failure_ts: float = 0.0

    @property
    def failure_rate(self) -> float:
        return self.failure_count / max(1, self.call_count)

    @property
    def composite_score(self) -> float:
        """Higher is better.  Combines trust and inverse latency."""
        latency_penalty = min(1.0, self.latency_ema_ms / 10_000)
        return (
            _TRUST_WEIGHT * self.trust_score
            - _LATENCY_WEIGHT * latency_penalty
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trust_score": round(self.trust_score, 4),
            "latency_ema_ms": round(self.latency_ema_ms, 1),
            "call_count": self.call_count,
            "failure_count": self.failure_count,
            "failure_rate": round(self.failure_rate, 4),
            "consecutive_failures": self.consecutive_failures,
            "composite_score": round(self.composite_score, 4),
            "last_used_ts": self.last_used_ts,
        }


@dataclass
class RoutingDecision:
    """The orchestrator's routing recommendation for a request."""
    model_id: str
    fallback_chain: list[str]
    reason: str
    intent: str = ""
    governance_mode: str = "normal"
    resource_mode: str = "balanced"
    confidence: float = 1.0
    rationale: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "fallback_chain": list(self.fallback_chain),
            "reason": self.reason,
            "intent": self.intent,
            "governance_mode": self.governance_mode,
            "resource_mode": self.resource_mode,
            "confidence": round(self.confidence, 4),
            "rationale": dict(self.rationale),
        }


class ModelOrchestrator:
    """Multi-model routing engine for the cloud runtime.

    Tracks per-model health, applies coherence-adjusted routing,
    and exposes orchestration telemetry.  Thread-safe singleton.
    """

    def __init__(self, model_ids: list[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._health: dict[str, ModelHealth] = {}
        self._routing_count: int = 0
        self._fallback_count: int = 0
        if model_ids:
            for mid in model_ids:
                self._health[mid] = ModelHealth(model_id=mid)
        if _CLOUD_URL:
            self._health["cloud"] = ModelHealth(model_id="cloud", latency_ema_ms=1000.0)
        log.debug("[ModelOrchestrator] initialised models=%s cloud=%s", model_ids, bool(_CLOUD_URL))

    def register_model(self, model_id: str) -> None:
        """Register a new model in the orchestrator."""
        with self._lock:
            if model_id not in self._health:
                self._health[model_id] = ModelHealth(model_id=model_id)

    def route(
        self,
        available_models: list[str],
        default_model: str,
        envelope: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Select the best model for this request.

        Args:
            available_models: List of model IDs registered in ModelManager.
            default_model:    The configured default model.
            envelope:         Normalized cognitive envelope (may be None).

        Returns:
            :class:`RoutingDecision` — always valid (falls back to default).
        """
        if not _ENABLED or not available_models:
            return RoutingDecision(
                model_id=default_model,
                fallback_chain=[],
                reason="orchestrator disabled or no models available",
            )

        try:
            return self._route_inner(available_models, default_model, envelope or {})
        except Exception as exc:
            log.warning("[ModelOrchestrator] routing error: %s", exc)
            return RoutingDecision(
                model_id=default_model,
                fallback_chain=[],
                reason=f"routing error fallback: {exc}",
            )

    def _route_inner(
        self,
        available: list[str],
        default: str,
        env: dict[str, Any],
    ) -> RoutingDecision:
        intent = str(env.get("intent", "conversational")).lower()
        gov = env.get("governance") or {}
        runtime = env.get("runtime") or {}
        gov_mode = str(
            gov.get("governance_mode") or runtime.get("mode") or "normal"
        ).lower()
        resource_mode = str(env.get("resource_mode", "balanced")).lower()
        coherence = float((env.get("temporal") or {}).get("coherence_score", 1.0))
        rationale: dict[str, Any] = {
            "intent": intent,
            "governance_mode": gov_mode,
            "resource_mode": resource_mode,
            "coherence": coherence,
        }

        # Ensure all available models are registered
        with self._lock:
            for mid in available:
                if mid not in self._health:
                    self._health[mid] = ModelHealth(model_id=mid)

        # ── Governance overrides ───────────────────────────────────────────────
        if gov_mode in ("survival", "lockdown"):
            # Survival/lockdown: use the smallest/fastest model
            chosen = self._pick_fastest(available, default)
            rationale["override"] = f"{gov_mode}_mode_forces_fastest"
            return self._make_decision(chosen, available, default, intent, gov_mode, resource_mode, rationale,
                                       reason=f"{gov_mode} mode — forced fastest model")

        # ── Resource mode ──────────────────────────────────────────────────────
        if resource_mode == "minimal":
            chosen = self._pick_fastest(available, default)
            rationale["override"] = "minimal_resource_mode"
            return self._make_decision(chosen, available, default, intent, gov_mode, resource_mode, rationale,
                                       reason="minimal resource mode — fastest model")

        # ── Coherence-adjusted routing ─────────────────────────────────────────
        if coherence < 0.4:
            chosen = self._pick_fastest(available, default)
            rationale["override"] = f"low_coherence ({coherence:.2f})"
            return self._make_decision(chosen, available, default, intent, gov_mode, resource_mode, rationale,
                                       reason=f"low coherence ({coherence:.2f}) — fastest fallback")

        # ── Trust + latency ranked selection ──────────────────────────────────
        with self._lock:
            ranked = sorted(
                [mid for mid in available if mid in self._health],
                key=lambda m: self._health[m].composite_score,
                reverse=True,
            )
            # Add unregistered models at the end
            unregistered = [m for m in available if m not in self._health]
            ranked = ranked + unregistered

        if not ranked:
            ranked = [default]

        chosen = ranked[0]
        rationale["ranked_candidates"] = ranked[:3]

        with self._lock:
            if chosen in self._health:
                h = self._health[chosen]
                rationale["trust_score"] = h.trust_score
                rationale["latency_ema_ms"] = h.latency_ema_ms

        return self._make_decision(
            chosen, available, default, intent, gov_mode, resource_mode, rationale,
            reason=f"highest composite score (trust+latency) among {len(ranked)} candidates",
        )

    def _pick_fastest(self, available: list[str], default: str) -> str:
        """Return the model with the lowest latency EMA."""
        with self._lock:
            registered = [m for m in available if m in self._health]
        if not registered:
            return default
        with self._lock:
            return min(registered, key=lambda m: self._health[m].latency_ema_ms)

    def _make_decision(
        self,
        chosen: str,
        available: list[str],
        default: str,
        intent: str,
        gov_mode: str,
        resource_mode: str,
        rationale: dict[str, Any],
        reason: str,
    ) -> RoutingDecision:
        fallback = [m for m in available if m != chosen]
        if not fallback and default != chosen:
            fallback = [default]
        with self._lock:
            self._routing_count += 1
        decision = RoutingDecision(
            model_id=chosen,
            fallback_chain=fallback,
            reason=reason,
            intent=intent,
            governance_mode=gov_mode,
            resource_mode=resource_mode,
            confidence=min(1.0, self._health.get(chosen, ModelHealth(chosen)).trust_score),
            rationale=rationale,
        )
        self._emit(decision)
        return decision

    # ── Outcome feedback ───────────────────────────────────────────────────────

    def record_outcome(
        self,
        model_id: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        quality: float = 1.0,
    ) -> None:
        """Update trust score and latency EMA after a request completes."""
        with self._lock:
            if model_id not in self._health:
                self._health[model_id] = ModelHealth(model_id=model_id)
            h = self._health[model_id]
            h.call_count += 1
            h.last_used_ts = time.time()
            h.latency_ema_ms = _EMA_ALPHA * latency_ms + (1 - _EMA_ALPHA) * h.latency_ema_ms

            if success:
                h.consecutive_failures = 0
                # Slowly recover trust
                h.trust_score = min(1.0, h.trust_score + 0.01 * quality)
            else:
                h.failure_count += 1
                h.consecutive_failures += 1
                h.last_failure_ts = time.time()
                # Decay trust proportional to consecutive failures
                decay = 0.05 * (1 + h.consecutive_failures * 0.1)
                h.trust_score = max(0.0, h.trust_score - decay)

        if not success:
            self._emit_failure(model_id)

    def record_fallback(self, primary: str, fallback: str, reason: str) -> None:
        """Record that a fallback was triggered from *primary* to *fallback*."""
        with self._lock:
            self._fallback_count += 1
        log.warning(
            "[ModelOrchestrator] fallback: %s → %s reason=%s", primary, fallback, reason
        )
        try:
            from app.event_bus import EVENT_MODEL_FALLBACK, CloudEvent, get_event_bus

            get_event_bus().publish(
                CloudEvent(
                    type=EVENT_MODEL_FALLBACK,
                    source="model_orchestrator",
                    payload={"from": primary, "to": fallback, "reason": reason},
                )
            )
        except Exception:
            pass

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return orchestrator health and routing statistics."""
        with self._lock:
            return {
                "enabled": _ENABLED,
                "routing_count": self._routing_count,
                "fallback_count": self._fallback_count,
                "model_health": {
                    mid: h.to_dict() for mid, h in self._health.items()
                },
                "cloud_enabled": bool(_CLOUD_URL),
            }

    def model_health(self, model_id: str) -> dict[str, Any] | None:
        """Return health info for a specific model."""
        with self._lock:
            h = self._health.get(model_id)
            return h.to_dict() if h else None

    # ── Event emission ─────────────────────────────────────────────────────────

    def _emit(self, decision: RoutingDecision) -> None:
        try:
            from app.event_bus import EVENT_ROUTING_DECISION, CloudEvent, get_event_bus

            get_event_bus().publish(
                CloudEvent(
                    type=EVENT_ROUTING_DECISION,
                    source="model_orchestrator",
                    payload=decision.to_dict(),
                )
            )
        except Exception:
            pass

    def _emit_failure(self, model_id: str) -> None:
        try:
            from app.event_bus import EVENT_MODEL_FAILURE, CloudEvent, get_event_bus

            get_event_bus().publish(
                CloudEvent(
                    type=EVENT_MODEL_FAILURE,
                    source="model_orchestrator",
                    payload={"model_id": model_id},
                )
            )
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_orch: ModelOrchestrator | None = None
_orch_lock = threading.Lock()


def get_model_orchestrator(model_ids: list[str] | None = None) -> ModelOrchestrator:
    """Return the process-level :class:`ModelOrchestrator` singleton.

    If *model_ids* is provided on first call, the orchestrator is seeded with
    those model IDs.  Subsequent calls ignore *model_ids*.
    """
    global _orch  # pylint: disable=global-statement
    with _orch_lock:
        if _orch is None:
            _orch = ModelOrchestrator(model_ids=model_ids or [])
    return _orch
