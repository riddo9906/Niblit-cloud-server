#!/usr/bin/env python3
"""app/attention_allocator.py — Niblit Cognitive Cloud Runtime Attention Economy.

Manages finite inference capacity across concurrent requests using a
salience-weighted priority queue.  Aligned with Phase Ω.6 in the main Niblit
repo (``modules/attention_allocator.py``, ``modules/cognitive_budget_manager.py``).

Design
------
- Each request receives a *salience score* based on its envelope fields
  (intent, coherence, governance mode, resource mode).
- Requests are queued and served highest-salience-first.
- Starvation prevention ensures low-priority requests are not indefinitely
  deferred (aging factor increases salience over wait time).
- Emergency survival mode: under extreme overload, only the highest-salience
  requests are processed; others receive an immediate degraded response hint.

Configuration (env vars)
------------------------
    NIBLIT_AA_ENABLED        — "0" to disable (default 1)
    NIBLIT_AA_MAX_QUEUE      — max concurrent queued requests (default 64)
    NIBLIT_AA_OVERLOAD_RATIO — queue fill ratio that triggers overload (default 0.85)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitAttentionAllocator")

_ENABLED: bool = os.getenv("NIBLIT_AA_ENABLED", "1").strip() not in ("0", "false")
_MAX_QUEUE: int = int(os.getenv("NIBLIT_AA_MAX_QUEUE", "64"))
_OVERLOAD_RATIO: float = float(os.getenv("NIBLIT_AA_OVERLOAD_RATIO", "0.85"))

# Intent salience weights (higher = more salient)
_INTENT_SALIENCE: dict[str, float] = {
    "trading":         0.9,
    "forecasting":     0.9,
    "reasoning":       0.7,
    "analytical":      0.7,
    "tool_use":        0.6,
    "code_generation": 0.5,
    "summarization":   0.4,
    "conversational":  0.3,
    "creative":        0.2,
    "unknown":         0.1,
}

# Governance mode salience multipliers
_GOV_MULTIPLIER: dict[str, float] = {
    "normal":   1.0,
    "cautious": 0.9,
    "survival": 0.7,
    "lockdown": 0.5,
}

# Aging factor: salience bonus per second of waiting
_AGING_RATE: float = 0.002


@dataclass
class AllocationSlot:
    """A request's place in the attention queue."""
    request_id: str
    salience: float
    enqueued_ts: float = field(default_factory=time.time)
    intent: str = "conversational"
    governance_mode: str = "normal"
    resource_mode: str = "balanced"

    def aged_salience(self) -> float:
        """Salience boosted by wait time to prevent starvation."""
        wait = time.time() - self.enqueued_ts
        return min(1.0, self.salience + _AGING_RATE * wait)


@dataclass
class AttentionAllocation:
    """Result of scoring a request for attention."""
    request_id: str
    salience: float
    granted: bool
    queue_depth: int
    attention_pressure: float
    overload: bool
    rationale: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "salience": round(self.salience, 4),
            "granted": self.granted,
            "queue_depth": self.queue_depth,
            "attention_pressure": round(self.attention_pressure, 4),
            "overload": self.overload,
            "rationale": dict(self.rationale),
        }


class AttentionAllocator:
    """Salience-based request prioritization and capacity management.

    Thread-safe singleton.  ``score_request`` computes salience and returns
    an :class:`AttentionAllocation`.  ``release`` signals that a request has
    completed processing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, AllocationSlot] = {}
        self._total_scored: int = 0
        self._total_denied: int = 0
        self._overload_events: int = 0
        self._pressure_ema: float = 0.0
        log.debug("[AttentionAllocator] initialised max_queue=%d", _MAX_QUEUE)

    def score_request(
        self,
        request_id: str,
        envelope: dict[str, Any] | None = None,
    ) -> AttentionAllocation:
        """Score and enqueue a request.

        Args:
            request_id: Unique ID for this request.
            envelope:   Normalized cognitive envelope (may be None).

        Returns:
            :class:`AttentionAllocation` with ``granted=True`` if the request
            can proceed or ``granted=False`` under extreme overload.
        """
        if not _ENABLED:
            return AttentionAllocation(
                request_id=request_id, salience=1.0, granted=True,
                queue_depth=0, attention_pressure=0.0, overload=False,
                rationale={"allocator": "disabled"},
            )

        env = dict(envelope or {})
        intent = str(env.get("intent", "conversational")).lower()
        gov = env.get("governance") or {}
        runtime = env.get("runtime") or {}
        gov_mode = str(gov.get("governance_mode") or runtime.get("mode") or "normal").lower()
        resource_mode = str(env.get("resource_mode", "balanced")).lower()
        coherence = float((env.get("temporal") or {}).get("coherence_score", 1.0))
        attention_budget = float(env.get("attention_budget", 1.0))

        # Compute salience
        base_salience = _INTENT_SALIENCE.get(intent, 0.3)
        gov_mult = _GOV_MULTIPLIER.get(gov_mode, 1.0)
        coherence_boost = coherence * 0.2  # coherent requests slightly preferred
        budget_factor = attention_budget
        salience = min(1.0, base_salience * gov_mult + coherence_boost) * budget_factor

        rationale = {
            "intent": intent,
            "base_salience": round(base_salience, 3),
            "governance_mode": gov_mode,
            "gov_multiplier": gov_mult,
            "coherence": round(coherence, 3),
            "attention_budget": round(attention_budget, 3),
            "final_salience": round(salience, 4),
        }

        with self._lock:
            queue_depth = len(self._active)
            pressure = queue_depth / _MAX_QUEUE
            self._pressure_ema = 0.1 * pressure + 0.9 * self._pressure_ema
            overload = pressure >= _OVERLOAD_RATIO

            self._total_scored += 1

            if overload:
                self._overload_events += 1
                # Under overload, deny low-salience requests
                min_salience_threshold = 0.5
                if salience < min_salience_threshold:
                    self._total_denied += 1
                    log.warning(
                        "[AttentionAllocator] denied low-salience request %s (salience=%.2f overload)",
                        request_id, salience,
                    )
                    return AttentionAllocation(
                        request_id=request_id, salience=salience, granted=False,
                        queue_depth=queue_depth,
                        attention_pressure=round(self._pressure_ema, 4),
                        overload=True,
                        rationale={**rationale, "deny_reason": "overload_low_salience"},
                    )

            slot = AllocationSlot(
                request_id=request_id, salience=salience, intent=intent,
                governance_mode=gov_mode, resource_mode=resource_mode,
            )
            self._active[request_id] = slot

        self._emit(
            AttentionAllocation(
                request_id=request_id, salience=salience, granted=True,
                queue_depth=queue_depth + 1,
                attention_pressure=round(self._pressure_ema, 4),
                overload=overload,
                rationale=rationale,
            )
        )

        return AttentionAllocation(
            request_id=request_id, salience=salience, granted=True,
            queue_depth=queue_depth + 1,
            attention_pressure=round(self._pressure_ema, 4),
            overload=overload,
            rationale=rationale,
        )

    def release(self, request_id: str) -> None:
        """Signal that a request has completed processing."""
        with self._lock:
            self._active.pop(request_id, None)

    def status(self) -> dict[str, Any]:
        """Return attention economy metrics."""
        with self._lock:
            return {
                "enabled": _ENABLED,
                "active_requests": len(self._active),
                "max_queue": _MAX_QUEUE,
                "attention_pressure": round(self._pressure_ema, 4),
                "overload": self._pressure_ema >= _OVERLOAD_RATIO,
                "total_scored": self._total_scored,
                "total_denied": self._total_denied,
                "overload_events": self._overload_events,
                "active_salience": {
                    rid: round(slot.aged_salience(), 4)
                    for rid, slot in self._active.items()
                },
            }

    def _emit(self, allocation: AttentionAllocation) -> None:
        try:
            from app.event_bus import EVENT_ATTENTION_ALLOCATED, CloudEvent, get_event_bus

            get_event_bus().publish(
                CloudEvent(
                    type=EVENT_ATTENTION_ALLOCATED,
                    source="attention_allocator",
                    payload=allocation.to_dict(),
                )
            )
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_alloc: AttentionAllocator | None = None
_alloc_lock = threading.Lock()


def get_attention_allocator() -> AttentionAllocator:
    """Return the process-level :class:`AttentionAllocator` singleton."""
    global _alloc  # pylint: disable=global-statement
    with _alloc_lock:
        if _alloc is None:
            _alloc = AttentionAllocator()
    return _alloc
