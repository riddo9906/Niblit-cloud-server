#!/usr/bin/env python3
"""app/event_bus.py — Niblit Cognitive Cloud Runtime Event Bus.

Lightweight pub/sub event bus for the cloud runtime.  Naming conventions and
event type constants are intentionally aligned with ``modules/event_bus.py``
in the main Niblit repo (Phase Ω through Ω.7) so that event streams can be
compared, replayed, or forwarded across runtime boundaries.

Public API
----------
``CloudEvent``
    Dataclass carrying ``type``, ``source``, ``payload``, and ``timestamp``.

``CloudEventBus``
    Thread-safe pub/sub hub.  ``subscribe`` / ``publish`` / ``last_event`` /
    ``stats``.

``get_event_bus() → CloudEventBus``
    Process-level singleton.

Well-known cloud event type constants
--------------------------------------
Aligned with Niblit Phase Ω through Ω.7 naming conventions.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitCloudEventBus")

# ── Cloud-local events (mirror Niblit naming where applicable) ─────────────────

# Orchestration
EVENT_MODEL_SELECTED           = "orchestration.model_selected"
EVENT_MODEL_FALLBACK           = "orchestration.model_fallback"
EVENT_MODEL_FAILURE            = "orchestration.model_failure"
EVENT_ROUTING_DECISION         = "orchestration.routing_decision"

# Governance
EVENT_GOVERNANCE_CHECKED       = "governance.checked"
EVENT_GOVERNANCE_VETOED        = "governance.vetoed"
EVENT_CONSTITUTION_CHECKED     = "constitution.checked"

# Cognitive envelope
EVENT_ENVELOPE_RECEIVED        = "envelope.received"
EVENT_ENVELOPE_INVALID         = "envelope.invalid"
EVENT_ENVELOPE_NORMALIZED      = "envelope.normalized"

# Coherence / temporal (aligned with Niblit Ω.5)
EVENT_COHERENCE_EVALUATED      = "coherence.evaluated"
EVENT_TEMPORAL_CAUSAL_UPDATED  = "temporal.causal.updated"
EVENT_EPOCH_SYNCED             = "epoch.synced"
EVENT_COHERENCE_DRIFT          = "coherence.drift"

# Reflection (aligned with Niblit Ω)
EVENT_REFLECTION_COMPLETE      = "reflection.complete"
EVENT_QUALITY_RECORDED         = "reflection.quality_recorded"
EVENT_HALLUCINATION_FLAGGED    = "reflection.hallucination_flagged"

# Attention economy (aligned with Niblit Ω.6)
EVENT_ATTENTION_ALLOCATED      = "attention.allocated"
EVENT_SALIENCE_SCORED          = "salience.scored"
EVENT_OVERLOAD_DETECTED        = "attention.overload_detected"
EVENT_STARVATION_PREVENTED     = "attention.starvation_prevented"

# Trading cognition (aligned with Niblit Ω.7)
EVENT_EXECUTION_ENVELOPE_PUBLISHED = "execution_envelope.published"
EVENT_TRADE_REFLECTION_INGESTED    = "trade_reflection.ingested"
EVENT_MARKET_EPISODE_INGESTED      = "market_episode.ingested"
EVENT_RUNTIME_MODE_CHANGED         = "runtime_mode.changed"
EVENT_FORECAST_CONSENSUS_UPDATED   = "trading.forecast_consensus_updated"
EVENT_REGIME_CHANGED               = "trading.regime_changed"

# Shared architecture event contract aliases
EVENT_MARKET_DATA_RECEIVED = "market_data.received"
EVENT_INDICATOR_UPDATED = "indicator.updated"
EVENT_SIGNAL_GENERATED = "signal.generated"
EVENT_AI_INFERENCE_REQUESTED = "ai.inference.requested"
EVENT_AI_INFERENCE_COMPLETED = "ai.inference.completed"
EVENT_CONTEXT_UPDATED = "context.updated"
EVENT_LEARNING_EVENT = "learning.event"
EVENT_MEMORY_STORED = "memory.stored"
EVENT_DECISION_REQUESTED = "decision.requested"
EVENT_DECISION_APPROVED = "decision.approved"
EVENT_DECISION_REJECTED = "decision.rejected"
EVENT_RISK_EVALUATION_COMPLETED = "risk.evaluation.completed"
EVENT_TRADE_PROPOSED = "trade.proposed"
EVENT_TRADE_APPROVED = "trade.approved"
EVENT_TRADE_EXECUTED = "trade.executed"
EVENT_TRADE_REJECTED = "trade.rejected"
EVENT_ORDER_FILLED = "order.filled"
EVENT_POSITION_CHANGED = "position.changed"
EVENT_PERFORMANCE_UPDATED = "performance.updated"
EVENT_FEEDBACK_RECEIVED = "feedback.received"
EVENT_STRATEGY_UPDATED = "strategy.updated"
EVENT_ERROR_OCCURRED = "error.occurred"
EVENT_HEARTBEAT = "heartbeat"

# Runtime health / cluster
EVENT_RUNTIME_HEALTH_UPDATED   = "runtime.health_updated"
EVENT_NODE_IDENTITY_SET        = "cluster.node_identity_set"

# Request lifecycle
EVENT_REQUEST_RECEIVED         = "request.received"
EVENT_REQUEST_COMPLETE         = "request.complete"
EVENT_REQUEST_FAILED           = "request.failed"
EVENT_CONTEXT_COMPRESSED       = "request.context_compressed"


@dataclass
class CloudEvent:
    """A single pub/sub event for the cloud runtime.

    Attributes
    ----------
    type:      Dot-namespaced event type string.
    source:    Name of the module that emitted this event.
    payload:   Arbitrary dict of event-specific data.
    timestamp: UNIX timestamp of emission (auto-set on creation).
    """

    type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


_Handler = Callable[[CloudEvent], None]


class CloudEventBus:
    """Thread-safe in-process pub/sub event bus.

    Multiple handlers may be registered per event type.  Handler errors are
    isolated and logged at DEBUG level so one failing handler never blocks
    subsequent handlers.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[_Handler]] = {}
        self._last: dict[str, CloudEvent] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        log.debug("[CloudEventBus] initialised")

    def subscribe(self, event_type: str, handler: _Handler) -> None:
        """Register *handler* for *event_type*.  Duplicate registrations are no-ops."""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: _Handler) -> None:
        """Remove *handler* from *event_type* (silent if absent)."""
        with self._lock:
            if event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(handler)
                except ValueError:
                    pass

    def publish(self, event: CloudEvent) -> None:
        """Dispatch *event* to all subscribed handlers."""
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))
            self._last[event.type] = event
            self._counts[event.type] = self._counts.get(event.type, 0) + 1

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                log.debug(
                    "[CloudEventBus] handler error for %s in %s: %s",
                    event.type,
                    getattr(handler, "__qualname__", repr(handler)),
                    exc,
                )

    def last_event(self, event_type: str) -> CloudEvent | None:
        """Return the most recently published event of *event_type*, or None."""
        with self._lock:
            return self._last.get(event_type)

    def stats(self) -> dict[str, Any]:
        """Return a copy of publish-count statistics keyed by event type."""
        with self._lock:
            return dict(self._counts)


# ── Singleton ──────────────────────────────────────────────────────────────────

_bus: CloudEventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> CloudEventBus:
    """Return the process-level :class:`CloudEventBus` singleton."""
    global _bus  # pylint: disable=global-statement
    with _bus_lock:
        if _bus is None:
            _bus = CloudEventBus()
    return _bus
