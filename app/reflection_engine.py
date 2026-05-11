#!/usr/bin/env python3
"""app/reflection_engine.py — Niblit Cognitive Cloud Runtime Reflection Layer.

Tracks per-request quality, coherence, latency, and governance outcomes to
produce rolling self-critique reports.  Aligned with the Reflection Engine in
the main Niblit repo (Phase Ω, ``modules/reflection_engine.py``).

Telemetry collected
-------------------
- request quality (0.0–1.0, derived from latency + coherence + model confidence)
- coherence drift (EMA of per-request coherence scores)
- governance veto rate
- model disagreement (when multiple models used)
- latency trends (rolling EMA)
- hallucination likelihood (proxy: low coherence + high token count)
- overload pressure (queue depth indicator)

Persisted output
----------------
Reflection snapshots are appended to a JSONL file (configurable via
``NIBLIT_CLOUD_REFLECTION_FILE``).  Rolling in-memory summaries are also
kept for the ``/metrics/cognitive`` endpoint.

Configuration (env vars)
------------------------
    NIBLIT_RE_ENABLED             — "0" to disable (default 1)
    NIBLIT_RE_CADENCE             — reflections per N requests (default 50)
    NIBLIT_CLOUD_REFLECTION_FILE  — JSONL output path (default /tmp/niblit_cloud_reflection.jsonl)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque

log = logging.getLogger("NiblitCloudReflection")

_ENABLED: bool = os.getenv("NIBLIT_RE_ENABLED", "1").strip() not in ("0", "false")
_CADENCE: int = int(os.getenv("NIBLIT_RE_CADENCE", "50"))
_REFLECTION_FILE: str = os.getenv(
    "NIBLIT_CLOUD_REFLECTION_FILE",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "niblit_cloud_reflection.jsonl"),
)
_EMA = 0.1


@dataclass
class TurnRecord:
    """A single inference request's telemetry snapshot."""
    quality: float
    latency_ms: float
    coherence: float
    model_id: str
    intent: str
    governance_vetoed: bool
    token_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReflectionSnapshot:
    """Structured self-critique from the cloud reflection engine."""
    summary: str
    overall_health: float
    quality_ema: float
    latency_ema_ms: float
    coherence_ema: float
    governance_veto_rate: float
    hallucination_risk: float
    failures_detected: list[str]
    strategy_drifts: list[str]
    adaptation_proposals: list[str]
    governance_notes: list[str]
    model_stats: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "overall_health": round(self.overall_health, 4),
            "quality_ema": round(self.quality_ema, 4),
            "latency_ema_ms": round(self.latency_ema_ms, 2),
            "coherence_ema": round(self.coherence_ema, 4),
            "governance_veto_rate": round(self.governance_veto_rate, 4),
            "hallucination_risk": round(self.hallucination_risk, 4),
            "failures_detected": self.failures_detected,
            "strategy_drifts": self.strategy_drifts,
            "adaptation_proposals": self.adaptation_proposals,
            "governance_notes": self.governance_notes,
            "model_stats": self.model_stats,
            "timestamp": self.timestamp,
        }


class ReflectionEngine:
    """Continuous telemetry and self-reflection layer for the cloud runtime.

    Records per-request outcomes and periodically generates
    :class:`ReflectionSnapshot` instances that inform orchestration and
    governance.  Thread-safe singleton.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: Deque[TurnRecord] = deque(maxlen=500)
        self._turn_count: int = 0
        self._reflect_count: int = 0
        self._quality_ema: float = 0.9
        self._latency_ema: float = 500.0  # ms
        self._coherence_ema: float = 1.0
        self._veto_count: int = 0
        self._model_calls: dict[str, int] = defaultdict(int)
        self._model_failures: dict[str, int] = defaultdict(int)
        self._model_latency: dict[str, float] = {}
        self._recent_snapshots: Deque[ReflectionSnapshot] = deque(maxlen=10)
        log.debug("[ReflectionEngine] initialised (cadence=%d)", _CADENCE)

    # ── Observation intake ─────────────────────────────────────────────────────

    def record_turn(
        self,
        *,
        quality: float = 1.0,
        latency_ms: float = 0.0,
        coherence: float = 1.0,
        model_id: str = "",
        intent: str = "",
        governance_vetoed: bool = False,
        token_count: int = 0,
    ) -> None:
        """Record the outcome of one inference request."""
        if not _ENABLED:
            return
        quality = max(0.0, min(1.0, float(quality)))
        coherence = max(0.0, min(1.0, float(coherence)))
        with self._lock:
            self._turn_count += 1
            self._quality_ema = _EMA * quality + (1 - _EMA) * self._quality_ema
            self._latency_ema = _EMA * latency_ms + (1 - _EMA) * self._latency_ema
            self._coherence_ema = _EMA * coherence + (1 - _EMA) * self._coherence_ema
            if governance_vetoed:
                self._veto_count += 1
            if model_id:
                self._model_calls[model_id] += 1
                if quality < 0.3:
                    self._model_failures[model_id] += 1
                prev = self._model_latency.get(model_id, latency_ms)
                self._model_latency[model_id] = _EMA * latency_ms + (1 - _EMA) * prev
            self._turns.append(TurnRecord(
                quality=quality, latency_ms=latency_ms, coherence=coherence,
                model_id=model_id, intent=intent, governance_vetoed=governance_vetoed,
                token_count=token_count,
            ))

        if self.should_reflect():
            self.reflect()

    def should_reflect(self) -> bool:
        """Return True when it is time for an automatic reflection cycle."""
        if not _ENABLED:
            return False
        with self._lock:
            return self._turn_count > 0 and (self._turn_count % _CADENCE == 0)

    # ── Reflection ─────────────────────────────────────────────────────────────

    def reflect(self) -> ReflectionSnapshot:
        """Generate a full :class:`ReflectionSnapshot`."""
        if not _ENABLED:
            return ReflectionSnapshot(
                summary="Reflection engine disabled.",
                overall_health=1.0, quality_ema=1.0, latency_ema_ms=0.0,
                coherence_ema=1.0, governance_veto_rate=0.0, hallucination_risk=0.0,
                failures_detected=[], strategy_drifts=[], adaptation_proposals=[],
                governance_notes=[], model_stats={},
            )

        with self._lock:
            quality_ema = self._quality_ema
            latency_ema = self._latency_ema
            coherence_ema = self._coherence_ema
            turn_count = self._turn_count
            veto_count = self._veto_count
            model_calls = dict(self._model_calls)
            model_failures = dict(self._model_failures)
            model_latency = dict(self._model_latency)
            turns = list(self._turns)
            self._reflect_count += 1
            reflect_num = self._reflect_count

        failures: list[str] = []
        drifts: list[str] = []
        proposals: list[str] = []
        governance_notes: list[str] = []

        # Quality trend
        if quality_ema < 0.5:
            failures.append(f"low_quality_ema ({quality_ema:.2f})")
            proposals.append("review model selection or reduce context noise")

        # Latency trend
        if latency_ema > 5000:
            failures.append(f"high_latency_ema ({latency_ema:.0f}ms)")
            proposals.append("prefer faster/smaller models under load")
            drifts.append("latency_degradation")

        # Coherence drift
        if coherence_ema < 0.6:
            drifts.append(f"coherence_drift (ema={coherence_ema:.2f})")
            proposals.append("increase coherence scoring or switch to deterministic mode")

        # Governance veto rate
        veto_rate = veto_count / max(1, turn_count)
        if veto_rate > 0.2:
            failures.append(f"high_governance_veto_rate ({veto_rate:.0%})")
            governance_notes.append("veto_rate_above_threshold — review governance thresholds")

        # Model failure rates
        for mid, fail_count in model_failures.items():
            calls = model_calls.get(mid, 1)
            fail_rate = fail_count / calls
            if fail_rate > 0.3:
                failures.append(f"model_high_failure:{mid} ({fail_rate:.0%})")
                proposals.append(f"reduce routing weight for {mid}")
                governance_notes.append(f"flag model {mid} for trust review")

        # Hallucination risk proxy: recent low coherence + high token counts
        recent = list(turns)[-20:] if turns else []
        if recent:
            avg_tokens = sum(t.token_count for t in recent) / len(recent)
            low_coh_frac = sum(1 for t in recent if t.coherence < 0.5) / len(recent)
            hallucination_risk = min(1.0, low_coh_frac * (avg_tokens / 1024))
        else:
            hallucination_risk = 0.0

        if hallucination_risk > 0.4:
            failures.append(f"hallucination_risk_elevated ({hallucination_risk:.2f})")
            proposals.append("reduce max_tokens or add coherence checks")

        health = max(0.0, min(1.0, quality_ema - len(failures) * 0.07 - len(drifts) * 0.04))

        model_stats = {
            mid: {
                "calls": model_calls.get(mid, 0),
                "failures": model_failures.get(mid, 0),
                "latency_ema_ms": round(model_latency.get(mid, 0.0), 1),
            }
            for mid in set(list(model_calls.keys()) + list(model_failures.keys()))
        }

        summary = (
            f"Cloud reflection #{reflect_num}: "
            f"health={health:.2f}, quality_ema={quality_ema:.2f}, "
            f"latency_ema={latency_ema:.0f}ms, coherence_ema={coherence_ema:.2f}, "
            f"veto_rate={veto_rate:.0%}, failures={len(failures)}"
        )

        snapshot = ReflectionSnapshot(
            summary=summary,
            overall_health=health,
            quality_ema=quality_ema,
            latency_ema_ms=latency_ema,
            coherence_ema=coherence_ema,
            governance_veto_rate=veto_rate,
            hallucination_risk=hallucination_risk,
            failures_detected=failures,
            strategy_drifts=drifts,
            adaptation_proposals=proposals,
            governance_notes=governance_notes,
            model_stats=model_stats,
        )

        with self._lock:
            self._recent_snapshots.append(snapshot)

        self._persist(snapshot)
        self._emit(snapshot)
        log.info("[ReflectionEngine] %s", summary)
        return snapshot

    def last_snapshot(self) -> ReflectionSnapshot | None:
        """Return the most recent reflection snapshot, or None."""
        with self._lock:
            return self._recent_snapshots[-1] if self._recent_snapshots else None

    def status(self) -> dict[str, Any]:
        """Return current telemetry status."""
        with self._lock:
            return {
                "enabled": _ENABLED,
                "turn_count": self._turn_count,
                "reflect_count": self._reflect_count,
                "quality_ema": round(self._quality_ema, 4),
                "latency_ema_ms": round(self._latency_ema, 2),
                "coherence_ema": round(self._coherence_ema, 4),
                "veto_count": self._veto_count,
                "veto_rate": round(self._veto_count / max(1, self._turn_count), 4),
                "model_calls": dict(self._model_calls),
                "model_failures": dict(self._model_failures),
                "model_latency_ema": {
                    k: round(v, 1) for k, v in self._model_latency.items()
                },
                "cadence": _CADENCE,
                "reflection_file": _REFLECTION_FILE,
            }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist(self, snapshot: ReflectionSnapshot) -> None:
        """Append snapshot to JSONL file (best-effort)."""
        try:
            with open(_REFLECTION_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(snapshot.to_dict()) + "\n")
        except Exception as exc:
            log.debug("[ReflectionEngine] persist failed: %s", exc)

    def _emit(self, snapshot: ReflectionSnapshot) -> None:
        try:
            from app.event_bus import EVENT_REFLECTION_COMPLETE, CloudEvent, get_event_bus

            get_event_bus().publish(
                CloudEvent(
                    type=EVENT_REFLECTION_COMPLETE,
                    source="reflection_engine",
                    payload={
                        "health": snapshot.overall_health,
                        "failures": len(snapshot.failures_detected),
                        "quality_ema": snapshot.quality_ema,
                    },
                )
            )
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_re: ReflectionEngine | None = None
_re_lock = threading.Lock()


def get_reflection_engine() -> ReflectionEngine:
    """Return the process-level :class:`ReflectionEngine` singleton."""
    global _re  # pylint: disable=global-statement
    with _re_lock:
        if _re is None:
            _re = ReflectionEngine()
    return _re
