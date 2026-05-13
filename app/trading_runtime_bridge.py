#!/usr/bin/env python3
"""app/trading_runtime_bridge.py — Niblit Trading Cognition Integration.

Bridges the cloud runtime with Niblit's trading cognition layer, ingesting
forecast consensus signals, market regime state, and governance telemetry
from the niblit-lean-algos ecosystem (Phase Ω.7, PR#20).

The bridge is purely optional.  If no signal file is configured or
present, all trading telemetry defaults to safe neutral values.

Signal sources (read-only)
--------------------------
- ``NIBLIT_SIGNAL_FILE``      — schema-v2 cognitive execution envelope (JSON)
- ``NIBLIT_REFLECTION_FILE``  — trade reflection JSONL sidecar
- ``NIBLIT_EPISODES_FILE``    — market episode JSONL sidecar

Exposed runtime state
---------------------
- ``forecast_consensus``    — directional forecast + agreement + uncertainty
- ``market_regime``         — current regime string
- ``governance_mode``       — normal / cautious / survival / lockdown
- ``trading_confidence``    — composite confidence score (0.0–1.0)
- ``volatility_scale``      — relative volatility indicator (0.0–1.0)
- ``risk_mode``             — inference scaling hint for volatile periods

Configuration (env vars)
------------------------
    NIBLIT_SIGNAL_FILE          — path to live signal JSON (optional)
    NIBLIT_SIGNAL_MAX_AGE       — max signal age in seconds (default 300)
    NIBLIT_REFLECTION_FILE      — trade reflection JSONL path (optional)
    NIBLIT_EPISODES_FILE        — market episodes JSONL path (optional)
    NIBLIT_TRADING_BRIDGE_ENABLED — "0" to disable (default 1)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitTradingBridge")

_ENABLED: bool = os.getenv("NIBLIT_TRADING_BRIDGE_ENABLED", "1").strip() not in ("0", "false")
_SIGNAL_FILE: str = os.getenv(
    "NIBLIT_SIGNAL_FILE",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "niblit_lean_signal.json"),
)
_SIGNAL_MAX_AGE: int = int(os.getenv("NIBLIT_SIGNAL_MAX_AGE", "300"))
_REFLECTION_FILE: str = os.getenv(
    "NIBLIT_REFLECTION_FILE",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "niblit_trade_reflection.jsonl"),
)
_EPISODES_FILE: str = os.getenv(
    "NIBLIT_EPISODES_FILE",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "niblit_market_episodes.jsonl"),
)


@dataclass
class TradingState:
    """Current trading cognition state snapshot."""
    signal: str = "HOLD"                  # BUY | SELL | HOLD
    confidence: float = 0.5
    market_regime: str = "ranging"
    governance_mode: str = "normal"
    forecast_direction: str = "NEUTRAL"   # UP | DOWN | NEUTRAL
    forecast_agreement: float = 0.5
    forecast_uncertainty: float = 0.5
    volatility_scale: float = 0.3
    risk_mode: str = "balanced"           # conservative | balanced | aggressive
    survival_mode: bool = False
    constitution_passed: bool = True
    schema_version: str = "2.0"
    envelope_age_secs: float = 0.0
    envelope_fresh: bool = False
    reflection_count: int = 0
    episode_count: int = 0
    last_updated_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "market_regime": self.market_regime,
            "governance_mode": self.governance_mode,
            "forecast_direction": self.forecast_direction,
            "forecast_agreement": round(self.forecast_agreement, 4),
            "forecast_uncertainty": round(self.forecast_uncertainty, 4),
            "volatility_scale": round(self.volatility_scale, 4),
            "risk_mode": self.risk_mode,
            "survival_mode": self.survival_mode,
            "constitution_passed": self.constitution_passed,
            "schema_version": self.schema_version,
            "envelope_age_secs": round(self.envelope_age_secs, 1),
            "envelope_fresh": self.envelope_fresh,
            "reflection_count": self.reflection_count,
            "episode_count": self.episode_count,
            "last_updated_ts": self.last_updated_ts,
        }

    @property
    def inference_scale(self) -> float:
        """Return a 0.0–1.0 inference scaling hint based on volatility + risk."""
        if self.governance_mode in ("survival", "lockdown"):
            return 0.2
        if self.survival_mode:
            return 0.3
        if self.governance_mode == "cautious":
            return 0.6
        return max(0.2, 1.0 - self.volatility_scale * 0.5)


class TradingRuntimeBridge:
    """Read-only bridge to the niblit-lean-algos trading cognition layer.

    Periodically refreshes the trading state from disk.  All reads are
    best-effort; failures default to safe neutral values.  Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: TradingState = TradingState()
        self._refresh_count: int = 0
        self._last_refresh_ts: float = 0.0
        self._reflection_cursor: int = 0
        self._episode_cursor: int = 0
        log.debug("[TradingBridge] initialised signal_file=%s", _SIGNAL_FILE)

    def refresh(self) -> TradingState:
        """Read the latest signal from disk and update internal state.

        Returns:
            The refreshed :class:`TradingState`.
        """
        if not _ENABLED:
            return self._state

        envelope = self._read_signal_file()
        reflections = self._read_jsonl_tail(_REFLECTION_FILE, cursor=self._reflection_cursor, limit=50)
        episodes = self._read_jsonl_tail(_EPISODES_FILE, cursor=self._episode_cursor, limit=50)

        with self._lock:
            self._refresh_count += 1
            self._last_refresh_ts = time.time()
            if reflections:
                self._reflection_cursor += len(reflections)
                self._emit_events("reflection", reflections)
            if episodes:
                self._episode_cursor += len(episodes)
                self._emit_events("episode", episodes)

            if envelope:
                self._state = self._envelope_to_state(envelope)
                self._state.reflection_count = self._reflection_cursor
                self._state.episode_count = self._episode_cursor
            else:
                # Keep stale state but mark as not fresh
                self._state.envelope_fresh = False
                self._state.reflection_count = self._reflection_cursor
                self._state.episode_count = self._episode_cursor

        return self._state

    def state(self) -> TradingState:
        """Return the last known trading state (does NOT refresh from disk)."""
        with self._lock:
            return self._state

    def status(self) -> dict[str, Any]:
        """Return bridge status and trading telemetry."""
        with self._lock:
            state_dict = self._state.to_dict()
            return {
                "enabled": _ENABLED,
                "signal_file": _SIGNAL_FILE,
                "reflection_file": _REFLECTION_FILE,
                "episodes_file": _EPISODES_FILE,
                "refresh_count": self._refresh_count,
                "last_refresh_ts": self._last_refresh_ts,
                "current_state": state_dict,
                "inference_scale": self._state.inference_scale,
            }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _read_signal_file(self) -> dict[str, Any] | None:
        try:
            if not os.path.isfile(_SIGNAL_FILE):
                return None
            with open(_SIGNAL_FILE, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        ts = float(payload.get("timestamp", 0))
        age = time.time() - ts if ts else float("inf")
        if age > _SIGNAL_MAX_AGE:
            return None  # stale

        return payload

    def _read_jsonl_tail(
        self, path: str, cursor: int, limit: int
    ) -> list[dict[str, Any]]:
        """Read up to *limit* new lines from *path* starting at *cursor*."""
        records: list[dict[str, Any]] = []
        try:
            if not os.path.isfile(path):
                return records
            with open(path, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i < cursor:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                    if len(records) >= limit:
                        break
        except OSError:
            pass
        return records

    def _envelope_to_state(self, payload: dict[str, Any]) -> TradingState:
        """Convert a raw signal payload into a :class:`TradingState`."""
        ts = float(payload.get("timestamp", 0))
        age = time.time() - ts if ts else 0.0

        gov = payload.get("governance") or {}
        forecast = payload.get("forecast_consensus") or {}
        runtime = payload.get("runtime") or {}
        temporal = payload.get("temporal") or {}

        signal = str(payload.get("signal", "HOLD")).upper()
        if signal not in {"BUY", "SELL", "HOLD"}:
            signal = "HOLD"

        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))
        regime = str(payload.get("market_regime", "ranging"))

        gov_mode = str(
            gov.get("governance_mode") or runtime.get("mode") or "normal"
        ).lower()
        if gov_mode not in ("normal", "cautious", "survival", "lockdown"):
            gov_mode = "normal"

        direction = str(forecast.get("direction", "NEUTRAL")).upper()
        if direction not in ("UP", "DOWN", "NEUTRAL"):
            direction = "NEUTRAL"

        agreement = max(0.0, min(1.0, float(forecast.get("agreement", confidence))))
        uncertainty = max(0.0, min(1.0, float(forecast.get("uncertainty", 1.0 - confidence))))

        # Volatility proxy: uncertainty + instability
        instability = float(runtime.get("instability", 0.0))
        volatility = min(1.0, (uncertainty + instability) / 2)

        # Risk mode based on governance
        if gov_mode in ("survival", "lockdown"):
            risk_mode = "conservative"
        elif gov_mode == "cautious":
            risk_mode = "balanced"
        else:
            risk_mode = "balanced" if volatility < 0.5 else "conservative"

        return TradingState(
            signal=signal,
            confidence=confidence,
            market_regime=regime,
            governance_mode=gov_mode,
            forecast_direction=direction,
            forecast_agreement=agreement,
            forecast_uncertainty=uncertainty,
            volatility_scale=volatility,
            risk_mode=risk_mode,
            survival_mode=bool(gov.get("survival_mode", False)),
            constitution_passed=bool(gov.get("constitution_passed", True)),
            schema_version=str(payload.get("schema_version", "2.0")),
            envelope_age_secs=age,
            envelope_fresh=True,
            last_updated_ts=time.time(),
        )

    def _emit_events(self, kind: str, records: list[dict[str, Any]]) -> None:
        try:
            from app.event_bus import (
                EVENT_TRADE_REFLECTION_INGESTED,
                EVENT_MARKET_EPISODE_INGESTED,
                CloudEvent,
                get_event_bus,
            )

            etype = (
                EVENT_TRADE_REFLECTION_INGESTED
                if kind == "reflection"
                else EVENT_MARKET_EPISODE_INGESTED
            )
            get_event_bus().publish(
                CloudEvent(
                    type=etype,
                    source="trading_runtime_bridge",
                    payload={"count": len(records), "kind": kind},
                )
            )
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_bridge: TradingRuntimeBridge | None = None
_bridge_lock = threading.Lock()


def get_trading_bridge() -> TradingRuntimeBridge:
    """Return the process-level :class:`TradingRuntimeBridge` singleton."""
    global _bridge  # pylint: disable=global-statement
    with _bridge_lock:
        if _bridge is None:
            _bridge = TradingRuntimeBridge()
    return _bridge
