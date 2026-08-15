"""Decision Orchestrator — unified trade decision layer."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class DecisionOrchestrator:
    """Aggregate inputs from multiple cognitive services before trading.

    Inputs:
    - Market Intelligence
    - Portfolio Intelligence
    - Risk Engine
    - Reflection Engine
    - Experience Memory
    - Strategy Evolution
    - Trade Confidence

    Produces:
    - Unified decision object passed to Freqtrade
    """

    def __init__(self) -> None:
        self._history: list[Dict[str, Any]] = []

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        decision = {
            "timestamp": logging.Formatter().formatTime(logging.LogRecord(
                name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
            )),
            "action": payload.get("action", "HOLD"),
            "symbol": payload.get("symbol"),
            "confidence": payload.get("confidence", 0.0),
            "risk_score": payload.get("risk_score", 0.0),
            "rationale": payload.get("rationale", ""),
        }
        self._history.append(decision)
        return decision

    def get_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        return self._history[-limit:]