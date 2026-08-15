"""Reflection Engine — continuous learning from trading outcomes."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class ReflectionEngine:
    """Continuous reflection on completed trades.

    Produces:
    - Reflection
    - Lessons
    - Confidence adjustment
    - Strategy adjustment
    - Memory update
    - Risk update

    Feeds:
    - Market Intelligence
    - Strategy Evolution
    - BrainMemory
    - Risk Engine
    """

    def __init__(self) -> None:
        self._history: list[Dict[str, Any]] = []

    def reflect(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        reflection = {
            "trade_id": trade.get("trade_id"),
            "timestamp": time.time(),
            "reflection": self._synthesize(trade, outcome),
            "lessons": self._extract_lessons(trade, outcome),
            "confidence_delta": self._confidence_delta(outcome),
            "strategy_adjustment": self._strategy_adjustment(outcome),
            "risk_update": self._risk_update(outcome),
        }
        self._history.append(reflection)
        return reflection

    def _synthesize(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> str:
        pnl = outcome.get("pnl", 0.0)
        return f"Trade {trade.get('trade_id')} finished with PnL {pnl:.2f}"

    def _extract_lessons(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> list[str]:
        lessons: list[str] = []
        if outcome.get("slippage", 0) > 0.01:
            lessons.append("High slippage detected")
        if outcome.get("duration", 0) > 86400:
            lessons.append("Trade held longer than expected")
        return lessons

    def _confidence_delta(self, outcome: Dict[str, Any]) -> float:
        pnl = outcome.get("pnl", 0.0)
        return max(-1.0, min(1.0, pnl / 100.0))

    def _strategy_adjustment(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "none", "reason": "placeholder"}

    def _risk_update(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        drawdown = abs(outcome.get("max_drawdown", 0.0))
        return {"drawdown": drawdown, "action": "none"}

    def get_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        return self._history[-limit:]