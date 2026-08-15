"""EvaluationEngine — lightweight production-safe implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class EvaluationEngine:
    """Evaluate trading outcomes and system performance.
    
    Lightweight implementation for runtime stabilization.
    """
    
    def __init__(self) -> None:
        self._evaluations: list[Dict[str, Any]] = []
    
    def evaluate_trade(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = {
            "trade_id": trade.get("trade_id"),
            "timestamp": logging.Formatter().formatTime(logging.LogRecord(
                name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
            )),
            "score": self._score(outcome),
            "passed": self._passed(outcome),
            "metrics": self._compute_metrics(outcome),
        }
        self._evaluations.append(evaluation)
        return evaluation
    
    def _score(self, outcome: Dict[str, Any]) -> float:
        pnl = outcome.get("pnl", 0.0)
        return max(0.0, min(1.0, 0.5 + pnl / 200.0))
    
    def _passed(self, outcome: Dict[str, Any]) -> bool:
        return outcome.get("pnl", 0.0) > 0
    
    def _compute_metrics(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pnl": outcome.get("pnl", 0.0),
            "duration": outcome.get("duration", 0),
            "slippage": outcome.get("slippage", 0.0),
        }
    
    def get_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        return self._evaluations[-limit:]