"""Strategy Evolution Engine — autonomous strategy improvement."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class StrategyEvolutionEngine:
    """Evaluate and evolve trading strategies.

    Responsibilities:
    - score strategy performance
    - detect degradation
    - adapt parameters
    - rank signals
    - detect market regimes
    - evolve confidence
    - recommend strategy replacements
    - maintain historical effectiveness metrics
    """

    def __init__(self) -> None:
        self._history: list[Dict[str, Any]] = []
        self._rankings: Dict[str, float] = {}

    def record_trade(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        entry = {
            "trade_id": trade.get("trade_id"),
            "strategy": trade.get("strategy", "unknown"),
            "timestamp": time.time(),
            "pnl": outcome.get("pnl", 0.0),
            "duration": outcome.get("duration", 0),
        }
        self._history.append(entry)
        self._update_ranking(entry)

    def _update_ranking(self, entry: Dict[str, Any]) -> None:
        strategy = entry["strategy"]
        pnl = entry.get("pnl", 0.0)
        current = self._rankings.get(strategy, 0.0)
        self._rankings[strategy] = current * 0.9 + pnl * 0.1

    def get_rankings(self) -> Dict[str, float]:
        return dict(self._rankings)

    def recommend(self) -> Optional[str]:
        if not self._rankings:
            return None
        return max(self._rankings, key=self._rankings.get)

    def get_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        return self._history[-limit:]