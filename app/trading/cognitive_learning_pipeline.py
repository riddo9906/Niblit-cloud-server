"""Cognitive Learning Pipeline — continuous feedback loop from trades to decisions."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class CognitiveLearningPipeline:
    """Continuous learning loop for trading intelligence.

    Flow:
    ExecutionFeedbackLoop
        ↓
    ReflectionEngine
        ↓
    TradeMemory
        ↓
    ExperienceMemory
        ↓
    StrategyEvolution
        ↓
    RiskEngine
        ↓
    Future decisions
    """

    def __init__(
        self,
        reflection_engine: Optional[Any] = None,
        strategy_evolution: Optional[Any] = None,
        portfolio_intelligence: Optional[Any] = None,
        decision_orchestrator: Optional[Any] = None,
    ) -> None:
        self.reflection_engine = reflection_engine
        self.strategy_evolution = strategy_evolution
        self.portfolio_intelligence = portfolio_intelligence
        self.decision_orchestrator = decision_orchestrator
        self._history: list[Dict[str, Any]] = []

    def process_trade_outcome(self, trade: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        reflection = {}
        if self.reflection_engine:
            reflection = self.reflection_engine.reflect(trade, outcome)

        if self.strategy_evolution:
            self.strategy_evolution.record_trade(trade, outcome)

        entry = {
            "trade_id": trade.get("trade_id"),
            "outcome": outcome,
            "reflection": reflection,
        }
        self._history.append(entry)
        return entry

    def get_decision_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {"history_count": len(self._history)}
        if self.portfolio_intelligence:
            context["portfolio"] = self.portfolio_intelligence.get_snapshot()
        if self.strategy_evolution:
            context["strategy_rankings"] = self.strategy_evolution.get_rankings()
            context["recommended_strategy"] = self.strategy_evolution.recommend()
        return context