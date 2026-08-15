"""Phase 10 — Trading Pipeline Orchestrator.

Coordinates the complete cognitive trading pipeline:

Exchange → Market Data → Market Intelligence → Memory → Lean-Algos →
GGUF Validation → Risk Engine → Trade Intent → Freqtrade → Feedback → Memory

This is the central coordinator. It does NOT replace individual module
responsibilities — it chains them together.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from .models import (
    ExecutionResult,
    GGUFValidation,
    MarketSnapshot,
    MarketState,
    PortfolioState,
    RiskAssessment,
    TradeIntent,
    TradeProposal,
)

logger = logging.getLogger(__name__)


class TradingPipeline:
    """Orchestrates the complete trading cognition pipeline.

    Each stage is delegated to its dedicated module. This class only
    chains them together and manages the flow.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline_id = uuid.uuid4().hex[:8]
        self._paused = False
        self._cycle_count = 0
        self._last_cycle_ts: float = 0.0
        logger.info("TradingPipeline initialized (id=%s)", self._pipeline_id)

    def run_cycle(
        self,
        symbol: str,
        timeframe: str = "1h",
    ) -> dict[str, Any]:
        """Execute one complete trading pipeline cycle for a symbol.

        Returns a dict with the result of each stage.
        """
        if self._paused:
            return {"status": "paused", "cycle": self._cycle_count}

        cycle_id = f"cycle-{self._cycle_count}-{uuid.uuid4().hex[:8]}"
        start_ts = time.time()
        logger.info("[%s] Pipeline cycle start: %s/%s", cycle_id, symbol, timeframe)

        result: dict[str, Any] = {
            "cycle_id": cycle_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "stages": {},
            "duration_ms": 0.0,
        }

        # Stage 1: Market Data
        try:
            from .market_adapter import get_market_data_adapter
            adapter = get_market_data_adapter()
            snapshot = adapter.get_snapshot(symbol, timeframe)
            if snapshot is None:
                result["stages"]["market_data"] = {
                    "status": "skipped",
                    "reason": "no_data_available",
                }
                result["status"] = "insufficient_data"
                return result
            result["stages"]["market_data"] = {"status": "ok", "candles": len(snapshot.candles)}
        except Exception as exc:
            result["stages"]["market_data"] = {"status": "error", "error": str(exc)}

        # Stage 2: Market Intelligence
        state = None
        try:
            from .market_intelligence import get_market_intelligence
            engine = get_market_intelligence()
            state = engine.analyze(snapshot)
            result["stages"]["market_intelligence"] = {
                "status": "ok",
                "regime": state.regime,
                "trend": state.trend,
                "confidence": state.confidence,
            }
        except Exception as exc:
            result["stages"]["market_intelligence"] = {"status": "error", "error": str(exc)}

        # Stage 3: Lean-Algos
        proposals: list[TradeProposal] = []
        try:
            from .lean_bridge import get_lean_bridge
            bridge = get_lean_bridge()
            proposals = bridge.evaluate(snapshot, state)
            result["stages"]["lean_algos"] = {
                "status": "ok",
                "proposals": len(proposals),
            }
        except Exception as exc:
            result["stages"]["lean_algos"] = {"status": "error", "error": str(exc)}

        if not proposals:
            result["stages"]["gguf_validation"] = {"status": "skipped", "reason": "no_proposals"}
            result["stages"]["risk_engine"] = {"status": "skipped", "reason": "no_proposals"}
            result["stages"]["trade_intent"] = {"status": "skipped", "reason": "no_proposals"}
            result["status"] = "no_opportunity"
            result["duration_ms"] = (time.time() - start_ts) * 1000
            self._record_cycle(result)
            return result

        # Stage 4: GGUF Validation
        validation: GGUFValidation | None = None
        try:
            from .gguf_validator import get_gguf_validator
            validator = get_gguf_validator()
            from .risk_engine import get_risk_engine
            portfolio = get_risk_engine().get_portfolio()
            validation = validator.validate(proposals[0], snapshot, state, portfolio)
            result["stages"]["gguf_validation"] = {
                "status": "ok",
                "approved": validation.approved,
                "confidence": validation.confidence,
                "risk_level": validation.risk_level,
            }
        except Exception as exc:
            result["stages"]["gguf_validation"] = {"status": "error", "error": str(exc)}

        # Stage 5: Risk Engine
        risk: RiskAssessment | None = None
        try:
            from .risk_engine import get_risk_engine
            risk_engine = get_risk_engine()
            from .models import PortfolioState
            portfolio = risk_engine.get_portfolio()
            risk = risk_engine.assess(proposals[0], portfolio)
            result["stages"]["risk_engine"] = {
                "status": "ok",
                "allowed": risk.allowed,
                "violations": len(risk.violations),
                "recommended_stake": risk.recommended_stake,
            }
        except Exception as exc:
            result["stages"]["risk_engine"] = {"status": "error", "error": str(exc)}

        # Stage 6: Trade Intent
        intent: TradeIntent | None = None
        try:
            from .trade_intent import get_trade_intent_factory
            factory = get_trade_intent_factory()
            intent = factory.create(
                proposal=proposals[0],
                validation=validation or GGUFValidation(approved=True, confidence=0.5),
                risk=risk or RiskAssessment(allowed=True),
            )
            result["stages"]["trade_intent"] = {
                "status": "ok",
                "approved": intent.approved,
                "side": intent.side,
                "quantity": intent.quantity,
            }
        except Exception as exc:
            result["stages"]["trade_intent"] = {"status": "error", "error": str(exc)}

        # Stage 7: Freqtrade Execution
        if intent and intent.approved:
            try:
                from .freqtrade_adapter import get_freqtrade_adapter
                ft = get_freqtrade_adapter()
                exec_result = ft.execute(intent)
                result["stages"]["freqtrade"] = {
                    "status": exec_result.get("status", "unknown"),
                    "trade_id": exec_result.get("trade_id", ""),
                }
            except Exception as exc:
                result["stages"]["freqtrade"] = {"status": "error", "error": str(exc)}
        else:
            result["stages"]["freqtrade"] = {"status": "skipped", "reason": "not_approved"}

        result["status"] = "completed"
        result["duration_ms"] = (time.time() - start_ts) * 1000
        self._record_cycle(result)
        return result

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            logger.info("Trading pipeline paused")

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            logger.info("Trading pipeline resumed")

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pipeline_id": self._pipeline_id,
                "paused": self._paused,
                "cycle_count": self._cycle_count,
                "last_cycle_ts": self._last_cycle_ts,
            }

    def _record_cycle(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._cycle_count += 1
            self._last_cycle_ts = time.time()


# ── Singleton ────────────────────────────────────────────────────────────────

_pipeline: TradingPipeline | None = None
_pipeline_lock = threading.Lock()


def get_trading_pipeline() -> TradingPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = TradingPipeline()
    return _pipeline