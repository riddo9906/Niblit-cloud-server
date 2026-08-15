"""Phase 12 — Trading API Endpoints.

REST endpoints for the complete trading pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from .models import PortfolioState, TradeProposal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading", tags=["trading"])


@router.get("/market/state")
def get_market_state(symbol: str | None = None) -> dict[str, Any]:
    from .market_intelligence import get_market_intelligence
    engine = get_market_intelligence()
    if symbol:
        state = engine.get_latest_state(symbol)
        if state is None:
            raise HTTPException(status_code=404, detail=f"No market state for {symbol}")
        return state.to_dict()
    return {s: st.to_dict() for s, st in engine.get_all_states().items()}


@router.get("/market/snapshot")
def get_market_snapshot(symbol: str, timeframe: str = "1h") -> dict[str, Any]:
    from .market_adapter import get_market_data_adapter
    adapter = get_market_data_adapter()
    snapshot = adapter.get_snapshot(symbol, timeframe)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No snapshot for {symbol}/{timeframe}")
    return {
        "symbol": snapshot.symbol,
        "timeframe": snapshot.timeframe,
        "candles": len(snapshot.candles),
        "latest_price": snapshot.latest_price,
        "spread": snapshot.bid_ask_spread,
        "atr": snapshot.atr,
        "volatility": snapshot.realized_volatility,
        "timestamp": snapshot.timestamp,
    }


@router.get("/portfolio")
def get_portfolio() -> dict[str, Any]:
    from .risk_engine import get_risk_engine
    return get_risk_engine().get_portfolio().to_dict()


@router.get("/trades/open")
def get_open_trades() -> dict[str, Any]:
    from .freqtrade_adapter import get_freqtrade_adapter
    adapter = get_freqtrade_adapter()
    return {"open_trades": adapter.get_open_trades()}


@router.get("/trades/history")
def get_trade_history(limit: int = 50) -> list[dict[str, Any]]:
    from .execution_feedback import get_execution_feedback
    feedback = get_execution_feedback()
    return [r.to_dict() for r in feedback.get_results(limit=limit)]


@router.post("/trade/proposal")
def create_proposal(symbol: str, timeframe: str = "1h") -> dict[str, Any]:
    from .pipeline import get_trading_pipeline
    pipeline = get_trading_pipeline()
    result = pipeline.run_cycle(symbol, timeframe)
    return result


@router.post("/trade/validate")
def validate_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    from .gguf_validator import get_gguf_validator
    from .models import TradeProposal
    proposal = TradeProposal(
        symbol=payload.get("symbol", ""),
        side=payload.get("side", "long"),
        entry_price=float(payload.get("entry_price", 0)),
        stop_loss=float(payload.get("stop_loss", 0)),
        take_profit=float(payload.get("take_profit", 0)),
        expected_risk_reward=float(payload.get("expected_risk_reward", 0)),
        confidence=float(payload.get("confidence", 0.5)),
        rationale=payload.get("rationale", ""),
    )
    validator = get_gguf_validator()
    validation = validator.validate(proposal)
    return validation.to_dict()


@router.post("/trade/execute")
def execute_trade(payload: dict[str, Any]) -> dict[str, Any]:
    from .trade_intent import get_trade_intent_factory
    from .freqtrade_adapter import get_freqtrade_adapter
    from .models import TradeIntent
    intent = TradeIntent(
        symbol=payload.get("symbol", ""),
        side=payload.get("side", "long"),
        quantity=float(payload.get("quantity", 0)),
        entry_price=float(payload.get("entry_price", 0)),
        stop_loss=float(payload.get("stop_loss", 0)),
        take_profit=float(payload.get("take_profit", 0)),
        confidence=float(payload.get("confidence", 0.5)),
        approved=bool(payload.get("approved", False)),
    )
    adapter = get_freqtrade_adapter()
    return adapter.execute(intent)


@router.get("/risk/status")
def get_risk_status() -> dict[str, Any]:
    from .risk_engine import get_risk_engine
    return get_risk_engine().status()


@router.get("/pipeline/status")
def get_pipeline_status() -> dict[str, Any]:
    from .pipeline import get_trading_pipeline
    return get_trading_pipeline().status()


@router.post("/pipeline/pause")
def pause_pipeline() -> dict[str, str]:
    from .pipeline import get_trading_pipeline
    get_trading_pipeline().pause()
    return {"status": "paused"}


@router.post("/pipeline/resume")
def resume_pipeline() -> dict[str, str]:
    from .pipeline import get_trading_pipeline
    get_trading_pipeline().resume()
    return {"status": "resumed"}


@router.get("/feedback/statistics")
def get_feedback_statistics() -> dict[str, Any]:
    from .execution_feedback import get_execution_feedback
    return get_execution_feedback().get_statistics()


@router.get("/proposals/history")
def get_proposal_history(limit: int = 50) -> list[dict[str, Any]]:
    from .lean_bridge import get_lean_bridge
    return [p.to_dict() for p in get_lean_bridge().get_proposal_history(limit)]


@router.get("/validations/history")
def get_validation_history(limit: int = 50) -> list[dict[str, Any]]:
    from .gguf_validator import get_gguf_validator
    return [v.to_dict() for v in get_gguf_validator().get_history(limit)]


@router.get("/intents/history")
def get_intent_history(limit: int = 50) -> list[dict[str, Any]]:
    from .trade_intent import get_trade_intent_factory
    return [i.to_dict() for i in get_trade_intent_factory().get_history(limit)]


@router.get("/health")
def trading_health() -> dict[str, Any]:
    from .market_adapter import get_market_data_adapter
    from .market_intelligence import get_market_intelligence
    from .freqtrade_adapter import get_freqtrade_adapter
    from .risk_engine import get_risk_engine
    from .pipeline import get_trading_pipeline
    return {
        "market_adapter": get_market_data_adapter().status(),
        "market_intelligence": get_market_intelligence().status(),
        "freqtrade": get_freqtrade_adapter().status(),
        "risk_engine": get_risk_engine().status(),
        "pipeline": get_trading_pipeline().status(),
    }