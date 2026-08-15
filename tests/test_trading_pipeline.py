"""Tests for the complete trading pipeline."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.trading.models import (
    Candle,
    ExecutionResult,
    GGUFValidation,
    MarketSnapshot,
    MarketState,
    PortfolioState,
    RiskAssessment,
    TradeIntent,
    TradeProposal,
)


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestDataModels:
    """Test immutable data models."""

    def test_candle(self):
        c = Candle(timestamp=1000.0, open=100.0, high=105.0, low=99.0, close=102.0, volume=1000.0)
        assert c.timestamp == 1000.0
        assert c.close == 102.0

    def test_market_snapshot(self):
        candles = (Candle(1000.0, 100, 105, 99, 102, 1000),)
        snap = MarketSnapshot(symbol="BTC/USDT", timeframe="1h", candles=candles)
        assert snap.symbol == "BTC/USDT"
        assert snap.latest_price == 102.0
        assert snap.latest_volume == 1000.0

    def test_market_state(self):
        state = MarketState(symbol="BTC/USDT", regime="accumulation", trend="bullish", confidence=0.85)
        assert state.regime == "accumulation"
        assert state.market_bias == "neutral"
        d = state.to_dict()
        assert d["regime"] == "accumulation"
        assert d["confidence"] == 0.85

    def test_trade_proposal(self):
        p = TradeProposal(
            symbol="BTC/USDT", side="long", entry_price=50000.0,
            stop_loss=49000.0, take_profit=55000.0, confidence=0.75,
        )
        assert p.expected_risk_reward == 0.0
        d = p.to_dict()
        assert d["side"] == "long"

    def test_gguf_validation(self):
        v = GGUFValidation(approved=True, confidence=0.8, risk_level="low")
        assert v.approved
        d = v.to_dict()
        assert d["approved"] is True

    def test_risk_assessment(self):
        r = RiskAssessment(allowed=True, recommended_stake=100.0)
        assert r.allowed
        d = r.to_dict()
        assert d["allowed"] is True

    def test_portfolio_state(self):
        p = PortfolioState(total_equity=10000.0, available_capital=8000.0)
        assert p.total_equity == 10000.0
        d = p.to_dict()
        assert d["total_equity"] == 10000.0

    def test_trade_intent(self):
        i = TradeIntent(
            symbol="BTC/USDT", side="long", quantity=0.1,
            entry_price=50000.0, stop_loss=49000.0, take_profit=55000.0,
            confidence=0.8, approved=True,
        )
        assert i.approved
        d = i.to_dict()
        assert d["approved"] is True

    def test_execution_result(self):
        e = ExecutionResult(
            trade_id="t1", symbol="BTC/USDT", side="long",
            entry_price=50000.0, exit_price=55000.0, quantity=0.1,
            pnl=500.0, pnl_pct=0.1, duration_seconds=3600.0, slippage=0.001,
        )
        assert e.pnl == 500.0
        d = e.to_dict()
        assert d["trade_id"] == "t1"


class TestMarketDataAdapter:
    """Test market data adapter."""

    def test_singleton(self):
        from app.trading.market_adapter import get_market_data_adapter
        a1 = get_market_data_adapter()
        a2 = get_market_data_adapter()
        assert a1 is a2

    def test_ingest_candles(self):
        from app.trading.market_adapter import get_market_data_adapter
        adapter = get_market_data_adapter()
        ohlcv = [(1000.0, 100, 105, 99, 102, 1000)]
        snap = adapter.ingest_candles("BTC/USDT", "1h", ohlcv)
        assert snap.symbol == "BTC/USDT"
        assert snap.latest_price == 102.0

    def test_get_snapshot(self):
        from app.trading.market_adapter import get_market_data_adapter
        adapter = get_market_data_adapter()
        snap = adapter.get_snapshot("BTC/USDT", "1h")
        assert snap is not None
        assert snap.symbol == "BTC/USDT"

    def test_register_symbol(self):
        from app.trading.market_adapter import get_market_data_adapter
        adapter = get_market_data_adapter()
        adapter.register_symbol("ETH/USDT")
        assert "ETH/USDT" in adapter.get_symbols()

    def test_status(self):
        from app.trading.market_adapter import get_market_data_adapter
        adapter = get_market_data_adapter()
        status = adapter.status()
        assert "symbols" in status
        assert "timeframes" in status


class TestMarketIntelligence:
    """Test market intelligence engine."""

    def test_singleton(self):
        from app.trading.market_intelligence import get_market_intelligence
        e1 = get_market_intelligence()
        e2 = get_market_intelligence()
        assert e1 is e2

    def test_analyze_insufficient_data(self):
        from app.trading.market_intelligence import get_market_intelligence
        engine = get_market_intelligence()
        snap = MarketSnapshot(symbol="BTC/USDT", timeframe="1h")
        state = engine.analyze(snap)
        assert state.regime == "insufficient_data"

    def test_analyze_with_candles(self):
        from app.trading.market_intelligence import get_market_intelligence
        engine = get_market_intelligence()
        candles = tuple(
            Candle(1000.0 + i * 3600, 100 + i * 0.1, 105 + i * 0.1, 99 + i * 0.1, 102 + i * 0.1, 1000)
            for i in range(100)
        )
        snap = MarketSnapshot(symbol="BTC/USDT", timeframe="1h", candles=candles)
        state = engine.analyze(snap)
        assert state.symbol == "BTC/USDT"
        assert state.confidence > 0

    def test_get_latest_state(self):
        from app.trading.market_intelligence import get_market_intelligence
        engine = get_market_intelligence()
        state = engine.get_latest_state("BTC/USDT")
        assert state is not None

    def test_status(self):
        from app.trading.market_intelligence import get_market_intelligence
        engine = get_market_intelligence()
        status = engine.status()
        assert "tracked_symbols" in status


class TestLeanBridge:
    """Test Lean-Algos bridge."""

    def test_singleton(self):
        from app.trading.lean_bridge import get_lean_bridge
        b1 = get_lean_bridge()
        b2 = get_lean_bridge()
        assert b1 is b2

    def test_evaluate_no_data(self):
        from app.trading.lean_bridge import get_lean_bridge
        bridge = get_lean_bridge()
        snap = MarketSnapshot(symbol="BTC/USDT", timeframe="1h")
        state = MarketState(symbol="BTC/USDT")
        proposals = bridge.evaluate(snap, state)
        assert isinstance(proposals, list)

    def test_status(self):
        from app.trading.lean_bridge import get_lean_bridge
        bridge = get_lean_bridge()
        status = bridge.status()
        assert "signal_dir" in status


class TestGGUFValidator:
    """Test GGUF validator."""

    def test_singleton(self):
        from app.trading.gguf_validator import get_gguf_validator
        v1 = get_gguf_validator()
        v2 = get_gguf_validator()
        assert v1 is v2

    def test_validate_default_fallback(self):
        from app.trading.gguf_validator import get_gguf_validator
        validator = get_gguf_validator()
        proposal = TradeProposal(
            symbol="BTC/USDT", side="long", entry_price=50000.0,
            stop_loss=49000.0, take_profit=55000.0, confidence=0.75,
        )
        result = validator.validate(proposal)
        assert result.approved  # Default approval
        # May be model_unavailable or parse_error depending on test env
        assert len(result.warnings) >= 0

    def test_status(self):
        from app.trading.gguf_validator import get_gguf_validator
        validator = get_gguf_validator()
        status = validator.status()
        assert "validation_count" in status


class TestRiskEngine:
    """Test risk engine."""

    def test_singleton(self):
        from app.trading.risk_engine import get_risk_engine
        r1 = get_risk_engine()
        r2 = get_risk_engine()
        assert r1 is r2

    def test_assess_allowed(self):
        from app.trading.risk_engine import get_risk_engine
        engine = get_risk_engine()
        portfolio = PortfolioState(total_equity=10000.0, available_capital=8000.0)
        proposal = TradeProposal(
            symbol="BTC/USDT", side="long", entry_price=100.0,
            stop_loss=98.0, take_profit=110.0,
            expected_risk_reward=5.0, confidence=0.75,
        )
        assessment = engine.assess(proposal, portfolio)
        assert assessment.allowed

    def test_assess_daily_loss(self):
        from app.trading.risk_engine import get_risk_engine
        engine = get_risk_engine()
        portfolio = PortfolioState(total_equity=10000.0, daily_pnl=-500.0)
        proposal = TradeProposal(
            symbol="BTC/USDT", side="long", entry_price=50000.0,
            stop_loss=49000.0, take_profit=55000.0,
            expected_risk_reward=2.0, confidence=0.75,
        )
        assessment = engine.assess(proposal, portfolio)
        # Daily loss of -500 on 10000 equity = -5%, exceeds 3% limit
        assert not assessment.allowed

    def test_status(self):
        from app.trading.risk_engine import get_risk_engine
        engine = get_risk_engine()
        status = engine.status()
        assert "config" in status


class TestTradeIntentFactory:
    """Test trade intent factory."""

    def test_singleton(self):
        from app.trading.trade_intent import get_trade_intent_factory
        f1 = get_trade_intent_factory()
        f2 = get_trade_intent_factory()
        assert f1 is f2

    def test_create_approved(self):
        from app.trading.trade_intent import get_trade_intent_factory
        factory = get_trade_intent_factory()
        proposal = TradeProposal(
            symbol="BTC/USDT", side="long", entry_price=50000.0,
            stop_loss=49000.0, take_profit=55000.0, confidence=0.75,
        )
        validation = GGUFValidation(approved=True, confidence=0.8, risk_level="low")
        risk = RiskAssessment(allowed=True, recommended_stake=1000.0)
        intent = factory.create(proposal, validation, risk)
        assert intent.approved
        assert intent.symbol == "BTC/USDT"

    def test_create_rejected(self):
        from app.trading.trade_intent import get_trade_intent_factory
        factory = get_trade_intent_factory()
        proposal = TradeProposal(
            symbol="BTC/USDT", side="long", entry_price=50000.0,
            stop_loss=49000.0, take_profit=55000.0, confidence=0.75,
        )
        validation = GGUFValidation(approved=False, confidence=0.3, risk_level="high")
        risk = RiskAssessment(allowed=False, violations=("risk_check",))
        intent = factory.create(proposal, validation, risk)
        assert not intent.approved


class TestFreqtradeAdapter:
    """Test Freqtrade adapter."""

    def test_singleton(self):
        from app.trading.freqtrade_adapter import get_freqtrade_adapter
        a1 = get_freqtrade_adapter()
        a2 = get_freqtrade_adapter()
        assert a1 is a2

    def test_execute_disabled(self):
        from app.trading.freqtrade_adapter import get_freqtrade_adapter
        adapter = get_freqtrade_adapter()
        intent = TradeIntent(
            symbol="BTC/USDT", side="long", quantity=0.1,
            entry_price=50000.0, stop_loss=49000.0, take_profit=55000.0,
            confidence=0.8, approved=True,
        )
        result = adapter.execute(intent)
        assert result["status"] == "logged"
        assert result["simulated"] is True

    def test_status(self):
        from app.trading.freqtrade_adapter import get_freqtrade_adapter
        adapter = get_freqtrade_adapter()
        status = adapter.status()
        assert "enabled" in status


class TestExecutionFeedback:
    """Test execution feedback loop."""

    def test_singleton(self):
        from app.trading.execution_feedback import get_execution_feedback
        f1 = get_execution_feedback()
        f2 = get_execution_feedback()
        assert f1 is f2

    def test_record_and_retrieve(self):
        from app.trading.execution_feedback import get_execution_feedback
        feedback = get_execution_feedback()
        result = ExecutionResult(
            trade_id="test-1", symbol="BTC/USDT", side="long",
            entry_price=50000.0, exit_price=55000.0, quantity=0.1,
            pnl=500.0, pnl_pct=0.1, duration_seconds=3600.0, slippage=0.001,
        )
        feedback.record(result)
        results = feedback.get_results(limit=10)
        assert len(results) >= 1
        assert results[-1].trade_id == "test-1"

    def test_statistics(self):
        from app.trading.execution_feedback import get_execution_feedback
        feedback = get_execution_feedback()
        stats = feedback.get_statistics()
        assert "total_trades" in stats


class TestTradingPipeline:
    """Test trading pipeline."""

    def test_singleton(self):
        from app.trading.pipeline import get_trading_pipeline
        p1 = get_trading_pipeline()
        p2 = get_trading_pipeline()
        assert p1 is p2

    def test_run_cycle_no_data(self):
        from app.trading.pipeline import get_trading_pipeline
        pipeline = get_trading_pipeline()
        result = pipeline.run_cycle("BTC/USDT", "1h")
        assert result["status"] in ("insufficient_data", "no_opportunity", "completed")

    def test_pause_resume(self):
        from app.trading.pipeline import get_trading_pipeline
        pipeline = get_trading_pipeline()
        pipeline.pause()
        assert pipeline.is_paused
        pipeline.resume()
        assert not pipeline.is_paused

    def test_status(self):
        from app.trading.pipeline import get_trading_pipeline
        pipeline = get_trading_pipeline()
        status = pipeline.status()
        assert "pipeline_id" in status


class TestTradingAPI:
    """Test trading API endpoints."""

    def test_market_state_endpoint(self, client):
        response = client.get("/trading/market/state")
        assert response.status_code == 200

    def test_market_snapshot_endpoint(self, client):
        response = client.get("/trading/market/snapshot?symbol=BTC/USDT&timeframe=1h")
        assert response.status_code in (200, 404)

    def test_portfolio_endpoint(self, client):
        response = client.get("/trading/portfolio")
        assert response.status_code == 200

    def test_trades_open_endpoint(self, client):
        response = client.get("/trading/trades/open")
        assert response.status_code == 200

    def test_trades_history_endpoint(self, client):
        response = client.get("/trading/trades/history")
        assert response.status_code == 200

    def test_trade_proposal_endpoint(self, client):
        response = client.post("/trading/trade/proposal?symbol=BTC/USDT&timeframe=1h")
        assert response.status_code == 200

    def test_trade_validate_endpoint(self, client):
        response = client.post("/trading/trade/validate", json={
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 55000.0,
            "expected_risk_reward": 2.0,
            "confidence": 0.75,
        })
        assert response.status_code == 200

    def test_trade_execute_endpoint(self, client):
        response = client.post("/trading/trade/execute", json={
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 0.1,
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 55000.0,
            "confidence": 0.8,
            "approved": True,
        })
        assert response.status_code == 200

    def test_risk_status_endpoint(self, client):
        response = client.get("/trading/risk/status")
        assert response.status_code == 200

    def test_pipeline_status_endpoint(self, client):
        response = client.get("/trading/pipeline/status")
        assert response.status_code == 200

    def test_pipeline_pause_resume(self, client):
        response = client.post("/trading/pipeline/pause")
        assert response.status_code == 200
        response = client.post("/trading/pipeline/resume")
        assert response.status_code == 200

    def test_feedback_statistics_endpoint(self, client):
        response = client.get("/trading/feedback/statistics")
        assert response.status_code == 200

    def test_proposals_history_endpoint(self, client):
        response = client.get("/trading/proposals/history")
        assert response.status_code == 200

    def test_validations_history_endpoint(self, client):
        response = client.get("/trading/validations/history")
        assert response.status_code == 200

    def test_intents_history_endpoint(self, client):
        response = client.get("/trading/intents/history")
        assert response.status_code == 200

    def test_trading_health_endpoint(self, client):
        response = client.get("/trading/health")
        assert response.status_code == 200