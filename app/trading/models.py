"""Trading data models — immutable snapshots for the entire pipeline.

Every object in this module is a frozen dataclass. No mutation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ── Market Data ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candle:
    """Single OHLCV candle."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    bids: tuple[OrderBookLevel, ...] = ()
    asks: tuple[OrderBookLevel, ...] = ()


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable snapshot of market state at a point in time."""
    symbol: str
    timeframe: str
    candles: tuple[Candle, ...] = ()
    orderbook: OrderBook = field(default_factory=OrderBook)
    bid_ask_spread: float = 0.0
    vwap: float | None = None
    atr: float | None = None
    realized_volatility: float | None = None
    volume_profile: dict[str, float] = field(default_factory=dict)
    funding_rate: float | None = None
    open_interest: float | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def latest_price(self) -> float | None:
        if self.candles:
            return self.candles[-1].close
        return None

    @property
    def latest_volume(self) -> float | None:
        if self.candles:
            return self.candles[-1].volume
        return None


# ── Market Intelligence ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketState:
    """Canonical description of current market behaviour."""
    symbol: str
    regime: str = "unknown"          # Accumulation, Distribution, Trending, Ranging, Breakout
    trend: str = "neutral"           # Bullish, Bearish, Neutral
    momentum: str = "neutral"        # Positive, Negative, Neutral
    volatility: str = "low"          # Low, Medium, High
    liquidity: str = "medium"        # Low, Medium, High
    market_structure: str = "unknown"
    support: float | None = None
    resistance: float | None = None
    swing_high: float | None = None
    swing_low: float | None = None
    fair_value_gap: tuple[float, float] | None = None
    structural_break: bool = False
    liquidity_sweep: bool = False
    compression: bool = False
    expansion: bool = False
    mean_reversion_signal: bool = False
    trend_continuation: bool = False
    trend_exhaustion: bool = False
    confidence: float = 0.5
    market_bias: str = "neutral"     # Long, Short, Neutral
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "regime": self.regime,
            "trend": self.trend,
            "momentum": self.momentum,
            "volatility": self.volatility,
            "liquidity": self.liquidity,
            "market_structure": self.market_structure,
            "support": self.support,
            "resistance": self.resistance,
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "fair_value_gap": list(self.fair_value_gap) if self.fair_value_gap else None,
            "structural_break": self.structural_break,
            "liquidity_sweep": self.liquidity_sweep,
            "compression": self.compression,
            "expansion": self.expansion,
            "mean_reversion_signal": self.mean_reversion_signal,
            "trend_continuation": self.trend_continuation,
            "trend_exhaustion": self.trend_exhaustion,
            "confidence": round(self.confidence, 4),
            "market_bias": self.market_bias,
            "timestamp": self.timestamp,
        }


# ── Lean-Algos Integration ───────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeProposal:
    """Recommendation from Niblit-Lean-Algos. Not an execution order."""
    symbol: str
    side: str                           # long, short
    entry_price: float
    stop_loss: float
    take_profit: float
    expected_duration: str = "unknown"  # scalping, intraday, swing, position
    expected_risk_reward: float = 0.0
    confidence: float = 0.5
    rationale: str = ""
    algorithm: str = "unknown"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "expected_duration": self.expected_duration,
            "expected_risk_reward": round(self.expected_risk_reward, 2),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
        }


# ── GGUF Validation ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GGUFValidation:
    """Structured validation result from the local GGUF model."""
    approved: bool
    confidence: float
    risk_level: str = "medium"         # low, medium, high, critical
    warnings: tuple[str, ...] = ()
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "warnings": list(self.warnings),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


# ── Risk & Portfolio ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskAssessment:
    """Deterministic risk check result. Never AI."""
    allowed: bool
    violations: tuple[str, ...] = ()
    max_position_size: float = 0.0
    recommended_stake: float = 0.0
    max_leverage: float = 1.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violations": list(self.violations),
            "max_position_size": self.max_position_size,
            "recommended_stake": self.recommended_stake,
            "max_leverage": self.max_leverage,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class PortfolioState:
    """Current portfolio state snapshot."""
    total_equity: float = 0.0
    available_capital: float = 0.0
    open_positions: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_drawdown: float = 0.0
    max_drawdown: float = 0.0
    current_exposure: float = 0.0
    position_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_equity": self.total_equity,
            "available_capital": self.available_capital,
            "open_positions": self.open_positions,
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_drawdown": round(self.daily_drawdown, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "current_exposure": round(self.current_exposure, 4),
            "position_count": self.position_count,
            "timestamp": self.timestamp,
        }


# ── Trade Intent ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TradeIntent:
    """The ONLY object allowed to reach execution. Immutable after creation."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    approved: bool
    proposal_id: str = ""
    validation_id: str = ""
    risk_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": round(self.confidence, 4),
            "approved": self.approved,
            "proposal_id": self.proposal_id,
            "validation_id": self.validation_id,
            "risk_id": self.risk_id,
            "timestamp": self.timestamp,
        }


# ── Execution Feedback ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a completed trade execution."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_seconds: float
    slippage: float
    market_state: MarketState | None = None
    proposal: TradeProposal | None = None
    validation: GGUFValidation | None = None
    risk_assessment: RiskAssessment | None = None
    intent: TradeIntent | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "duration_seconds": self.duration_seconds,
            "slippage": round(self.slippage, 6),
            "market_state": self.market_state.to_dict() if self.market_state else None,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "risk_assessment": self.risk_assessment.to_dict() if self.risk_assessment else None,
            "intent": self.intent.to_dict() if self.intent else None,
            "timestamp": self.timestamp,
        }