"""Phase 2 — Market Intelligence Engine.

Computes market behaviour understanding from raw MarketSnapshot data.
No execution logic. Only analysis.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from .models import Candle, MarketSnapshot, MarketState

logger = logging.getLogger(__name__)


def _compute_sma(prices: tuple[float, ...], period: int) -> float | None:
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def _compute_ema(prices: tuple[float, ...], period: int) -> float | None:
    if len(prices) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema


def _compute_rsi(prices: tuple[float, ...], period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = prices[i] - prices[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(candles: tuple[Candle, ...], period: int = 14) -> float | None:
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(candles)):
        high, low = candles[i].high, candles[i].low
        prev_close = candles[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return None
    return sum(trs[-period:]) / period


def _detect_swing_points(candles: tuple[Candle, ...], window: int = 5):
    """Detect swing highs and lows."""
    if len(candles) < window * 2 + 1:
        return None, None
    swing_high = None
    swing_low = None
    for i in range(window, len(candles) - window):
        if all(candles[i].high >= candles[j].high for j in range(i - window, i + window + 1)):
            if swing_high is None or candles[i].high > swing_high:
                swing_high = candles[i].high
        if all(candles[i].low <= candles[j].low for j in range(i - window, i + window + 1)):
            if swing_low is None or candles[i].low < swing_low:
                swing_low = candles[i].low
    return swing_high, swing_low


class MarketIntelligenceEngine:
    """Analyzes MarketSnapshots and produces MarketState objects.

    Computes trend, momentum, volatility, market regime identification,
    support/resistance, and structural analysis.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_states: dict[str, MarketState] = {}
        self._state_history: dict[str, list[MarketState]] = {}
        logger.info("MarketIntelligenceEngine initialized")

    def analyze(self, snapshot: MarketSnapshot) -> MarketState:
        """Produce a MarketState from a MarketSnapshot."""
        symbol = snapshot.symbol
        candles = snapshot.candles

        if len(candles) < 20:
            state = MarketState(symbol=symbol, regime="insufficient_data")
            self._store_state(state)
            return state

        closes = tuple(c.close for c in candles)
        highs = tuple(c.high for c in candles)
        lows = tuple(c.low for c in candles)

        # ── Trend ──────────────────────────────────────────────────────────
        sma_short = _compute_sma(closes, 10)
        sma_long = _compute_sma(closes, 50)
        ema_short = _compute_ema(closes, 12)
        ema_long = _compute_ema(closes, 26)

        if sma_short is not None and sma_long is not None:
            if sma_short > sma_long:
                trend = "bullish"
            elif sma_short < sma_long:
                trend = "bearish"
            else:
                trend = "neutral"
        else:
            trend = "neutral"

        # ── Momentum ───────────────────────────────────────────────────────
        rsi = _compute_rsi(closes)
        if rsi is not None:
            if rsi > 60:
                momentum = "positive"
            elif rsi < 40:
                momentum = "negative"
            else:
                momentum = "neutral"
        else:
            momentum = "neutral"

        # ── Volatility ─────────────────────────────────────────────────────
        atr = _compute_atr(candles)
        if atr is not None and snapshot.vwap is not None and snapshot.vwap > 0:
            vol_ratio = atr / snapshot.vwap
            if vol_ratio < 0.01:
                volatility = "low"
            elif vol_ratio < 0.03:
                volatility = "medium"
            else:
                volatility = "high"
        else:
            volatility = "unknown"

        # ── Market Regime ──────────────────────────────────────────────────
        regime = "ranging"
        if rsi is not None and len(candles) >= 50:
            sma_50 = _compute_sma(closes, 50)
            sma_200 = _compute_sma(closes, 200)
            if sma_50 is not None and sma_200 is not None:
                bullish_cross = closes[-1] > sma_50 > sma_200
                bearish_cross = closes[-1] < sma_50 < sma_200
                if bullish_cross and momentum == "positive":
                    regime = "accumulation"
                elif bearish_cross and momentum == "negative":
                    regime = "distribution"
                elif closes[-1] > sma_50:
                    regime = "trending_up"
                elif closes[-1] < sma_50:
                    regime = "trending_down"
                else:
                    regime = "ranging"

        # ── Breakout / Compression / Expansion ──────────────────────────────
        compression = False
        expansion = False
        if len(candles) >= 20:
            recent_range = max(highs[-20:]) - min(lows[-20:])
            avg_candle = sum(c.high - c.low for c in candles[-20:]) / 20
            if avg_candle > 0 and (recent_range / avg_candle) < 2:
                compression = True
            if avg_candle > 0 and (recent_range / avg_candle) > 5:
                expansion = True

        # ── Support / Resistance / Swing Points ────────────────────────────
        swing_high, swing_low = _detect_swing_points(candles)

        # ── Structural Break ────────────────────────────────────────────────
        structural_break = False
        if len(candles) >= 50 and swing_high and swing_low:
            if closes[-1] > swing_high or closes[-1] < swing_low:
                structural_break = True

        # ── Market Bias ─────────────────────────────────────────────────────
        if trend == "bullish" and momentum == "positive":
            bias = "long"
        elif trend == "bearish" and momentum == "negative":
            bias = "short"
        else:
            bias = "neutral"

        # ── Confidence ─────────────────────────────────────────────────────
        signals = 0
        total = 0
        for cond, weight in [
            (trend != "neutral", 0.2),
            (momentum != "neutral", 0.15),
            (regime != "ranging", 0.2),
            (swing_high is not None, 0.1),
            (swing_low is not None, 0.1),
            (not compression, 0.1),
            (volatility != "unknown", 0.15),
        ]:
            total += weight
            if cond:
                signals += weight
        confidence = signals / total if total > 0 else 0.5

        # ── Liquidity ──────────────────────────────────────────────────────
        spread = snapshot.bid_ask_spread
        if spread > 0:
            if spread < 0.001:
                liquidity = "high"
            elif spread < 0.01:
                liquidity = "medium"
            else:
                liquidity = "low"
        else:
            liquidity = "unknown"

        state = MarketState(
            symbol=symbol,
            regime=regime,
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            liquidity=liquidity,
            support=swing_low,
            resistance=swing_high,
            swing_high=swing_high,
            swing_low=swing_low,
            compression=compression,
            expansion=expansion,
            structural_break=structural_break,
            confidence=min(1.0, max(0.0, confidence)),
            market_bias=bias,
            timestamp=time.time(),
        )
        self._store_state(state)
        return state

    def get_latest_state(self, symbol: str) -> MarketState | None:
        with self._lock:
            return self._latest_states.get(symbol)

    def get_state_history(self, symbol: str, limit: int = 100) -> list[MarketState]:
        with self._lock:
            history = self._state_history.get(symbol, [])
            return history[-limit:]

    def get_all_states(self) -> dict[str, MarketState]:
        with self._lock:
            return dict(self._latest_states)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tracked_symbols": list(self._latest_states.keys()),
                "history_depth": {s: len(h) for s, h in self._state_history.items()},
            }

    def _store_state(self, state: MarketState) -> None:
        with self._lock:
            self._latest_states[state.symbol] = state
            if state.symbol not in self._state_history:
                self._state_history[state.symbol] = []
            self._state_history[state.symbol].append(state)
            if len(self._state_history[state.symbol]) > 1000:
                self._state_history[state.symbol] = self._state_history[state.symbol][-500:]


# ── Singleton ────────────────────────────────────────────────────────────────

_engine: MarketIntelligenceEngine | None = None
_engine_lock = threading.Lock()


def get_market_intelligence() -> MarketIntelligenceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = MarketIntelligenceEngine()
    return _engine