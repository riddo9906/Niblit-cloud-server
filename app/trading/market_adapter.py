"""Phase 1 — Market Data Adapter.

Collects and normalizes market data from exchange/Freqtrade into
immutable MarketSnapshot objects. No AI logic exists here.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from .models import Candle, MarketSnapshot, OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)


class MarketDataAdapter:
    """Responsible only for acquiring and normalizing market data.

    This layer has zero AI. It acquires OHLCV, order book, trades,
    and metadata from exchange sources and normalizes into snapshots.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._symbols: set[str] = set()
        self._timeframes: set[str] = {"1m", "5m", "15m", "1h", "4h", "1d"}
        self._latest_snapshots: dict[str, dict[str, MarketSnapshot]] = {}
        self._subscribers: list[Callable[[MarketSnapshot], None]] = []
        self._running = False
        self._refresh_interval: float = 60.0
        logger.info("MarketDataAdapter initialized")

    @property
    def supported_timeframes(self) -> tuple[str, ...]:
        return tuple(sorted(self._timeframes))

    def subscribe(self, callback: Callable[[MarketSnapshot], None]) -> None:
        self._subscribers.append(callback)

    def register_symbol(self, symbol: str) -> None:
        with self._lock:
            self._symbols.add(symbol)
            self._latest_snapshots[symbol] = {}

    def register_timeframe(self, tf: str) -> None:
        if tf in ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"):
            with self._lock:
                self._timeframes.add(tf)

    def get_snapshot(self, symbol: str, timeframe: str) -> MarketSnapshot | None:
        with self._lock:
            sym_data = self._latest_snapshots.get(symbol, {})
            return sym_data.get(timeframe)

    def get_latest_snapshots(self) -> dict[str, dict[str, MarketSnapshot]]:
        with self._lock:
            return {s: dict(tf) for s, tf in self._latest_snapshots.items()}

    def ingest_candles(
        self,
        symbol: str,
        timeframe: str,
        ohlcv: list[tuple[float, float, float, float, float, float]],
    ) -> MarketSnapshot:
        candles = tuple(
            Candle(ts, o, h, l, c, v) for ts, o, h, l, c, v in ohlcv
        )
        snapshot = MarketSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            timestamp=time.time(),
        )
        with self._lock:
            if symbol not in self._latest_snapshots:
                self._latest_snapshots[symbol] = {}
            self._latest_snapshots[symbol][timeframe] = snapshot
            self._symbols.add(symbol)
        for cb in self._subscribers:
            try:
                cb(snapshot)
            except Exception:
                logger.exception("MarketDataAdapter subscriber error")
        return snapshot

    def ingest_orderbook(
        self,
        symbol: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        ob = OrderBook(
            bids=tuple(OrderBookLevel(p, s) for p, s in bids),
            asks=tuple(OrderBookLevel(p, s) for p, s in asks),
        )
        spread = abs(bids[0][0] - asks[0][0]) if bids and asks else 0.0
        with self._lock:
            sym_data = self._latest_snapshots.get(symbol, {})
            for tf in list(sym_data):
                old = sym_data[tf]
                updated = MarketSnapshot(
                    symbol=old.symbol,
                    timeframe=old.timeframe,
                    candles=old.candles,
                    orderbook=ob,
                    bid_ask_spread=spread,
                    vwap=old.vwap,
                    atr=old.atr,
                    realized_volatility=old.realized_volatility,
                    volume_profile=dict(old.volume_profile),
                    funding_rate=old.funding_rate,
                    open_interest=old.open_interest,
                    timestamp=time.time(),
                )
                sym_data[tf] = updated

    def get_symbols(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._symbols))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "symbols": list(self._symbols),
                "timeframes": list(self._timeframes),
                "snapshot_count": sum(len(tf) for tf in self._latest_snapshots.values()),
                "subscribers": len(self._subscribers),
            }


# ── Singleton ────────────────────────────────────────────────────────────────

_adapter: MarketDataAdapter | None = None
_adapter_lock = threading.Lock()


def get_market_data_adapter() -> MarketDataAdapter:
    global _adapter
    with _adapter_lock:
        if _adapter is None:
            _adapter = MarketDataAdapter()
    return _adapter