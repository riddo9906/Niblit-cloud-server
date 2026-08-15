"""Phase 8 — Freqtrade Adapter.

Pure translation layer between TradeIntent and Freqtrade RPC/API.
No strategy logic. No AI. No decision-making.

Converts TradeIntent → Freqtrade commands.
Monitors open trades.
Receives execution feedback.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable

from .models import ExecutionResult, TradeIntent

logger = logging.getLogger(__name__)


class FreqtradeAdapter:
    """Adapter between Niblit TradeIntents and Freqtrade execution.

    Responsibilities:
    - Convert TradeIntent to Freqtrade RPC commands
    - Monitor open trades via Freqtrade API
    - Receive execution feedback (fills, cancellations, stop-loss, take-profit)
    - Report trade results back to the pipeline

    No strategy logic. Pure translation layer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._freqtrade_url: str = os.getenv("FREQTRADE_RPC_URL", "http://localhost:8080")
        self._freqtrade_username: str = os.getenv("FREQTRADE_RPC_USERNAME", "")
        self._freqtrade_password: str = os.getenv("FREQTRADE_RPC_PASSWORD", "")
        self._enabled: bool = os.getenv("FREQTRADE_ENABLED", "0").strip() in ("1", "true")
        self._open_trades: dict[str, dict[str, Any]] = {}
        self._execution_history: list[ExecutionResult] = []
        self._max_history = 500
        self._subscribers: list[Callable[[ExecutionResult], None]] = []
        self._last_sync_ts: float = 0.0
        logger.info(
            "FreqtradeAdapter initialized (url=%s, enabled=%s)",
            self._freqtrade_url, self._enabled,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def subscribe(self, callback: Callable[[ExecutionResult], None]) -> None:
        self._subscribers.append(callback)

    def execute(self, intent: TradeIntent) -> dict[str, Any]:
        """Execute a TradeIntent through Freqtrade.

        Returns execution response dict with status and trade_id.
        """
        if not self._enabled:
            logger.info("Freqtrade disabled — trade %s/%s logged only", intent.symbol, intent.side)
            return {"status": "logged", "trade_id": f"dry-{uuid.uuid4().hex[:12]}", "simulated": True}

        if not intent.approved:
            logger.warning("Rejected unapproved trade: %s/%s", intent.symbol, intent.side)
            return {"status": "rejected", "reason": "not_approved"}

        # Build Freqtrade RPC payload
        payload = {
            "method": "forcebuy" if intent.side == "long" else "forcesell",
            "pair": intent.symbol,
            "side": intent.side,
            "amount": intent.quantity,
            "limit": intent.entry_price,
            "stoploss": intent.stop_loss,
            "takeprofit": intent.take_profit,
        }

        try:
            import requests
            response = requests.post(
                f"{self._freqtrade_url}/api/v1/forceentry",
                json=payload,
                auth=(self._freqtrade_username, self._freqtrade_password)
                if self._freqtrade_username else None,
                timeout=30,
            )
            result = response.json()
            trade_id = result.get("trade_id", f"ft-{uuid.uuid4().hex[:12]}")
            logger.info("Trade executed via Freqtrade: %s/%s id=%s", intent.symbol, intent.side, trade_id)
            return {"status": "executed", "trade_id": trade_id, "response": result}
        except Exception as exc:
            logger.error("Freqtrade execution failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

    def sync_open_trades(self) -> list[dict[str, Any]]:
        """Sync open trades from Freqtrade."""
        if not self._enabled:
            return list(self._open_trades.values())

        try:
            import requests
            response = requests.get(
                f"{self._freqtrade_url}/api/v1/status",
                auth=(self._freqtrade_username, self._freqtrade_password)
                if self._freqtrade_username else None,
                timeout=30,
            )
            trades = response.json()
            with self._lock:
                for t in trades:
                    pair = t.get("pair", "")
                    self._open_trades[pair] = t
                self._last_sync_ts = time.time()
            return trades
        except Exception as exc:
            logger.warning("Freqtrade sync failed: %s", exc)
            return list(self._open_trades.values())

    def get_open_trades(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._open_trades)

    def record_execution(self, result: ExecutionResult) -> None:
        """Record an execution result and notify subscribers."""
        with self._lock:
            self._execution_history.append(result)
            if len(self._execution_history) > self._max_history:
                self._execution_history = self._execution_history[-self._max_history:]
            if result.symbol in self._open_trades:
                del self._open_trades[result.symbol]
        for cb in self._subscribers:
            try:
                cb(result)
            except Exception:
                logger.exception("FreqtradeAdapter subscriber error")

    def get_execution_history(self, limit: int = 50) -> list[ExecutionResult]:
        with self._lock:
            return self._execution_history[-limit:]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "url": self._freqtrade_url,
                "open_trades": len(self._open_trades),
                "execution_count": len(self._execution_history),
                "last_sync_ts": self._last_sync_ts,
            }


# ── Singleton ────────────────────────────────────────────────────────────────

_adapter: FreqtradeAdapter | None = None
_adapter_lock = threading.Lock()


def get_freqtrade_adapter() -> FreqtradeAdapter:
    global _adapter
    with _adapter_lock:
        if _adapter is None:
            _adapter = FreqtradeAdapter()
    return _adapter