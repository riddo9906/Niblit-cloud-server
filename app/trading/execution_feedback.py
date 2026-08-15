"""Phase 9 — Execution Feedback Loop.

Every completed trade returns to Niblit for memory storage and analysis.
Stores entry, exit, PnL, duration, slippage, market state, proposal,
GGUF validation, risk assessment, and final outcome.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable

from .models import ExecutionResult, MarketState, TradeProposal

logger = logging.getLogger(__name__)


class ExecutionFeedbackLoop:
    """Captures and stores every completed trade for future reference.

    Every trade result feeds back into:
    - Niblit memory (via event bus)
    - Market intelligence (pattern recognition)
    - Risk engine (portfolio updates)
    - Lean algorithms (strategy improvement)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: list[ExecutionResult] = []
        self._max_history = 1000
        self._subscribers: list[Callable[[ExecutionResult], None]] = []
        self._feedback_dir: str = os.getenv(
            "NIBLIT_FEEDBACK_DIR",
            os.path.join(os.getcwd(), ".niblit", "feedback"),
        )
        os.makedirs(self._feedback_dir, exist_ok=True)
        logger.info("ExecutionFeedbackLoop initialized (dir=%s)", self._feedback_dir)

    def subscribe(self, callback: Callable[[ExecutionResult], None]) -> None:
        self._subscribers.append(callback)

    def record(self, result: ExecutionResult) -> None:
        """Record a completed trade execution."""
        with self._lock:
            self._results.append(result)
            if len(self._results) > self._max_history:
                self._results = self._results[-self._max_history:]

        # Persist to disk
        self._persist(result)

        # Notify subscribers
        for cb in self._subscribers:
            try:
                cb(result)
            except Exception:
                logger.exception("FeedbackLoop subscriber error")

        # Emit to event bus
        self._emit_event(result)

        logger.info(
            "Trade recorded: %s/%s PnL=%.2f duration=%.0fs",
            result.symbol, result.side, result.pnl, result.duration_seconds,
        )

    def get_results(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[ExecutionResult]:
        with self._lock:
            if symbol:
                return [r for r in self._results if r.symbol == symbol][-limit:]
            return self._results[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            if not self._results:
                return {"total_trades": 0}
            total = len(self._results)
            winning = [r for r in self._results if r.pnl > 0]
            losing = [r for r in self._results if r.pnl < 0]
            total_pnl = sum(r.pnl for r in self._results)
            return {
                "total_trades": total,
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": len(winning) / total if total > 0 else 0.0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / total, 2) if total > 0 else 0.0,
                "avg_duration": sum(r.duration_seconds for r in self._results) / total if total > 0 else 0.0,
                "avg_slippage": sum(r.slippage for r in self._results) / total if total > 0 else 0.0,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_trades": len(self._results),
                "feedback_dir": self._feedback_dir,
                "subscribers": len(self._subscribers),
            }

    def _persist(self, result: ExecutionResult) -> None:
        """Persist execution result to disk as JSONL."""
        try:
            path = os.path.join(self._feedback_dir, f"{result.symbol.replace('/', '_')}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Feedback persistence failed: %s", exc)

    def _emit_event(self, result: ExecutionResult) -> None:
        """Emit trade completion event to the event bus."""
        try:
            from app.event_bus import (
                CloudEvent,
                get_event_bus,
            )
            get_event_bus().publish(
                CloudEvent(
                    type="trade.execution.completed",
                    source="execution_feedback_loop",
                    payload={
                        "trade_id": result.trade_id,
                        "symbol": result.symbol,
                        "side": result.side,
                        "pnl": result.pnl,
                        "duration_seconds": result.duration_seconds,
                    },
                )
            )
        except Exception:
            pass


# ── Singleton ────────────────────────────────────────────────────────────────

_loop: ExecutionFeedbackLoop | None = None
_loop_lock = threading.Lock()


def get_execution_feedback() -> ExecutionFeedbackLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = ExecutionFeedbackLoop()
    return _loop