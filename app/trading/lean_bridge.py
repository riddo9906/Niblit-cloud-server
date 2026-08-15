"""Phase 4 — Niblit-Lean-Algos Integration.

Bridges the Cloud Server trading pipeline with the niblit-lean-algos
ecosystem. Consumes signals from Lean strategies and converts them
into standardized TradeProposal objects.

Quantitative reasoning only. No execution.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from .models import MarketSnapshot, MarketState, TradeProposal

logger = logging.getLogger(__name__)


class LeanAlgosBridge:
    """Bridge to niblit-lean-algos quantitative strategies.

    Receives MarketSnapshot + MarketState + historical context,
    routes to Lean strategy engine, returns TradeProposals.

    Currently reads signal files from the Lean bridge filesystem.
    When fully integrated with QuantConnect, this will use the
    Lean REST API or signal bus.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signal_dir: str = os.getenv(
            "NIBLIT_LEAN_SIGNAL_DIR",
            os.path.join(os.getcwd(), ".niblit", "signals"),
        )
        self._last_signal_ts: dict[str, float] = {}
        self._proposal_history: list[TradeProposal] = []
        self._max_history = 500
        logger.info("LeanAlgosBridge initialized (signal_dir=%s)", self._signal_dir)

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        state: MarketState,
        portfolio_context: dict[str, Any] | None = None,
    ) -> list[TradeProposal]:
        """Evaluate current market conditions through Lean strategies.

        Returns a list of TradeProposals (usually 0 or 1 for live trading).
        """
        proposals: list[TradeProposal] = []

        # 1. Try to read live signal file from Lean bridge
        signal = self._read_signal_file(snapshot.symbol)
        if signal is not None:
            proposal = self._signal_to_proposal(signal, snapshot, state)
            proposals.append(proposal)

        # 2. Generate rule-based proposals based on market state
        rule_proposals = self._rule_based_proposals(snapshot, state)
        for rp in rule_proposals:
            # Avoid duplicates with signal file
            if not any(p.symbol == rp.symbol and p.side == rp.side for p in proposals):
                proposals.append(rp)

        # Store in history
        with self._lock:
            for p in proposals:
                self._proposal_history.append(p)
            if len(self._proposal_history) > self._max_history:
                self._proposal_history = self._proposal_history[-self._max_history:]

        return proposals

    def get_proposal_history(self, limit: int = 50) -> list[TradeProposal]:
        with self._lock:
            return self._proposal_history[-limit:]

    def _read_signal_file(self, symbol: str) -> dict[str, Any] | None:
        """Read a signal JSON file from the Lean bridge directory."""
        signal_path = os.path.join(self._signal_dir, f"{symbol.replace('/', '_')}.json")
        try:
            if not os.path.isfile(signal_path):
                return None
            with open(signal_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            age = time.time() - float(data.get("timestamp", 0))
            if age > 300:  # 5 minute max age
                return None
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _signal_to_proposal(
        self,
        signal: dict[str, Any],
        snapshot: MarketSnapshot,
        state: MarketState,
    ) -> TradeProposal:
        """Convert a Lean signal dict to a TradeProposal."""
        price = snapshot.latest_price or 0.0
        stop_distance = float(signal.get("stop_loss", 0.02))
        tp_distance = float(signal.get("take_profit", 0.06))
        return TradeProposal(
            symbol=signal.get("symbol", snapshot.symbol),
            side=str(signal.get("side", "long")),
            entry_price=float(signal.get("entry_price", price)),
            stop_loss=float(signal.get("stop_loss_price", price * (1 - stop_distance))),
            take_profit=float(signal.get("take_profit_price", price * (1 + tp_distance))),
            expected_duration=str(signal.get("duration", "intraday")),
            expected_risk_reward=float(signal.get("risk_reward", tp_distance / max(stop_distance, 0.001))),
            confidence=float(signal.get("confidence", 0.5)),
            rationale=str(signal.get("reason", "Lean signal")),
            algorithm=str(signal.get("algorithm", "lean")),
            timestamp=time.time(),
        )

    def _rule_based_proposals(
        self,
        snapshot: MarketSnapshot,
        state: MarketState,
    ) -> list[TradeProposal]:
        """Generate proposals based on MarketState analysis rules.

        These are simple quantitative rules, not AI decisions.
        They serve as a fallback when no Lean signal file is present.
        """
        proposals: list[TradeProposal] = []
        price = snapshot.latest_price
        if price is None or price <= 0:
            return proposals

        atr_value = snapshot.atr or (price * 0.01)
        min_confidence = 0.55

        if state.confidence < min_confidence:
            return proposals

        # Breakout proposal
        if state.structural_break and state.confidence > 0.6:
            side = "long" if state.market_bias == "long" else "short"
            stop_dist = atr_value * 1.5
            tp_dist = atr_value * 4.0
            proposals.append(TradeProposal(
                symbol=snapshot.symbol,
                side=side,
                entry_price=price,
                stop_loss=price - stop_dist if side == "long" else price + stop_dist,
                take_profit=price + tp_dist if side == "long" else price - tp_dist,
                expected_duration="swing",
                expected_risk_reward=4.0 / 1.5,
                confidence=state.confidence * 0.9,
                rationale=f"Structural break {state.regime} (rule-based)",
                algorithm="market_intelligence_rule",
            ))

        # Trend continuation proposal
        if state.trend_continuation and state.confidence > 0.65:
            side = "long" if state.trend == "bullish" else "short"
            stop_dist = atr_value * 2.0
            tp_dist = atr_value * 3.0
            rr = tp_dist / max(stop_dist, 0.001)
            if rr >= 1.5:
                proposals.append(TradeProposal(
                    symbol=snapshot.symbol,
                    side=side,
                    entry_price=price,
                    stop_loss=price - stop_dist if side == "long" else price + stop_dist,
                    take_profit=price + tp_dist if side == "long" else price - tp_dist,
                    expected_duration="swing",
                    expected_risk_reward=rr,
                    confidence=state.confidence * 0.85,
                    rationale=f"Trend continuation {state.trend} (rule-based)",
                    algorithm="market_intelligence_rule",
                ))

        # Mean reversion proposal
        if state.mean_reversion_signal and state.confidence > 0.5:
            side = "long" if state.market_bias == "short" else "short"
            stop_dist = atr_value * 1.0
            tp_dist = atr_value * 2.0
            proposals.append(TradeProposal(
                symbol=snapshot.symbol,
                side=side,
                entry_price=price,
                stop_loss=price - stop_dist if side == "long" else price + stop_dist,
                take_profit=price + tp_dist if side == "long" else price - tp_dist,
                expected_duration="intraday",
                expected_risk_reward=2.0,
                confidence=state.confidence * 0.75,
                rationale=f"Mean reversion signal on {state.regime} (rule-based)",
                algorithm="market_intelligence_rule",
            ))

        return proposals

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "signal_dir": self._signal_dir,
                "proposal_history_count": len(self._proposal_history),
            }


# ── Singleton ────────────────────────────────────────────────────────────────

_bridge: LeanAlgosBridge | None = None
_bridge_lock = threading.Lock()


def get_lean_bridge() -> LeanAlgosBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = LeanAlgosBridge()
    return _bridge