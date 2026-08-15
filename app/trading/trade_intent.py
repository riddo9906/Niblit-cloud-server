"""Phase 7 — Trade Intent.

Converts validated trades into immutable TradeIntent objects.
TradeIntent is the ONLY object allowed to reach execution.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from .models import (
    GGUFValidation,
    RiskAssessment,
    TradeIntent,
    TradeProposal,
)

logger = logging.getLogger(__name__)


class TradeIntentFactory:
    """Creates immutable TradeIntent objects from validated pipeline results.

    TradeIntent is the single source of truth for execution.
    No trade reaches Freqtrade without a signed TradeIntent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._intent_history: list[TradeIntent] = []
        self._max_history = 500
        logger.info("TradeIntentFactory initialized")

    def create(
        self,
        proposal: TradeProposal,
        validation: GGUFValidation,
        risk: RiskAssessment,
        quantity: float | None = None,
        approved_override: bool | None = None,
    ) -> TradeIntent:
        """Create a TradeIntent from validated pipeline outputs.

        Args:
            proposal: The original TradeProposal
            validation: GGUF model validation result
            risk: Deterministic risk assessment
            quantity: Override quantity (defaults to risk.recommended_stake / entry_price)
            approved_override: Force approval status

        Returns:
            An immutable TradeIntent

        Raises:
            ValueError: If trade is not approved by any required stage
        """
        approved = (
            approved_override
            if approved_override is not None
            else (validation.approved and risk.allowed)
        )

        if not approved:
            logger.warning(
                "TradeIntent rejected: approved=%s validation=%s risk=%s",
                approved, validation.approved, risk.allowed,
            )

        # Compute quantity from recommended stake if not provided
        if quantity is None and proposal.entry_price > 0:
            if risk.recommended_stake > 0:
                quantity = risk.recommended_stake / proposal.entry_price
            else:
                quantity = 0.0

        intent = TradeIntent(
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=max(0.0, quantity or 0.0),
            entry_price=proposal.entry_price,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            confidence=min(validation.confidence, proposal.confidence),
            approved=approved,
            proposal_id=uuid.uuid4().hex[:12],
            validation_id=uuid.uuid4().hex[:12],
            risk_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
        )

        self._store(intent)
        return intent

    def get_history(self, limit: int = 50) -> list[TradeIntent]:
        with self._lock:
            return self._intent_history[-limit:]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "intent_count": len(self._intent_history),
                "recent_intents": [
                    {"symbol": i.symbol, "side": i.side, "approved": i.approved}
                    for i in self._intent_history[-10:]
                ],
            }

    def _store(self, intent: TradeIntent) -> None:
        with self._lock:
            self._intent_history.append(intent)
            if len(self._intent_history) > self._max_history:
                self._intent_history = self._intent_history[-self._max_history:]


# ── Singleton ────────────────────────────────────────────────────────────────

_factory: TradeIntentFactory | None = None
_factory_lock = threading.Lock()


def get_trade_intent_factory() -> TradeIntentFactory:
    global _factory
    with _factory_lock:
        if _factory is None:
            _factory = TradeIntentFactory()
    return _factory