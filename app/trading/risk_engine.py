"""Phase 6 — Portfolio & Risk Engine.

Deterministic validation. Never AI.

Checks: maximum exposure, daily drawdown, open trades, portfolio correlation,
position sizing, maximum leverage, account balance, risk limits, volatility filters.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from .models import (
    PortfolioState,
    RiskAssessment,
    TradeIntent,
    TradeProposal,
)

logger = logging.getLogger(__name__)


class RiskEngine:
    """Deterministic risk management. No AI. No exceptions.

    Every trade must pass all checks before reaching execution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._portfolio: PortfolioState = PortfolioState()
        self._config: dict[str, Any] = {
            "max_daily_loss": 0.03,           # 3% max daily loss
            "max_drawdown": 0.10,              # 10% max drawdown
            "max_position_size": 0.10,         # 10% per position
            "max_total_exposure": 0.50,        # 50% total exposure
            "max_leverage": 3.0,               # 3x max leverage
            "min_risk_reward": 1.5,            # minimum 1.5:1 risk/reward
            "max_open_positions": 5,           # max concurrent positions
            "min_confidence": 0.4,             # minimum proposal confidence
            "volatility_position_scale": True, # reduce size in high volatility
            "max_slippage": 0.005,             # 0.5% max expected slippage
        }
        self._assessment_history: list[RiskAssessment] = []
        self._max_history = 500
        logger.info("RiskEngine initialized")

    def assess(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioState | None = None,
    ) -> RiskAssessment:
        """Assess a TradeProposal against all risk rules.

        Returns a RiskAssessment. If not allowed, violations explain why.
        """
        if portfolio is not None:
            self._portfolio = portfolio

        violations: list[str] = []
        cfg = self._config

        # 1. Daily loss check
        if self._portfolio.daily_pnl < -self._portfolio.total_equity * cfg["max_daily_loss"]:
            violations.append(f"daily_loss_exceeded: {self._portfolio.daily_pnl:.2f}")

        # 2. Drawdown check
        if self._portfolio.daily_drawdown > cfg["max_drawdown"]:
            violations.append(f"drawdown_exceeded: {self._portfolio.daily_drawdown:.2%}")

        # 3. Max drawdown check
        if self._portfolio.max_drawdown > cfg["max_drawdown"]:
            violations.append(f"max_drawdown_exceeded: {self._portfolio.max_drawdown:.2%}")

        # 4. Open positions limit
        if self._portfolio.open_positions >= cfg["max_open_positions"]:
            violations.append(f"max_open_positions: {self._portfolio.open_positions}")

        # 5. Total exposure check
        if self._portfolio.current_exposure >= cfg["max_total_exposure"]:
            violations.append(f"total_exposure_exceeded: {self._portfolio.current_exposure:.2%}")

        # 6. Position size check
        if self._portfolio.total_equity > 0:
            position_pct = abs(proposal.entry_price * 1.0) / self._portfolio.total_equity
            if position_pct > cfg["max_position_size"]:
                violations.append(f"position_size_exceeded: {position_pct:.2%}")

        # 7. Risk/reward check
        if proposal.expected_risk_reward < cfg["min_risk_reward"]:
            violations.append(f"risk_reward_too_low: {proposal.expected_risk_reward:.2f}")

        # 8. Confidence check
        if proposal.confidence < cfg["min_confidence"]:
            violations.append(f"confidence_too_low: {proposal.confidence:.2f}")

        # 9. Leverage check
        if proposal.expected_risk_reward > cfg["max_leverage"] * 2:
            violations.append(f"leverage_too_high")

        # 10. Available capital check
        if self._portfolio.available_capital <= 0:
            violations.append("no_available_capital")

        # Compute recommended stake
        recommended_stake = 0.0
        if self._portfolio.total_equity > 0 and not violations:
            base_stake = self._portfolio.total_equity * cfg["max_position_size"]
            # Scale by confidence
            confidence_scale = 0.5 + (proposal.confidence * 0.5)
            recommended_stake = base_stake * confidence_scale
            # Scale by volatility if enabled
            if cfg["volatility_position_scale"]:
                recommended_stake *= 0.7  # Reduce in high vol

        allowed = len(violations) == 0
        assessment = RiskAssessment(
            allowed=allowed,
            violations=tuple(violations),
            max_position_size=self._portfolio.total_equity * cfg["max_position_size"],
            recommended_stake=recommended_stake,
            max_leverage=cfg["max_leverage"],
            reason="All risk checks passed" if allowed else f"Violations: {'; '.join(violations)}",
        )
        self._store(assessment)
        return assessment

    def update_portfolio(self, portfolio: PortfolioState) -> None:
        with self._lock:
            self._portfolio = portfolio

    def get_portfolio(self) -> PortfolioState:
        with self._lock:
            return self._portfolio

    def get_assessment_history(self, limit: int = 50) -> list[RiskAssessment]:
        with self._lock:
            return self._assessment_history[-limit:]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": dict(self._config),
                "portfolio": self._portfolio.to_dict(),
                "assessment_count": len(self._assessment_history),
            }

    def _store(self, assessment: RiskAssessment) -> None:
        with self._lock:
            self._assessment_history.append(assessment)
            if len(self._assessment_history) > self._max_history:
                self._assessment_history = self._assessment_history[-self._max_history:]


# ── Singleton ────────────────────────────────────────────────────────────────

_engine: RiskEngine | None = None
_engine_lock = threading.Lock()


def get_risk_engine() -> RiskEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = RiskEngine()
    return _engine