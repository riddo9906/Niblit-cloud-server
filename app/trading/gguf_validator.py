"""Phase 5 — Local GGUF Validation.

Every TradeProposal must be validated by the locally loaded GGUF model.
The model evaluates structured data only — it never invents trades.

Output is strict JSON. No natural language. No free-form reasoning.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any

from .models import (
    GGUFValidation,
    MarketSnapshot,
    MarketState,
    PortfolioState,
    TradeProposal,
)

logger = logging.getLogger(__name__)

_VALIDATION_PROMPT_TEMPLATE = """You are a trade validation system. Evaluate the following trade proposal and respond with ONLY a JSON object. No explanation. No natural language.

Market Snapshot:
{symbol} ({timeframe})
Price: {price}
ATR: {atr}
Spread: {spread}
Volume: {volume}

Market State:
Regime: {regime}
Trend: {trend}
Momentum: {momentum}
Volatility: {volatility}
Confidence: {market_confidence}

Trade Proposal:
Side: {side}
Entry: {entry}
Stop: {stop}
Target: {target}
Risk/Reward: {risk_reward}
Proposal Confidence: {proposal_confidence}

Portfolio:
Equity: {equity}
Open Positions: {open_positions}
Current Exposure: {exposure}
Daily PnL: {daily_pnl}

Respond with JSON only:
{{"approve": true/false, "confidence": 0.0-1.0, "risk_level": "low/medium/high/critical", "warnings": [], "reason": "..."}}"""


class GGUFValidator:
    """Validates TradeProposals using the local GGUF model.

    The model receives structured market data and returns a structured
    JSON validation. No free-form reasoning. No trade invention.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._validation_history: list[GGUFValidation] = []
        self._max_history = 500
        self._model_manager = None
        logger.info("GGUFValidator initialized")

    def _get_model(self) -> Any:
        """Get the local GGUF model manager."""
        if self._model_manager is None:
            try:
                from app.main import ModelManager
                from app.config import get_config
                cfg = get_config()
                self._model_manager = ModelManager(cfg.model_map, cfg.default_model, config=cfg)
            except Exception as exc:
                logger.warning("GGUFValidator: cannot access model: %s", exc)
                return None
        return self._model_manager

    def validate(
        self,
        proposal: TradeProposal,
        snapshot: MarketSnapshot | None = None,
        state: MarketState | None = None,
        portfolio: PortfolioState | None = None,
    ) -> GGUFValidation:
        """Validate a TradeProposal through the local GGUF model.

        Returns a structured GGUFValidation. If the model is unavailable,
        returns a default validation with reduced confidence.
        """
        model = self._get_model()
        if model is None:
            result = GGUFValidation(
                approved=True,
                confidence=0.5,
                risk_level="medium",
                warnings=("local_model_unavailable",),
                reason="Model unavailable — default approval with reduced confidence",
            )
            self._store(result)
            return result

        prompt = self._build_prompt(proposal, snapshot, state, portfolio)

        try:
            response = model.chat(
                model_id=model.get_active_model_id() or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            raw = response.text.strip()
            # Extract JSON from response
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            parsed = json.loads(raw)
            result = GGUFValidation(
                approved=bool(parsed.get("approve", False)),
                confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
                risk_level=str(parsed.get("risk_level", "medium")),
                warnings=tuple(str(w) for w in parsed.get("warnings", [])),
                reason=str(parsed.get("reason", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("GGUFValidator: parse error: %s", exc)
            result = GGUFValidation(
                approved=True,
                confidence=0.4,
                risk_level="medium",
                warnings=("validation_parse_error", str(exc)),
                reason="Validation response could not be parsed",
            )
        except Exception as exc:
            logger.error("GGUFValidator: inference error: %s", exc)
            result = GGUFValidation(
                approved=True,
                confidence=0.3,
                risk_level="medium",
                warnings=("validation_inference_error", str(exc)),
                reason="Validation inference failed",
            )

        self._store(result)
        return result

    def get_history(self, limit: int = 50) -> list[GGUFValidation]:
        with self._lock:
            return self._validation_history[-limit:]

    def _build_prompt(
        self,
        proposal: TradeProposal,
        snapshot: MarketSnapshot | None,
        state: MarketState | None,
        portfolio: PortfolioState | None,
    ) -> str:
        """Build a structured prompt for the GGUF model."""
        price = snapshot.latest_price if snapshot else 0.0
        atr = snapshot.atr if snapshot else 0.0
        spread = snapshot.bid_ask_spread if snapshot else 0.0
        volume = snapshot.latest_volume if snapshot else 0.0
        regime = state.regime if state else "unknown"
        trend = state.trend if state else "neutral"
        momentum = state.momentum if state else "neutral"
        volatility = state.volatility if state else "unknown"
        market_conf = state.confidence if state else 0.5
        equity = portfolio.total_equity if portfolio else 0.0
        open_pos = portfolio.open_positions if portfolio else 0
        exposure = portfolio.current_exposure if portfolio else 0.0
        daily_pnl = portfolio.daily_pnl if portfolio else 0.0

        return _VALIDATION_PROMPT_TEMPLATE.format(
            symbol=proposal.symbol,
            timeframe="1h",
            price=price,
            atr=atr,
            spread=spread,
            volume=volume,
            regime=regime,
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            market_confidence=market_conf,
            side=proposal.side,
            entry=proposal.entry_price,
            stop=proposal.stop_loss,
            target=proposal.take_profit,
            risk_reward=proposal.expected_risk_reward,
            proposal_confidence=proposal.confidence,
            equity=equity,
            open_positions=open_pos,
            exposure=exposure,
            daily_pnl=daily_pnl,
        )

    def _store(self, result: GGUFValidation) -> None:
        with self._lock:
            self._validation_history.append(result)
            if len(self._validation_history) > self._max_history:
                self._validation_history = self._validation_history[-self._max_history:]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "validation_count": len(self._validation_history),
                "model_available": self._model_manager is not None,
            }


# ── Singleton ────────────────────────────────────────────────────────────────

_validator: GGUFValidator | None = None
_validator_lock = threading.Lock()


def get_gguf_validator() -> GGUFValidator:
    global _validator
    with _validator_lock:
        if _validator is None:
            _validator = GGUFValidator()
    return _validator