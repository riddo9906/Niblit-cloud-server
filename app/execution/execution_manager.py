"""ExecutionManager — runtime execution layer owned by LocalBrain.

This is a runtime interface only. No live trading execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class ExchangeManager:
    """Exchange interface — abstracts exchange connectivity."""

    def __init__(self) -> None:
        self.connected = False

    def start(self) -> Dict[str, Any]:
        log.info("[ExchangeManager] Starting (interface only)")
        return {"status": "interface"}

    def get_state(self) -> Dict[str, Any]:
        return {"connected": self.connected}


class MarketDataManager:
    """Market data interface."""

    def __init__(self) -> None:
        self.streaming = False

    def start(self) -> Dict[str, Any]:
        log.info("[MarketDataManager] Starting (interface only)")
        return {"status": "interface"}


class StrategyManager:
    """Strategy interface."""

    def __init__(self) -> None:
        self.active_strategies: list[str] = []

    def start(self) -> Dict[str, Any]:
        log.info("[StrategyManager] Starting (interface only)")
        return {"status": "interface"}


class RiskManager:
    """Risk manager interface."""

    def __init__(self) -> None:
        self.limits = {"max_drawdown": 0.2, "max_leverage": 3.0}

    def start(self) -> Dict[str, Any]:
        log.info("[RiskManager] Starting (interface only)")
        return {"status": "interface"}


class PortfolioManager:
    """Portfolio manager interface."""

    def __init__(self) -> None:
        self.portfolio: Dict[str, Any] = {}

    def start(self) -> Dict[str, Any]:
        log.info("[PortfolioManager] Starting (interface only)")
        return {"status": "interface"}


class TradeJournal:
    """Trade journal interface."""

    def __init__(self) -> None:
        self.trades: list[Dict[str, Any]] = []

    def start(self) -> Dict[str, Any]:
        log.info("[TradeJournal] Starting (interface only)")
        return {"status": "interface"}


class ExecutionManager:
    """ExecutionManager — orchestrates exchange, market data, strategy, risk, portfolio, journal.

    Owned by LocalBrain.  Created during LocalBrain.start().
    """

    def __init__(self) -> None:
        self.exchange = ExchangeManager()
        self.market_data = MarketDataManager()
        self.strategy = StrategyManager()
        self.risk = RiskManager()
        self.portfolio = PortfolioManager()
        self.journal = TradeJournal()
        self.ready = False

    def start(self) -> Dict[str, Any]:
        log.info("[ExecutionManager] Starting subsystems...")
        self.exchange.start()
        self.market_data.start()
        self.strategy.start()
        self.risk.start()
        self.portfolio.start()
        self.journal.start()
        self.ready = True
        log.info("[ExecutionManager] Ready (interface mode)")
        return {"status": "ready"}

    def get_state(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "exchange_connected": self.exchange.connected,
            "market_data_streaming": self.market_data.streaming,
            "active_strategies": len(self.strategy.active_strategies),
        }