"""Scheduler Expansion — recurring cognitive and trading jobs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class SchedulerExpansion:
    """Define recurring jobs for the runtime scheduler.

    Jobs:
    - market polling
    - portfolio updates
    - trade supervision
    - reflection
    - memory consolidation
    - strategy scoring
    - Freqtrade synchronization
    - portfolio reconciliation
    - runtime health verification
    - model registry refresh
    """

    def __init__(self) -> None:
        self._jobs: List[Dict[str, Any]] = []

    def register_defaults(self) -> List[Dict[str, Any]]:
        self._jobs = [
            {"name": "market_polling", "interval_seconds": 60, "enabled": True},
            {"name": "portfolio_updates", "interval_seconds": 120, "enabled": True},
            {"name": "trade_supervision", "interval_seconds": 10, "enabled": True},
            {"name": "reflection", "interval_seconds": 300, "enabled": True},
            {"name": "memory_consolidation", "interval_seconds": 600, "enabled": True},
            {"name": "strategy_scoring", "interval_seconds": 900, "enabled": True},
            {"name": "freqtrade_sync", "interval_seconds": 30, "enabled": True},
            {"name": "portfolio_reconciliation", "interval_seconds": 180, "enabled": True},
            {"name": "runtime_health", "interval_seconds": 15, "enabled": True},
            {"name": "model_registry_refresh", "interval_seconds": 3600, "enabled": True},
        ]
        return list(self._jobs)

    def get_jobs(self) -> List[Dict[str, Any]]:
        return list(self._jobs)

    def get_enabled(self) -> List[Dict[str, Any]]:
        return [job for job in self._jobs if job.get("enabled")]