"""Freqtrade Adapters — abstract Freqtrade completely.

No live execution.  Runtime interfaces only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class FreqtradeAdapter:
    """Primary Freqtrade adapter — abstracts the Freqtrade bot."""

    def __init__(self) -> None:
        self.connected = False
        self.bot_running = False

    def connect(self) -> Dict[str, Any]:
        log.info("[FreqtradeAdapter] Connect (interface)")
        return {"status": "interface"}

    def start_bot(self) -> Dict[str, Any]:
        log.info("[FreqtradeAdapter] Start bot (interface)")
        return {"status": "interface"}

    def stop_bot(self) -> Dict[str, Any]:
        log.info("[FreqtradeAdapter] Stop bot (interface)")
        return {"status": "interface"}

    def get_state(self) -> Dict[str, Any]:
        return {"connected": self.connected, "bot_running": self.bot_running}


class FreqtradeStrategyAdapter:
    """Freqtrade strategy adapter."""

    def __init__(self) -> None:
        self.strategies: List[str] = []

    def list_strategies(self) -> List[str]:
        return self.strategies


class FreqtradeExchangeAdapter:
    """Freqtrade exchange adapter."""

    def __init__(self) -> None:
        self.exchanges: List[str] = []

    def list_exchanges(self) -> List[str]:
        return self.exchanges


class FreqtradeDataAdapter:
    """Freqtrade data adapter."""

    def __init__(self) -> None:
        self.pairs: List[str] = []

    def list_pairs(self) -> List[str]:
        return self.pairs