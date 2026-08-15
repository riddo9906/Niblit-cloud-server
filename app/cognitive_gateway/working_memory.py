"""Working Memory — active cognitive context for decisions."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class WorkingMemory:
    """Short-lived active context for decision-making.

    Holds:
    - active objectives
    - recent context
    - current trade decisions
    - temporary state
    """

    def __init__(self, max_items: int = 100) -> None:
        self.max_items = max_items
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []
        self._objectives: List[str] = []

    def push(self, item: Dict[str, Any]) -> None:
        with self._lock:
            self._items.append(item)
            if len(self._items) > self.max_items:
                self._items = self._items[-self.max_items:]

    def set_objectives(self, objectives: List[str]) -> None:
        with self._lock:
            self._objectives = list(objectives)

    def get_context(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "objectives": list(self._objectives),
                "recent_items": list(self._items[-10:]),
            }