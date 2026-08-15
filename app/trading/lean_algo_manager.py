"""LeanAlgoManager — integrate LEAN algorithms into CloudRuntime."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class LeanAlgoManager:
    """Manage LEAN algorithm lifecycle from CloudRuntime.

    Responsibilities:
    - discover LEAN home
    - validate configuration
    - register with RuntimeManager
    - expose service health
    """

    def __init__(self, lean_home: Optional[str] = None) -> None:
        self.lean_home = lean_home or os.environ.get("LEAN_HOME", "")
        self._available = bool(self.lean_home and os.path.isdir(self.lean_home))

    def initialize(self) -> Dict[str, Any]:
        if not self._available:
            log.warning("[LeanAlgoManager] LEAN_HOME not found or invalid: %s", self.lean_home)
            return {"status": "degraded", "reason": "LEAN_HOME invalid"}
        log.info("[LeanAlgoManager] Initialized with LEAN_HOME=%s", self.lean_home)
        return {"status": "ok", "lean_home": self.lean_home}

    def get_health(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "lean_home": self.lean_home,
        }