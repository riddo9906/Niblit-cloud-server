"""Phase 8 — Freqtrade Adapter.

Pure translation layer between TradeIntent and Freqtrade RPC/API.
No strategy logic. No AI. No decision-making.

Converts TradeIntent → Freqtrade commands.
Monitors open trades.
Receives execution feedback.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable

from .models import ExecutionResult, TradeIntent

logger = logging.getLogger(__name__)


class FreqtradeAdapter:
    """Adapter between Niblit TradeIntents and Freqtrade execution.

    Responsibilities:
    - Convert TradeIntent to Freqtrade RPC commands
    - Monitor open trades via Freqtrade API
    - Receive execution feedback (fills, cancellations, stop-loss, take-profit)
    - Report trade results back to the pipeline

    No strategy logic. Pure translation layer.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._freqtrade_url: str = os.getenv("FREQTRADE_RPC_URL", "http://localhost:8080")
        self._freqtrade_username: str = os.getenv("FREQTRADE_RPC_USERNAME", "")
        self._freqtrade_password: str = os.getenv("FREQTRADE_RPC_PASSWORD", "")
        self._enabled: bool = os.getenv("FREQTRADE_ENABLED", "0").strip() in ("1", "true")
        self._open_trades: dict[str, dict[str, Any]] = {}
        self._execution_history: list[ExecutionResult] = []
        self._max_history = 500
        self._subscribers: list[Callable[[ExecutionResult], None]] = []
        self._last_sync_ts: float = 0.0
        logger.info(
            "FreqtradeAdapter initialized (url=%s, enabled=%s)",
            self._freqtrade_url, self._enabled,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def subscribe(self, callback: Callable[[ExecutionResult], None]) -> None:
        self._subscribers.append(callback)

    def execute(self, intent: TradeIntent) -> dict[str, Any]:
        """Legacy proposals are analysis, never transferable Core approval."""
        return {"status":"execution_blocked", "reason":"core_human_ticket_required",
                "execution_authorized":False}

    def confirm_core_ticket(self, confirmation, desk_token):
        """Delegate an existing human ticket to its Core owner, never a broker.

        No approval issuer, ticket minting, broker credentials, cache or retry is
        provided. Core atomically verifies binding, expiry, account/mode, risk,
        current human consent and durable idempotency before any submission.
        """
        def blocked(reason):
            return {"status":"execution_blocked","reason":reason,"execution_authorized":False}
        if not self._enabled:
            return blocked("cloud_execution_disabled")
        try:
            # Freeze the request before checking/sending; reject non-JSON/NaN.
            confirmation=json.loads(json.dumps(confirmation,allow_nan=False))
            if not isinstance(confirmation,dict) or set(confirmation)!={
                    "ticket","acknowledgement","mode","intent","identity"}:
                return blocked("malformed_core_confirmation")
            if (confirmation["mode"]!="dry-run" or
                    not isinstance(confirmation["identity"],dict) or
                    confirmation["identity"].get("dry_run") is not True):
                return blocked("explicit_non_live_mode_required")
            intent=confirmation["intent"]
            if (not isinstance(intent,dict) or intent.get("action") not in ("enter_long","exit")
                    or not isinstance(intent.get("pair"),str)):
                return blocked("unsupported_core_operation")
            if confirmation["acknowledgement"]!="CONFIRM_SIMULATED_ORDER":
                return blocked("human_acknowledgement_required")
            import re
            if (not isinstance(confirmation["ticket"],str)
                    or re.fullmatch(r"[A-Za-z0-9_-]{32}",confirmation["ticket"]) is None
                    or not isinstance(desk_token,str)
                    or re.fullmatch(r"[A-Za-z0-9_-]{32,128}",desk_token) is None):
                return blocked("core_human_authorization_required")
        except (TypeError,ValueError):
            return blocked("malformed_core_confirmation")

        try:
            import httpx
            # Fixed existing Core-owned human gateway. Never follow redirects or
            # environment proxies with the human capability header.
            port=int(os.getenv("NIBLIT_API_PORT","5000"))
            if not 1 <= port <= 65535:
                return blocked("core_endpoint_unverifiable")
            with httpx.Client(base_url=f"http://127.0.0.1:{port}",trust_env=False,
                              follow_redirects=False,timeout=httpx.Timeout(20,connect=2)) as client:
                response=client.post("/api/trading/cloud/confirm",json=confirmation,headers={
                    "x-core-desk-token":desk_token})
            if 400 <= response.status_code < 500:
                return blocked("core_denied_or_unverifiable")
            # Redirects, 5xx, bad JSON and transport errors are never accepted
            # and never automatically resent. Delivery may already have begun.
            if response.status_code!=200:
                return {"status":"unknown","reason":"core_confirmation_unverified",
                        "execution_authorized":False,"retry_allowed":False}
            body=response.json()
            if body.get("mode")!="dry-run" or body.get("authority")!="core_human_ticket":
                raise ValueError("Unverified Core confirmation")
            state=body.get("state")
            if state in ("submitted","reconciled"):
                return {"status":state,"core_receipt":body,"execution_authorized":True,
                        "notice":"Core submission evidence; not proof of a broker fill."}
            if state in ("rejected","revoked","denied"):
                return blocked("core_rejected")
            return {"status":"unknown","core_receipt":body,
                    "execution_authorized":False,"retry_allowed":False}
        except Exception as exc:
            return {"status":"unknown","reason":"core_unavailable_or_delivery_uncertain",
                    "error_type":type(exc).__name__,"execution_authorized":False,"retry_allowed":False}

    def sync_open_trades(self) -> list[dict[str, Any]]:
        """Sync open trades from Freqtrade."""
        if not self._enabled:
            return list(self._open_trades.values())

        try:
            import requests
            response = requests.get(
                f"{self._freqtrade_url}/api/v1/status",
                auth=(self._freqtrade_username, self._freqtrade_password)
                if self._freqtrade_username else None,
                timeout=30,
            )
            trades = response.json()
            with self._lock:
                for t in trades:
                    pair = t.get("pair", "")
                    self._open_trades[pair] = t
                self._last_sync_ts = time.time()
            return trades
        except Exception as exc:
            logger.warning("Freqtrade sync failed: %s", exc)
            return list(self._open_trades.values())

    def get_open_trades(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._open_trades)

    def record_execution(self, result: ExecutionResult) -> None:
        """Record an execution result and notify subscribers."""
        with self._lock:
            self._execution_history.append(result)
            if len(self._execution_history) > self._max_history:
                self._execution_history = self._execution_history[-self._max_history:]
            if result.symbol in self._open_trades:
                del self._open_trades[result.symbol]
        for cb in self._subscribers:
            try:
                cb(result)
            except Exception:
                logger.exception("FreqtradeAdapter subscriber error")

    def get_execution_history(self, limit: int = 50) -> list[ExecutionResult]:
        with self._lock:
            return self._execution_history[-limit:]

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "url": self._freqtrade_url,
                "open_trades": len(self._open_trades),
                "execution_count": len(self._execution_history),
                "last_sync_ts": self._last_sync_ts,
            }


# ── Singleton ────────────────────────────────────────────────────────────────

_adapter: FreqtradeAdapter | None = None
_adapter_lock = threading.Lock()


def get_freqtrade_adapter() -> FreqtradeAdapter:
    global _adapter
    with _adapter_lock:
        if _adapter is None:
            _adapter = FreqtradeAdapter()
    return _adapter
