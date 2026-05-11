#!/usr/bin/env python3
"""app/temporal_sync.py — Niblit Cognitive Cloud Runtime Temporal Coherence.

Tracks epoch state, synchronization status, request ordering, and coherence
lag for the cloud runtime.  Aligned with the Temporal Coherence Layer in the
main Niblit repo (Phase Ω.5, ``modules/causal_temporal_engine.py``).

Design
------
- Epoch counter increments on every request cycle.
- Coherence score is a rolling EMA of per-request coherence inputs.
- Drift is tracked as the delta between requested epoch and current epoch.
- Sync status transitions: ``unsynced`` → ``syncing`` → ``synced`` → ``drifted``.

Configuration (env vars)
------------------------
    NIBLIT_TS_ENABLED     — "0" to disable (default 1)
    NIBLIT_TS_EMA_ALPHA   — EMA smoothing factor for coherence (default 0.1)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitTemporalSync")

_ENABLED: bool = os.getenv("NIBLIT_TS_ENABLED", "1").strip() not in ("0", "false")
_EMA_ALPHA: float = float(os.getenv("NIBLIT_TS_EMA_ALPHA", "0.1"))

_SYNC_STATES = frozenset({"unsynced", "syncing", "synced", "drifted"})


@dataclass
class EpochState:
    """Current epoch and coherence information."""
    epoch_id: int
    sync_status: str      # unsynced | syncing | synced | drifted
    coherence_ema: float  # rolling EMA of coherence scores
    coherence_lag: float  # absolute drift from last known good coherence
    drift_epochs: int     # how many epochs behind/ahead vs last sync
    last_sync_ts: float   # UNIX timestamp of last sync
    request_count: int    # total requests processed in this epoch

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "sync_status": self.sync_status,
            "coherence_ema": round(self.coherence_ema, 4),
            "coherence_lag": round(self.coherence_lag, 4),
            "drift_epochs": self.drift_epochs,
            "last_sync_ts": self.last_sync_ts,
            "request_count": self.request_count,
        }


class TemporalSync:
    """Epoch tracking and temporal coherence manager.

    Thread-safe singleton.  Each ``record_request`` call contributes coherence
    data; ``sync_epoch`` advances the epoch.  The ``status`` method returns
    a full snapshot of the current temporal state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch_id: int = int(time.time())
        self._sync_status: str = "unsynced"
        self._coherence_ema: float = 1.0
        self._last_sync_ts: float = time.time()
        self._request_count: int = 0
        self._total_request_count: int = 0
        self._epoch_history: list[dict[str, Any]] = []
        log.debug("[TemporalSync] initialised epoch=%d", self._epoch_id)

    def record_request(self, coherence: float = 1.0, epoch_tag: str = "") -> None:
        """Record a request's coherence contribution.

        Args:
            coherence:  Coherence score from the request envelope (0.0–1.0).
            epoch_tag:  Optional epoch identifier from the request envelope.
        """
        if not _ENABLED:
            return
        coherence = max(0.0, min(1.0, float(coherence)))
        with self._lock:
            self._coherence_ema = (
                _EMA_ALPHA * coherence + (1 - _EMA_ALPHA) * self._coherence_ema
            )
            self._request_count += 1
            self._total_request_count += 1

            # Detect drift: if envelope supplies an epoch_tag and it diverges
            # significantly from the current epoch, mark as drifted.
            if epoch_tag:
                try:
                    # epoch_tag format: "epoch_<int>" or arbitrary string
                    parts = epoch_tag.rsplit("_", 1)
                    req_epoch = int(parts[-1]) if parts[-1].isdigit() else None
                    if req_epoch is not None:
                        delta = abs(req_epoch - self._epoch_id)
                        if delta > 3600 and self._sync_status == "synced":
                            self._sync_status = "drifted"
                            log.warning(
                                "[TemporalSync] epoch drift detected: request=%d current=%d",
                                req_epoch,
                                self._epoch_id,
                            )
                except (ValueError, IndexError):
                    pass

            # Auto-advance to synced after first few requests
            if self._sync_status == "unsynced" and self._total_request_count >= 1:
                self._sync_status = "syncing"
            if self._sync_status == "syncing" and self._total_request_count >= 3:
                self._sync_status = "synced"
                self._last_sync_ts = time.time()

        self._emit()

    def sync_epoch(self, new_epoch_id: int | None = None) -> EpochState:
        """Advance to a new epoch, resetting per-epoch counters.

        Args:
            new_epoch_id: Target epoch ID.  If None, auto-increments.

        Returns:
            :class:`EpochState` snapshot after the sync.
        """
        with self._lock:
            if new_epoch_id is not None:
                drift = abs(new_epoch_id - self._epoch_id)
            else:
                drift = 0
                new_epoch_id = int(time.time())

            # Archive current epoch
            self._epoch_history.append({
                "epoch_id": self._epoch_id,
                "requests": self._request_count,
                "coherence_ema": self._coherence_ema,
                "closed_ts": time.time(),
            })
            if len(self._epoch_history) > 100:
                self._epoch_history = self._epoch_history[-100:]

            self._epoch_id = new_epoch_id
            self._request_count = 0
            self._last_sync_ts = time.time()
            self._sync_status = "synced"
            coherence_lag = max(0.0, 1.0 - self._coherence_ema)
            state = EpochState(
                epoch_id=self._epoch_id,
                sync_status=self._sync_status,
                coherence_ema=self._coherence_ema,
                coherence_lag=coherence_lag,
                drift_epochs=drift,
                last_sync_ts=self._last_sync_ts,
                request_count=self._request_count,
            )

        log.info("[TemporalSync] epoch synced → %d (drift=%d)", new_epoch_id, drift)
        self._emit()
        return state

    def current_epoch(self) -> int:
        """Return the current epoch ID."""
        with self._lock:
            return self._epoch_id

    def coherence(self) -> float:
        """Return the current coherence EMA."""
        with self._lock:
            return self._coherence_ema

    def status(self) -> dict[str, Any]:
        """Return full temporal status snapshot."""
        with self._lock:
            return {
                "enabled": _ENABLED,
                "epoch_id": self._epoch_id,
                "sync_status": self._sync_status,
                "coherence_ema": round(self._coherence_ema, 4),
                "coherence_lag": round(max(0.0, 1.0 - self._coherence_ema), 4),
                "last_sync_ts": self._last_sync_ts,
                "request_count": self._request_count,
                "total_request_count": self._total_request_count,
                "epoch_history_count": len(self._epoch_history),
            }

    def _emit(self) -> None:
        try:
            from app.event_bus import EVENT_EPOCH_SYNCED, CloudEvent, get_event_bus

            get_event_bus().publish(
                CloudEvent(
                    type=EVENT_EPOCH_SYNCED,
                    source="temporal_sync",
                    payload={
                        "epoch_id": self._epoch_id,
                        "sync_status": self._sync_status,
                        "coherence_ema": round(self._coherence_ema, 4),
                    },
                )
            )
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_ts: TemporalSync | None = None
_ts_lock = threading.Lock()


def get_temporal_sync() -> TemporalSync:
    """Return the process-level :class:`TemporalSync` singleton."""
    global _ts  # pylint: disable=global-statement
    with _ts_lock:
        if _ts is None:
            _ts = TemporalSync()
    return _ts
