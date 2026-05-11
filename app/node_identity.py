#!/usr/bin/env python3
"""app/node_identity.py — Niblit Cognitive Cloud Runtime Node Identity.

Maintains a stable node fingerprint, capability advertisement, and readiness
state for future distributed/swarm cognition participation.

This module only prepares the architecture — full federation is NOT implemented.
The API layer exposes ``/cluster/status``, ``/cluster/identity``, and
``/cluster/capabilities`` as documented stubs for future swarm coordination.

Configuration (env vars)
------------------------
    NIBLIT_NODE_ID       — override auto-generated node ID (optional)
    NIBLIT_NODE_REGION   — geographic region hint (optional)
    NIBLIT_NODE_ROLE     — node role (default: "inference")
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitNodeIdentity")

_NODE_ID: str = os.getenv(
    "NIBLIT_NODE_ID",
    hashlib.sha256(
        f"{socket.gethostname()}-{os.getpid()}".encode()
    ).hexdigest()[:16],
)
_NODE_REGION: str = os.getenv("NIBLIT_NODE_REGION", "local")
_NODE_ROLE: str = os.getenv("NIBLIT_NODE_ROLE", "inference")


@dataclass
class NodeCapabilities:
    """Advertised capabilities for this node."""
    max_models: int = 8
    supports_gguf: bool = True
    supports_cognitive_envelope: bool = True
    supports_trading_bridge: bool = True
    supports_governance: bool = True
    supports_temporal_sync: bool = True
    supports_reflection: bool = True
    swarm_ready: bool = False          # federation not yet implemented
    api_version: str = "omega.7"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_models": self.max_models,
            "supports_gguf": self.supports_gguf,
            "supports_cognitive_envelope": self.supports_cognitive_envelope,
            "supports_trading_bridge": self.supports_trading_bridge,
            "supports_governance": self.supports_governance,
            "supports_temporal_sync": self.supports_temporal_sync,
            "supports_reflection": self.supports_reflection,
            "swarm_ready": self.swarm_ready,
            "api_version": self.api_version,
        }


@dataclass
class NodeIdentitySnapshot:
    """Full node identity and health snapshot."""
    node_id: str
    region: str
    role: str
    started_ts: float
    capabilities: NodeCapabilities
    uptime_secs: float = 0.0
    request_count: int = 0
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "region": self.region,
            "role": self.role,
            "started_ts": self.started_ts,
            "uptime_secs": round(self.uptime_secs, 1),
            "request_count": self.request_count,
            "fingerprint": self.fingerprint,
            "capabilities": self.capabilities.to_dict(),
        }


class NodeIdentity:
    """Stable node identity for distributed cognition readiness.

    Thread-safe singleton.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_ts = time.time()
        self._node_id = _NODE_ID
        self._fingerprint = str(uuid.uuid4().hex[:12])
        self._request_count = 0
        self._capabilities = NodeCapabilities()
        log.info("[NodeIdentity] node_id=%s region=%s role=%s", _NODE_ID, _NODE_REGION, _NODE_ROLE)
        self._emit_init()

    def increment_request(self) -> None:
        with self._lock:
            self._request_count += 1

    def snapshot(self) -> NodeIdentitySnapshot:
        with self._lock:
            return NodeIdentitySnapshot(
                node_id=self._node_id,
                region=_NODE_REGION,
                role=_NODE_ROLE,
                started_ts=self._started_ts,
                uptime_secs=time.time() - self._started_ts,
                request_count=self._request_count,
                fingerprint=self._fingerprint,
                capabilities=self._capabilities,
            )

    def cluster_status(self) -> dict[str, Any]:
        """Return cluster status (single-node — federation not implemented)."""
        snap = self.snapshot()
        return {
            "status": "single_node",
            "federation_ready": False,
            "note": "Distributed swarm federation is not yet implemented. "
                    "This node is operating in standalone mode.",
            "node": snap.to_dict(),
        }

    def _emit_init(self) -> None:
        try:
            from app.event_bus import EVENT_NODE_IDENTITY_SET, CloudEvent, get_event_bus

            get_event_bus().publish(
                CloudEvent(
                    type=EVENT_NODE_IDENTITY_SET,
                    source="node_identity",
                    payload={"node_id": self._node_id, "region": _NODE_REGION},
                )
            )
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────

_identity: NodeIdentity | None = None
_identity_lock = threading.Lock()


def get_node_identity() -> NodeIdentity:
    """Return the process-level :class:`NodeIdentity` singleton."""
    global _identity  # pylint: disable=global-statement
    with _identity_lock:
        if _identity is None:
            _identity = NodeIdentity()
    return _identity
