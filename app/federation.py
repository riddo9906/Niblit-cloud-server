#!/usr/bin/env python3
"""Federation coordination foundation for Niblit cloud runtime.

This module remains deterministic and additive (no active networking) while
exposing readiness, topology, governance compatibility, and capability
negotiation metadata aligned with Ω.7/schema-v2 contracts.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("NiblitFederation")

_ENABLED = os.getenv("NIBLIT_FEDERATION_ENABLED", "0").strip().lower() not in {"", "0", "false"}
_REGISTRY = os.getenv("NIBLIT_FEDERATION_REGISTRY", "")
_MAX_PEERS = int(os.getenv("NIBLIT_FEDERATION_MAX_PEERS", "8"))
_HEARTBEAT_INTERVAL = int(os.getenv("NIBLIT_HEARTBEAT_INTERVAL", "30"))

_NOT_IMPLEMENTED_NOTE = (
    "Federation networking is not yet implemented; this foundation preserves "
    "coordination contracts for deterministic runtime orchestration."
)

_COMPATIBILITY = {
    "schema_version": "2.x",
    "event_contract_version": "omega-7",
    "governance_contract_version": "1.x",
    "advisor_protocol_version": "2.x",
    "runtime_mode_contract": "2026.05",
}

_CANONICAL_MODES = {"normal", "cautious", "survival", "lockdown"}


def _normalize_mode(mode: object) -> str:
    value = str(mode or "normal").strip().lower()
    if value in {"minimal", "constrained"}:
        value = "cautious"
    if value not in _CANONICAL_MODES:
        value = "normal"
    return value


@dataclass
class NodeRegistration:
    node_id: str
    region: str
    role: str
    base_url: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    epoch_id: int = 0
    coherence: float = 1.0
    governance_mode: str = "normal"
    registered_ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "region": self.region,
            "role": self.role,
            "base_url": self.base_url,
            "capabilities": self.capabilities,
            "epoch_id": self.epoch_id,
            "coherence": round(self.coherence, 4),
            "governance_mode": _normalize_mode(self.governance_mode),
            "registered_ts": self.registered_ts,
        }


@dataclass
class HeartbeatRecord:
    node_id: str
    epoch_id: int
    coherence: float
    runtime_health: float
    request_count: int
    attention_pressure: float
    governance_mode: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "epoch_id": self.epoch_id,
            "coherence": round(self.coherence, 4),
            "runtime_health": round(self.runtime_health, 4),
            "request_count": self.request_count,
            "attention_pressure": round(self.attention_pressure, 4),
            "governance_mode": _normalize_mode(self.governance_mode),
            "timestamp": self.timestamp,
        }


@dataclass
class ClusterPeer:
    node_id: str
    region: str
    role: str
    base_url: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    last_seen_ts: float = field(default_factory=time.time)
    trust_score: float = 1.0
    epoch_id: int = 0
    coherence: float = 1.0
    governance_mode: str = "normal"
    is_healthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "region": self.region,
            "role": self.role,
            "base_url": self.base_url,
            "capabilities": self.capabilities,
            "last_seen_ts": self.last_seen_ts,
            "trust_score": round(self.trust_score, 4),
            "epoch_id": self.epoch_id,
            "coherence": round(self.coherence, 4),
            "governance_mode": _normalize_mode(self.governance_mode),
            "is_healthy": self.is_healthy,
        }


@dataclass
class FederationState:
    enabled: bool
    status: str
    peer_count: int
    peers: list[ClusterPeer] = field(default_factory=list)
    registry_url: str = ""
    last_sync_ts: float = 0.0
    epoch_consensus: int = 0
    governance_sync: bool = False
    compatibility: dict[str, str] = field(default_factory=lambda: dict(_COMPATIBILITY))
    topology: dict[str, Any] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    note: str = _NOT_IMPLEMENTED_NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "peer_count": self.peer_count,
            "peers": [p.to_dict() for p in self.peers],
            "registry_url": self.registry_url,
            "last_sync_ts": self.last_sync_ts,
            "epoch_consensus": self.epoch_consensus,
            "governance_sync": self.governance_sync,
            "compatibility": self.compatibility,
            "topology": self.topology,
            "readiness": self.readiness,
            "note": self.note,
        }


class FederationManager:
    """Deterministic federation foundation manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peers: dict[str, ClusterPeer] = {}
        self._registration_count = 0
        self._heartbeat_count = 0
        self._governance_sync_count = 0
        self._epoch_sync_count = 0
        self._last_sync_ts = 0.0
        self._runtime_lineage_id = f"runtime-lineage-{int(time.time())}"
        self._node_capabilities = {
            "runtime_orchestration_authority": True,
            "governance_coordination": True,
            "schema_v2_runtime_bridge": True,
            "federation_foundation": True,
            "replay_safe_telemetry": True,
        }

    def _topology(self) -> dict[str, Any]:
        peer_count = len(self._peers)
        connectivity = "connected" if _ENABLED and peer_count > 0 else "standalone"
        if not _ENABLED:
            connectivity = "disabled"

        return {
            "node_count": 1 + peer_count,
            "peer_count": peer_count,
            "connectivity": connectivity,
            "runtime_lineage_id": self._runtime_lineage_id,
            "heartbeat_interval": _HEARTBEAT_INTERVAL,
            "max_peers": _MAX_PEERS,
            "registry": _REGISTRY,
        }

    def _readiness(self) -> dict[str, Any]:
        return {
            "federation_foundation_ready": True,
            "networking_ready": False,
            "governance_compatibility_ready": True,
            "topology_sync_contract_ready": True,
            "runtime_capability_negotiation_ready": True,
        }

    def _compatibility_summary(self) -> dict[str, Any]:
        incompatible: list[dict[str, Any]] = []
        for p in self._peers.values():
            peer_mode = _normalize_mode(p.governance_mode)
            if peer_mode not in _CANONICAL_MODES:
                incompatible.append({"node_id": p.node_id, "reason": "mode_contract_mismatch"})

        return {
            "compatible": len(incompatible) == 0,
            "contract": dict(_COMPATIBILITY),
            "incompatible_peers": incompatible,
        }

    def negotiate_capabilities(self, peer_capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
        peer = dict(peer_capabilities or {})
        missing = [k for k, v in self._node_capabilities.items() if v and not bool(peer.get(k, False))]
        return {
            "accepted": len(missing) == 0,
            "missing_capabilities": missing,
            "required_capabilities": dict(self._node_capabilities),
        }

    def state(self) -> FederationState:
        with self._lock:
            peers = list(self._peers.values())
            epoch_consensus = int(sum(p.epoch_id for p in peers) / len(peers)) if peers else 0
            return FederationState(
                enabled=_ENABLED,
                status="joined" if _ENABLED and peers else "standalone",
                peer_count=len(peers),
                peers=peers,
                registry_url=_REGISTRY,
                last_sync_ts=self._last_sync_ts,
                epoch_consensus=epoch_consensus,
                governance_sync=self._governance_sync_count > 0,
                compatibility=self._compatibility_summary(),
                topology=self._topology(),
                readiness=self._readiness(),
            )

    def register_peer(self, registration: NodeRegistration) -> dict[str, Any]:
        with self._lock:
            self._registration_count += 1
            peer = ClusterPeer(
                node_id=registration.node_id,
                region=registration.region,
                role=registration.role,
                base_url=registration.base_url,
                capabilities=dict(registration.capabilities),
                last_seen_ts=time.time(),
                epoch_id=registration.epoch_id,
                coherence=registration.coherence,
                governance_mode=_normalize_mode(registration.governance_mode),
                is_healthy=True,
            )
            self._peers[registration.node_id] = peer
            self._last_sync_ts = time.time()

            negotiation = self.negotiate_capabilities(peer.capabilities)
            return {
                "accepted": True,
                "registration": registration.to_dict(),
                "capability_negotiation": negotiation,
                "compatibility": self._compatibility_summary(),
                "note": _NOT_IMPLEMENTED_NOTE,
            }

    def record_heartbeat(self, heartbeat: HeartbeatRecord) -> dict[str, Any]:
        with self._lock:
            self._heartbeat_count += 1
            peer = self._peers.get(heartbeat.node_id)
            if peer:
                peer.last_seen_ts = heartbeat.timestamp
                peer.epoch_id = heartbeat.epoch_id
                peer.coherence = heartbeat.coherence
                peer.governance_mode = _normalize_mode(heartbeat.governance_mode)
                peer.is_healthy = heartbeat.runtime_health >= 0.4
            self._last_sync_ts = time.time()
        return {
            "acknowledged": True,
            "heartbeat": heartbeat.to_dict(),
            "topology": self._topology(),
            "note": _NOT_IMPLEMENTED_NOTE,
        }

    def list_peers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [p.to_dict() for p in self._peers.values()]

    def discover_peers(self) -> dict[str, Any]:
        return {
            "discovered": 0,
            "topology": self._topology(),
            "compatibility": self._compatibility_summary(),
            "note": _NOT_IMPLEMENTED_NOTE,
        }

    def sync_governance(self, governance_state: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._governance_sync_count += 1
            self._last_sync_ts = time.time()
            incoming_mode = _normalize_mode(governance_state.get("governance_mode", "normal"))
            return {
                "synced_to": len(self._peers),
                "governance_mode": incoming_mode,
                "governance_sync_count": self._governance_sync_count,
                "compatibility": self._compatibility_summary(),
                "topology": self._topology(),
                "note": _NOT_IMPLEMENTED_NOTE,
            }

    def sync_epoch(self, epoch_id: int) -> dict[str, Any]:
        with self._lock:
            self._epoch_sync_count += 1
            self._last_sync_ts = time.time()
            return {
                "synced_to": len(self._peers),
                "epoch_id": epoch_id,
                "epoch_sync_count": self._epoch_sync_count,
                "topology": self._topology(),
                "note": _NOT_IMPLEMENTED_NOTE,
            }

    def emit_heartbeat(self) -> dict[str, Any]:
        try:
            from app.attention_allocator import get_attention_allocator
            from app.node_identity import get_node_identity
            from app.temporal_sync import get_temporal_sync

            snap = get_node_identity().snapshot()
            ts = get_temporal_sync()
            aa = get_attention_allocator().status()
            hb = HeartbeatRecord(
                node_id=snap.node_id,
                epoch_id=ts.current_epoch(),
                coherence=ts.coherence(),
                runtime_health=1.0,
                request_count=snap.request_count,
                attention_pressure=float(aa.get("attention_pressure", 0.0)),
                governance_mode="normal",
            )
        except Exception:
            hb = HeartbeatRecord(
                node_id="unknown",
                epoch_id=0,
                coherence=1.0,
                runtime_health=1.0,
                request_count=0,
                attention_pressure=0.0,
                governance_mode="normal",
            )
        return self.record_heartbeat(hb)

    def status(self) -> dict[str, Any]:
        state = self.state().to_dict()
        state.update(
            {
                "registration_count": self._registration_count,
                "heartbeat_count": self._heartbeat_count,
                "governance_sync_count": self._governance_sync_count,
                "epoch_sync_count": self._epoch_sync_count,
                "runtime_lineage_id": self._runtime_lineage_id,
                "node_capabilities": dict(self._node_capabilities),
            }
        )
        return state


_fed: FederationManager | None = None
_fed_lock = threading.Lock()


def get_federation_manager() -> FederationManager:
    global _fed
    with _fed_lock:
        if _fed is None:
            _fed = FederationManager()
    return _fed
