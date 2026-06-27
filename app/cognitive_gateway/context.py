"""Per-request state carried through the Cognitive Gateway pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatewayContext:
    """Mutable request-scoped context — created per request, not shared."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_type: str = "general_chat"
    compute_priority: str = "medium"
    requested_model: str | None = None
    routed_model: str | None = None
    routing_source: str = "default"
    fallback_used: bool = False
    stream: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    memory_context: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    stream_started_at: float | None = None
    response_model: str | None = None
    error: str | None = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def stream_duration_ms(self) -> float | None:
        if self.stream_started_at is None:
            return None
        return (time.perf_counter() - self.stream_started_at) * 1000

    def to_hook_context(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_type": self.request_type,
            "compute_priority": self.compute_priority,
            "requested_model": self.requested_model,
            "routed_model": self.routed_model,
            "routing_source": self.routing_source,
            "fallback_used": self.fallback_used,
            "stream": self.stream,
            "messages": self.messages,
            "memory_context": dict(self.memory_context),
        }
