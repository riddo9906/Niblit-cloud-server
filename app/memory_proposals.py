"""Memory proposal interface — preparation for the future memory phase.

Models NEVER write to Qdrant or any store directly. They may only emit a
structured, provenance-carrying MemoryProposal that a Niblit memory controller
validates before anything becomes durable. Persistence is intentionally out of
scope for this phase (no Qdrant writes here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_MEMORY_TYPES = {"episodic_summary", "fact_candidate", "workspace_note", "interaction_summary"}


@dataclass
class MemoryProposal:
    content: str
    memory_type: str = "episodic_summary"
    tags: list[str] = field(default_factory=list)
    # Provenance — proposals without source are rejected.
    source: str = ""
    model: str = ""
    request_id: str = ""
    timestamp: float = 0.0
    workspace_ref: str = ""
    confidence: float = 0.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.content.strip():
            errors.append("content_empty")
        if self.memory_type not in VALID_MEMORY_TYPES:
            errors.append(f"invalid_memory_type:{self.memory_type}")
        if not self.source:
            errors.append("missing_source")
        if not self.model:
            errors.append("missing_model")
        if not self.request_id:
            errors.append("missing_request_id")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append("confidence_out_of_range")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "tags": list(self.tags),
            "provenance": {
                "source": self.source,
                "model": self.model,
                "request_id": self.request_id,
                "timestamp": self.timestamp,
                "workspace_ref": self.workspace_ref,
                "confidence": self.confidence,
            },
        }