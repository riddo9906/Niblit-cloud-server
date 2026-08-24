"""Deterministic small-agent task routing + bounded delegation.

The primary model (qwen2.5-coder-3b) may delegate LOW-COMPLEXITY subtasks to
the lightweight agents. Routing is a static table — no LLM-in-the-loop for
routing decisions.

Delegation limits (enforced by callers):
    max_depth = 1          (children may never delegate)
    bounded subtask count, token caps, input size
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.local_models import (
    CONTEXT_MODEL_ID,
    PRIMARY_MODEL_ID,
    SUMMARIZER_MODEL_ID,
)

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 1
MAX_SUBTASKS_PER_REQUEST = 4
MAX_DELEGATION_INPUT_CHARS = 24_000
MAX_DELEGATION_TOKENS = 512

# Deterministic task-type -> agent model mapping.
TASK_ROUTING: dict[str, str] = {
    "classification": CONTEXT_MODEL_ID,
    "tagging": CONTEXT_MODEL_ID,
    "entity_extraction": CONTEXT_MODEL_ID,
    "context_filtering": CONTEXT_MODEL_ID,
    "relevance_scoring": CONTEXT_MODEL_ID,
    "intent_classification": CONTEXT_MODEL_ID,
    "memory_tagging": CONTEXT_MODEL_ID,
    "short_summary": SUMMARIZER_MODEL_ID,
    "workspace_summary": SUMMARIZER_MODEL_ID,
    "retrieval_compression": SUMMARIZER_MODEL_ID,
    "conversation_compression": SUMMARIZER_MODEL_ID,
    "command_output_summary": SUMMARIZER_MODEL_ID,
    "activity_summary": SUMMARIZER_MODEL_ID,
    "coding": PRIMARY_MODEL_ID,
    "debugging": PRIMARY_MODEL_ID,
    "architecture": PRIMARY_MODEL_ID,
    "multi_file_reasoning": PRIMARY_MODEL_ID,
    "planning": PRIMARY_MODEL_ID,
    "synthesis": PRIMARY_MODEL_ID,
}


@dataclass
class DelegationResult:
    model_id: str
    task_type: str
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "ok"
    error: str = ""


def route_task(task_type: str) -> str:
    """Deterministically select the agent model for a task type."""
    return TASK_ROUTING.get(str(task_type or "").strip().lower(), PRIMARY_MODEL_ID)


def check_delegation_allowed(depth: int, subtask_count: int) -> str | None:
    """Return a rejection reason, or None when delegation is permitted."""
    if depth >= MAX_DELEGATION_DEPTH:
        return f"delegation depth {depth} exceeds limit {MAX_DELEGATION_DEPTH}"
    if subtask_count >= MAX_SUBTASKS_PER_REQUEST:
        return f"subtask count {subtask_count} exceeds limit {MAX_SUBTASKS_PER_REQUEST}"
    return None
