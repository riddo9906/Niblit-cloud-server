"""Request classification — lightweight heuristic classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CODING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(function|class|def |import |const |let |var |=>|```)",
        r"\b(debug|refactor|implement|fix bug|unit test|typescript|python|rust)\b",
        r"\b(code|syntax|compile|linter|stack trace|error on line)\b",
    )
)

_REASONING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(why|explain|analyze|compare|evaluate|prove|reason|think step)\b",
        r"\b(trade-?off|pros and cons|implications|consequence)\b",
        r"\b(plan|strategy|architecture|design)\b",
    )
)

_MEMORY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(remember|recall|previous|earlier|last time|you said|context from)\b",
        r"\b(memory|history|conversation log|what did we)\b",
    )
)

_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\btool_calls?\b",
        r"\b(function_call|invoke tool|run tool|use the .+ tool)\b",
    )
)

_PRIORITY_BY_TYPE: dict[str, str] = {
    "coding": "high",
    "reasoning": "high",
    "tool_usage": "medium",
    "memory_lookup": "low",
    "general_chat": "low",
}


@dataclass(frozen=True)
class ClassificationResult:
    request_type: str
    compute_priority: str
    signals: tuple[str, ...] = ()


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role", "")).lower() in ("user", "developer"):
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join(parts)
    return ""


def _has_tool_signals(messages: list[dict[str, Any]], payload: Any) -> bool:
    if getattr(payload, "tools", None):
        return True
    if getattr(payload, "tool_choice", None):
        return True
    for msg in messages:
        if msg.get("tool_calls"):
            return True
        if str(msg.get("role", "")).lower() == "tool":
            return True
    return False


def _score_patterns(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for pattern in patterns if pattern.search(text))


def classify_request(messages: list[dict[str, Any]], payload: Any) -> ClassificationResult:
    """Classify inbound chat request type and compute priority."""
    if _has_tool_signals(messages, payload):
        return ClassificationResult(
            request_type="tool_usage",
            compute_priority=_PRIORITY_BY_TYPE["tool_usage"],
            signals=("tool_payload",),
        )

    text = _last_user_text(messages)
    if not text.strip():
        return ClassificationResult(
            request_type="general_chat",
            compute_priority=_PRIORITY_BY_TYPE["general_chat"],
        )

    scores = {
        "coding": _score_patterns(text, _CODING_PATTERNS),
        "reasoning": _score_patterns(text, _REASONING_PATTERNS),
        "memory_lookup": _score_patterns(text, _MEMORY_PATTERNS),
        "tool_usage": _score_patterns(text, _TOOL_PATTERNS),
    }
    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] == 0:
        best_type = "general_chat"

    return ClassificationResult(
        request_type=best_type,
        compute_priority=_PRIORITY_BY_TYPE.get(best_type, "medium"),
        signals=tuple(k for k, v in scores.items() if v > 0),
    )
