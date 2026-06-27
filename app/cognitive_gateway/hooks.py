"""Memory and event hooks for the Cognitive Gateway (structural skeleton)."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("niblit.cognitive_gateway.hooks")

GatewayEventHandler = Callable[[dict[str, Any]], None]
MemoryHookHandler = Callable[[dict[str, Any]], dict[str, Any] | None]


class MemoryHook:
    """Placeholder memory hook — inject/retrieve context later."""

    def before_request(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def after_response(self, context: dict[str, Any]) -> None:
        return None


class PlaceholderMemoryHook(MemoryHook):
    """No-op memory hook used when MEMORY_HOOK_ENABLED=0."""

    pass


class GatewayEventBus:
    """Synchronous, request-scoped event dispatch — no background workers."""

    EVENT_REQUEST_RECEIVED = "on_request_received"
    EVENT_MODEL_SELECTED = "on_model_selected"
    EVENT_RESPONSE_GENERATED = "on_response_generated"
    EVENT_ERROR = "on_error"

    def __init__(self) -> None:
        self._subscribers: dict[str, list[GatewayEventHandler]] = {}

    def subscribe(self, event: str, handler: GatewayEventHandler) -> None:
        if event not in self._subscribers:
            self._subscribers[event] = []
        if handler not in self._subscribers[event]:
            self._subscribers[event].append(handler)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        for handler in self._subscribers.get(event, ()):
            try:
                handler(payload)
            except Exception:
                logger.exception("gateway event handler failed: %s", event)


def logging_event_handler(event_name: str) -> GatewayEventHandler:
    def _handler(payload: dict[str, Any]) -> None:
        logger.debug("gateway event %s: %s", event_name, payload)

    return _handler
