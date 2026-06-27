"""Cognitive Gateway orchestrator — classification, routing, hooks, observability."""

from __future__ import annotations

from typing import Any

from app.cognitive_gateway.classifier import classify_request
from app.cognitive_gateway.config import CognitiveGatewayConfig, load_cognitive_gateway_config
from app.cognitive_gateway.context import GatewayContext
from app.cognitive_gateway.hooks import (
    GatewayEventBus,
    MemoryHook,
    PlaceholderMemoryHook,
    logging_event_handler,
)
from app.cognitive_gateway.observability import (
    log_fallback_triggered,
    log_model_selected,
    log_request_classified,
    log_request_complete,
    log_upstream_error,
)
from app.cognitive_gateway.router import route_model


class CognitiveGateway:
    """Decision layer between external clients and inference handlers."""

    def __init__(self, config: CognitiveGatewayConfig | None = None):
        self.config = config or load_cognitive_gateway_config()
        self.events = GatewayEventBus()
        self.memory_hook: MemoryHook = PlaceholderMemoryHook()
        self._wire_default_subscribers()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _wire_default_subscribers(self) -> None:
        for event in (
            GatewayEventBus.EVENT_REQUEST_RECEIVED,
            GatewayEventBus.EVENT_MODEL_SELECTED,
            GatewayEventBus.EVENT_RESPONSE_GENERATED,
            GatewayEventBus.EVENT_ERROR,
        ):
            self.events.subscribe(event, logging_event_handler(event))

    def prepare_chat_request(
        self,
        payload: Any,
        *,
        path_model: str | None,
        available_models: list[str],
        default_model: str | None,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, GatewayContext]:
        """Classify, route, and run pre-inference hooks. Returns (payload, context)."""
        msg_list = messages if messages is not None else list(getattr(payload, "messages", []) or [])
        requested = path_model or getattr(payload, "model", None)
        stream = bool(getattr(payload, "stream", False))

        ctx = GatewayContext(
            requested_model=str(requested) if requested else None,
            stream=stream,
            messages=msg_list,
        )

        if not self.config.enabled:
            ctx.routed_model = str(requested) if requested else default_model
            ctx.routing_source = "cognitive_disabled"
            return payload, ctx

        classification = classify_request(msg_list, payload)
        ctx.request_type = classification.request_type
        ctx.compute_priority = classification.compute_priority
        log_request_classified(ctx)

        decision = route_model(
            config=self.config,
            request_type=ctx.request_type,
            available_models=available_models,
            requested_model=ctx.requested_model,
            default_model=default_model,
        )
        ctx.routed_model = decision.model_id
        ctx.routing_source = decision.source
        ctx.fallback_used = decision.fallback_used
        if decision.fallback_used:
            log_fallback_triggered(
                ctx,
                from_model=ctx.requested_model,
                to_model=decision.model_id,
            )
        log_model_selected(ctx)

        if self.config.memory_hook_enabled:
            injected = self.memory_hook.before_request(ctx.to_hook_context())
            if isinstance(injected, dict):
                ctx.memory_context.update(injected)

        self.events.emit(
            GatewayEventBus.EVENT_REQUEST_RECEIVED,
            {
                "request_id": ctx.request_id,
                "request_type": ctx.request_type,
                "compute_priority": ctx.compute_priority,
                "stream": ctx.stream,
            },
        )
        self.events.emit(
            GatewayEventBus.EVENT_MODEL_SELECTED,
            {
                "request_id": ctx.request_id,
                "requested_model": ctx.requested_model,
                "routed_model": ctx.routed_model,
                "routing_source": ctx.routing_source,
                "fallback_used": ctx.fallback_used,
            },
        )

        if decision.model_id and hasattr(payload, "model_copy"):
            payload = payload.model_copy(update={"model": decision.model_id})
        elif decision.model_id:
            if hasattr(payload, "model_dump"):
                data = payload.model_dump()
                data["model"] = decision.model_id
                return data, ctx
            setattr(payload, "model", decision.model_id)

        return payload, ctx

    def finalize_chat_response(
        self,
        ctx: GatewayContext,
        *,
        response: dict[str, Any] | None = None,
        status_code: int = 200,
        error: str | None = None,
    ) -> None:
        """Run post-inference hooks and observability after response."""
        if error:
            ctx.error = error
            log_upstream_error(ctx, status_code=status_code, detail=error)
            self.events.emit(
                GatewayEventBus.EVENT_ERROR,
                {
                    "request_id": ctx.request_id,
                    "status_code": status_code,
                    "detail": error,
                    "model": ctx.routed_model,
                },
            )
            return

        if response:
            ctx.response_model = str(response.get("model") or ctx.routed_model or "")

        if self.config.memory_hook_enabled and response is not None:
            hook_ctx = ctx.to_hook_context()
            hook_ctx["response"] = response
            self.memory_hook.after_response(hook_ctx)

        log_request_complete(ctx, status_code=status_code)
        self.events.emit(
            GatewayEventBus.EVENT_RESPONSE_GENERATED,
            {
                "request_id": ctx.request_id,
                "model": ctx.response_model or ctx.routed_model,
                "latency_ms": round(ctx.elapsed_ms(), 1),
                "stream": ctx.stream,
            },
        )

    def mark_stream_started(self, ctx: GatewayContext) -> None:
        import time

        ctx.stream_started_at = time.perf_counter()


_cognitive_gateway: CognitiveGateway | None = None


def get_cognitive_gateway() -> CognitiveGateway:
    global _cognitive_gateway  # pylint: disable=global-statement
    if _cognitive_gateway is None:
        _cognitive_gateway = CognitiveGateway()
    return _cognitive_gateway
