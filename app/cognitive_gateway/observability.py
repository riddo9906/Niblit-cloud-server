"""Structured observability for the Cognitive Gateway Layer."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.cognitive_gateway.context import GatewayContext

logger = logging.getLogger("niblit.cognitive_gateway")


def log_request_classified(ctx: GatewayContext) -> None:
    logger.info(
        "cognitive classified request_id=%s type=%s priority=%s requested_model=%s",
        ctx.request_id,
        ctx.request_type,
        ctx.compute_priority,
        ctx.requested_model or "(default)",
    )


def log_model_selected(ctx: GatewayContext) -> None:
    logger.info(
        "cognitive model selected request_id=%s model=%s source=%s fallback=%s",
        ctx.request_id,
        ctx.routed_model or "(default)",
        ctx.routing_source,
        ctx.fallback_used,
    )


def log_request_complete(ctx: GatewayContext, *, status_code: int) -> None:
    payload: dict[str, Any] = {
        "request_id": ctx.request_id,
        "request_type": ctx.request_type,
        "compute_priority": ctx.compute_priority,
        "requested_model": ctx.requested_model,
        "routed_model": ctx.routed_model,
        "response_model": ctx.response_model,
        "routing_source": ctx.routing_source,
        "fallback_used": ctx.fallback_used,
        "stream": ctx.stream,
        "status_code": status_code,
        "latency_ms": round(ctx.elapsed_ms(), 1),
    }
    stream_ms = ctx.stream_duration_ms()
    if stream_ms is not None:
        payload["stream_duration_ms"] = round(stream_ms, 1)
    logger.info("cognitive request complete %s", json.dumps(payload, sort_keys=True))


def log_upstream_error(ctx: GatewayContext, *, status_code: int, detail: str) -> None:
    logger.warning(
        "cognitive upstream error request_id=%s model=%s status=%d detail=%s fallback=%s",
        ctx.request_id,
        ctx.routed_model or "(default)",
        status_code,
        detail,
        ctx.fallback_used,
    )


def log_fallback_triggered(ctx: GatewayContext, *, from_model: str | None, to_model: str | None) -> None:
    logger.info(
        "cognitive fallback triggered request_id=%s from=%s to=%s",
        ctx.request_id,
        from_model or "(none)",
        to_model or "(none)",
    )
