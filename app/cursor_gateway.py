"""Cursor LLM Gateway Adapter — OpenAI-compatible normalization for Cursor IDE.

Thin layer that sits in front of niblit-cloud-server /v1 endpoints.  In embedded
mode (default) it normalizes Cursor requests and enriches responses in-process.
In proxy mode it forwards HTTP to an upstream base URL with streaming passthrough.

When the Cognitive Gateway is enabled, this module also performs request
classification, rule-based model routing, hooks, and structured observability.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Generator, Mapping

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from app.cognitive_gateway.context import GatewayContext
from app.cognitive_gateway.gateway import CognitiveGateway, get_cognitive_gateway

logger = logging.getLogger("niblit.cursor_gateway")

# Roles Cursor / OpenAI may send that llama.cpp chat templates understand.
_ROLE_ALIASES: dict[str, str] = {
    "developer": "system",
}

_SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CursorGatewayConfig:
    enabled: bool = True
    upstream_base_url: str = "http://localhost:8000/v1"
    proxy_mode: bool = False
    request_timeout_s: float = 600.0
    model_passthrough: bool = True
    owned_by: str = "niblit"


def load_cursor_gateway_config() -> CursorGatewayConfig:
    upstream = os.getenv("CURSOR_GATEWAY_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    return CursorGatewayConfig(
        enabled=_env_bool("CURSOR_GATEWAY_ENABLED", True),
        upstream_base_url=upstream,
        proxy_mode=_env_bool("CURSOR_GATEWAY_PROXY_MODE", False),
        request_timeout_s=_env_float("CURSOR_GATEWAY_TIMEOUT_S", 600.0),
        model_passthrough=_env_bool("CURSOR_GATEWAY_MODEL_PASSTHROUGH", True),
        owned_by=os.getenv("CURSOR_GATEWAY_OWNED_BY", "niblit"),
    )


def content_to_string(content: Any) -> str:
    """Flatten OpenAI/Cursor message content into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = str(item.get("type", "text"))
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type == "image_url":
                parts.append("[image]")
            elif item_type == "input_audio":
                parts.append("[audio]")
            else:
                text = item.get("text")
                parts.append(str(text) if text is not None else json.dumps(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_cursor_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert Cursor/OpenAI message shapes into llama.cpp-compatible messages."""
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    normalized: list[dict[str, str]] = []
    for item in messages:
        role = str(item.get("role", "user")).strip().lower()
        role = _ROLE_ALIASES.get(role, role)

        content = content_to_string(item.get("content"))
        if not content and role == "assistant" and item.get("tool_calls"):
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list):
                content = json.dumps(tool_calls)
        if not content and role == "tool":
            content = content_to_string(item.get("content")) or str(item.get("tool_call_id", ""))

        normalized.append({"role": role, "content": content})
    return normalized


def openai_error_body(
    message: str,
    *,
    status_code: int,
    error_type: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    err_type = error_type or _error_type_for_status(status_code)
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": err_type,
        }
    }
    if code:
        body["error"]["code"] = code
    return body


def _error_type_for_status(status_code: int) -> str:
    if status_code == 401:
        return "invalid_request_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "invalid_request_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _detail_to_message(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        if "message" in detail:
            return str(detail["message"])
        return json.dumps(detail)
    if isinstance(detail, list):
        return json.dumps(detail)
    return str(detail)


class CursorGateway:
    """Cursor-facing adapter — embedded normalization or HTTP upstream proxy."""

    def __init__(self, config: CursorGatewayConfig | None = None):
        self.config = config or load_cursor_gateway_config()
        self._model_epoch = int(time.time())
        self._cognitive = get_cognitive_gateway()

    @property
    def cognitive(self) -> CognitiveGateway:
        return self._cognitive

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def stream_headers(self) -> dict[str, str]:
        return dict(_SSE_HEADERS)

    def log_incoming(self, request: Request | None, *, path: str, model: str | None) -> float:
        start = time.perf_counter()
        logger.info(
            "cursor request: %s %s model=%s",
            "POST" if request is None else request.method,
            path,
            model or "(default)",
        )
        return start

    def log_complete(self, start: float, *, status_code: int, model: str | None = None) -> None:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "cursor response: model=%s status=%d latency_ms=%.1f",
            model or "(default)",
            status_code,
            latency_ms,
        )

    def log_upstream_error(self, *, status_code: int, detail: str, model: str | None = None) -> None:
        logger.warning(
            "cursor upstream error: model=%s status=%d detail=%s",
            model or "(default)",
            status_code,
            detail,
        )

    def adapt_chat_payload(self, payload: Any) -> Any:
        """Normalize a ChatCompletionRequest for the existing inference handlers."""
        if not self.config.enabled:
            return payload

        updates: dict[str, Any] = {}

        if hasattr(payload, "messages") and payload.messages:
            updates["messages"] = normalize_cursor_messages(list(payload.messages))

        max_completion = getattr(payload, "max_completion_tokens", None)
        if max_completion is not None and int(max_completion) > 0:
            updates["max_tokens"] = int(max_completion)

        if updates and hasattr(payload, "model_copy"):
            return payload.model_copy(update=updates)
        if updates:
            data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
            data.update(updates)
            return data
        return payload

    def prepare_chat(
        self,
        payload: Any,
        *,
        path_model: str | None,
        available_models: list[str],
        default_model: str | None,
    ) -> tuple[Any, GatewayContext]:
        """Normalize, classify, route, and run pre-inference cognitive hooks."""
        adapted = self.adapt_chat_payload(payload)
        messages = list(getattr(adapted, "messages", []) or [])
        return self._cognitive.prepare_chat_request(
            adapted,
            path_model=path_model,
            available_models=available_models,
            default_model=default_model,
            messages=messages,
        )

    def finalize_chat(
        self,
        ctx: GatewayContext,
        *,
        response: Mapping[str, Any] | None = None,
        status_code: int = 200,
        error: str | None = None,
    ) -> None:
        """Post-inference cognitive hooks and structured observability."""
        self._cognitive.finalize_chat_response(
            ctx,
            response=dict(response) if response is not None else None,
            status_code=status_code,
            error=error,
        )

    def resolve_request_model(self, payload: Any, path_model: str | None = None) -> str | None:
        model = path_model or getattr(payload, "model", None)
        if isinstance(model, str) and model.strip():
            return model.strip()
        return None

    def adapt_models_response(self, response: Mapping[str, Any]) -> dict[str, Any]:
        """Enrich /v1/models payload with OpenAI fields Cursor expects."""
        if not self.config.enabled:
            return dict(response)

        data: list[dict[str, Any]] = []
        for entry in response.get("data", []):
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id", ""))
            if not model_id:
                continue
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(entry.get("created", self._model_epoch)),
                    "owned_by": str(entry.get("owned_by", self.config.owned_by)),
                }
            )
        return {"object": "list", "data": data}

    def adapt_chat_response(self, response: Mapping[str, Any]) -> dict[str, Any]:
        """Strip non-OpenAI fields (e.g. cognitive metadata) for Cursor clients."""
        if not self.config.enabled:
            return dict(response)

        payload = dict(response)
        payload.pop("cognitive", None)
        return payload

    def http_exception_to_openai(self, exc: HTTPException) -> JSONResponse:
        message = _detail_to_message(exc.detail)
        self.log_upstream_error(status_code=exc.status_code, detail=message)
        return JSONResponse(
            status_code=exc.status_code,
            content=openai_error_body(message, status_code=exc.status_code),
        )

    def wrap_stream(
        self,
        chunks: Generator[str, None, None],
        *,
        start: float,
        model: str | None,
        ctx: GatewayContext | None = None,
    ) -> Generator[str, None, None]:
        if ctx is not None:
            self._cognitive.mark_stream_started(ctx)
        try:
            yield from chunks
            if ctx is not None:
                self.finalize_chat(ctx, response={"model": model}, status_code=200)
            else:
                self.log_complete(start, status_code=200, model=model)
        except Exception as exc:
            detail = str(exc)
            if ctx is not None:
                self.finalize_chat(ctx, status_code=502, error=detail)
            else:
                self.log_upstream_error(status_code=502, detail=detail, model=model)
            raise

    # ── HTTP proxy mode (standalone gateway) ──────────────────────────────────

    def _upstream_url(self, path: str) -> str:
        base = self.config.upstream_base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        if suffix.startswith("/v1/"):
            return f"{base}{suffix[len('/v1'):]}"
        return f"{base}{suffix}"

    def _forward_headers(self, request: Request) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth = request.headers.get("authorization")
        if auth:
            headers["Authorization"] = auth
        api_key = request.headers.get("x-api-key")
        if api_key:
            headers["X-Api-Key"] = api_key
        return headers

    async def proxy_list_models(self, request: Request) -> Response:
        start = self.log_incoming(request, path="/v1/models", model=None)
        url = self._upstream_url("/models")
        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout_s) as client:
                upstream = await client.get(url, headers=self._forward_headers(request))
        except httpx.HTTPError as exc:
            self.log_upstream_error(status_code=502, detail=str(exc))
            return JSONResponse(
                status_code=502,
                content=openai_error_body(f"Upstream unreachable: {exc}", status_code=502),
            )

        if upstream.status_code >= 400:
            self.log_upstream_error(
                status_code=upstream.status_code,
                detail=upstream.text[:500],
            )
            return JSONResponse(status_code=upstream.status_code, content=upstream.json())

        body = self.adapt_models_response(upstream.json())
        self.log_complete(start, status_code=200)
        return JSONResponse(content=body)

    async def proxy_chat_completions(self, request: Request) -> Response:
        try:
            raw = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content=openai_error_body("Invalid JSON body", status_code=400),
            )
        if not isinstance(raw, dict):
            return JSONResponse(
                status_code=400,
                content=openai_error_body("Request body must be a JSON object", status_code=400),
            )

        model = str(raw.get("model") or "")
        start = self.log_incoming(request, path="/v1/chat/completions", model=model or None)

        messages = raw.get("messages")
        if isinstance(messages, list):
            raw["messages"] = normalize_cursor_messages(messages)

        max_completion = raw.get("max_completion_tokens")
        if max_completion is not None and "max_tokens" not in raw:
            raw["max_tokens"] = max_completion

        if self.config.model_passthrough and model:
            raw["model"] = model

        url = self._upstream_url("/chat/completions")
        stream = bool(raw.get("stream"))

        try:
            client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
            if stream:
                req = client.build_request(
                    "POST",
                    url,
                    headers=self._forward_headers(request),
                    json=raw,
                )
                upstream = await client.send(req, stream=True)

                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    await upstream.aclose()
                    await client.aclose()
                    detail = body.decode("utf-8", errors="replace")[:500]
                    self.log_upstream_error(status_code=upstream.status_code, detail=detail, model=model)
                    try:
                        content = json.loads(body)
                    except json.JSONDecodeError:
                        content = openai_error_body(detail, status_code=upstream.status_code)
                    return JSONResponse(status_code=upstream.status_code, content=content)

                async def stream_body() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk
                    finally:
                        await upstream.aclose()
                        await client.aclose()
                        self.log_complete(start, status_code=200, model=model)

                return StreamingResponse(
                    stream_body(),
                    media_type="text/event-stream",
                    headers=self.stream_headers,
                    background=BackgroundTask(lambda: None),
                )

            upstream = await client.post(
                url,
                headers=self._forward_headers(request),
                json=raw,
            )
            await client.aclose()
        except httpx.HTTPError as exc:
            self.log_upstream_error(status_code=502, detail=str(exc), model=model)
            return JSONResponse(
                status_code=502,
                content=openai_error_body(f"Upstream unreachable: {exc}", status_code=502),
            )

        if upstream.status_code >= 400:
            self.log_upstream_error(
                status_code=upstream.status_code,
                detail=upstream.text[:500],
                model=model,
            )
            try:
                content = upstream.json()
            except json.JSONDecodeError:
                content = openai_error_body(upstream.text, status_code=upstream.status_code)
            return JSONResponse(status_code=upstream.status_code, content=content)

        try:
            body = self.adapt_chat_response(upstream.json())
        except json.JSONDecodeError:
            return JSONResponse(status_code=502, content=openai_error_body("Invalid upstream JSON", status_code=502))

        self.log_complete(start, status_code=200, model=model)
        return JSONResponse(content=body)


_gateway: CursorGateway | None = None


def get_cursor_gateway() -> CursorGateway:
    global _gateway  # pylint: disable=global-statement
    if _gateway is None:
        _gateway = CursorGateway()
    return _gateway
