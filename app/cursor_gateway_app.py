"""Standalone Cursor LLM Gateway — HTTP proxy to niblit-cloud-server.

Run when Cursor should connect to a dedicated gateway port while inference stays
on the main server:

    uvicorn app.cursor_gateway_app:app --host 0.0.0.0 --port 8001

Point Cursor at ``http://localhost:8001/v1`` and set
``CURSOR_GATEWAY_BASE_URL=http://localhost:8000/v1``.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request

from app.cursor_gateway import CursorGateway, CursorGatewayConfig, load_cursor_gateway_config

_base = load_cursor_gateway_config()
_config = CursorGatewayConfig(
    enabled=True,
    upstream_base_url=_base.upstream_base_url,
    proxy_mode=True,
    request_timeout_s=_base.request_timeout_s,
    model_passthrough=_base.model_passthrough,
    owned_by=_base.owned_by,
)
_gateway = CursorGateway(_config)

app = FastAPI(
    title="Niblit Cursor LLM Gateway",
    version="1.0.0",
    description="OpenAI-compatible proxy for Cursor IDE → niblit-cloud-server",
)


@app.get("/health")
@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "gateway": "cursor"}


@app.get("/v1/models")
async def list_models(request: Request):
    return await _gateway.proxy_list_models(request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _gateway.proxy_chat_completions(request)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CURSOR_GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("CURSOR_GATEWAY_PORT", "8001"))
    uvicorn.run("app.cursor_gateway_app:app", host=host, port=port, reload=False)
