import json

import httpx
from starlette.requests import Request

from app.cursor_gateway import (
    CursorGateway,
    CursorGatewayConfig,
    content_to_string,
    get_cursor_gateway,
    normalize_cursor_messages,
)
from app.main import ModelEngineResult, ModelManager, create_app
from fastapi.testclient import TestClient


class FakeModelManager(ModelManager):
    def __init__(self):
        super().__init__(
            model_map={"demo-model": "/tmp/demo.gguf", "qwen": "/tmp/qwen.gguf"},
            default_model="demo-model",
        )
        self.last_chat_call = None

    def chat(self, model_id, messages, temperature, max_tokens):
        self.last_chat_call = {
            "model_id": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return ModelEngineResult(
            text=f"echo:{messages[-1]['content']}",
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )


def make_client() -> tuple[TestClient, FakeModelManager]:
    manager = FakeModelManager()
    app = create_app(model_manager=manager)
    return TestClient(app), manager


def test_content_to_string_multimodal():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
    ]
    assert content_to_string(content) == "hello\n[image]"


def test_normalize_cursor_messages_developer_role():
    messages = normalize_cursor_messages(
        [{"role": "developer", "content": "You are helpful."}]
    )
    assert messages == [{"role": "system", "content": "You are helpful."}]


def test_normalize_cursor_messages_tool_calls():
    messages = normalize_cursor_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "read", "arguments": "{}"}}],
            }
        ]
    )
    assert messages[0]["role"] == "assistant"
    assert "call_1" in messages[0]["content"]


def test_models_list_openai_fields():
    client, _ = make_client()
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    model = payload["data"][0]
    assert model["id"] == "demo-model"
    assert model["object"] == "model"
    assert "created" in model
    assert model["owned_by"] == "niblit"


def test_chat_completions_max_completion_tokens():
    client, manager = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 48,
        },
    )
    assert response.status_code == 200
    assert manager.last_chat_call["max_tokens"] == 48


def test_chat_completions_multimodal_content():
    client, manager = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "explain this"}],
                }
            ],
        },
    )
    assert response.status_code == 200
    assert manager.last_chat_call["messages"][-1]["content"] == "explain this"


def test_chat_completions_openai_error_shape():
    client, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "missing", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["message"]
    assert payload["error"]["type"]


def test_chat_completions_stream_sse_headers():
    client, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-cache"
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "data: [DONE]" in response.text


def test_adapt_chat_response_strips_cognitive():
    gateway = CursorGateway(CursorGatewayConfig(enabled=True))
    adapted = gateway.adapt_chat_response(
        {
            "id": "x",
            "choices": [],
            "cognitive": {"intent": "test"},
        }
    )
    assert "cognitive" not in adapted


def test_proxy_list_models(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://upstream.test/v1/models")
        return httpx.Response(200, json={"object": "list", "data": [{"id": "m1", "object": "model"}]})

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs):
        kwargs["transport"] = transport
        return real_async_client(**kwargs)

    monkeypatch.setattr("app.cursor_gateway.httpx.AsyncClient", fake_async_client)

    gateway = CursorGateway(
        CursorGatewayConfig(
            enabled=True,
            upstream_base_url="http://upstream.test/v1",
            proxy_mode=True,
        )
    )

    import asyncio

    async def run():
        scope = {"type": "http", "method": "GET", "path": "/v1/models", "headers": []}
        request = Request(scope)
        return await gateway.proxy_list_models(request)

    response = asyncio.run(run())
    assert response.status_code == 200
    model = json.loads(response.body)["data"][0]
    assert model["id"] == "m1"
    assert model["owned_by"] == "niblit"


def test_proxy_chat_completions_stream(monkeypatch):
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n",
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs):
        kwargs["transport"] = transport
        return real_async_client(**kwargs)

    monkeypatch.setattr("app.cursor_gateway.httpx.AsyncClient", fake_async_client)

    gateway = CursorGateway(
        CursorGatewayConfig(
            enabled=True,
            upstream_base_url="http://localhost:8000/v1",
            proxy_mode=True,
        )
    )

    async def run():
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
        }
        body = json.dumps(
            {
                "model": "demo-model",
                "messages": [{"role": "developer", "content": "hi"}],
                "stream": True,
            }
        ).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(scope, receive)
        response = await gateway.proxy_chat_completions(request)
        chunks = b""
        async for chunk in response.body_iterator:
            chunks += chunk
        return response.status_code, chunks

    import asyncio

    status_code, body = asyncio.run(run())
    assert status_code == 200
    assert b"data: [DONE]" in body
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["body"]["messages"][0]["role"] == "system"


def test_gateway_disabled_skips_model_enrichment(monkeypatch):
    monkeypatch.setenv("CURSOR_GATEWAY_ENABLED", "0")
    import app.cursor_gateway as cg

    cg._gateway = None
    gateway = get_cursor_gateway()
    assert gateway.enabled is False
    raw = {"object": "list", "data": [{"id": "x", "object": "model"}]}
    assert gateway.adapt_models_response(raw) == raw
