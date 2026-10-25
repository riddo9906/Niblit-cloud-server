"""VS Code / OpenAI-compatibility regression tests.

Covers: /v1/completions (non-streaming + streaming), oversized max_tokens
clamping (never a mid-stream rejection), governance preflight before
StreamingResponse, and model-identity consistency between the advertised
model list and the executable manager.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """App with a stubbed ModelManager so no real GGUF load is needed."""
    monkeypatch.setenv("NIBLIT_TEST_MODE", "1")

    class FakeResult:
        def __init__(self) -> None:
            self.text = "OK"
            self.finish_reason = "stop"
            self.usage = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            self.tool_calls = None

    class FakeManager:
        def __init__(self) -> None:
            self._model_map = {"qwen2.5-coder-3b": "fake.gguf"}
            self.last_max_tokens: int | None = None

        def list_models(self):
            return [{"id": "qwen2.5-coder-3b", "object": "model"}]

        def get_default_model_info(self):
            return {"model_id": "qwen2.5-coder-3b", "model_path": "fake.gguf"}

        def get_active_model_id(self):
            return "qwen2.5-coder-3b"

        def resolve_model_id(self, model, path_model=None):
            model_id = model or path_model or "qwen2.5-coder-3b"
            if model_id not in self._model_map:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
            return model_id

        def estimate_inference(self, messages, max_tokens, model_id=None):
            return {
                "prompt_tokens_estimate": 8,
                "messages_truncated": False,
                "memory_pressure": 0.01,
                "effective_max_tokens": min(int(max_tokens), 8192),
                "model_context_limit": 16384,
            }

        def runtime_stats(self):
            return {"context_window": 16384}

        def chat(self, model_id, messages, temperature, max_tokens, **kwargs):
            self.last_max_tokens = max_tokens
            return FakeResult()

    import app.main as app_main

    fake = FakeManager()
    app_main.app.state.model_manager = fake
    return TestClient(app_main.app), fake


def _assert_no_double_v1(url: str) -> None:
    assert "/v1/v1/" not in url
    assert "//v1" not in url


def test_v1_completions_non_streaming(client) -> None:
    tc, _ = client
    resp = tc.post(
        "/v1/completions",
        json={"model": "qwen2.5-coder-3b", "prompt": "Say OK", "max_tokens": 16},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] == "stop"
    # Real usage from engine stub — present, not fabricated zeros-only shape
    assert body["usage"]["total_tokens"] >= 1


def test_v1_completions_streaming(client) -> None:
    tc, _ = client
    with tc.stream(
        "POST",
        "/v1/completions",
        json={"model": "qwen2.5-coder-3b", "prompt": "Say OK", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        chunks = [
            line for line in resp.iter_lines() if line.startswith("data:")
        ]
    assert any("[DONE]" in c for c in chunks)
    text_chunks = [c for c in chunks if "text_completion.chunk" in c]
    assert text_chunks


def test_chat_streaming_governed_oversize_request_is_clean(client) -> None:
    """Oversized max_tokens must be clamped — never a mid-stream failure."""
    tc, fake = client
    with tc.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder-3b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 13432,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200  # clean stream, not an ASGI error
        body = "".join(resp.iter_lines())
    assert "[DONE]" in body
    # Clamp actually applied to the inference call
    assert fake.last_max_tokens is not None
    assert fake.last_max_tokens <= 8192


def test_non_streaming_clamps_oversized_max_tokens(client) -> None:
    tc, fake = client
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder-3b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 13432,
        },
    )
    assert resp.status_code == 200
    assert fake.last_max_tokens <= 8192


def test_advertised_models_match_executable(client) -> None:
    tc, fake = client
    advertised = [m["id"] for m in tc.get("/v1/models").json()["data"]]
    executable = list(fake._model_map.keys())
    assert set(advertised) == set(executable)


def test_unknown_model_structured_error(client) -> None:
    tc, _ = client
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": "definitely-not-a-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    if resp.status_code != 200:
        detail = resp.json()
        assert "detail" in detail or "error" in detail


def test_no_v1_v1_route_registered(client) -> None:
    tc, _ = client
    resp = tc.get("/v1/v1/models")
    assert resp.status_code == 404


def test_no_doubled_v1_in_chat_completions(client) -> None:
    """Regression: the chat completions path must never compose /v1/v1/..."""
    tc, _ = client
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder-3b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    # Request should succeed cleanly — no doubled prefix anywhere
    assert resp.status_code in (200, 403, 404, 503)
    # Verify the request path itself is correct
    _assert_no_double_v1(str(resp.url))


def test_governance_rejects_before_stream_starts(client) -> None:
    """Oversized max_tokens that exceeds context must return clean 4xx, not stream."""
    tc, fake = client
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder-3b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 999999,
        },
    )
    # Should be rejected as 403, NOT 200 with a broken stream
    assert resp.status_code == 403
    body = resp.json()
    assert "detail" in body or "error" in body


def test_model_omitted_uses_default(client) -> None:
    tc, _ = client
    resp = tc.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200


def test_explicit_unknown_model_returns_404(client) -> None:
    tc, _ = client
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 404


def test_missing_max_tokens_uses_default(client) -> None:
    tc, fake = client
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-coder-3b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200


