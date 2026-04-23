from fastapi.testclient import TestClient

from app.main import ModelEngineResult, ModelManager, create_app


class FakeModelManager(ModelManager):
    def __init__(self):
        super().__init__(
            model_map={"demo-model": "/tmp/demo.gguf"},
            default_model="demo-model",
        )
        self.last_chat_call = None

    def chat(self, model_id, messages, temperature, max_tokens):
        if not messages:
            raise AssertionError("messages must not be empty in tests")
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


def test_models_list():
    client, _ = make_client()
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    assert payload["data"] == [{"id": "demo-model", "object": "model"}]


def test_chat_completions_hf_route():
    client, manager = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.1,
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "demo-model"
    assert payload["choices"][0]["message"]["content"] == "echo:hello"
    assert payload["usage"]["total_tokens"] == 8
    assert manager.last_chat_call["model_id"] == "demo-model"
    assert manager.last_chat_call["temperature"] == 0.1
    assert manager.last_chat_call["max_tokens"] == 32


def test_chat_completions_compat_hf_prefix_route():
    client, _ = make_client()
    response = client.post(
        "/hf/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "compat"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "echo:compat"


def test_chat_completions_path_model_route():
    client, manager = make_client()
    response = client.post(
        "/v1/models/demo-model/chat/completions",
        json={"messages": [{"role": "user", "content": "path-model"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "echo:path-model"
    assert manager.last_chat_call["model_id"] == "demo-model"


def test_inference_api_format():
    client, manager = make_client()
    response = client.post(
        "/models/demo-model",
        json={"inputs": "inference format", "parameters": {"max_new_tokens": 12}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["generated_text"] == "echo:inference format"
    assert payload["model"] == "demo-model"
    assert manager.last_chat_call["max_tokens"] == 12


def test_unknown_model_is_404():
    client, _ = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404


# ── Niblit local_brain compatibility tests ────────────────────────────────────

def test_model_alias_local_resolves_to_default():
    """Niblit QwenLocalBrain always sends model='local'; must map to default."""
    client, manager = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "niblit local alias"}],
            "temperature": 0.7,
            "max_tokens": 200,
            "stop": ["<|im_end|>"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "echo:niblit local alias"
    assert manager.last_chat_call["model_id"] == "demo-model"


def test_health_probe_endpoint():
    """Niblit _check_server_url probes /health as its first target."""
    client, _ = make_client()
    assert client.get("/health").status_code == 200
    assert client.get("/healthz").status_code == 200


def test_props_probe_endpoint():
    """Niblit _check_server_url probes /props as its legacy fallback target."""
    client, _ = make_client()
    response = client.get("/props")
    assert response.status_code == 200
    assert "total_slots" in response.json()


def test_legacy_completion_endpoint():
    """Niblit _generate_http_legacy calls POST /completion with prompt+n_predict."""
    client, manager = make_client()
    response = client.post(
        "/completion",
        json={
            "prompt": "legacy prompt text",
            "n_predict": 64,
            "temperature": 0.7,
            "stop": ["<|im_end|>"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["content"] == "echo:legacy prompt text"
    assert manager.last_chat_call["max_tokens"] == 64


def test_chat_completions_request_defaults():
    """Omitting temperature/max_tokens should use the shared defaults."""
    client, manager = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "demo-model", "messages": [{"role": "user", "content": "defaults"}]},
    )
    assert response.status_code == 200
    # Default temperature and max_tokens from _DEFAULT_TEMPERATURE / _DEFAULT_MAX_TOKENS
    assert manager.last_chat_call["temperature"] == 0.2
    assert manager.last_chat_call["max_tokens"] == 256


def test_legacy_completion_request_defaults():
    """Omitting n_predict/temperature should use the shared defaults."""
    client, manager = make_client()
    response = client.post(
        "/completion",
        json={"prompt": "default check"},
    )
    assert response.status_code == 200
    assert manager.last_chat_call["temperature"] == 0.2
    assert manager.last_chat_call["max_tokens"] == 256


def test_niblit_full_flow_model_local_stop_tokens():
    """Full round-trip matching what Niblit's _generate_http() sends."""
    client, manager = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": "You are Niblit."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "choices" in payload
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert manager.last_chat_call["model_id"] == "demo-model"
    assert manager.last_chat_call["max_tokens"] == 200
