from fastapi.testclient import TestClient

from app.main import ModelEngineResult, ModelManager, create_app


class FakeModelManager(ModelManager):
    def __init__(self):
        super().__init__(
            model_map={"demo-model": "/tmp/demo.gguf"},
            default_model="demo-model",
        )

    def chat(self, model_id, messages, temperature, max_tokens):
        return ModelEngineResult(
            text=f"echo:{messages[-1]['content']}",
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )


def make_client() -> TestClient:
    app = create_app(model_manager=FakeModelManager())
    return TestClient(app)


def test_models_list():
    client = make_client()
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "demo-model"


def test_chat_completions_hf_route():
    client = make_client()
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


def test_chat_completions_compat_hf_prefix_route():
    client = make_client()
    response = client.post(
        "/hf/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "compat"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "echo:compat"


def test_inference_api_format():
    client = make_client()
    response = client.post(
        "/models/demo-model",
        json={"inputs": "inference format", "parameters": {"max_new_tokens": 12}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["generated_text"] == "echo:inference format"
    assert payload["model"] == "demo-model"


def test_unknown_model_is_404():
    client = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404
