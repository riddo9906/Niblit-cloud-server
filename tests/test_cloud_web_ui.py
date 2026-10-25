from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


class FakeManager:
    def list_models(self):
        return [
            {"id": "qwen2.5-coder-3b", "object": "model"},
            {"id": "llama3.2-1b", "object": "model"},
            {"id": "qwen2.5-0.5b", "object": "model"},
        ]

    def get_default_model_info(self):
        return {"model_id": "qwen2.5-coder-3b", "model_path": "local"}


def test_root_serves_cloud_console():
    client = TestClient(create_app(model_manager=FakeManager()))
    response = client.get("/")
    assert response.status_code == 200
    assert "Niblit Cloud" in response.text
    assert "/ui/app.js" in response.text


def test_cloud_ui_assets_are_reachable():
    client = TestClient(create_app(model_manager=FakeManager()))
    assert client.get("/ui/styles.css").status_code == 200
    assert client.get("/ui/app.js").status_code == 200


def test_cloud_api_routes_remain_available():
    client = TestClient(create_app(model_manager=FakeManager()))
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert [item["id"] for item in models.json()["data"]] == [
        "qwen2.5-coder-3b",
        "llama3.2-1b",
        "qwen2.5-0.5b",
    ]


def test_ui_model_selector_uses_api_inventory_not_hardcoded_models():
    script = Path(__file__).parents[1] / "app" / "static" / "cloud_ui" / "app.js"
    content = script.read_text(encoding="utf-8")
    assert "fetch('/v1/models')" in content
    assert "option.value = model.id" in content
    assert "qwen2.5-coder-3b" not in content
    assert "llama3.2-1b" not in content
    assert "qwen2.5-0.5b" not in content


def test_ui_sends_selected_canonical_model_to_existing_chat_route():
    script = Path(__file__).parents[1] / "app" / "static" / "cloud_ui" / "app.js"
    content = script.read_text(encoding="utf-8")
    assert "fetch('/v1/chat/completions'" in content
    assert "model: selectedModel()" in content
    assert "stream: true" in content


def test_ui_has_no_duplicate_v1_path():
    for name in ("index.html", "app.js", "styles.css"):
        content = (Path(__file__).parents[1] / "app" / "static" / "cloud_ui" / name).read_text(encoding="utf-8")
        assert "/v1/v1" not in content
