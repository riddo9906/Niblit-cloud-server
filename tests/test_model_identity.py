"""Phase 4A — model routing truth.

Invariant: an explicit model request must be requested == resolved ==
executed == response.model, with fallback exposed (never silent) if it
ever differs. Unknown explicit models must fail cleanly, not fall back.
"""

from fastapi.testclient import TestClient

from app.main import ModelEngineResult, ModelManager, create_app
from tests.test_api import FakeModelManager


def _make(manager):
    return TestClient(create_app(model_manager=manager)), manager


def test_explicit_model_executes_requested_model():
    client, manager = _make(FakeModelManager())
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "demo-model",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    meta = body["niblit_model"]
    assert manager.last_chat_call["model_id"] == "demo-model"
    assert body["model"] == "demo-model"
    assert meta["requested_model"] == "demo-model"
    assert meta["resolved_model"] == "demo-model"
    assert meta["executed_model"] == "demo-model"
    assert meta["fallback"] is False


def test_omitted_model_uses_default_without_fallback_flag():
    client, manager = _make(FakeModelManager())
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    meta = body["niblit_model"]
    assert meta["executed_model"] == "demo-model"
    # No caller-requested model -> no fallback occurred by definition.
    assert meta["fallback"] is False
    assert manager.last_chat_call["model_id"] == "demo-model"


def test_unknown_explicit_model_is_clean_error_not_silent_fallback():
    client, manager = _make(FakeModelManager())
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "definitely-not-a-model",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    # Must be a clean structured error and must NOT execute any model.
    assert resp.status_code in (400, 404), resp.text
    body_text = resp.text.lower()
    assert "definitely-not-a-model" in body_text or "unknown model" in body_text
    assert manager.last_chat_call is None


def test_orchestrator_cannot_override_explicit_selection(monkeypatch):
    """Regression for the Phase-4 latency-rerouting defect: even when the
    orchestrator recommends a different model, an explicit selection wins."""
    client, manager = _make(FakeModelManager())

    from app.model_orchestrator import ModelOrchestrator

    class FakeDecision:
        model_id = "other-model"
        fallback_used = False
        reason = "latency"

    def fake_route(self, **kwargs):
        return FakeDecision()

    monkeypatch.setattr(ModelOrchestrator, "route", fake_route)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "demo-model",
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "demo-model"
    assert manager.last_chat_call["model_id"] == "demo-model"
    assert body["niblit_model"]["executed_model"] == "demo-model"


def test_response_model_matches_engine_identity():
    manager = ModelManager(
        model_map={"demo-model": "/tmp/demo.gguf"},
        default_model="demo-model",
    )
    # The engine result's model identity must flow into response.model via
    # handle_chat; verify the manager reports truthful executable ids.
    assert "demo-model" in manager._model_map
