import json

import pytest

from app.cognitive_gateway.classifier import classify_request
from app.cognitive_gateway.config import CognitiveGatewayConfig, load_cognitive_gateway_config
from app.cognitive_gateway.context import GatewayContext
from app.cognitive_gateway.gateway import CognitiveGateway
from app.cognitive_gateway.hooks import GatewayEventBus, PlaceholderMemoryHook
from app.cognitive_gateway.router import route_model
from app.main import ModelEngineResult, ModelManager, create_app
from fastapi.testclient import TestClient


class FakeModelManager(ModelManager):
    def __init__(self):
        super().__init__(
            model_map={
                "demo-model": "/tmp/demo.gguf",
                "code-model": "/tmp/code.gguf",
                "reason-model": "/tmp/reason.gguf",
            },
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


def make_client(monkeypatch=None) -> tuple[TestClient, FakeModelManager]:
    if monkeypatch is not None:
        monkeypatch.setenv("ENABLE_COGNITIVE_GATEWAY", "1")
        monkeypatch.setenv("NIBLIT_MO_ENABLED", "0")
        monkeypatch.setattr("app.model_orchestrator._ENABLED", False)
        monkeypatch.setenv(
            "COGNITIVE_GATEWAY_ROUTING_JSON",
            json.dumps(
                {
                    "coding": "code-model",
                    "reasoning": "reason-model",
                    "general_chat": "demo-model",
                    "fallback": "demo-model",
                }
            ),
        )
        import app.cognitive_gateway.gateway as cg
        import app.cursor_gateway as gw

        cg._cognitive_gateway = None
        gw._gateway = None

    manager = FakeModelManager()
    app = create_app(model_manager=manager)
    return TestClient(app), manager


def test_classify_coding_request():
    result = classify_request(
        [{"role": "user", "content": "refactor this python function to use async"}],
        payload=type("P", (), {"tools": None, "tool_choice": None})(),
    )
    assert result.request_type == "coding"
    assert result.compute_priority == "high"


def test_classify_tool_usage_from_tools_field():
    result = classify_request(
        [{"role": "user", "content": "hello"}],
        payload=type("P", (), {"tools": [{"type": "function"}], "tool_choice": "auto"})(),
    )
    assert result.request_type == "tool_usage"


def test_route_model_dynamic_coding():
    config = CognitiveGatewayConfig(
        enabled=True,
        routing_strategy="dynamic",
        routing_rules={"coding": "code-model", "fallback": "demo-model"},
        fallback_model="demo-model",
    )
    decision = route_model(
        config=config,
        request_type="coding",
        available_models=["demo-model", "code-model"],
        requested_model=None,
        default_model="demo-model",
    )
    assert decision.model_id == "code-model"
    assert decision.source == "rule:coding"


def test_route_model_respects_explicit_model():
    config = CognitiveGatewayConfig(
        enabled=True,
        routing_strategy="dynamic",
        routing_rules={"coding": "code-model"},
        respect_explicit_model=True,
    )
    decision = route_model(
        config=config,
        request_type="coding",
        available_models=["demo-model", "code-model"],
        requested_model="demo-model",
        default_model="demo-model",
    )
    assert decision.model_id == "demo-model"
    assert decision.source == "explicit"


def test_route_model_fallback_when_rule_missing():
    config = CognitiveGatewayConfig(
        enabled=True,
        routing_strategy="dynamic",
        routing_rules={"fallback": "demo-model"},
        fallback_model="demo-model",
    )
    decision = route_model(
        config=config,
        request_type="reasoning",
        available_models=["demo-model"],
        requested_model=None,
        default_model="demo-model",
    )
    assert decision.model_id == "demo-model"
    assert decision.fallback_used is True


def test_gateway_event_bus_is_synchronous():
    bus = GatewayEventBus()
    seen: list[str] = []

    def handler(payload: dict) -> None:
        seen.append(payload["event"])

    bus.subscribe(GatewayEventBus.EVENT_REQUEST_RECEIVED, handler)
    bus.emit(GatewayEventBus.EVENT_REQUEST_RECEIVED, {"event": "received"})
    assert seen == ["received"]


def test_memory_hook_placeholder_is_noop():
    hook = PlaceholderMemoryHook()
    assert hook.before_request({}) == {}
    assert hook.after_response({}) is None


def test_cognitive_gateway_prepare_routes_coding():
    from app.main import ChatCompletionRequest

    gateway = CognitiveGateway(
        CognitiveGatewayConfig(
            enabled=True,
            routing_strategy="dynamic",
            routing_rules={"coding": "code-model", "fallback": "demo-model"},
            fallback_model="demo-model",
        )
    )
    payload = ChatCompletionRequest(
        messages=[{"role": "user", "content": "fix this bug in my typescript code"}],
    )
    adapted, ctx = gateway.prepare_chat_request(
        payload,
        path_model=None,
        available_models=["demo-model", "code-model"],
        default_model="demo-model",
    )
    assert ctx.request_type == "coding"
    assert adapted.model == "code-model"


def test_chat_routes_coding_to_code_model(monkeypatch):
    client, manager = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "implement a rust function for parsing json"}],
        },
    )
    assert response.status_code == 200
    assert manager.last_chat_call["model_id"] == "code-model"


def test_chat_respects_explicit_model_selection(monkeypatch):
    client, manager = make_client(monkeypatch)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo-model",
            "messages": [{"role": "user", "content": "refactor this python module"}],
        },
    )
    assert response.status_code == 200
    assert manager.last_chat_call["model_id"] == "demo-model"


def test_cognitive_gateway_disabled_passthrough(monkeypatch):
    monkeypatch.setenv("ENABLE_COGNITIVE_GATEWAY", "0")
    monkeypatch.setenv("NIBLIT_MO_ENABLED", "0")
    monkeypatch.setattr("app.model_orchestrator._ENABLED", False)
    import app.cognitive_gateway.gateway as cg
    import app.cursor_gateway as gw

    cg._cognitive_gateway = None
    gw._gateway = None

    client, manager = make_client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello there"}],
        },
    )
    assert response.status_code == 200
    assert manager.last_chat_call["model_id"] == "demo-model"
