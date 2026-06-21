from app.main import create_app
from fastapi.testclient import TestClient


def test_shared_contract_bridge_endpoint_returns_completed_message():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/bridge/inference",
        json={
            "message_type": "ai.inference.requested",
            "source": "niblit",
            "target": "niblit-cloud-server",
            "schema_version": "1.0",
            "correlation_id": "corr-bridge-1",
            "payload": {
                "model_id": "demo-model",
                "prompt": "Summarize the market regime",
                "market_snapshot": {"symbol": "BTCUSDT", "price": 50000.0, "volume": 120.0},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message_type"] == "ai.inference.completed"
    assert payload["correlation_id"] == "corr-bridge-1"
    assert payload["payload"]["response_text"].startswith("Bridge response")
    assert payload["payload"]["model_id"] == "demo-model"
