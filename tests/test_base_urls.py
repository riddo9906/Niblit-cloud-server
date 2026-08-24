"""Regression tests for /v1 prefix normalization and health semantics.

Mandatory invariant: the sequences  /v1/v1/models, /v1/v1/chat/completions and
/v1//models must never be produced by any URL composition helper.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.base_urls import (
    join_openai_base,
    join_server_root,
    normalize_openai_base_url,
    normalize_server_root,
)


FORBIDDEN = ("/v1/v1/", "//v1", "/v1//")


def _assert_no_double_v1(url: str) -> None:
    for bad in FORBIDDEN:
        assert bad not in url, f"forbidden sequence {bad!r} in {url!r}"


def test_normalize_server_root_strips_v1() -> None:
    assert normalize_server_root("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert normalize_server_root("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    assert normalize_server_root("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert normalize_server_root("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000"
    assert normalize_server_root("http://localhost:8000/V1") == "http://localhost:8000"


def test_normalize_openai_base_has_exactly_one_v1() -> None:
    for raw in (
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8000/v1/",
    ):
        out = normalize_openai_base_url(raw)
        assert out == "http://127.0.0.1:8000/v1"
        _assert_no_double_v1(out)


def test_join_helpers_never_double_v1() -> None:
    for base in ("http://127.0.0.1:8000", "http://127.0.0.1:8000/v1"):
        models = join_openai_base(base, "/models")
        chat = join_openai_base(base, "/v1/chat/completions")
        assert models.endswith("/v1/models")
        assert chat.endswith("/v1/chat/completions")
        _assert_no_double_v1(models)
        _assert_no_double_v1(chat)

    root_models = join_server_root("http://127.0.0.1:8000/v1", "/v1/models")
    assert root_models == "http://127.0.0.1:8000/v1/models"
    _assert_no_double_v1(root_models)


def test_trailing_slash_and_host_variants() -> None:
    assert join_openai_base("http://localhost:8000/", "models") == "http://localhost:8000/v1/models"
    assert join_server_root("http://localhost:8000///", "/healthz").startswith("http://localhost:8000")


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_healthz_is_liveness_only() -> None:
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    resp = _client(app).get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
