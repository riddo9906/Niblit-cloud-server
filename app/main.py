import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _load_models_from_env() -> dict[str, str]:
    raw = os.getenv("GGUF_MODELS_JSON", "{}").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GGUF_MODELS_JSON is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("GGUF_MODELS_JSON must be a JSON object")
    return {str(k): str(v) for k, v in loaded.items()}


@dataclass
class ModelEngineResult:
    text: str
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None


class GGUFEngine:
    def __init__(self, model_path: str, n_ctx: int, n_threads: int):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to run GGUF inference."
            ) from exc
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)

    def chat(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> ModelEngineResult:
        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        choice = response["choices"][0]
        text = choice["message"]["content"]
        usage = response.get("usage")
        return ModelEngineResult(
            text=text,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=usage,
        )


class ModelManager:
    def __init__(self, model_map: dict[str, str], default_model: str | None):
        self._model_map = model_map
        self._default_model = default_model or (next(iter(model_map), None))
        self._engines: dict[str, GGUFEngine] = {}
        self._n_ctx = int(os.getenv("N_CTX", "4096"))
        self._n_threads = int(os.getenv("N_THREADS", "4"))

    def list_models(self) -> list[dict[str, str]]:
        return [{"id": model_id, "object": "model"} for model_id in self._model_map]

    def get_model(self, model_id: str) -> dict[str, str]:
        if model_id not in self._model_map:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return {"id": model_id, "object": "model"}

    def resolve_model_id(self, request_model: str | None, path_model: str | None) -> str:
        model_id = request_model or path_model or self._default_model
        if not model_id:
            raise HTTPException(status_code=400, detail="No model provided.")
        if model_id not in self._model_map:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return model_id

    def chat(
        self, model_id: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> ModelEngineResult:
        if model_id not in self._engines:
            self._engines[model_id] = GGUFEngine(
                model_path=self._model_map[model_id],
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
            )
        return self._engines[model_id].chat(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 256


class InferenceRequest(BaseModel):
    inputs: str | None = None
    messages: list[dict[str, Any]] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    normalized: list[dict[str, str]] = []
    for item in messages:
        role = str(item.get("role", "user"))
        content = item.get("content")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="message content must be a string")
        normalized.append({"role": role, "content": content})
    return normalized


def _build_chat_response(model_id: str, result: ModelEngineResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
    }
    if result.usage:
        payload["usage"] = result.usage
    return payload


def create_app(model_manager: ModelManager | None = None) -> FastAPI:
    app = FastAPI(title="Niblit Cloud GGUF Server", version="0.1.0")
    models = _load_models_from_env()
    default_model = os.getenv("DEFAULT_MODEL_ID")
    app.state.model_manager = model_manager or ModelManager(models, default_model)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    @app.get("/models")
    def list_models() -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        return {"object": "list", "data": manager.list_models()}

    @app.get("/v1/models/{model_id}")
    @app.get("/models/{model_id}/info")
    def get_model(model_id: str) -> dict[str, str]:
        manager: ModelManager = app.state.model_manager
        return manager.get_model(model_id)

    def handle_chat(payload: ChatCompletionRequest, path_model: str | None = None) -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        model_id = manager.resolve_model_id(payload.model, path_model)
        messages = _normalize_messages(payload.messages)
        result = manager.chat(
            model_id=model_id,
            messages=messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        return _build_chat_response(model_id, result)

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    @app.post("/v1/models/{path_model}/chat/completions")
    def chat_completions(payload: ChatCompletionRequest, path_model: str | None = None) -> dict[str, Any]:
        return handle_chat(payload, path_model)

    @app.post("/models/{path_model}")
    def inference_api(path_model: str, payload: InferenceRequest) -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        model_id = manager.resolve_model_id(payload.model, path_model)
        if payload.messages:
            messages = _normalize_messages(payload.messages)
        elif payload.inputs:
            messages = [{"role": "user", "content": payload.inputs}]
        else:
            raise HTTPException(status_code=400, detail="Provide either inputs or messages")

        temperature = float(payload.parameters.get("temperature", 0.2))
        max_tokens = int(payload.parameters.get("max_new_tokens", 256))
        result = manager.chat(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {"generated_text": result.text, "model": model_id}

    compat_prefixes = [
        p.strip().strip("/")
        for p in os.getenv("COMPAT_PREFIXES", "hf,local,kimi,claude").split(",")
        if p.strip()
    ]
    for prefix in compat_prefixes:
        app.add_api_route(
            f"/{prefix}/v1/chat/completions",
            chat_completions,
            methods=["POST"],
        )
        app.add_api_route(
            f"/{prefix}/chat/completions",
            chat_completions,
            methods=["POST"],
        )
        app.add_api_route(
            f"/{prefix}/models/{{path_model}}",
            inference_api,
            methods=["POST"],
        )
        app.add_api_route(
            f"/{prefix}/v1/models",
            list_models,
            methods=["GET"],
        )

    return app


app = create_app()
