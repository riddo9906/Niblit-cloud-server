import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Niblit's QwenLocalBrain._generate_http() always sends "model": "local" in
# every request so the cloud server can act as a drop-in llama-server
# replacement.  Any of these aliases resolves to the configured default model.
_LOCAL_MODEL_ALIASES: frozenset[str] = frozenset({"local", "llama", "default"})

# Shared defaults for request schemas.
_DEFAULT_TEMPERATURE: float = 0.2
_DEFAULT_MAX_TOKENS: int = 256


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
        logger.info("Loading model from %s (n_ctx=%d, n_threads=%d)", model_path, n_ctx, n_threads)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to run GGUF inference."
            ) from exc
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        logger.info("Model loaded successfully from %s", model_path)

    def chat(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> ModelEngineResult:
        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        choices = response.get("choices")
        if not isinstance(choices, list):
            raise RuntimeError(
                "Model returned unexpected response format: choices must be a list "
                f"(got {type(choices).__name__})."
            )
        if not choices:
            raise RuntimeError(
                "Model returned unexpected response format: choices must not be empty "
                f"(response keys: {sorted(response.keys())})."
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise RuntimeError(
                "Model returned unexpected response format: message.content must be a string."
            )
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
        logger.info(
            "ModelManager initialized: models=%s default=%s n_ctx=%d n_threads=%d",
            list(model_map.keys()),
            self._default_model,
            self._n_ctx,
            self._n_threads,
        )

    def get_default_model_info(self) -> dict[str, str]:
        """Return {model_id, model_path} for the default model (empty strings if none)."""
        model_id = self._default_model or ""
        model_path = self._model_map.get(model_id, "") if model_id else ""
        return {"model_id": model_id, "model_path": model_path}

    def list_models(self) -> list[dict[str, str]]:
        return [{"id": model_id, "object": "model"} for model_id in self._model_map]

    def get_model(self, model_id: str) -> dict[str, str]:
        if model_id not in self._model_map:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return {"id": model_id, "object": "model"}

    def resolve_model_id(self, request_model: str | None, path_model: str | None) -> str:
        """Resolve model precedence as request model > path model > default model.

        The string ``"local"`` (and other well-known aliases set in
        _LOCAL_MODEL_ALIASES) is mapped to the configured default model so that
        Niblit's ``QwenLocalBrain`` HTTP backend — which always sends
        ``"model": "local"`` — works without any client-side configuration.
        """
        model_id = request_model or path_model or self._default_model
        if not model_id:
            raise HTTPException(status_code=400, detail="No model provided.")
        # Resolve well-known aliases (e.g. "local" sent by Niblit's local_brain)
        if model_id.lower() in _LOCAL_MODEL_ALIASES:
            if not self._default_model:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Model alias '{model_id}' received but no default model is "
                        "configured. Set DEFAULT_MODEL_ID and GGUF_MODELS_JSON."
                    ),
                )
            return self._default_model
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
        logger.debug(
            "chat request: model=%s messages=%d temperature=%s max_tokens=%d",
            model_id,
            len(messages),
            temperature,
            max_tokens,
        )
        try:
            result = self._engines[model_id].chat(
                messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except RuntimeError as exc:
            logger.error("Inference error for model %s: %s", model_id, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        logger.debug(
            "chat response: model=%s finish_reason=%s",
            model_id,
            result.finish_reason,
        )
        return result


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = _DEFAULT_TEMPERATURE
    max_tokens: int = _DEFAULT_MAX_TOKENS
    # Accepted for llama-server API compatibility; passed through to the engine.
    stop: list[str] | None = None
    # Tool schemas (Niblit generate_with_tools); accepted and ignored if not supported.
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None


class CompletionRequest(BaseModel):
    """Legacy llama-server /completion endpoint schema.

    Niblit's ``QwenLocalBrain._generate_http_legacy()`` falls back to this
    endpoint when ``POST /v1/chat/completions`` returns HTTP 404.  The
    request body uses ``prompt`` (plain string) and ``n_predict`` (max tokens).
    """

    prompt: str
    n_predict: int = _DEFAULT_MAX_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE
    stop: list[str] | None = None
    model: str | None = None


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
    models = _load_models_from_env()
    default_model = os.getenv("DEFAULT_MODEL_ID")
    _manager = model_manager or ModelManager(models, default_model)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        info = _manager.get_default_model_info()
        logger.info(
            "Niblit Cloud GGUF Server starting — models=%s default=%s",
            list(models.keys()),
            info["model_id"] or "(none)",
        )
        yield
        logger.info("Niblit Cloud GGUF Server shutdown complete.")

    app = FastAPI(title="Niblit Cloud GGUF Server", version="0.1.0", lifespan=lifespan)
    app.state.model_manager = _manager

    @app.middleware("http")
    async def log_requests(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.get("/healthz")
    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check — responds to both /health (llama-server probe) and /healthz."""
        return {"status": "ok"}

    @app.get("/props")
    def props() -> dict[str, Any]:
        """Legacy llama-server /props probe endpoint.

        Niblit's ``_check_server_url`` probes /props as a last resort when
        /health and /v1/models are unavailable.  Returns a minimal JSON object
        so the probe succeeds.
        """
        manager: ModelManager = app.state.model_manager
        info = manager.get_default_model_info()
        return {
            "model_path": info["model_path"],
            "total_slots": 1,
        }

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

    @app.post("/completion")
    def legacy_completion(payload: CompletionRequest) -> dict[str, Any]:
        """Legacy llama-server ``POST /completion`` endpoint.

        Niblit's ``QwenLocalBrain._generate_http_legacy()`` falls back here
        when the /v1/chat/completions endpoint returned HTTP 404.  The prompt
        is wrapped into a single-user chat message and forwarded to the model.
        The response uses the ``{"content": "..."}`` shape that llama-server
        returns.
        """
        manager: ModelManager = app.state.model_manager
        model_id = manager.resolve_model_id(payload.model, None)
        messages: list[dict[str, str]] = [{"role": "user", "content": payload.prompt}]
        result = manager.chat(
            model_id=model_id,
            messages=messages,
            temperature=payload.temperature,
            max_tokens=payload.n_predict,
        )
        return {"content": result.text}

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

        temperature = float(payload.parameters.get("temperature", _DEFAULT_TEMPERATURE))
        max_tokens = int(payload.parameters.get("max_new_tokens", _DEFAULT_MAX_TOKENS))
        result = manager.chat(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {"generated_text": result.text, "model": model_id}

    # Strip spaces and slashes so values like " hf ", "/local/" remain valid.
    compat_prefixes = [
        p.strip(" /")
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
            f"/{prefix}/completion",
            legacy_completion,
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
        app.add_api_route(
            f"/{prefix}/health",
            health,
            methods=["GET"],
        )

    return app


app = create_app()
