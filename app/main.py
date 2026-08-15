import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.cursor_gateway import _detail_to_message, get_cursor_gateway, openai_error_body

logger = logging.getLogger(__name__)

def _try_import(module_path: str) -> Any:
    try:
        import importlib
        return importlib.import_module(module_path)
    except Exception:
        return None

_LOCAL_MODEL_ALIASES: frozenset[str] = frozenset({"local", "llama", "default", "primary"})
_DEFAULT_TEMPERATURE: float = 0.2
_DEFAULT_MAX_TOKENS: int = 256
_CANONICAL_MODES: tuple[str, ...] = ("normal", "cautious", "survival", "lockdown")
_MESSAGE_OVERHEAD_CHARS: int = 8


def _get_config() -> Any:
    from app.config import get_config
    return get_config()


def _normalize_runtime_mode(mode: object, default: str = "normal") -> str:
    candidate = str(mode or default).strip().lower()
    if candidate in ("minimal", "constrained"):
        candidate = "cautious"
    if candidate not in _CANONICAL_MODES:
        return default
    return candidate


@dataclass
class ModelEngineResult:
    text: str
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None


class GGUFEngine:
    """GGUF inference engine with crash protection and diagnostics.

    Root cause of segfault: unsupported constructor parameters (n_batch,
    n_ubatch, rope_freq_base, rope_freq_scale) passed to Llama() in
    llama-cpp-python 0.3.16 on Windows cause native memory corruption.
    Additional fix: disable mmap on Windows (known segfault with GGUF),
    validate prompt length before inference, wrap all calls in try/except.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        runtime_options: dict[str, Any] | None = None,
    ):
        logger.info(
            "Loading model from %s (n_ctx=%d, n_threads=%d)",
            model_path, n_ctx, n_threads,
        )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to run GGUF inference."
            ) from exc

        # ── Safe parameters only ──────────────────────────────────────────
        # Avoid: n_batch, n_ubatch, rope_freq_base, rope_freq_scale
        # (unsupported in 0.3.16, cause segfault via native memory corruption).
        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "use_mmap": False,            # Windows + mmap = segfault
            "use_mlock": False,           # safest: no mlock
            "seed": 42,                    # avoids undefined native state
            "verbose": False,              # suppress excessive stderr
        }
        gpu_layers = os.getenv("NIBLIT_N_GPU_LAYERS", "").strip()
        if gpu_layers:
            try:
                kwargs["n_gpu_layers"] = int(gpu_layers)
            except (TypeError, ValueError):
                pass

        logger.info("Llama kwargs: %s", {k: v for k, v in kwargs.items() if k != "model_path"})
        try:
            self._llm = Llama(**kwargs)
        except Exception as exc:
            logger.error("Llama init failed: %s", exc)
            raise RuntimeError(f"Model initialization failed: {exc}") from exc

        self._model_meta: dict[str, Any] = {
            "file": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
        }
        self._lock = threading.Lock()
        logger.info("Model loaded successfully from %s", model_path)

    def _reload_model(self) -> None:
        """Reload the GGUF model to recover from native state corruption.

        Called when llama_decode returns -1 (GGML_ASSERT failure), which
        indicates the native llama.cpp context has been corrupted by a
        previous large generation. Reloading creates a fresh context.
        """
        model_path = self._model_meta["file"]
        n_ctx = self._model_meta["n_ctx"]
        n_threads = self._model_meta["n_threads"]
        logger.info("Reloading model: %s", model_path)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is required.") from exc
        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "use_mmap": False,
            "use_mlock": False,
            "seed": 42,
            "verbose": False,
        }
        gpu_layers = os.getenv("NIBLIT_N_GPU_LAYERS", "").strip()
        if gpu_layers:
            try:
                kwargs["n_gpu_layers"] = int(gpu_layers)
            except (TypeError, ValueError):
                pass
        with self._lock:
            self._llm = Llama(**kwargs)
        logger.info("Model reloaded successfully: %s", model_path)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._model_meta)

    def _estimate_tokens_rough(self, messages: list[dict[str, str]]) -> int:
        total = sum(len(m.get("content", "")) for m in messages)
        return max(1, total // 4)

    def validate_prompt(self, messages: list[dict[str, str]], max_tokens: int) -> None:
        prompt_tokens = self._estimate_tokens_rough(messages)
        available = self._model_meta.get("n_ctx", 16384)
        reserve = 256
        budget = available - reserve
        if prompt_tokens > budget:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "context_overflow",
                    "message": (
                        f"Prompt ~{prompt_tokens} tokens exceeds "
                        f"budget of {budget} (context={available}, reserve={reserve})."
                    ),
                    "prompt_tokens": prompt_tokens,
                    "max_context": available,
                    "budget": budget,
                },
            )

    def chat(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> ModelEngineResult:
        self.validate_prompt(messages, max_tokens)

        safe_messages: list[dict[str, str]] = []
        for msg in messages:
            safe_messages.append({
                "role": str(msg.get("role", "user")).strip(),
                "content": str(msg.get("content", " ")).strip() or " ",
            })

        logger.info(
            "Inference: messages=%d prompt~%d max_tokens=%d temp=%.2f",
            len(safe_messages), self._estimate_tokens_rough(safe_messages),
            max_tokens, temperature,
        )

        # llama-cpp-python 0.3.16 Llama object is NOT thread-safe.
        # Serialize all create_chat_completion() calls to prevent
        # concurrent native access that causes llama_decode returned -1
        # and GGML_ASSERT failures.
        with self._lock:
            try:
                response = self._llm.create_chat_completion(
                messages=safe_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            except RuntimeError as exc:
                exc_str = str(exc)
                logger.error("llama.cpp error: %s", exc_str)
                if "llama_decode returned" in exc_str or "GGML_ASSERT" in exc_str:
                    logger.info("Attempting model reload after decode error...")
                    try:
                        self._reload_model()
                        response = self._llm.create_chat_completion(
                            messages=safe_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=False,
                        )
                    except Exception as retry_exc:
                        logger.error("Retry after reload failed: %s", retry_exc)
                        raise HTTPException(
                            status_code=503,
                            detail={"error": "inference_failed", "message": str(retry_exc)},
                        ) from retry_exc
                else:
                    raise HTTPException(
                        status_code=503,
                        detail={"error": "inference_failed", "message": exc_str},
                    ) from exc
            except ValueError as exc:
                logger.error("llama.cpp value error: %s", exc)
                raise HTTPException(status_code=503, detail={"error": "inference_failed", "message": str(exc)}) from exc
            except MemoryError as exc:
                logger.error("llama.cpp OOM: %s", exc)
                raise HTTPException(status_code=503, detail={"error": "out_of_memory", "message": str(exc)}) from exc

            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise HTTPException(status_code=502, detail={"error": "empty_response", "message": "Model returned empty choices."})
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise HTTPException(status_code=502, detail={"error": "invalid_response", "message": "message.content must be a string."})
            usage = response.get("usage")
            logger.info("Inference OK: finish=%s tokens=%s", choice.get("finish_reason"), usage.get("total_tokens") if usage else "?")
            return ModelEngineResult(
                text=text,
                finish_reason=choice.get("finish_reason", "stop"),
                usage=usage,
            )


class ModelManager:
    def __init__(self, model_map: dict[str, str], default_model: str | None, *, config: Any | None = None):
        self._model_map = model_map
        self._default_model = default_model or (next(iter(model_map), None))
        self._config = config
        self._engines: dict[str, GGUFEngine] = {}
        if config is not None:
            self._n_ctx = config.n_ctx
            self._n_threads = config.n_threads
            self._n_batch = config.n_batch
            self._n_ubatch = config.n_ubatch
            self._context_reserve_tokens = config.context_reserve_tokens
            self._min_generation_tokens = config.min_generation_tokens
            self._char_to_token_ratio = config.char_per_token
            self._memory_guard_ratio = config.memory_guard_ratio
            self._runtime_options = {
                "n_batch": config.n_batch,
                "n_ubatch": config.n_ubatch,
                "rope_freq_base": config.rope_freq_base,
                "rope_freq_scale": config.rope_freq_scale,
            }
        else:
            def _env_int(name: str, default: int) -> int:
                raw = os.getenv(name)
                if raw is None:
                    return default
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return default
            def _env_float(name: str, default: float | None = None) -> float | None:
                raw = os.getenv(name)
                if raw is None:
                    return default
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return default
            self._n_ctx = _env_int("N_CTX", _env_int("NIBLIT_CONTEXT_WINDOW", 16384))
            self._n_threads = _env_int("N_THREADS", 4)
            self._n_batch = _env_int("NIBLIT_N_BATCH", _env_int("N_BATCH", 1024))
            self._n_ubatch = _env_int("NIBLIT_N_UBATCH", _env_int("N_UBATCH", 512))
            self._context_reserve_tokens = _env_int("NIBLIT_CONTEXT_RESERVE_TOKENS", 512)
            self._min_generation_tokens = _env_int("NIBLIT_MIN_GENERATION_TOKENS", 64)
            self._char_to_token_ratio = max(1, _env_int("NIBLIT_CHAR_PER_TOKEN", 4))
            self._memory_guard_ratio = max(
                0.5, min(0.98, _env_float("NIBLIT_MEMORY_GUARD_RATIO", 0.92) or 0.92),
            )
            self._runtime_options = {
                "n_batch": self._n_batch,
                "n_ubatch": self._n_ubatch,
                "rope_freq_base": _env_float("NIBLIT_ROPE_FREQ_BASE", None),
                "rope_freq_scale": _env_float("NIBLIT_ROPE_FREQ_SCALE", None),
            }
        self._stats: dict[str, Any] = {
            "requests_total": 0,
            "context_trim_events": 0,
            "max_token_clamp_events": 0,
            "last_prompt_tokens_estimate": 0,
            "last_effective_max_tokens": 0,
            "last_context_usage_ratio": 0.0,
        }
        self._lock = threading.Lock()
        logger.info(
            "ModelManager initialized: models=%s default=%s n_ctx=%d n_threads=%d",
            list(model_map.keys()),
            self._default_model,
            self._n_ctx,
            self._n_threads,
        )

    def get_default_model_info(self) -> dict[str, str]:
        with self._lock:
            model_id = self._default_model or ""
            model_path = self._model_map.get(model_id, "") if model_id else ""
        return {"model_id": model_id, "model_path": model_path}

    def get_active_model_id(self) -> str | None:
        with self._lock:
            return self._default_model

    def set_active_model(self, model_id: str) -> str:
        with self._lock:
            if model_id not in self._model_map:
                raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
            previous = self._default_model
            self._default_model = model_id
        logger.info("Active model switched: %s -> %s", previous, model_id)
        return previous or ""

    def reload_model(self, model_id: str) -> bool:
        with self._lock:
            if model_id not in self._model_map:
                raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
            model_path = self._model_map[model_id]
        if not os.path.isfile(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        try:
            engine = GGUFEngine(
                model_path=model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                runtime_options=self._runtime_options,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        with self._lock:
            self._engines[model_id] = engine
        logger.info("Model reloaded successfully: %s", model_id)
        return True

    def list_models(self) -> list[dict[str, str]]:
        return [{"id": model_id, "object": "model"} for model_id in self._model_map]

    def get_model(self, model_id: str) -> dict[str, str]:
        if model_id not in self._model_map:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return {"id": model_id, "object": "model"}

    def list_models_detailed(self) -> list[dict[str, Any]]:
        result = []
        for mid, path in self._model_map.items():
            result.append({
                "id": mid,
                "object": "model",
                "path": path,
                "loaded": mid in self._engines,
            })
        return result

    def resolve_model_id(self, request_model: str | None, path_model: str | None) -> str:
        with self._lock:
            default = self._default_model
        model_id = request_model or path_model or default
        if not model_id:
            raise HTTPException(status_code=400, detail="No model provided.")
        raw = model_id.lower()
        if raw in _LOCAL_MODEL_ALIASES:
            if not default:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Model alias '{model_id}' received but no default model is "
                        "configured. Set DEFAULT_MODEL_ID and GGUF_MODELS_JSON."
                    ),
                )
            return default
        if model_id not in self._model_map:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return model_id

    def chat(
        self, model_id: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> ModelEngineResult:
        with self._lock:
            if model_id not in self._engines:
                self._engines[model_id] = GGUFEngine(
                    model_path=self._model_map[model_id],
                    n_ctx=self._n_ctx,
                    n_threads=self._n_threads,
                    runtime_options=self._runtime_options,
                )
            engine = self._engines[model_id]
        plan = self._prepare_inference(messages=messages, max_tokens=max_tokens)
        logger.debug(
            "chat request: model=%s messages=%d temperature=%s max_tokens=%d effective_max_tokens=%d",
            model_id, len(messages), temperature, max_tokens, plan["effective_max_tokens"],
        )
        try:
            result = engine.chat(
                messages=plan["messages"], temperature=temperature, max_tokens=plan["effective_max_tokens"]
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            logger.error("Inference error for model %s: %s", model_id, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        with self._lock:
            self._stats["requests_total"] += 1
            if plan["messages_truncated"]:
                self._stats["context_trim_events"] += 1
            if plan["effective_max_tokens"] < max_tokens:
                self._stats["max_token_clamp_events"] += 1
            self._stats["last_prompt_tokens_estimate"] = plan["prompt_tokens_estimate"]
            self._stats["last_effective_max_tokens"] = plan["effective_max_tokens"]
            self._stats["last_context_usage_ratio"] = round(plan["context_usage_ratio"], 4)
        return result

    def estimate_inference(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        return self._prepare_inference(messages=messages, max_tokens=max_tokens)

    def runtime_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "context_window": self._n_ctx,
                "n_batch": self._n_batch,
                "n_ubatch": self._n_ubatch,
                "context_reserve_tokens": self._context_reserve_tokens,
                "memory_guard_ratio": self._memory_guard_ratio,
            }

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        total_chars = 0
        for msg in messages:
            total_chars += (
                len(msg.get("content", ""))
                + len(msg.get("role", ""))
                + _MESSAGE_OVERHEAD_CHARS
            )
        return max(1, total_chars // self._char_to_token_ratio)

    def _prepare_inference(self, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        source_messages = [dict(m) for m in messages]
        safe_messages = [dict(m) for m in messages]
        requested_max_tokens = max(1, int(max_tokens))
        target_ctx_budget = max(256, int(self._n_ctx * self._memory_guard_ratio))
        prompt_tokens = self._estimate_tokens(safe_messages)
        min_generation = max(16, min(self._min_generation_tokens, requested_max_tokens))
        max_prompt_budget = max(64, target_ctx_budget - max(self._context_reserve_tokens, min_generation))
        messages_truncated = False
        if prompt_tokens > max_prompt_budget and safe_messages:
            kept: list[dict[str, str]] = []
            running_tokens = 0
            for msg in reversed(safe_messages):
                msg_tokens = self._estimate_tokens([msg])
                if kept and (running_tokens + msg_tokens) > max_prompt_budget:
                    messages_truncated = True
                    continue
                kept.append(msg)
                running_tokens += msg_tokens
                if running_tokens >= max_prompt_budget:
                    break
            safe_messages = list(reversed(kept)) or [source_messages[-1]]
            if safe_messages:
                prompt_tokens = self._estimate_tokens(safe_messages)
            if prompt_tokens > max_prompt_budget and safe_messages:
                last = dict(safe_messages[-1])
                max_chars = max(64, max_prompt_budget * self._char_to_token_ratio)
                if len(last.get("content", "")) > max_chars:
                    last["content"] = last["content"][-max_chars:]
                    safe_messages[-1] = last
                    messages_truncated = True
                prompt_tokens = self._estimate_tokens(safe_messages)
        available_for_generation = max(1, target_ctx_budget - prompt_tokens - self._context_reserve_tokens)
        effective_max_tokens = max(1, min(requested_max_tokens, available_for_generation))
        context_usage_ratio = (prompt_tokens + effective_max_tokens) / max(1, self._n_ctx)
        memory_pressure = min(1.0, (prompt_tokens + requested_max_tokens) / max(1, target_ctx_budget))
        return {
            "messages": safe_messages,
            "messages_truncated": messages_truncated,
            "prompt_tokens_estimate": prompt_tokens,
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "context_usage_ratio": context_usage_ratio,
            "memory_pressure": memory_pressure,
            "target_ctx_budget": target_ctx_budget,
        }


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = _DEFAULT_TEMPERATURE
    max_tokens: int = _DEFAULT_MAX_TOKENS
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    intent: str | None = None
    execution_mode: str | None = None
    coherence_score: float | None = None
    constitutional_priority: str | None = None
    attention_budget: float | None = None
    resource_mode: str | None = None
    epoch_tag: str | None = None
    forecast_context: dict[str, Any] | None = None
    governance_context: dict[str, Any] | None = None
    tool_context: dict[str, Any] | None = None
    reflection_context: dict[str, Any] | None = None
    identity_context: dict[str, Any] | None = None
    governance: dict[str, Any] | None = None
    temporal: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
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


def create_app(model_manager: Any | None = None, config: Any | None = None) -> FastAPI:
    if config is None:
        config = _get_config()
    if model_manager is not None:
        _manager = model_manager
    else:
        models = config.model_map
        default_model = config.default_model
        _manager = ModelManager(models, default_model, config=config)
    _model_registry = _try_import_model_registry()

    _event_bus_mod   = _try_import("app.event_bus")
    _envelope_mod    = _try_import("app.cognitive_envelope")
    _governance_mod  = _try_import("app.cloud_governance")
    _temporal_mod    = _try_import("app.temporal_sync")
    _reflection_mod  = _try_import("app.reflection_engine")
    _orchestrator_mod = _try_import("app.model_orchestrator")
    _attention_mod   = _try_import("app.attention_allocator")
    _trading_mod     = _try_import("app.trading_runtime_bridge")
    _identity_mod    = _try_import("app.node_identity")
    _federation_mod  = _try_import("app.federation")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        info = _manager.get_default_model_info()
        logger.info(
            "Niblit Cognitive Cloud Runtime: models=%s default=%s",
            list(models.keys()), info["model_id"] or "(none)",
        )
        if _orchestrator_mod:
            _orchestrator_mod.get_model_orchestrator(model_ids=list(models.keys()))
        if _identity_mod:
            _identity_mod.get_node_identity()
        if _federation_mod:
            _federation_mod.get_federation_manager()
        yield
        logger.info("Niblit Cognitive Cloud Runtime shutdown.")

    app = FastAPI(
        title="Niblit Cognitive Cloud Runtime",
        version="0.9.0",
        description="Niblit Cloud Server — inference layer. Backward compatible with OpenAI, HF, llama.cpp.",
        lifespan=lifespan,
    )
    app.state.model_manager = _manager
    app.state.cloud_config = config
    _cursor_gateway = get_cursor_gateway()

    @app.exception_handler(HTTPException)
    async def openai_http_exception_handler(request: Request, exc: HTTPException):
        if not _cursor_gateway.enabled:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        if not request.url.path.startswith("/v1/"):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        detail = exc.detail
        if isinstance(detail, dict) and "message" in detail:
            message = str(detail["message"])
        elif isinstance(detail, str):
            message = detail
        else:
            message = json.dumps(detail)
        _cursor_gateway.log_upstream_error(status_code=exc.status_code, detail=message)
        return JSONResponse(
            status_code=exc.status_code,
            content=openai_error_body(message, status_code=exc.status_code),
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, duration_ms)
        if _identity_mod:
            try:
                _identity_mod.get_node_identity().increment_request()
            except Exception:
                pass
        return response

    @app.get("/healthz")
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/props")
    def props() -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        info = manager.get_default_model_info()
        return {"model_path": info["model_path"], "total_slots": 1}

    @app.get("/v1/models")
    @app.get("/models")
    def list_models(request: Request) -> dict[str, Any]:
        start = _cursor_gateway.log_incoming(request, path="/v1/models", model=None)
        manager: ModelManager = app.state.model_manager
        payload = {"object": "list", "data": manager.list_models()}
        adapted = _cursor_gateway.adapt_models_response(payload)
        _cursor_gateway.log_complete(start, status_code=200)
        return adapted

    @app.get("/v1/models/detail")
    def list_models_detailed() -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        config: Any = getattr(app.state, "cloud_config", None)
        aliases = {alias: (config.default_model_id or "") for alias in sorted(_LOCAL_MODEL_ALIASES)}
        return {
            "default": config.default_model_id or "",
            "aliases": {k: v for k, v in aliases.items() if v},
            "providers": {
                "qwen_server": bool(config.qwen_model_path and config.qwen_backend_bin),
                "ollama": bool(config.ollama_base_url),
                "openai": bool(config.openai_api_base or config.openai_api_key),
                "anthropic": bool(config.anthropic_api_key),
                "huggingface": bool(config.hf_api_key),
                "llama_cpp": bool(config.llama_cpp_server_url),
                "vllm": bool(config.vllm_api_base),
                "remote": bool(config.remote_api_base_url),
            },
            "models": manager.list_models_detailed(),
        }

    @app.get("/v1/config")
    def get_config_endpoint() -> dict[str, Any]:
        config: Any = getattr(app.state, "cloud_config", None)
        if config is None:
            return {"error": "config_not_loaded"}
        return config.summary()

    @app.get("/v1/models/{model_id}")
    @app.get("/models/{model_id}/info")
    def get_model(model_id: str) -> dict[str, str]:
        manager: ModelManager = app.state.model_manager
        return manager.get_model(model_id)

    def _extract_envelope(payload: ChatCompletionRequest) -> dict[str, Any]:
        if _envelope_mod is None:
            return {}
        raw: dict[str, Any] = {}
        if payload.intent is not None:
            raw["intent"] = payload.intent
        if payload.execution_mode is not None:
            raw["execution_mode"] = payload.execution_mode
        if payload.coherence_score is not None:
            raw["coherence_score"] = payload.coherence_score
        if payload.constitutional_priority is not None:
            raw["constitutional_priority"] = payload.constitutional_priority
        if payload.attention_budget is not None:
            raw["attention_budget"] = payload.attention_budget
        if payload.resource_mode is not None:
            raw["resource_mode"] = payload.resource_mode
        if payload.epoch_tag is not None:
            raw["epoch_tag"] = payload.epoch_tag
        if payload.forecast_context is not None:
            raw["forecast_context"] = payload.forecast_context
        if payload.governance_context is not None:
            raw["governance_context"] = payload.governance_context
        if payload.governance is not None:
            raw["governance"] = payload.governance
        if payload.temporal is not None:
            raw["temporal"] = payload.temporal
        if payload.runtime is not None:
            raw["runtime"] = payload.runtime
        if payload.tool_context is not None:
            raw["tool_context"] = payload.tool_context
        if payload.reflection_context is not None:
            raw["reflection_context"] = payload.reflection_context
        if payload.identity_context is not None:
            raw["identity_context"] = payload.identity_context
        return _envelope_mod.normalize_envelope(raw)

    @app.get("/v1/runtime")
    def runtime_detail() -> dict[str, Any]:
        """Health diagnostic endpoint for inference runtime."""
        manager: ModelManager = app.state.model_manager
        engine_info = {}
        for mid in manager._model_map:
            engine = manager._engines.get(mid)
            if engine:
                engine_info[mid] = {
                    "loaded": True,
                    "model_path": manager._model_map[mid],
                    "n_ctx": manager._n_ctx,
                    "n_threads": manager._n_threads,
                    "metadata": getattr(engine, "metadata", {}),
                }
            else:
                engine_info[mid] = {
                    "loaded": False,
                    "model_path": manager._model_map[mid],
                }
        return {
            "runtime": "niblit_cognitive_cloud_runtime",
            "models": manager.list_models(),
            "default_model": manager.get_default_model_info(),
            "engines": engine_info,
            "context_runtime": manager.runtime_stats(),
        }

    def handle_chat(payload: ChatCompletionRequest, path_model: str | None = None) -> dict[str, Any]:
        start_ms = time.perf_counter() * 1000
        request_id = uuid.uuid4().hex
        manager: ModelManager = app.state.model_manager
        messages = _normalize_messages(payload.messages)
        inference_plan = manager.estimate_inference(messages=messages, max_tokens=payload.max_tokens)
        envelope = _extract_envelope(payload)
        if inference_plan["memory_pressure"] >= 0.85:
            envelope["resource_mode"] = "conservative"
            runtime_ctx = dict(envelope.get("runtime") or {})
            runtime_ctx["mode"] = "cautious"
            runtime_ctx["attention_pressure"] = min(1.0, inference_plan["memory_pressure"])
            envelope["runtime"] = runtime_ctx
        allocation = None
        if _attention_mod:
            try:
                allocation = _attention_mod.get_attention_allocator().score_request(
                    request_id, envelope=envelope
                )
                if not allocation.granted:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "attention_overload",
                            "message": "Runtime is overloaded. Request denied by attention allocator.",
                            "rationale": allocation.to_dict(),
                        },
                    )
            except HTTPException:
                raise
            except Exception:
                pass
        governance_vetoed = False
        if _governance_mod:
            try:
                verdict = _governance_mod.get_cloud_governance().validate(
                    action="chat_inference",
                    context={
                        "prompt_tokens_estimate": inference_plan["prompt_tokens_estimate"],
                        "estimated_total_tokens": inference_plan["prompt_tokens_estimate"] + payload.max_tokens,
                        "context_window": manager.runtime_stats().get("context_window"),
                    },
                    envelope=envelope,
                    messages=messages,
                    max_tokens=payload.max_tokens,
                )
                if not verdict.allowed:
                    governance_vetoed = True
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "governance_violation",
                            "message": verdict.reason,
                            "violated": verdict.violated,
                            "rationale": verdict.rationale,
                        },
                    )
            except HTTPException:
                raise
            except Exception:
                pass
        if _temporal_mod:
            try:
                _temporal_mod.get_temporal_sync().record_request(
                    coherence=float(envelope.get("coherence_score", 1.0)),
                    epoch_tag=str(envelope.get("epoch_tag", "")),
                )
            except Exception:
                pass
        available_models = list(manager._model_map.keys())
        request_model_id = manager.resolve_model_id(payload.model, path_model)
        if _orchestrator_mod and len(available_models) > 1:
            try:
                decision = _orchestrator_mod.get_model_orchestrator().route(
                    available_models=available_models,
                    default_model=request_model_id,
                    envelope=envelope,
                )
                if decision.model_id in available_models:
                    request_model_id = decision.model_id
            except Exception:
                pass
        result = manager.chat(
            model_id=request_model_id,
            messages=messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        end_ms = time.perf_counter() * 1000
        latency_ms = end_ms - start_ms
        token_count = (result.usage or {}).get("completion_tokens", 0)
        if _orchestrator_mod:
            try:
                _orchestrator_mod.get_model_orchestrator().record_outcome(
                    request_model_id, success=True, latency_ms=latency_ms, quality=0.8
                )
            except Exception:
                pass
        if _reflection_mod:
            try:
                quality = min(1.0, max(0.0, 1.0 - latency_ms / 10_000))
                _reflection_mod.get_reflection_engine().record_turn(
                    quality=quality,
                    latency_ms=latency_ms,
                    coherence=float(envelope.get("coherence_score", 1.0)),
                    model_id=request_model_id,
                    intent=str(envelope.get("intent", "conversational")),
                    governance_vetoed=governance_vetoed,
                    token_count=token_count,
                )
            except Exception:
                pass
        if _attention_mod and allocation:
            try:
                _attention_mod.get_attention_allocator().release(request_id)
            except Exception:
                pass
        response = _build_chat_response(request_model_id, result)
        if envelope.get("intent"):
            response["cognitive"] = {
                "request_id": request_id,
                "intent": envelope.get("intent"),
                "governance_mode": (envelope.get("governance") or {}).get("governance_mode", "normal"),
                "latency_ms": round(latency_ms, 1),
                "context": {
                    "window": manager.runtime_stats().get("context_window"),
                    "prompt_tokens_estimate": inference_plan["prompt_tokens_estimate"],
                    "effective_max_tokens": manager.runtime_stats().get("last_effective_max_tokens"),
                    "messages_truncated": bool(inference_plan["messages_truncated"]),
                    "memory_pressure": round(inference_plan["memory_pressure"], 4),
                },
            }
        return response

    def stream_chat_completion(
        payload: ChatCompletionRequest, path_model: str | None = None
    ) -> Generator[str, None, None]:
        response = handle_chat(payload, path_model)
        model_id = str(response.get("model", payload.model or ""))
        choice = ((response.get("choices") or [{}])[0]) if isinstance(response, dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_ts = int(time.time())
        first = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
        }
        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(first)}\n\n"
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    @app.post("/v1/models/{path_model}/chat/completions")
    def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        path_model: str | None = None,
    ) -> Any:
        manager: ModelManager = app.state.model_manager
        available_models = [entry["id"] for entry in manager.list_models()]
        default_model = manager.get_active_model_id()
        adapted, gw_ctx = _cursor_gateway.prepare_chat(
            payload, path_model=path_model,
            available_models=available_models, default_model=default_model,
        )
        model = _cursor_gateway.resolve_request_model(adapted, path_model)
        start = _cursor_gateway.log_incoming(request, path="/v1/chat/completions", model=model)
        try:
            if adapted.stream:
                return StreamingResponse(
                    _cursor_gateway.wrap_stream(
                        stream_chat_completion(adapted, path_model),
                        start=start, model=model, ctx=gw_ctx,
                    ),
                    media_type="text/event-stream",
                    headers=_cursor_gateway.stream_headers,
                )
            response = _cursor_gateway.adapt_chat_response(handle_chat(adapted, path_model))
            _cursor_gateway.finalize_chat(gw_ctx, response=response, status_code=200)
            _cursor_gateway.log_complete(start, status_code=200, model=model)
            return response
        except HTTPException as exc:
            _cursor_gateway.finalize_chat(gw_ctx, status_code=exc.status_code, error=_detail_to_message(exc.detail))
            raise

    @app.post("/completion")
    def legacy_completion(payload: CompletionRequest) -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        model_id = manager.resolve_model_id(payload.model, None)
        messages: list[dict[str, str]] = [{"role": "user", "content": payload.prompt}]
        result = manager.chat(
            model_id=model_id, messages=messages,
            temperature=payload.temperature, max_tokens=payload.n_predict,
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
            model_id=model_id, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return {"generated_text": result.text, "model": model_id}

    @app.post("/v1/cognitive/chat")
    def cognitive_chat(payload: ChatCompletionRequest) -> dict[str, Any]:
        return handle_chat(payload)

    # ── Runtime status endpoints ───────────────────────────────────────────

    @app.get("/v1/runtime/status")
    def runtime_status() -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        status: dict[str, Any] = {
            "runtime": "niblit_cognitive_cloud_runtime",
            "version": "0.9.0",
            "phase": "omega.9",
            "models": manager.list_models(),
            "default_model": manager.get_default_model_info(),
            "context_runtime": manager.runtime_stats(),
        }
        if _identity_mod:
            try:
                status["node"] = _identity_mod.get_node_identity().snapshot().to_dict()
            except Exception:
                pass
        if _temporal_mod:
            try:
                status["temporal"] = _temporal_mod.get_temporal_sync().status()
            except Exception:
                pass
        if _attention_mod:
            try:
                status["attention"] = _attention_mod.get_attention_allocator().status()
            except Exception:
                pass
        return status

    @app.get("/v1/runtime/coherence")
    def runtime_coherence() -> dict[str, Any]:
        result: dict[str, Any] = {}
        if _temporal_mod:
            try:
                result = _temporal_mod.get_temporal_sync().status()
            except Exception:
                result = {"error": "temporal_sync_unavailable"}
        return result

    @app.get("/v1/runtime/governance")
    def runtime_governance() -> dict[str, Any]:
        result: dict[str, Any] = {}
        if _governance_mod:
            try:
                result = _governance_mod.get_cloud_governance().status()
            except Exception:
                result = {"error": "governance_unavailable"}
        return result

    @app.get("/v1/runtime/attention")
    def runtime_attention() -> dict[str, Any]:
        result: dict[str, Any] = {}
        if _attention_mod:
            try:
                result = _attention_mod.get_attention_allocator().status()
            except Exception:
                result = {"error": "attention_allocator_unavailable"}
        return result

    @app.get("/v1/runtime/models")
    def runtime_models() -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        result: dict[str, Any] = {"registered_models": manager.list_models()}
        if _orchestrator_mod:
            try:
                result["orchestrator"] = _orchestrator_mod.get_model_orchestrator().status()
            except Exception:
                result["orchestrator_error"] = "unavailable"
        return result

    class ModelSwitchRequest(BaseModel):
        model_id: str

    @app.get("/v1/runtime/model/active")
    def get_active_model() -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        active = manager.get_active_model_id()
        return {"active_model": active or "", "available_models": [m["id"] for m in manager.list_models()]}

    @app.post("/v1/runtime/model/switch")
    def switch_model(payload: ModelSwitchRequest) -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        reloaded = manager.reload_model(payload.model_id)
        previous = manager.set_active_model(payload.model_id)
        if _orchestrator_mod:
            try:
                _orchestrator_mod.get_model_orchestrator().register_model(payload.model_id)
            except Exception as exc:
                logger.warning("switch_model: orchestrator registration failed: %s", exc)
        logger.info("Model switch requested: %s -> %s", previous, payload.model_id)
        return {"status": "switched", "active_model": payload.model_id, "previous_model": previous, "reloaded": reloaded}

    @app.get("/v1/runtime/reflection")
    def runtime_reflection() -> dict[str, Any]:
        result: dict[str, Any] = {}
        if _reflection_mod:
            try:
                re = _reflection_mod.get_reflection_engine()
                status = re.status()
                snap = re.last_snapshot()
                result = {**status, "last_snapshot": snap.to_dict() if snap else None}
            except Exception:
                result = {"error": "reflection_unavailable"}
        return result

    @app.get("/v1/runtime/trading")
    def runtime_trading() -> dict[str, Any]:
        result: dict[str, Any] = {}
        if _trading_mod:
            try:
                bridge = _trading_mod.get_trading_bridge()
                bridge.refresh()
                result = bridge.status()
            except Exception:
                result = {"error": "trading_bridge_unavailable"}
        return result

    @app.get("/v1/runtime/epoch")
    def runtime_epoch() -> dict[str, Any]:
        if _temporal_mod:
            try:
                ts = _temporal_mod.get_temporal_sync()
                return {"epoch_id": ts.current_epoch(), "coherence": round(ts.coherence(), 4), "status": ts.status()}
            except Exception:
                pass
        return {"epoch_id": int(time.time()), "coherence": 1.0}

    @app.get("/v1/runtime/mode")
    def runtime_mode() -> dict[str, Any]:
        mode = _normalize_runtime_mode(os.getenv("NIBLIT_RUNTIME_MODE", "normal"))
        strict = None
        reasons: list[str] = []
        if _governance_mod:
            try:
                g = _governance_mod.get_cloud_governance().status()
                strict = g.get("strict_mode")
                if g.get("block_rate", 0.0) > 0.2:
                    mode = "cautious"
                    reasons.append("governance_block_rate")
            except Exception:
                pass
        if _attention_mod:
            try:
                if _attention_mod.get_attention_allocator().status().get("overload"):
                    mode = "survival"
                    reasons.append("attention_overload")
            except Exception:
                pass
        if _federation_mod:
            try:
                fed = _federation_mod.get_federation_manager().status()
                connectivity = ((fed.get("topology") or {}).get("connectivity") or "").lower()
                if connectivity in {"offline", "disconnected"}:
                    mode = "lockdown"
                    reasons.append("federation_disconnected")
            except Exception:
                pass
        mode = _normalize_runtime_mode(mode)
        return {
            "mode": mode,
            "governance_mode": mode,
            "strict_governance": strict,
            "phase": "omega.9",
            "resource_adaptation": "attention_economy",
            "reasons": reasons,
        }

    @app.get("/v1/runtime/node")
    def runtime_node() -> dict[str, Any]:
        node: dict[str, Any] = {}
        if _identity_mod:
            try:
                node = _identity_mod.get_node_identity().snapshot().to_dict()
            except Exception:
                node = {}
        if _federation_mod:
            try:
                node["federation"] = _federation_mod.get_federation_manager().status()
            except Exception:
                pass
        return node

    @app.get("/v1/runtime/topology")
    def runtime_topology() -> dict[str, Any]:
        profile = os.getenv("NIBLIT_PROFILE", "cloud-server")
        mode = runtime_mode().get("mode", "normal")
        topology: dict[str, Any] = {
            "profile": profile,
            "runtime_mode": _normalize_runtime_mode(mode),
            "governance_mode": _normalize_runtime_mode(mode),
            "compatibility": {
                "schema_version": "2.x",
                "event_contract_version": "omega-7",
                "governance_contract_version": "1.x",
                "advisor_protocol_version": "2.x",
                "runtime_mode_contract": "2026.05",
            },
        }
        if _identity_mod:
            try:
                snap = _identity_mod.get_node_identity().snapshot()
                topology["node"] = snap.to_dict()
            except Exception:
                pass
        if _federation_mod:
            try:
                fed = _federation_mod.get_federation_manager().status()
                topology["federation"] = fed
                topology["connectivity"] = ((fed.get("topology") or {}).get("connectivity") or "standalone")
            except Exception:
                topology["connectivity"] = "unknown"
        else:
            topology["connectivity"] = "unknown"
        return topology

    def _build_runtime_envelope_snapshot() -> dict[str, Any]:
        ts = int(time.time())
        mode_info = runtime_mode()
        mode = str(mode_info.get("mode", "normal"))
        base: dict[str, Any] = {}
        if _envelope_mod is not None:
            try:
                base = _envelope_mod.normalize_envelope({})
            except Exception:
                base = {}
        envelope: dict[str, Any] = {
            "schema_version": "2.x",
            "timestamp": ts,
            "runtime": dict(base.get("runtime") or {}),
            "governance": dict(base.get("governance") or {}),
            "temporal": dict(base.get("temporal") or {}),
            "resources": dict(base.get("resources") or {}),
            "reflection": dict(base.get("reflection") or {}),
            "risk": dict(base.get("risk") or {}),
        }
        envelope["runtime"]["mode"] = mode
        envelope["runtime"]["runtime_health"] = 0.95
        envelope["runtime"]["attention_pressure"] = 0.15
        envelope["runtime"]["runtime_pressure"] = 0.15
        envelope["governance"]["governance_mode"] = mode
        envelope["governance"]["survival_mode"] = mode in {"survival", "lockdown"}
        envelope["governance"]["constitution_passed"] = mode != "lockdown"
        envelope["temporal"]["epoch_id"] = ts
        envelope["temporal"]["coherence_score"] = 0.85
        envelope["temporal"]["coherence_drift"] = 0.0
        envelope["resources"]["cognitive_budget"] = 1.0
        envelope["resources"]["attention_available"] = 1.0
        if _temporal_mod:
            try:
                temporal_status = _temporal_mod.get_temporal_sync().status()
                if isinstance(temporal_status, dict):
                    envelope["temporal"].update(temporal_status)
                    envelope["temporal"]["coherence_score"] = float(
                        temporal_status.get("coherence", envelope["temporal"]["coherence_score"])
                    )
            except Exception:
                pass
        if _governance_mod:
            try:
                gov_status = _governance_mod.get_cloud_governance().status()
                if isinstance(gov_status, dict):
                    envelope["governance"].update(gov_status)
            except Exception:
                pass
        if _attention_mod:
            try:
                att = _attention_mod.get_attention_allocator().status()
                if isinstance(att, dict):
                    overload = bool(att.get("overload"))
                    envelope["resources"]["attention_available"] = 0.25 if overload else 1.0
                    envelope["runtime"]["attention_pressure"] = 0.85 if overload else 0.15
            except Exception:
                pass
        return envelope

    @app.get("/niblit/runtime")
    @app.get("/v1/runtime/envelope")
    def runtime_envelope_snapshot() -> dict[str, Any]:
        return _build_runtime_envelope_snapshot()

    @app.get("/v1/runtime/diagnostics")
    def runtime_diagnostics() -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        diagnostics: dict[str, Any] = {
            "runtime_health": 1.0,
            "inference_pressure": {},
            "attention_pressure": {},
            "context_runtime": {},
            "model_latency_ema": {},
            "governance_violations": {},
            "thermal_resource_state": {
                "cpu_load": None, "memory_pressure": None,
                "resource_mode": "balanced",
                "note": "Host thermal metrics are not available in-process.",
            },
            "reflection_statistics": {},
            "coherence_drift": {},
            "federation": {"status": "standalone"},
            "topology": {},
        }
        try:
            diagnostics["runtime_health"] = 1.0 if health().get("status") == "ok" else 0.0
        except Exception:
            diagnostics["runtime_health"] = 0.0
        if _attention_mod:
            try:
                a = _attention_mod.get_attention_allocator().status()
                diagnostics["attention_pressure"] = a
                diagnostics["inference_pressure"] = {
                    "active_requests": a.get("active_requests", 0),
                    "max_queue": a.get("max_queue", 0),
                    "pressure_ema": a.get("attention_pressure", 0.0),
                    "overload": a.get("overload", False),
                }
                if a.get("overload"):
                    diagnostics["thermal_resource_state"]["resource_mode"] = "minimal"
            except Exception:
                pass
        try:
            diagnostics["context_runtime"] = manager.runtime_stats()
            if diagnostics["context_runtime"].get("last_context_usage_ratio", 0.0) >= 0.9:
                diagnostics["thermal_resource_state"]["resource_mode"] = "conservative"
        except Exception:
            pass
        if _orchestrator_mod:
            try:
                ms = _orchestrator_mod.get_model_orchestrator().status()
                diagnostics["model_latency_ema"] = {
                    mid: h.get("latency_ema_ms") for mid, h in (ms.get("model_health") or {}).items()
                }
            except Exception:
                pass
        if _governance_mod:
            try:
                gs = _governance_mod.get_cloud_governance().status()
                diagnostics["governance_violations"] = gs.get("violation_counts", {})
                diagnostics["governance"] = {
                    "strict_mode": gs.get("strict_mode"),
                    "validation_count": gs.get("validation_count"),
                    "block_count": gs.get("block_count"),
                    "block_rate": gs.get("block_rate"),
                }
            except Exception:
                pass
        if _reflection_mod:
            try:
                rs = _reflection_mod.get_reflection_engine().status()
                diagnostics["reflection_statistics"] = {
                    "turn_count": rs.get("turn_count"),
                    "reflect_count": rs.get("reflect_count"),
                    "quality_ema": rs.get("quality_ema"),
                    "latency_ema_ms": rs.get("latency_ema_ms"),
                    "veto_rate": rs.get("veto_rate"),
                }
            except Exception:
                pass
        if _temporal_mod:
            try:
                ts = _temporal_mod.get_temporal_sync().status()
                diagnostics["coherence_drift"] = {
                    "coherence_ema": ts.get("coherence_ema"),
                    "coherence_lag": ts.get("coherence_lag"),
                    "sync_status": ts.get("sync_status"),
                    "epoch_id": ts.get("epoch_id"),
                }
            except Exception:
                pass
        if _federation_mod:
            try:
                diagnostics["federation"] = _federation_mod.get_federation_manager().status()
            except Exception:
                pass
        try:
            diagnostics["topology"] = runtime_topology()
        except Exception:
            pass
        return diagnostics

    @app.post("/v1/bridge/inference")
    async def bridge_inference(request: Request) -> dict[str, Any]:
        manager: ModelManager = app.state.model_manager
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        message_type = str(payload.get("message_type", "")).strip()
        source = str(payload.get("source", "unknown")).strip()
        target = str(payload.get("target", "unknown")).strip()
        schema_version = str(payload.get("schema_version", "1.0")).strip() or "1.0"
        correlation_id = payload.get("correlation_id")
        message_payload = payload.get("payload", {})
        if not isinstance(message_payload, dict):
            raise HTTPException(status_code=400, detail="payload must be an object")
        prompt = str(message_payload.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="payload.prompt is required")
        model_id = str(message_payload.get("model_id") or manager.get_active_model_id() or "").strip()
        resolved_model = model_id
        response_text = f"Bridge response: {prompt}"
        finish_reason = "fallback"
        usage: dict[str, int] | None = None
        if model_id:
            try:
                resolved_model = manager.resolve_model_id(request_model=model_id, path_model=None)
                result = manager.chat(
                    model_id=resolved_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=_DEFAULT_TEMPERATURE,
                    max_tokens=_DEFAULT_MAX_TOKENS,
                )
                response_text = f"Bridge response: {result.text}"
                finish_reason = result.finish_reason
                usage = result.usage
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                response_text = f"Bridge response: {prompt}"
                finish_reason = "fallback"
                usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": 0, "total_tokens": len(prompt.split())}
        else:
            response_text = f"Bridge response: {prompt}"
            usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": 0, "total_tokens": len(prompt.split())}
        return {
            "message_type": "ai.inference.completed",
            "source": "niblit-cloud-server",
            "target": target or source or "niblit",
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "payload": {
                "model_id": resolved_model or model_id,
                "response_text": response_text,
                "finish_reason": finish_reason,
                "usage": usage,
                "request_source": source,
                "request_message_type": message_type,
            },
        }

    @app.get("/metrics/cognitive")
    def metrics_cognitive() -> dict[str, Any]:
        result: dict[str, Any] = {}
        if _reflection_mod:
            try:
                result = _reflection_mod.get_reflection_engine().status()
            except Exception:
                pass
        return result

    @app.get("/metrics/coherence")
    def metrics_coherence() -> dict[str, Any]:
        if _temporal_mod:
            try:
                return _temporal_mod.get_temporal_sync().status()
            except Exception:
                pass
        return {}

    @app.get("/metrics/governance")
    def metrics_governance() -> dict[str, Any]:
        if _governance_mod:
            try:
                return _governance_mod.get_cloud_governance().status()
            except Exception:
                pass
        return {}

    @app.get("/metrics/models")
    def metrics_models() -> dict[str, Any]:
        if _orchestrator_mod:
            try:
                return _orchestrator_mod.get_model_orchestrator().status()
            except Exception:
                pass
        return {}

    @app.get("/cluster/status")
    def cluster_status() -> dict[str, Any]:
        fed_status: dict[str, Any] = {}
        if _federation_mod:
            try:
                fed_status = _federation_mod.get_federation_manager().status()
            except Exception:
                fed_status = {}
        if _identity_mod:
            try:
                result = _identity_mod.get_node_identity().cluster_status()
                if fed_status:
                    result["federation"] = fed_status
                return result
            except Exception:
                pass
        return {"status": "single_node", "federation_ready": False, "federation": fed_status}

    @app.get("/cluster/identity")
    def cluster_identity() -> dict[str, Any]:
        if _identity_mod:
            try:
                return _identity_mod.get_node_identity().snapshot().to_dict()
            except Exception:
                pass
        return {}

    @app.get("/cluster/capabilities")
    def cluster_capabilities() -> dict[str, Any]:
        if _identity_mod:
            try:
                return _identity_mod.get_node_identity().snapshot().capabilities.to_dict()
            except Exception:
                pass
        return {}

    @app.get("/federation/status")
    def federation_status() -> dict[str, Any]:
        if _federation_mod:
            try:
                return _federation_mod.get_federation_manager().status()
            except Exception:
                pass
        return {"enabled": False, "status": "standalone"}

    @app.get("/federation/peers")
    def federation_peers() -> dict[str, Any]:
        if _federation_mod:
            try:
                fm = _federation_mod.get_federation_manager()
                return {"peers": fm.list_peers(), "status": fm.status()}
            except Exception:
                pass
        return {"peers": []}

    @app.post("/federation/register")
    def federation_register(payload: dict[str, Any]) -> dict[str, Any]:
        if _federation_mod:
            try:
                reg = _federation_mod.NodeRegistration(
                    node_id=str(payload.get("node_id", "unknown")),
                    region=str(payload.get("region", "unknown")),
                    role=str(payload.get("role", "inference")),
                    base_url=str(payload.get("base_url", "")),
                    capabilities=dict(payload.get("capabilities") or {}),
                    epoch_id=int(payload.get("epoch_id", 0)),
                    coherence=float(payload.get("coherence", 1.0)),
                )
                return _federation_mod.get_federation_manager().register_peer(reg)
            except Exception:
                logger.exception("federation_register failed")
                return {"accepted": False, "error": "federation_register_failed"}
        return {"accepted": False, "note": "federation module unavailable"}

    @app.post("/federation/discover")
    def federation_discover() -> dict[str, Any]:
        if _federation_mod:
            try:
                return _federation_mod.get_federation_manager().discover_peers()
            except Exception:
                logger.exception("federation_discover failed")
                return {"discovered": 0, "error": "federation_discover_failed"}
        return {"discovered": 0}

    @app.post("/federation/heartbeat")
    def federation_heartbeat() -> dict[str, Any]:
        if _federation_mod:
            try:
                return _federation_mod.get_federation_manager().emit_heartbeat()
            except Exception:
                logger.exception("federation_heartbeat failed")
                return {"emitted_to": 0, "error": "federation_heartbeat_failed"}
        return {"emitted_to": 0}

    @app.post("/federation/governance/sync")
    def federation_governance_sync(payload: dict[str, Any]) -> dict[str, Any]:
        if _federation_mod:
            try:
                return _federation_mod.get_federation_manager().sync_governance(payload)
            except Exception:
                logger.exception("federation_governance_sync failed")
                return {"synced_to": 0, "error": "federation_governance_sync_failed"}
        return {"synced_to": 0}

    @app.post("/federation/epoch/sync")
    def federation_epoch_sync(payload: dict[str, Any]) -> dict[str, Any]:
        if _federation_mod:
            try:
                return _federation_mod.get_federation_manager().sync_epoch(
                    int(payload.get("epoch_id", int(time.time())))
                )
            except Exception:
                logger.exception("federation_epoch_sync failed")
                return {"synced_to": 0, "error": "federation_epoch_sync_failed"}
        return {"synced_to": 0}

    # ── Trading API router ──────────────────────────────────────────────────────
    try:
        from app.trading.router import router as trading_router
        app.include_router(trading_router)
        logger.info("Trading API router registered")
    except Exception as exc:
        logger.warning("Trading API router not available: %s", exc)

    compat_prefixes = [
        p.strip(" /")
        for p in os.getenv("COMPAT_PREFIXES", "hf,local,kimi,claude").split(",")
        if p.strip()
    ]
    for prefix in compat_prefixes:
        app.add_api_route(f"/{prefix}/v1/chat/completions", chat_completions, methods=["POST"])
        app.add_api_route(f"/{prefix}/chat/completions", chat_completions, methods=["POST"])
        app.add_api_route(f"/{prefix}/completion", legacy_completion, methods=["POST"])
        app.add_api_route(f"/{prefix}/models/{{path_model}}", inference_api, methods=["POST"])
        app.add_api_route(f"/{prefix}/v1/models", list_models, methods=["GET"])
        app.add_api_route(f"/{prefix}/health", health, methods=["GET"])

    return app


def _try_import_model_registry() -> Any:
    try:
        import importlib
        return importlib.import_module("app.model_registry")
    except Exception:
        return None


app = create_app()


def _print_diagnostics(config: Any, manager: Any, registry_count: int, provider_count: int) -> None:
    print("\nConfiguration Verification", flush=True)
    print("-------------------------", flush=True)
    print(f"  CloudConfig:", flush=True)
    print(f"    models = {len(config.model_map)}", flush=True)
    print(f"    default = {config.default_model}", flush=True)
    print(f"  ModelManager:", flush=True)
    print(f"    models = {len(manager.list_models())}", flush=True)
    print(f"    default = {manager.get_active_model_id()}", flush=True)
    print(f"  ModelRegistry:", flush=True)
    print(f"    models = {registry_count}", flush=True)
    print(f"  ProviderRegistry:", flush=True)
    print(f"    providers = {provider_count}", flush=True)
    models = config.model_map
    if models and config.default_model:
        for alias in sorted(_LOCAL_MODEL_ALIASES):
            print(f"  Alias: {alias} -> {config.default_model}", flush=True)
    else:
        print("  Aliases: (no default model configured)", flush=True)
    print(f"  API: http://{config.host}:{config.port}", flush=True)
    print(flush=True)


def _run_cli() -> None:
    import uvicorn
    from app.runtime import CloudRuntime
    from app.config import get_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = get_config()
    app.state.cloud_config = cfg

    runtime = CloudRuntime(config=cfg)
    ready = runtime.run()
    status = "READY" if ready else "DEGRADED"
    print(f"\nCloud {status}\n", flush=True)
    for stage in runtime.stages:
        print(
            f"  [{stage.stage.value}] {stage.status} ({stage.duration_ms:.1f} ms): {stage.message}",
            flush=True,
        )

    app.state.cloud_runtime = runtime
    manager = runtime.get_service("model_manager")
    if manager is not None:
        app.state.model_manager = manager
    else:
        from app.main import ModelManager as _MM
        manager = _MM(cfg.model_map, cfg.default_model, config=cfg)
        app.state.model_manager = manager

    registry = runtime.get_service("model_registry")
    if registry is not None:
        app.state.model_registry = registry

    p_reg = runtime.get_service("provider_registry")
    if p_reg is not None:
        app.state.provider_registry = p_reg

    print("\nObject Identity", flush=True)
    print("---------------", flush=True)
    print(f"  CloudConfig ......... {id(cfg)}", flush=True)
    print(f"  CloudRuntime ........ {id(runtime)}", flush=True)
    print(f"  ModelManager ........ {id(manager)}", flush=True)
    if registry:
        print(f"  ModelRegistry ....... {id(registry)}", flush=True)
    if p_reg:
        print(f"  ProviderRegistry .... {id(p_reg)}", flush=True)

    cfg_models = cfg.model_map
    mm_models = manager.list_models() if manager else []
    errs = []
    if len(mm_models) != len(cfg_models):
        errs.append(
            f"ModelManager sees {len(mm_models)} models but CloudConfig has {len(cfg_models)}"
        )
    if manager.get_active_model_id() != cfg.default_model:
        errs.append(
            f"ModelManager default '{manager.get_active_model_id()}' != CloudConfig default '{cfg.default_model}'"
        )
    if errs:
        raise RuntimeError("Configuration mismatch: " + "; ".join(errs))

    registry_models = runtime.get_service("registered_models") or []
    provider_count = len(p_reg.list_providers()) if p_reg else 0
    _print_diagnostics(cfg, manager, len(registry_models), provider_count)

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def extended_lifespan(application):
        async with original_lifespan(application):
            yield
        runtime.shutdown()

    app.router.lifespan_context = extended_lifespan

    host = cfg.host
    port = cfg.port
    logger.info("Launching uvicorn on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_config=None)


def main() -> None:
    if sys.gettrace() is not None:
        return
    _run_cli()


if __name__ == "__main__":
    main()