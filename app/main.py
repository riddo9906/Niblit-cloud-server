import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Cognitive runtime modules (best-effort imports) ───────────────────────────
# All new modules degrade gracefully if disabled or unavailable.

def _try_import(module_path: str) -> Any:
    """Best-effort import of a cloud runtime module."""
    try:
        import importlib
        return importlib.import_module(module_path)
    except Exception:
        return None

# Niblit's QwenLocalBrain._generate_http() always sends "model": "local" in
# every request so the cloud server can act as a drop-in llama-server
# replacement.  Any of these aliases resolves to the configured default model.
_LOCAL_MODEL_ALIASES: frozenset[str] = frozenset({"local", "llama", "default"})

# Shared defaults for request schemas.
_DEFAULT_TEMPERATURE: float = 0.2
_DEFAULT_MAX_TOKENS: int = 256
_CANONICAL_MODES: tuple[str, ...] = ("normal", "cautious", "survival", "lockdown")
_MESSAGE_OVERHEAD_CHARS: int = 8


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


def _normalize_runtime_mode(mode: object, default: str = "normal") -> str:
    candidate = str(mode or default).strip().lower()
    if candidate in ("minimal", "constrained"):
        candidate = "cautious"
    if candidate not in _CANONICAL_MODES:
        return default
    return candidate


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
    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        runtime_options: dict[str, Any] | None = None,
    ):
        logger.info("Loading model from %s (n_ctx=%d, n_threads=%d)", model_path, n_ctx, n_threads)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required to run GGUF inference."
            ) from exc
        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads,
        }
        for key, value in (runtime_options or {}).items():
            if value is not None:
                kwargs[key] = value
        try:
            self._llm = Llama(**kwargs)
        except TypeError:
            logger.warning(
                "llama_cpp runtime options unsupported by current build; falling back to core args."
            )
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
        self._n_ctx = _env_int("N_CTX", _env_int("NIBLIT_CONTEXT_WINDOW", 16384))
        self._n_threads = _env_int("N_THREADS", 4)
        self._n_batch = _env_int("NIBLIT_N_BATCH", _env_int("N_BATCH", 1024))
        self._n_ubatch = _env_int("NIBLIT_N_UBATCH", _env_int("N_UBATCH", 512))
        self._context_reserve_tokens = _env_int("NIBLIT_CONTEXT_RESERVE_TOKENS", 512)
        self._min_generation_tokens = _env_int("NIBLIT_MIN_GENERATION_TOKENS", 64)
        self._char_to_token_ratio = max(1, _env_int("NIBLIT_CHAR_PER_TOKEN", 4))
        self._memory_guard_ratio = max(
            0.5,
            min(0.98, _env_float("NIBLIT_MEMORY_GUARD_RATIO", 0.92) or 0.92),
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
            "ModelManager initialized: models=%s default=%s n_ctx=%d n_threads=%d n_batch=%d n_ubatch=%d",
            list(model_map.keys()),
            self._default_model,
            self._n_ctx,
            self._n_threads,
            self._n_batch,
            self._n_ubatch,
        )

    def get_default_model_info(self) -> dict[str, str]:
        """Return {model_id, model_path} for the default model (empty strings if none)."""
        with self._lock:
            model_id = self._default_model or ""
            model_path = self._model_map.get(model_id, "") if model_id else ""
        return {"model_id": model_id, "model_path": model_path}

    def get_active_model_id(self) -> str | None:
        """Return the current active/default model ID."""
        with self._lock:
            return self._default_model

    def set_active_model(self, model_id: str) -> str:
        """Switch the active (default) model to *model_id*.

        Returns the previous active model ID.  Raises HTTPException 404 if
        *model_id* is not registered.
        """
        with self._lock:
            if model_id not in self._model_map:
                raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
            previous = self._default_model
            self._default_model = model_id
        logger.info("Active model switched: %s -> %s", previous, model_id)
        return previous or ""

    def reload_model(self, model_id: str) -> bool:
        """Hot reload a specific model while the server stays online.

        Returns True when a fresh engine instance is loaded.

        Raises HTTPException:
        - 404 if model_id is unknown or the model file path does not exist.
        - 502 if the backend engine fails to initialize for the model.
        """
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

    def resolve_model_id(self, request_model: str | None, path_model: str | None) -> str:
        """Resolve model precedence as request model > path model > default model.

        The string ``"local"`` (and other well-known aliases set in
        _LOCAL_MODEL_ALIASES) is mapped to the configured default model so that
        Niblit's ``QwenLocalBrain`` HTTP backend — which always sends
        ``"model": "local"`` — works without any client-side configuration.
        """
        with self._lock:
            default = self._default_model
        model_id = request_model or path_model or default
        if not model_id:
            raise HTTPException(status_code=400, detail="No model provided.")
        # Resolve well-known aliases (e.g. "local" sent by Niblit's local_brain)
        if model_id.lower() in _LOCAL_MODEL_ALIASES:
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
        if model_id not in self._engines:
            self._engines[model_id] = GGUFEngine(
                model_path=self._model_map[model_id],
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                runtime_options=self._runtime_options,
            )
        plan = self._prepare_inference(messages=messages, max_tokens=max_tokens)
        logger.debug(
            "chat request: model=%s messages=%d temperature=%s max_tokens=%d effective_max_tokens=%d",
            model_id,
            len(messages),
            temperature,
            max_tokens,
            plan["effective_max_tokens"],
        )
        try:
            result = self._engines[model_id].chat(
                messages=plan["messages"], temperature=temperature, max_tokens=plan["effective_max_tokens"]
            )
        except RuntimeError as exc:
            logger.error("Inference error for model %s: %s", model_id, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        with self._lock:
            self._stats["requests_total"] += 1
            if plan["messages_truncated"]:
                self._stats["context_trim_events"] += 1
                logger.info(
                    "Context truncation applied for model=%s prompt_tokens=%d requested_max_tokens=%d effective_max_tokens=%d",
                    model_id,
                    plan["prompt_tokens_estimate"],
                    max_tokens,
                    plan["effective_max_tokens"],
                )
            if plan["effective_max_tokens"] < max_tokens:
                self._stats["max_token_clamp_events"] += 1
            self._stats["last_prompt_tokens_estimate"] = plan["prompt_tokens_estimate"]
            self._stats["last_effective_max_tokens"] = plan["effective_max_tokens"]
            self._stats["last_context_usage_ratio"] = round(plan["context_usage_ratio"], 4)
        logger.debug(
            "chat response: model=%s finish_reason=%s",
            model_id,
            result.finish_reason,
        )
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

        available_for_generation = max(
            1,
            target_ctx_budget - prompt_tokens - self._context_reserve_tokens,
        )
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
    stream: bool = False
    # Accepted for llama-server API compatibility; passed through to the engine.
    stop: list[str] | None = None
    # Tool schemas (Niblit generate_with_tools); accepted and ignored if not supported.
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    # ── Cognitive envelope fields (Phase Ω.7) — all optional ──────────────────
    # Plain OpenAI/HF requests that omit these fields remain fully compatible.
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

    # ── Boot cognitive runtime subsystems ─────────────────────────────────────
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
    async def lifespan(app: FastAPI):
        info = _manager.get_default_model_info()
        logger.info(
            "Niblit Cognitive Cloud Runtime starting — models=%s default=%s",
            list(models.keys()),
            info["model_id"] or "(none)",
        )
        # Seed orchestrator with known model IDs
        if _orchestrator_mod:
            _orchestrator_mod.get_model_orchestrator(model_ids=list(models.keys()))
        # Initialize node identity
        if _identity_mod:
            _identity_mod.get_node_identity()
        if _federation_mod:
            _federation_mod.get_federation_manager()
        yield
        logger.info("Niblit Cognitive Cloud Runtime shutdown complete.")

    app = FastAPI(
        title="Niblit Cognitive Cloud Runtime",
        version="0.7.0",
        description=(
            "Distributed cognitive execution + cognition + orchestration node. "
            "Backward compatible with HuggingFace-style APIs, llama.cpp, and "
            "QwenLocalBrain. Phase Ω.7."
        ),
        lifespan=lifespan,
    )
    app.state.model_manager = _manager

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
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
        # Increment node identity request counter
        if _identity_mod:
            try:
                _identity_mod.get_node_identity().increment_request()
            except Exception:
                pass
        return response

    # ── Existing health/probe endpoints (unchanged) ───────────────────────────

    @app.get("/healthz")
    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check — responds to both /health (llama-server probe) and /healthz."""
        return {"status": "ok"}

    @app.get("/props")
    def props() -> dict[str, Any]:
        """Legacy llama-server /props probe endpoint."""
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

    # ── Internal helper: extract envelope from request ────────────────────────

    def _extract_envelope(payload: ChatCompletionRequest) -> dict[str, Any]:
        """Build a cognitive envelope dict from the request payload fields."""
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

    # ── Core chat handler (shared by all chat endpoints) ─────────────────────

    def handle_chat(payload: ChatCompletionRequest, path_model: str | None = None) -> dict[str, Any]:
        start_ms = time.perf_counter() * 1000
        request_id = uuid.uuid4().hex

        manager: ModelManager = app.state.model_manager
        messages = _normalize_messages(payload.messages)
        inference_plan = manager.estimate_inference(messages=messages, max_tokens=payload.max_tokens)

        # ── Build cognitive envelope ───────────────────────────────────────────
        envelope = _extract_envelope(payload)
        if inference_plan["memory_pressure"] >= 0.85:
            envelope["resource_mode"] = "conservative"
            runtime_ctx = dict(envelope.get("runtime") or {})
            runtime_ctx["mode"] = "cautious"
            runtime_ctx["attention_pressure"] = min(1.0, inference_plan["memory_pressure"])
            envelope["runtime"] = runtime_ctx

        # ── Attention allocation ───────────────────────────────────────────────
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

        # ── Governance validation ──────────────────────────────────────────────
        governance_vetoed = False
        if _governance_mod:
            try:
                verdict = _governance_mod.get_cloud_governance().validate(
                    action="chat_inference",
                    context={
                        "prompt_tokens_estimate": inference_plan["prompt_tokens_estimate"],
                        "estimated_total_tokens": (
                            inference_plan["prompt_tokens_estimate"] + payload.max_tokens
                        ),
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

        # ── Temporal sync ──────────────────────────────────────────────────────
        if _temporal_mod:
            try:
                _temporal_mod.get_temporal_sync().record_request(
                    coherence=float(envelope.get("coherence_score", 1.0)),
                    epoch_tag=str(envelope.get("epoch_tag", "")),
                )
            except Exception:
                pass

        # ── Model orchestration ────────────────────────────────────────────────
        available_models = list(manager._model_map.keys())
        request_model_id = manager.resolve_model_id(payload.model, path_model)

        if _orchestrator_mod and len(available_models) > 1:
            try:
                decision = _orchestrator_mod.get_model_orchestrator().route(
                    available_models=available_models,
                    default_model=request_model_id,
                    envelope=envelope,
                )
                # Only use orchestrated model if it matches one we can resolve
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

        # ── Record outcome in subsystems ───────────────────────────────────────
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

        # Attach cognitive metadata if envelope was supplied
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
        payload: ChatCompletionRequest, path_model: str | None = None
    ) -> Any:
        if payload.stream:
            return StreamingResponse(
                stream_chat_completion(payload, path_model),
                media_type="text/event-stream",
            )
        return handle_chat(payload, path_model)

    @app.post("/completion")
    def legacy_completion(payload: CompletionRequest) -> dict[str, Any]:
        """Legacy llama-server ``POST /completion`` endpoint."""
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

    # ── Cognitive chat endpoint (enriched response) ───────────────────────────

    @app.post("/v1/cognitive/chat")
    def cognitive_chat(payload: ChatCompletionRequest) -> dict[str, Any]:
        """Cognitive chat endpoint — same as /v1/chat/completions with richer metadata."""
        return handle_chat(payload)

    # ── Runtime status endpoints ───────────────────────────────────────────────

    @app.get("/v1/runtime/status")
    def runtime_status() -> dict[str, Any]:
        """Full runtime status snapshot."""
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
        """Temporal coherence and epoch sync status."""
        result: dict[str, Any] = {}
        if _temporal_mod:
            try:
                result = _temporal_mod.get_temporal_sync().status()
            except Exception:
                result = {"error": "temporal_sync_unavailable"}
        return result

    @app.get("/v1/runtime/governance")
    def runtime_governance() -> dict[str, Any]:
        """Constitutional governance statistics."""
        result: dict[str, Any] = {}
        if _governance_mod:
            try:
                result = _governance_mod.get_cloud_governance().status()
            except Exception:
                result = {"error": "governance_unavailable"}
        return result

    @app.get("/v1/runtime/attention")
    def runtime_attention() -> dict[str, Any]:
        """Attention economy metrics."""
        result: dict[str, Any] = {}
        if _attention_mod:
            try:
                result = _attention_mod.get_attention_allocator().status()
            except Exception:
                result = {"error": "attention_allocator_unavailable"}
        return result

    @app.get("/v1/runtime/models")
    def runtime_models() -> dict[str, Any]:
        """Model orchestration health and routing statistics."""
        manager: ModelManager = app.state.model_manager
        result: dict[str, Any] = {
            "registered_models": manager.list_models(),
        }
        if _orchestrator_mod:
            try:
                result["orchestrator"] = _orchestrator_mod.get_model_orchestrator().status()
            except Exception:
                result["orchestrator_error"] = "unavailable"
        return result

    # ── Model switch endpoints ─────────────────────────────────────────────────

    class ModelSwitchRequest(BaseModel):
        model_id: str

    @app.get("/v1/runtime/model/active")
    def get_active_model() -> dict[str, Any]:
        """Return the currently active (default) model and all registered models."""
        manager: ModelManager = app.state.model_manager
        active = manager.get_active_model_id()
        return {
            "active_model": active or "",
            "available_models": [m["id"] for m in manager.list_models()],
        }

    @app.post("/v1/runtime/model/switch")
    def switch_model(payload: ModelSwitchRequest) -> dict[str, Any]:
        """Switch the active model to *payload.model_id* while the server is running.

        Both llama3 and qwen (or any other registered GGUF) model IDs are
        accepted.  The new default is applied immediately — subsequent requests
        that use the ``"local"`` alias will be routed to the switched model.

        Returns 404 if *model_id* is not registered in GGUF_MODELS_JSON.
        """
        manager: ModelManager = app.state.model_manager
        reloaded = manager.reload_model(payload.model_id)
        previous = manager.set_active_model(payload.model_id)
        if _orchestrator_mod:
            try:
                _orchestrator_mod.get_model_orchestrator().register_model(payload.model_id)
            except Exception as exc:
                logger.warning("switch_model: orchestrator registration failed: %s", exc)
        logger.info("Model switch requested: %s -> %s", previous, payload.model_id)
        return {
            "status": "switched",
            "active_model": payload.model_id,
            "previous_model": previous,
            "reloaded": reloaded,
        }

    @app.get("/v1/runtime/reflection")
    def runtime_reflection() -> dict[str, Any]:
        """Reflection engine telemetry."""
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
        """Trading cognition bridge status and market state."""
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
        """Current epoch and temporal ordering state."""
        if _temporal_mod:
            try:
                ts = _temporal_mod.get_temporal_sync()
                return {
                    "epoch_id": ts.current_epoch(),
                    "coherence": round(ts.coherence(), 4),
                    "status": ts.status(),
                }
            except Exception:
                pass
        return {"epoch_id": int(time.time()), "coherence": 1.0}

    @app.get("/v1/runtime/mode")
    def runtime_mode() -> dict[str, Any]:
        """Current runtime/governance mode and adaptation posture."""
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
        """Node identity + cluster/federation posture."""
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
        """Topology-aware runtime coordination summary."""
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

    @app.get("/v1/runtime/diagnostics")
    def runtime_diagnostics() -> dict[str, Any]:
        """Operational diagnostics for governance-aware runtime operations."""
        manager: ModelManager = app.state.model_manager
        diagnostics: dict[str, Any] = {
            "runtime_health": 1.0,
            "inference_pressure": {},
            "attention_pressure": {},
            "context_runtime": {},
            "model_latency_ema": {},
            "governance_violations": {},
            "thermal_resource_state": {
                "cpu_load": None,
                "memory_pressure": None,
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
                    mid: h.get("latency_ema_ms")
                    for mid, h in (ms.get("model_health") or {}).items()
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

    # ── Metrics endpoints ──────────────────────────────────────────────────────

    @app.get("/metrics/cognitive")
    def metrics_cognitive() -> dict[str, Any]:
        """Cognitive telemetry metrics."""
        result: dict[str, Any] = {}
        if _reflection_mod:
            try:
                result = _reflection_mod.get_reflection_engine().status()
            except Exception:
                pass
        return result

    @app.get("/metrics/coherence")
    def metrics_coherence() -> dict[str, Any]:
        """Coherence metrics."""
        if _temporal_mod:
            try:
                return _temporal_mod.get_temporal_sync().status()
            except Exception:
                pass
        return {}

    @app.get("/metrics/governance")
    def metrics_governance() -> dict[str, Any]:
        """Governance metrics."""
        if _governance_mod:
            try:
                return _governance_mod.get_cloud_governance().status()
            except Exception:
                pass
        return {}

    @app.get("/metrics/models")
    def metrics_models() -> dict[str, Any]:
        """Model routing and health metrics."""
        if _orchestrator_mod:
            try:
                return _orchestrator_mod.get_model_orchestrator().status()
            except Exception:
                pass
        return {}

    # ── Cluster / swarm readiness endpoints ───────────────────────────────────

    @app.get("/cluster/status")
    def cluster_status() -> dict[str, Any]:
        """Cluster status (single-node; federation not yet implemented)."""
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
        """Node identity and fingerprint."""
        if _identity_mod:
            try:
                return _identity_mod.get_node_identity().snapshot().to_dict()
            except Exception:
                pass
        return {}

    @app.get("/cluster/capabilities")
    def cluster_capabilities() -> dict[str, Any]:
        """Advertised node capabilities."""
        if _identity_mod:
            try:
                return _identity_mod.get_node_identity().snapshot().capabilities.to_dict()
            except Exception:
                pass
        return {}

    # ── Federation preparation endpoints (stubs) ──────────────────────────────

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

    # ── Compatibility prefix routes (unchanged) ────────────────────────────────

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
