"""Niblit Model Registry — Unified model discovery and metadata.

Supports:
- Local GGUF files
- HuggingFace models
- Remote providers (OpenAI, Anthropic, Ollama, vLLM, llama.cpp)
- OpenAI-compatible endpoints
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    LOCAL_GGUF = "local_gguf"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    REMOTE = "remote"


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelCapabilities:
    embedding: bool = False
    chat: bool = True
    vision: bool = False
    tools: bool = False
    streaming: bool = True


@dataclass
class RegisteredModel:
    name: str
    provider: ProviderType
    context: int = 4096
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    status: ModelStatus = ModelStatus.AVAILABLE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.name,
            "object": "model",
            "created": self.metadata.get("created", 0),
            "owned_by": self.provider.value,
            "context": self.context,
            "capabilities": {
                "embedding": self.capabilities.embedding,
                "chat": self.capabilities.chat,
                "vision": self.capabilities.vision,
                "tools": self.capabilities.tools,
                "streaming": self.capabilities.streaming,
            },
            "status": self.status.value,
        }


class ModelRegistry:
    """Discovers and tracks all available models across providers."""

    def __init__(self) -> None:
        self._models: dict[str, RegisteredModel] = {}
        self._lock = __import__("threading").Lock()
        logger.info("ModelRegistry instantiated")

    def register(self, model: RegisteredModel) -> None:
        with self._lock:
            self._models[model.name] = model
        logger.info("Model registered: %s (%s)", model.name, model.provider.value)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._models.pop(name, None)
        logger.info("Model unregistered: %s", name)

    def get(self, name: str) -> RegisteredModel | None:
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> list[RegisteredModel]:
        with self._lock:
            return list(self._models.values())

    def to_openai_list(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [m.to_openai() for m in self.list_models()],
        }

    def discover_from_env(self) -> None:
        """Auto-discover models from environment configuration."""
        # GGUF models from GGUF_MODELS_JSON
        import json
        raw = os.getenv("GGUF_MODELS_JSON", "").strip()
        if raw:
            try:
                models = json.loads(raw)
                if isinstance(models, dict):
                    for model_id, path in models.items():
                        self.register(RegisteredModel(
                            name=model_id,
                            provider=ProviderType.LOCAL_GGUF,
                            metadata={"path": path},
                        ))
            except json.JSONDecodeError:
                logger.warning("GGUF_MODELS_JSON is not valid JSON")

        # Single fallback path
        fallback = os.getenv("NIBLIT_DEFAULT_MODEL_PATH") or os.getenv("NIBLIT_MODEL_PATH")
        if fallback and not raw:
            model_id = os.getenv("NIBLIT_DEFAULT_MODEL_ID", "fallback")
            self.register(RegisteredModel(
                name=model_id,
                provider=ProviderType.LOCAL_GGUF,
                metadata={"path": fallback},
            ))

        # Remote OpenAI-compatible endpoint
        openai_url = os.getenv("OPENAI_API_BASE", "").strip()
        if openai_url:
            self.register(RegisteredModel(
                name="gpt-4o",
                provider=ProviderType.OPENAI,
                context=128000,
                capabilities=ModelCapabilities(chat=True, vision=True, tools=True, streaming=True),
                metadata={"base_url": openai_url},
            ))

        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY", "").strip():
            self.register(RegisteredModel(
                name="claude-3-5-sonnet",
                provider=ProviderType.ANTHROPIC,
                context=200000,
                capabilities=ModelCapabilities(chat=True, vision=True, tools=True, streaming=True),
            ))

        # Ollama
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        self.register(RegisteredModel(
            name="ollama/llama3",
            provider=ProviderType.OLLAMA,
            context=8192,
            metadata={"base_url": ollama_url},
        ))

        # vLLM
        vllm_url = os.getenv("VLLM_API_BASE", "").strip()
        if vllm_url:
            self.register(RegisteredModel(
                name="vllm/default",
                provider=ProviderType.VLLM,
                context=32768,
                metadata={"base_url": vllm_url},
            ))

        # llama.cpp server
        llama_url = os.getenv("LLAMA_CPP_SERVER_URL", "").strip()
        if llama_url:
            self.register(RegisteredModel(
                name="llama.cpp/default",
                provider=ProviderType.LLAMA_CPP,
                context=32768,
                metadata={"base_url": llama_url},
            ))

        logger.info("Discovered %d models from environment", len(self._models))


def get_model_registry() -> ModelRegistry:
    """Return the process-level ModelRegistry singleton."""
    global _registry
    try:
        return _registry
    except NameError:
        _registry = ModelRegistry()
        _registry.discover_from_env()
        return _registry