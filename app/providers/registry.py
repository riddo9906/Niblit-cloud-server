"""Provider registry — discovers, starts, and manages inference providers."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Discovers and manages inference providers."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}

    def register(self, name: str, provider: Any) -> None:
        self._providers[name] = provider
        logger.info("Provider registered: %s", name)

    def get(self, name: str) -> Any | None:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def discover_from_env(self) -> None:
        """Best-effort provider discovery from env/config."""
        # qwen_server.sh provider
        try:
            from app.providers.qwen_server_provider import get_qwen_server_provider
            self.register("qwen_server", get_qwen_server_provider())
        except Exception as exc:
            logger.debug("qwen_server provider not available: %s", exc)

        # Ollama provider
        if os.getenv("OLLAMA_BASE_URL", "").strip():
            try:
                from app.providers.ollama_provider import OllamaProvider
                self.register("ollama", OllamaProvider())
            except Exception as exc:
                logger.debug("ollama provider not available: %s", exc)

        # OpenAI-compatible provider
        if os.getenv("OPENAI_API_BASE", "").strip() or os.getenv("OPENAI_API_KEY", "").strip():
            try:
                from app.providers.openai_provider import OpenAIProvider
                self.register("openai", OpenAIProvider())
            except Exception as exc:
                logger.debug("openai provider not available: %s", exc)

        # Anthropic provider
        if os.getenv("ANTHROPIC_API_KEY", "").strip():
            try:
                from app.providers.anthropic_provider import AnthropicProvider
                self.register("anthropic", AnthropicProvider())
            except Exception as exc:
                logger.debug("anthropic provider not available: %s", exc)

        # HuggingFace provider
        if os.getenv("HF_API_KEY", "").strip():
            try:
                from app.providers.hf_provider import HFProvider
                self.register("huggingface", HFProvider())
            except Exception as exc:
                logger.debug("huggingface provider not available: %s", exc)

        # llama.cpp provider
        if os.getenv("LLAMA_CPP_SERVER_URL", "").strip():
            try:
                from app.providers.llamacpp_provider import LlamaCppProvider
                self.register("llama_cpp", LlamaCppProvider())
            except Exception as exc:
                logger.debug("llama_cpp provider not available: %s", exc)

        # vLLM provider
        if os.getenv("VLLM_API_BASE", "").strip():
            try:
                from app.providers.vllm_provider import VLLMProvider
                self.register("vllm", VLLMProvider())
            except Exception as exc:
                logger.debug("vllm provider not available: %s", exc)

        # Remote OpenAI-compatible provider
        if os.getenv("REMOTE_API_BASE_URL", "").strip():
            try:
                from app.providers.remote_provider import RemoteProvider
                self.register("remote", RemoteProvider())
            except Exception as exc:
                logger.debug("remote provider not available: %s", exc)

    def start_all(self) -> None:
        for name, provider in list(self._providers.items()):
            try:
                if hasattr(provider, "start") and callable(provider.start):
                    provider.start()
            except Exception as exc:
                logger.warning("Provider %s failed to start: %s", name, exc)

    def stop_all(self) -> None:
        for name, provider in list(self._providers.items()):
            try:
                if hasattr(provider, "stop") and callable(provider.stop):
                    provider.stop()
            except Exception as exc:
                logger.warning("Provider %s failed to stop: %s", name, exc)