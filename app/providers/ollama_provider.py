"""Ollama provider — delegates inference to Ollama."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Ollama inference provider."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.name = "ollama"

    def start(self) -> bool:
        logger.info("Ollama provider ready at %s", self.base_url)
        return True

    def stop(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "provider": self.name, "base_url": self.base_url}

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["options"] = {"num_predict": kwargs["max_tokens"]}

        # Best-effort HTTP call; degrade gracefully if unavailable.
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=__import__("json").dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                content = body.get("message", {}).get("content", "")
                return {"text": content, "finish_reason": "stop", "usage": None}
        except Exception as exc:
            logger.warning("Ollama provider chat failed: %s", exc)
            return {"text": "", "finish_reason": "error", "usage": None}

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        # Ollama /api/embeddings expects prompt/model separately; this is a stub.
        return []