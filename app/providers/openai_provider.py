"""OpenAI-compatible provider — delegates to OpenAI or compatible APIs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI-compatible inference provider."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.name = "openai"

    def start(self) -> bool:
        logger.info("OpenAI provider ready at %s", self.base_url)
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
            payload["max_tokens"] = kwargs["max_tokens"]

        try:
            import urllib.request
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message", {})
                return {
                    "text": message.get("content", ""),
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "usage": body.get("usage"),
                }
        except Exception as exc:
            logger.warning("OpenAI provider chat failed: %s", exc)
            return {"text": "", "finish_reason": "error", "usage": None}

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {"input": texts, "model": "text-embedding-3-small"}
        try:
            import urllib.request
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=json.dumps(payload).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                return [item.get("embedding", []) for item in body.get("data", [])]
        except Exception as exc:
            logger.warning("OpenAI provider embeddings failed: %s", exc)
            return []