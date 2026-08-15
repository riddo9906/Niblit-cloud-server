"""Anthropic provider — delegates inference to Anthropic."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic inference provider."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = "https://api.anthropic.com/v1"
        self.name = "anthropic"

    def start(self) -> bool:
        if not self.api_key:
            logger.warning("Anthropic provider disabled: missing ANTHROPIC_API_KEY")
            return False
        logger.info("Anthropic provider ready")
        return True

    def stop(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "ok" if self.api_key else "disabled", "provider": self.name}

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        if not self.api_key:
            return {"text": "", "finish_reason": "error", "usage": None}
        system = ""
        filtered = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system = msg.get("content", "")
            else:
                filtered.append({"role": role, "content": msg.get("content", "")})
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": filtered,
        }
        if system:
            payload["system"] = system
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]

        try:
            import urllib.request
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            req = urllib.request.Request(
                f"{self.base_url}/messages",
                data=json.dumps(payload).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                content = body.get("content", [{}])[0].get("text", "")
                usage = body.get("usage")
                return {"text": content, "finish_reason": "stop", "usage": usage}
        except Exception as exc:
            logger.warning("Anthropic provider chat failed: %s", exc)
            return {"text": "", "finish_reason": "error", "usage": None}

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        return []