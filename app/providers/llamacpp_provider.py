"""llama.cpp provider — delegates inference to a llama.cpp server."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LlamaCppProvider:
    """llama.cpp inference provider."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("LLAMA_CPP_SERVER_URL", "http://localhost:11434")).rstrip("/")
        self.name = "llama_cpp"

    def start(self) -> bool:
        logger.info("llama.cpp provider ready at %s", self.base_url)
        return True

    def stop(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "provider": self.name, "base_url": self.base_url}

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        prompt = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
        payload = {"model": model, "prompt": prompt, "stream": False}
        if "max_tokens" in kwargs:
            payload["n_predict"] = kwargs["max_tokens"]
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/completion",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                return {"text": body.get("content", ""), "finish_reason": "stop", "usage": None}
        except Exception as exc:
            logger.warning("llama.cpp provider chat failed: %s", exc)
            return {"text": "", "finish_reason": "error", "usage": None}

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        return []