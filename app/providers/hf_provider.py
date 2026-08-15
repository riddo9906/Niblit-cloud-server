"""HuggingFace provider — delegates inference to HF Inference API."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class HFProvider:
    """HuggingFace inference provider."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("HF_API_KEY", "")
        self.base_url = (base_url or os.getenv("HF_API_BASE_URL", "https://api-inference.huggingface.co")).rstrip("/")
        self.name = "huggingface"

    def start(self) -> bool:
        if not self.api_key:
            logger.warning("HF provider disabled: missing HF_API_KEY")
            return False
        logger.info("HF provider ready at %s", self.base_url)
        return True

    def stop(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "ok" if self.api_key else "disabled", "provider": self.name}

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        if not self.api_key:
            return {"text": "", "finish_reason": "error", "usage": None}
        prompt = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
        payload = {"inputs": prompt}
        if "max_tokens" in kwargs:
            payload["parameters"] = {"max_new_tokens": kwargs["max_tokens"]}
        try:
            import urllib.request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            req = urllib.request.Request(
                f"{self.base_url}/models/{model}",
                data=json.dumps(payload).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                if isinstance(body, list):
                    text = (body[0] or {}).get("generated_text", body[0])
                elif isinstance(body, dict):
                    text = body.get("generated_text", "")
                else:
                    text = str(body)
                return {"text": text, "finish_reason": "stop", "usage": None}
        except Exception as exc:
            logger.warning("HF provider chat failed: %s", exc)
            return {"text": "", "finish_reason": "error", "usage": None}

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        return []