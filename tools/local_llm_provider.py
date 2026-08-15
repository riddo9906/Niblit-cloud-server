#!/usr/bin/env python3
"""
Local LLM Provider — HTTP client wrapper for llama.cpp OpenAI-compatible server.

This module provides a unified interface for interacting with the local llama.cpp
server running on the configured endpoint. It supports:
- generate()
- chat()
- complete()
- edit()
- summarize()
- repository_chat()
- code_review()
- reasoning()
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Iterator, Optional

# Default configuration from environment
LOCAL_MODEL_ENDPOINT = os.getenv("LOCAL_MODEL_ENDPOINT", "http://127.0.0.1:8000").rstrip("/")
LOCAL_MODEL_CHAT_ENDPOINT = os.getenv("LOCAL_MODEL_CHAT_ENDPOINT", "/v1/chat/completions")
LOCAL_MODEL_COMPLETION_ENDPOINT = os.getenv("LOCAL_MODEL_COMPLETION_ENDPOINT", "/v1/completions")
LOCAL_MODEL_DEFAULT_TEMPERATURE = float(os.getenv("LOCAL_MODEL_DEFAULT_TEMPERATURE", "0.7"))
LOCAL_MODEL_DEFAULT_MAX_TOKENS = int(os.getenv("LOCAL_MODEL_DEFAULT_MAX_TOKENS", "1024"))
LOCAL_MODEL_STREAM = os.getenv("LOCAL_MODEL_STREAM", "true").lower() in ("1", "true", "yes")
LOCAL_MODEL_TIMEOUT = int(os.getenv("LOCAL_MODEL_TIMEOUT", "120"))


@dataclass
class LocalLLMResponse:
    """Response from the local LLM provider."""
    text: str
    model: str
    usage: dict[str, Any] | None = None
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    error: str = ""


class LocalLLMProvider:
    """
    HTTP client wrapper for llama.cpp OpenAI-compatible server.
    
    This class provides methods for various LLM operations while reusing
    a single HTTP connection for optimal performance.
    """
    
    _instance: Optional["LocalLLMProvider"] = None
    
    def __new__(cls) -> "LocalLLMProvider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._base_url = LOCAL_MODEL_ENDPOINT
            cls._instance._timeout = LOCAL_MODEL_TIMEOUT
            cls._instance._model_id: Optional[str] = None
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the provider by detecting the available model."""
        self._model_id = self._detect_model()
    
    def _detect_model(self) -> str:
        """Detect the model ID from the server's /v1/models endpoint."""
        try:
            url = f"{self._base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("data", []) or data.get("models", [])
                if models:
                    # Return the first available model
                    return models[0].get("id", "local")
        except Exception:
            pass
        return "local"  # Fallback to the alias that the server recognizes
    
    def get_model_id(self) -> str:
        """Return the detected or configured model ID."""
        return self._model_id or "local"
    
    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        stream: bool = False,
    ) -> LocalLLMResponse:
        """Make a POST request to the llama.cpp server."""
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                result = json.loads(resp.read().decode())
                return self._parse_response(result, self._model_id or "local")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode()
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("error", {}).get("message", str(exc))
            except Exception:
                error_msg = str(exc)
            return LocalLLMResponse(
                text="",
                model=self._model_id or "local",
                error=error_msg,
            )
        except Exception as exc:
            return LocalLLMResponse(
                text="",
                model=self._model_id or "local",
                error=str(exc),
            )
    
    def _parse_response(self, data: dict[str, Any], model_id: str) -> LocalLLMResponse:
        """Parse OpenAI-compatible response into LocalLLMResponse."""
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            if "message" in choice:
                text = choice["message"].get("content", "")
            else:
                text = choice.get("text", "")
            return LocalLLMResponse(
                text=text,
                model=data.get("model", model_id),
                usage=data.get("usage"),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        return LocalLLMResponse(
            text=data.get("generated_text", ""),
            model=model_id,
            usage=None,
        )
    
    # ── Core LLM Methods ─────────────────────────────────────────────────────────────
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Generate text from a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": temperature or LOCAL_MODEL_DEFAULT_TEMPERATURE,
            "max_tokens": max_tokens or LOCAL_MODEL_DEFAULT_MAX_TOKENS,
            "stream": False,
        }
        if kwargs:
            payload.update(kwargs)
        
        return self._post(LOCAL_MODEL_CHAT_ENDPOINT, payload)
    
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Chat completion with message history."""
        payload = {
            "model": "local",
            "messages": messages,
            "temperature": temperature or LOCAL_MODEL_DEFAULT_TEMPERATURE,
            "max_tokens": max_tokens or LOCAL_MODEL_DEFAULT_MAX_TOKENS,
            "stream": False,
        }
        if kwargs:
            payload.update(kwargs)
        
        return self._post(LOCAL_MODEL_CHAT_ENDPOINT, payload)
    
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Legacy completion endpoint (uses chat under the hood)."""
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    # ── Specialized Methods ────────────────────────────────────────────────────────
    
    def edit(
        self,
        code: str,
        instruction: str,
        language: str = "python",
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Edit code based on instruction."""
        system_prompt = f"Edit the following {language} code."
        prompt = f"{instruction}\n\n```language\n{code}\n```"
        
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False,
        }
        
        return self._post(LOCAL_MODEL_CHAT_ENDPOINT, payload)
    
    def summarize(
        self,
        text: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Summarize text."""
        prompt = f"Summarize the following text:\n\n{text}"
        return self.generate(
            prompt,
            system_prompt="You are a helpful assistant that summarizes text concisely.",
            max_tokens=max_tokens or 500,
            **kwargs,
        )
    
    def repository_chat(
        self,
        messages: list[dict[str, str]],
        context: str | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Chat with repository context."""
        if context:
            # Prepend repository context as a system message
            system_msg = {"role": "system", "content": f"Repository context:\n{context}"}
            all_messages = [system_msg] + messages
        else:
            all_messages = messages
        
        return self.chat(all_messages, **kwargs)
    
    def code_review(
        self,
        code: str,
        language: str = "python",
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Review code for issues and suggestions."""
        prompt = f"Review this {language} code for issues, bugs, and improvements:\n\n```language\n{code}\n```"
        return self.generate(
            prompt,
            system_prompt="You are an expert code reviewer. Provide specific feedback on code quality, potential bugs, and suggested improvements. Be concise but thorough.",
            **kwargs,
        )
    
    def reasoning(
        self,
        prompt: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> LocalLLMResponse:
        """Perform step-by-step reasoning."""
        if context:
            system_prompt = f"You are a reasoning assistant. Consider this context:\n{context}"
        else:
            system_prompt = "You are a reasoning assistant. Think step-by-step and provide clear logical analysis."
        
        return self.generate(
            f"Let me think through this step-by-step:\n\n{prompt}",
            system_prompt=system_prompt,
            temperature=0.2,
            **kwargs,
        )
    
    # ── Health & Status ────────────────────────────────────────────────────────────
    
    def health(self) -> dict[str, Any]:
        """Check server health."""
        url = f"{self._base_url}/health"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
    
    def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        url = f"{self._base_url}/v1/models"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data.get("data", []) or data.get("models", [])
        except Exception:
            return []


# ── Convenience Functions ─────────────────────────────────────────────────────────


_provider: Optional[LocalLLMProvider] = None


def get_provider() -> LocalLLMProvider:
    """Get the singleton LocalLLMProvider instance."""
    global _provider
    if _provider is None:
        _provider = LocalLLMProvider()
    return _provider


def generate(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    """Convenience function for generate()."""
    return get_provider().generate(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    ).text


def chat(messages: list[dict[str, str]], **kwargs: Any) -> str:
    """Convenience function for chat()."""
    return get_provider().chat(messages, **kwargs).text


if __name__ == "__main__":
    # Quick test
    provider = get_provider()
    print(f"Model: {provider.get_model_id()}")
    print(f"Health: {provider.health()}")
    result = provider.generate("Say hello!")
    print(f"Response: {result.text}")
    print(f"Error: {result.error if result.error else 'None'}")