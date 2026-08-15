"""Unified configuration bootstrap for Niblit Cloud Server.

This is the ONLY module that reads os.getenv(). Every other subsystem
receives configuration from this module, ensuring one authoritative
configuration source.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_dotenv(path: Path) -> dict[str, str]:
    """Best-effort dotenv loader. Returns dict of key/value pairs."""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as exc:
        logger.debug("Failed to load env file %s: %s", path, exc)
    return env


@dataclass
class CloudConfig:
    """Authoritative configuration object for the entire Cloud Server."""

    # ── env file loader bookkeeping ───────────────────────────────────────────
    loaded_env_files: list[str] = field(default_factory=list)

    # ── Model / registry config ───────────────────────────────────────────────
    gguf_models_json: str = ""
    default_model_id: str | None = None
    n_ctx: int = 16384
    n_threads: int = 4
    n_batch: int = 1024
    n_ubatch: int = 512
    context_reserve_tokens: int = 512
    min_generation_tokens: int = 64
    char_per_token: int = 4
    memory_guard_ratio: float = 0.92
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None

    # ── Host / port ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Storage ──────────────────────────────────────────────────────────────
    storage_root: str = ""

    # ── Providers ────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    openai_api_base: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    hf_api_key: str = ""
    llama_cpp_server_url: str = ""
    vllm_api_base: str = ""
    remote_api_base_url: str = ""
    remote_api_key: str = ""

    # ── qwen server provider ─────────────────────────────────────────────────
    qwen_model_path: str = ""
    qwen_backend_bin: str = ""
    qwen_host: str = "127.0.0.1"
    qwen_port: int = 8000
    qwen_n_ctx: int = 32768
    qwen_n_threads: int = 8
    qwen_n_gpu_layers: int = 35

    # ── Runtime ──────────────────────────────────────────────────────────────
    runtime_mode: str = "normal"

    # ── Plugins ──────────────────────────────────────────────────────────────
    plugins_dir: str = ""

    @classmethod
    def load(cls) -> CloudConfig:
        """Load configuration from env files then OS environment."""
        root = Path(os.getcwd())
        candidates = [
            root / ".env",
            root / ".env.local",
            root / ".env.local_llm",
        ]
        merged: dict[str, str] = {}
        loaded: list[str] = []

        # During tests, avoid loading local env files so testcases control state.
        testing = "pytest" in sys.modules or bool(os.getenv("PYTEST_CURRENT_TEST"))
        for path in candidates:
            if testing and path.name in {".env.local", ".env.local_llm"}:
                continue
            values = _load_dotenv(path)
            if values:
                merged.update(values)
                loaded.append(path.name)
                logger.debug("Loaded env file: %s", path.name)

        # OS environment variables override and extend file values
        for key in list(os.environ.keys()):
            merged[key] = os.environ[key]

        cfg = cls.from_dict(merged)
        cfg.loaded_env_files = loaded
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudConfig:
        """Construct config from a flat key/value mapping."""
        def _get_str(key: str, default: str = "") -> str:
            val = data.get(key)
            if val is None:
                return default
            if not isinstance(val, str):
                val = str(val)
            return val.strip()

        def _get_int(key: str, fallback: str, default: int) -> int:
            raw = _get_str(key, fallback)
            if not raw:
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        def _get_float(key: str, fallback: str, default: float | None = None) -> float | None:
            raw = _get_str(key, fallback)
            if not raw and default is None:
                return None
            if not raw:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        default_model_id = _get_str("DEFAULT_MODEL_ID", "") or _get_str("NIBLIT_DEFAULT_MODEL_ID", "") or None
        config = cls(
            loaded_env_files=[],
            gguf_models_json=_get_str("GGUF_MODELS_JSON", ""),
            default_model_id=default_model_id,
            n_ctx=_get_int("N_CTX", _get_str("NIBLIT_CONTEXT_WINDOW", ""), 16384),
            n_threads=_get_int("N_THREADS", "", 4),
            n_batch=_get_int("NIBLIT_N_BATCH", _get_str("N_BATCH", ""), 1024),
            n_ubatch=_get_int("NIBLIT_N_UBATCH", _get_str("N_UBATCH", ""), 512),
            context_reserve_tokens=_get_int("NIBLIT_CONTEXT_RESERVE_TOKENS", "", 512),
            min_generation_tokens=_get_int("NIBLIT_MIN_GENERATION_TOKENS", "", 64),
            char_per_token=max(1, _get_int("NIBLIT_CHAR_PER_TOKEN", "", 4)),
            memory_guard_ratio=max(
                0.5,
                min(0.98, _get_float("NIBLIT_MEMORY_GUARD_RATIO", "", 0.92) or 0.92),
            ),
            rope_freq_base=_get_float("NIBLIT_ROPE_FREQ_BASE", "", None),
            rope_freq_scale=_get_float("NIBLIT_ROPE_FREQ_SCALE", "", None),
            host=_get_str("NIBLIT_HOST", "0.0.0.0"),
            port=_get_int("NIBLIT_PORT", "", 8000),
            storage_root=_get_str("NIBLIT_STORAGE_ROOT", ""),
            ollama_base_url=_get_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            openai_api_base=_get_str("OPENAI_API_BASE", ""),
            openai_api_key=_get_str("OPENAI_API_KEY", ""),
            anthropic_api_key=_get_str("ANTHROPIC_API_KEY", ""),
            hf_api_key=_get_str("HF_API_KEY", ""),
            llama_cpp_server_url=_get_str("LLAMA_CPP_SERVER_URL", ""),
            vllm_api_base=_get_str("VLLM_API_BASE", ""),
            remote_api_base_url=_get_str("REMOTE_API_BASE_URL", ""),
            remote_api_key=_get_str("REMOTE_API_KEY", ""),
            qwen_model_path=_get_str("NIBLIT_MODEL_PATH", "") or _get_str("NIBLIT_DEFAULT_MODEL_PATH", ""),
            qwen_backend_bin=_get_str("NIBLIT_LLAMA_SERVER_BIN", ""),
            qwen_host=_get_str("NIBLIT_HOST", "127.0.0.1"),
            qwen_port=_get_int("NIBLIT_PORT", "", 8000),
            qwen_n_ctx=_get_int("NIBLIT_N_CTX", "", 32768),
            qwen_n_threads=_get_int("NIBLIT_N_THREADS", "", 8),
            qwen_n_gpu_layers=_get_int("NIBLIT_N_GPU_LAYERS", "", 35),
            runtime_mode=_get_str("NIBLIT_RUNTIME_MODE", "normal"),
            plugins_dir=_get_str("NIBLIT_PLUGINS_DIR", ""),
        )
        return config

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def model_map(self) -> dict[str, str]:
        if self.gguf_models_json:
            try:
                import json
                loaded = json.loads(self.gguf_models_json)
                if isinstance(loaded, dict):
                    return {str(k): str(v) for k, v in loaded.items()}
            except json.JSONDecodeError:
                pass
        fallback_path = self.qwen_model_path or os.getenv("NIBLIT_DEFAULT_MODEL_PATH", "")
        if not fallback_path:
            return {}
        model_id = self.default_model_id or "fallback"
        return {model_id: fallback_path}

    @property
    def default_model(self) -> str | None:
        if self.default_model_id and self.default_model_id in self.model_map:
            return self.default_model_id
        if self.model_map:
            return next(iter(self.model_map))
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "loaded_env_files": list(self.loaded_env_files),
            "default_model_id": self.default_model_id,
            "registered_models": list(self.model_map.keys()),
            "host": self.host,
            "port": self.port,
            "runtime_mode": self.runtime_mode,
            "providers": {
                "qwen_server": bool(self.qwen_model_path and self.qwen_backend_bin),
                "ollama": bool(self.ollama_base_url),
                "openai": bool(self.openai_api_base or self.openai_api_key),
                "anthropic": bool(self.anthropic_api_key),
                "huggingface": bool(self.hf_api_key),
                "llama_cpp": bool(self.llama_cpp_server_url),
                "vllm": bool(self.vllm_api_base),
                "remote": bool(self.remote_api_base_url),
            },
        }


# Singleton
_config: CloudConfig | None = None


def get_config() -> CloudConfig:
    """Return the process-level CloudConfig singleton."""
    global _config
    if _config is None:
        _config = CloudConfig.load()
    return _config


def reset_config() -> None:
    """Reset the singleton (mainly for tests)."""
    global _config
    _config = None