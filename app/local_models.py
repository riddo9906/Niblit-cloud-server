"""Central local GGUF model registry — ONE authoritative source.

Discovers the three required local GGUF models from the authoritative model
directory (NIBLIT_GGUF_MODEL_DIR). No filesystem scanning beyond that
directory; no downloads; no Ollama substitution.

Canonical IDs:
    qwen2.5-coder-3b   primary reasoning/coding model
    llama3.2-1b        lightweight summarization agent
    qwen2.5-0.5b       lightweight classification/context agent

Aliases are TRUTHFUL:
    llama3 -> llama3.2-1b   (an actual Llama family model — never Qwen)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Keep the development default relative to the current user profile while
# allowing packaged deployments to set NIBLIT_GGUF_MODEL_DIR explicitly.
DEFAULT_GGUF_DIR = str(Path.home() / "llama_migration" / "models")

PRIMARY_MODEL_ID = "qwen2.5-coder-3b"
SUMMARIZER_MODEL_ID = "llama3.2-1b"
CONTEXT_MODEL_ID = "qwen2.5-0.5b"

# Configured RUNTIME context limits per model (tokens). These cap what the
# runtime will allocate regardless of training-context capability; they are a
# resource decision, not a claim about the models' maximum support.
DEFAULT_RUNTIME_CONTEXT = {
    PRIMARY_MODEL_ID: 16384,
    SUMMARIZER_MODEL_ID: 8192,
    CONTEXT_MODEL_ID: 4096,
}

# Filename matchers, checked case-insensitively against *.gguf names.
_MATCHERS: list[tuple[str, tuple[str, ...]]] = [
    (PRIMARY_MODEL_ID, ("qwen2.5-coder", "3b")),
    (SUMMARIZER_MODEL_ID, ("llama-3.2", "1b")),
    (SUMMARIZER_MODEL_ID, ("llama3.2", "1b")),
    (CONTEXT_MODEL_ID, ("qwen2.5-0.5b",)),
]

ROLE_ALIASES: dict[str, str] = {
    "default": PRIMARY_MODEL_ID,
    "primary": PRIMARY_MODEL_ID,
    "local": PRIMARY_MODEL_ID,
    "coder": PRIMARY_MODEL_ID,
    "small-summary": SUMMARIZER_MODEL_ID,
    "summarizer": SUMMARIZER_MODEL_ID,
    "small-context": CONTEXT_MODEL_ID,
    "classifier": CONTEXT_MODEL_ID,
    # Truthful family alias: an explicit "llama3" request maps to the actual
    # local Llama model — never to a Qwen model.
    "llama": SUMMARIZER_MODEL_ID,
    "llama3": SUMMARIZER_MODEL_ID,
}

RECOMMENDED_ROLE = {
    PRIMARY_MODEL_ID: "primary",
    SUMMARIZER_MODEL_ID: "summarization-agent",
    CONTEXT_MODEL_ID: "context-agent",
}


@dataclass
class LocalModelSpec:
    id: str
    provider: str = "local_gguf"
    path: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    file_size_bytes: int = 0
    runtime_context_limit: int = 4096
    recommended_role: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    available: bool = False
    loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "path": self.path,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "file_size_bytes": self.file_size_bytes,
            "runtime_context_limit": self.runtime_context_limit,
            "recommended_role": self.recommended_role,
            "capabilities": dict(self.capabilities),
            "available": self.available,
            "loaded": self.loaded,
        }


def gguf_model_dir() -> str:
    """Authoritative local GGUF directory."""
    return os.getenv("NIBLIT_GGUF_MODEL_DIR", DEFAULT_GGUF_DIR).strip()


def _match_canonical_id(filename: str) -> str | None:
    lowered = filename.lower()
    for canonical_id, needles in _MATCHERS:
        if all(needle in lowered for needle in needles):
            return canonical_id
    return None


def _derive_metadata(filename: str) -> dict[str, str]:
    lowered = filename.lower()
    family = (
        "qwen2.5-coder" if "qwen2.5-coder" in lowered
        else "llama3.2" if "llama" in lowered
        else "qwen2.5"
    )
    parameter_size = (
        "3B" if "3b" in lowered else "1B" if "1b" in lowered else "0.5B" if "0.5b" in lowered else "unknown"
    )
    quant = "Q4_K_M" if "q4_k_m" in lowered else "unknown"
    return {"family": family, "parameter_size": parameter_size, "quantization": quant}


def discover_local_models(directory: str | None = None) -> dict[str, LocalModelSpec]:
    """Enumerate ONLY the authoritative GGUF directory and identify models.

    Ambiguous or unmatched files are NOT guessed into a canonical ID; they are
    logged once and excluded from the executable registry.
    """
    directory = directory or gguf_model_dir()
    specs: dict[str, LocalModelSpec] = {}
    if not directory or not os.path.isdir(directory):
        logger.warning("Local GGUF directory not found: %s", directory)
        return specs

    required_ids = {cid for cid, _ in _MATCHERS}
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".gguf"):
            continue
        full_path = os.path.join(directory, filename)
        canonical_id = _match_canonical_id(filename)
        if canonical_id is None:
            logger.info("Unidentified local GGUF (not registered): %s", filename)
            continue
        if canonical_id in specs:
            logger.warning(
                "Duplicate GGUF for %s; keeping first: %s", canonical_id, specs[canonical_id].path
            )
            continue
        meta = _derive_metadata(filename)
        try:
            size_bytes = os.path.getsize(full_path)
        except OSError:
            size_bytes = 0
        specs[canonical_id] = LocalModelSpec(
            id=canonical_id,
            path=full_path,
            family=meta["family"],
            parameter_size=meta["parameter_size"],
            quantization=meta["quantization"],
            file_size_bytes=size_bytes,
            runtime_context_limit=DEFAULT_RUNTIME_CONTEXT.get(canonical_id, 4096),
            recommended_role=RECOMMENDED_ROLE.get(canonical_id, "general"),
            # File present; executability is proven on first actual load.
            available=True,
            capabilities={"chat": True, "streaming": True},
        )
    logger.info(
        "Local GGUF discovery in %s: %d/%d required model(s) found (%s)",
        directory, len(specs), len(required_ids),
        ", ".join(sorted(specs)) or "none",
    )
    return specs


def resolve_alias(model_ref: str) -> str:
    """Resolve a known alias to its canonical ID; unknown refs pass through."""
    return ROLE_ALIASES.get(model_ref.strip().lower(), model_ref.strip())


def is_known_alias(model_ref: str) -> bool:
    return model_ref.strip().lower() in ROLE_ALIASES
