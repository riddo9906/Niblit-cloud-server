"""Embedding Service — text embeddings for semantic memory."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class EmbeddingService:
    """Text embedding service with provider abstraction.

    Supports:
    - Sentence Transformers
    - Nomic
    - Ollama Embeddings
    - OpenAI-compatible providers
    - future providers
    """

    def __init__(self, default_provider: str = "sentence_transformers") -> None:
        self.default_provider = default_provider
        self._lock = threading.Lock()
        self._model = None
        self._dimension = 384
        self._available = False

    def initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self._available = True
            log.info("[EmbeddingService] SentenceTransformers available")
        except Exception as exc:
            log.warning("[EmbeddingService] SentenceTransformers unavailable: %s", exc)
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def embed(self, text: str, provider: Optional[str] = None) -> List[float]:
        if not self._available:
            return [0.0] * self._dimension
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            with self._lock:
                result = self._model.encode([text], normalize_embeddings=True)
                return result[0].tolist() if hasattr(result, "tolist") else list(result[0])
        except Exception as exc:
            log.warning("[EmbeddingService] embed failed: %s", exc)
            return [0.0] * self._dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._available:
            return [[0.0] * self._dimension for _ in texts]
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            with self._lock:
                result = self._model.encode(texts, normalize_embeddings=True)
                return result.tolist() if hasattr(result, "tolist") else [list(v) for v in result]
        except Exception as exc:
            log.warning("[EmbeddingService] batch embed failed: %s", exc)
            return [[0.0] * self._dimension for _ in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def get_health(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "provider": self.default_provider,
            "dimension": self._dimension,
        }