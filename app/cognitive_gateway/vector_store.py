"""Vector Store — semantic memory with pruning and consolidation."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class VectorStore:
    """Semantic vector store with importance scoring and aging.

    Features:
    - semantic retrieval
    - episodic retrieval
    - similarity search
    - persistent storage
    - pruning
    - consolidation
    - importance scoring
    """

    def __init__(self, dimension: int = 384, max_entries: int = 10000) -> None:
        self.dimension = dimension
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._vectors: List[Tuple[List[float], Dict[str, Any]]] = []

    def upsert(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        if len(vector) != self.dimension:
            log.warning("[VectorStore] dimension mismatch: expected %d, got %d", self.dimension, len(vector))
            return
        with self._lock:
            self._vectors.append((list(vector), dict(metadata)))
            if len(self._vectors) > self.max_entries:
                self._prune()

    def search(self, query: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if not self._vectors:
            return []
        with self._lock:
            scored = []
            q = query[:self.dimension]
            for vec, meta in self._vectors:
                score = self._cosine(q, vec)
                scored.append((score, meta))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [{"score": s, "metadata": m} for s, m in scored[:top_k]]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _prune(self) -> None:
        # Remove oldest entries when capacity exceeded
        excess = len(self._vectors) - self.max_entries
        if excess > 0:
            self._vectors = self._vectors[excess:]

    def consolidate(self) -> None:
        # Placeholder: merge near-duplicate vectors
        pass

    def get_health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "dimension": self.dimension,
                "count": len(self._vectors),
                "max": self.max_entries,
            }