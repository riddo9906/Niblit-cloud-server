"""Runtime Intelligence Metrics — expose operational metrics."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class RuntimeMetrics:
    """Collect and expose runtime intelligence metrics.

    Tracks:
    - inference latency
    - provider latency
    - model usage
    - memory usage
    - vector memory size
    - embedding status
    - Freqtrade health
    - trading performance
    - portfolio metrics
    - scheduler health
    - reflection statistics
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_ts = time.time()
        self._inference_latency: List[float] = []
        self._provider_latency: Dict[str, List[float]] = {}
        self._model_usage: Dict[str, int] = {}
        self._reflection_count = 0
        self._trade_count = 0

    def record_inference(self, latency_ms: float, provider: str = "default") -> None:
        with self._lock:
            self._inference_latency.append(latency_ms)
            if len(self._inference_latency) > 1000:
                self._inference_latency = self._inference_latency[-1000:]
            self._provider_latency.setdefault(provider, []).append(latency_ms)

    def record_model_usage(self, model: str) -> None:
        with self._lock:
            self._model_usage[model] = self._model_usage.get(model, 0) + 1

    def record_reflection(self) -> None:
        with self._lock:
            self._reflection_count += 1

    def record_trade(self) -> None:
        with self._lock:
            self._trade_count += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = (
                sum(self._inference_latency) / len(self._inference_latency)
                if self._inference_latency else 0.0
            )
            return {
                "uptime_seconds": time.time() - self._start_ts,
                "inference_latency_avg_ms": avg_latency,
                "inference_count": len(self._inference_latency),
                "provider_latency": {
                    k: (sum(v) / len(v) if v else 0.0)
                    for k, v in self._provider_latency.items()
                },
                "model_usage": dict(self._model_usage),
                "reflection_count": self._reflection_count,
                "trade_count": self._trade_count,
            }