"""Provider abstraction for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class ProviderStatus(str, Enum):
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

@dataclass
class ProviderCapabilities:
    embedding: bool = False
    chat: bool = True
    completion: bool = True
    vision: bool = False
    tools: bool = False
    streaming: bool = True

@dataclass
class ModelInfo:
    name: str
    provider: str
    context: int = 4096
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    status: ProviderStatus = ProviderStatus.AVAILABLE
    metadata: dict[str, Any] = field(default_factory=dict)

class InferenceProvider(ABC):
    """Abstract base for all inference providers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = ProviderStatus.UNAVAILABLE

    @abstractmethod
    def start(self) -> bool:
        """Initialize and start the provider. Returns True if ready."""

    @abstractmethod
    def stop(self) -> None:
        """Shutdown the provider."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return provider health status."""

    @abstractmethod
    def chat(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Run chat completion."""

    @abstractmethod
    def embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings."""