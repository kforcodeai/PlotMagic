from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderHealth:
    provider_id: str
    available: bool
    capabilities: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RerankCandidate:
    candidate_id: str
    text: str
    base_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RerankResult:
    candidate_id: str
    score: float


class EmbeddingProvider(ABC):
    provider_id: str

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_document_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_batch(texts)

    @abstractmethod
    def health(self) -> ProviderHealth:
        raise NotImplementedError


class RerankerProvider(ABC):
    provider_id: str

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> ProviderHealth:
        raise NotImplementedError


class LLMProvider(ABC):
    provider_id: str

    @abstractmethod
    def generate_structured(
        self,
        *,
        task: str,
        payload: dict[str, Any],
        json_schema: dict[str, Any],
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> ProviderHealth:
        raise NotImplementedError
