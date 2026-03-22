from __future__ import annotations

from collections.abc import Callable

from src.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider
from src.providers.config import ProviderSettings
from src.providers.errors import ProviderConfigurationError


EmbeddingBuilder = Callable[[ProviderSettings], EmbeddingProvider]
RerankerBuilder = Callable[[ProviderSettings], RerankerProvider]
LLMBuilder = Callable[[ProviderSettings], LLMProvider]


class ProviderRegistry:
    def __init__(self) -> None:
        self._embedding_builders: dict[str, EmbeddingBuilder] = {}
        self._reranker_builders: dict[str, RerankerBuilder] = {}
        self._llm_builders: dict[str, LLMBuilder] = {}

    def register_embedding(self, provider_id: str, builder: EmbeddingBuilder) -> None:
        self._embedding_builders[provider_id] = builder

    def register_reranker(self, provider_id: str, builder: RerankerBuilder) -> None:
        self._reranker_builders[provider_id] = builder

    def register_llm(self, provider_id: str, builder: LLMBuilder) -> None:
        self._llm_builders[provider_id] = builder

    def build_embedding(self, provider_id: str, settings: ProviderSettings) -> EmbeddingProvider:
        builder = self._embedding_builders.get(provider_id)
        if not builder:
            raise ProviderConfigurationError(f"Embedding provider '{provider_id}' is not registered")
        return builder(settings)

    def build_reranker(self, provider_id: str, settings: ProviderSettings) -> RerankerProvider:
        builder = self._reranker_builders.get(provider_id)
        if not builder:
            raise ProviderConfigurationError(f"Reranker provider '{provider_id}' is not registered")
        return builder(settings)

    def build_llm(self, provider_id: str, settings: ProviderSettings) -> LLMProvider:
        builder = self._llm_builders.get(provider_id)
        if not builder:
            raise ProviderConfigurationError(f"LLM provider '{provider_id}' is not registered")
        return builder(settings)

    @property
    def embedding_provider_ids(self) -> list[str]:
        return sorted(self._embedding_builders.keys())

    @property
    def reranker_provider_ids(self) -> list[str]:
        return sorted(self._reranker_builders.keys())

    @property
    def llm_provider_ids(self) -> list[str]:
        return sorted(self._llm_builders.keys())
