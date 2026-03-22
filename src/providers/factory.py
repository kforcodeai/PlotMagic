from __future__ import annotations

from dataclasses import dataclass, field

from src.providers.base import EmbeddingProvider, LLMProvider, ProviderHealth, RerankerProvider
from src.providers.config import ProvidersConfig
from src.providers.errors import ProviderError
from src.providers.registry import ProviderRegistry


@dataclass(slots=True)
class ProviderFactory:
    registry: ProviderRegistry
    config: ProvidersConfig
    diagnostics: list[str] = field(default_factory=list)

    def create_embedding_provider(self) -> EmbeddingProvider:
        requested = self.config.embedding_provider_id
        settings = self.config.embedding()
        try:
            return self.registry.build_embedding(requested, settings)
        except ProviderError as exc:
            fallback_id = "hash_embedding"
            if requested == fallback_id:
                raise
            self.diagnostics.append(
                f"Embedding provider '{requested}' unavailable ({exc}); falling back to '{fallback_id}'."
            )
            fallback_settings = self.config.embedding_settings[fallback_id]
            return self.registry.build_embedding(fallback_id, fallback_settings)

    def create_reranker_provider(self) -> RerankerProvider:
        requested = self.config.reranker_provider_id
        settings = self.config.reranker()
        if not self.config.feature_flags.rerank_enabled:
            fallback_settings = self.config.reranker_settings["no_reranker"]
            return self.registry.build_reranker("no_reranker", fallback_settings)
        try:
            return self.registry.build_reranker(requested, settings)
        except ProviderError as exc:
            fallback_id = "no_reranker"
            if requested == fallback_id:
                raise
            self.diagnostics.append(
                f"Reranker provider '{requested}' unavailable ({exc}); falling back to '{fallback_id}'."
            )
            fallback_settings = self.config.reranker_settings[fallback_id]
            return self.registry.build_reranker(fallback_id, fallback_settings)

    def create_llm_provider(self) -> LLMProvider:
        requested = self.config.llm_provider_id
        settings = self.config.llm()
        if not self.config.feature_flags.llm_enabled:
            fallback_settings = self.config.llm_settings["no_llm"]
            return self.registry.build_llm("no_llm", fallback_settings)
        try:
            return self.registry.build_llm(requested, settings)
        except ProviderError as exc:
            fallback_id = "no_llm"
            if requested == fallback_id:
                raise
            self.diagnostics.append(
                f"LLM provider '{requested}' unavailable ({exc}); falling back to '{fallback_id}'."
            )
            fallback_settings = self.config.llm_settings[fallback_id]
            return self.registry.build_llm(fallback_id, fallback_settings)

    def health_snapshot(
        self,
        embedding_provider: EmbeddingProvider,
        reranker_provider: RerankerProvider,
        llm_provider: LLMProvider,
    ) -> dict[str, ProviderHealth]:
        return {
            "embedding": embedding_provider.health(),
            "reranker": reranker_provider.health(),
            "llm": llm_provider.health(),
        }
