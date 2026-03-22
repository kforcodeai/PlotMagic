from __future__ import annotations

from src.providers.adapters.cohere_reranker import CohereRerankerProvider
from src.providers.adapters.hash_embedding import HashEmbeddingProvider
from src.providers.adapters.identity_reranker import IdentityRerankerProvider, NoRerankerProvider
from src.providers.adapters.no_llm import NoLLMProvider
from src.providers.adapters.openai_embedding import OpenAIEmbeddingProvider
from src.providers.adapters.openai_llm_reranker import OpenAILLMRerankerProvider
from src.providers.adapters.openai_responses_llm import OpenAIResponsesLLMProvider
from src.providers.registry import ProviderRegistry


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register_embedding("hash_embedding", HashEmbeddingProvider)
    registry.register_embedding("openai_embedding", OpenAIEmbeddingProvider)

    registry.register_reranker("no_reranker", NoRerankerProvider)
    registry.register_reranker("identity_reranker", IdentityRerankerProvider)
    registry.register_reranker("cohere_reranker", CohereRerankerProvider)
    registry.register_reranker("openai_llm_reranker", OpenAILLMRerankerProvider)

    registry.register_llm("no_llm", NoLLMProvider)
    registry.register_llm("openai_responses_llm", OpenAIResponsesLLMProvider)
    return registry
