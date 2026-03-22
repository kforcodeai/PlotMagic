from __future__ import annotations

from src.providers.base import EmbeddingProvider
from src.providers.config import ProviderSettings
from src.providers.adapters.hash_embedding import HashEmbeddingProvider as _HashEmbeddingProvider


class HashEmbeddingProvider(_HashEmbeddingProvider):
    def __init__(self, dim: int = 256) -> None:
        super().__init__(ProviderSettings(provider_id="hash_embedding", dim=dim))
