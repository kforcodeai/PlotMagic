from __future__ import annotations

import hashlib
import math
import re

from src.providers.base import EmbeddingProvider, ProviderHealth
from src.providers.config import ProviderSettings


class HashEmbeddingProvider(EmbeddingProvider):
    provider_id = "hash_embedding"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.dim = settings.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0:
            return vec
        return [value / norm for value in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            available=True,
            capabilities=["embed", "embed_batch"],
            details={"dim": self.dim},
        )

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        lowered = re.sub(r"[^a-z0-9()\s]", " ", lowered)
        return [token for token in lowered.split() if token]
