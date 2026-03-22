from __future__ import annotations

import os
import time
from typing import Any

from src.providers.base import EmbeddingProvider, ProviderHealth
from src.providers.config import ProviderSettings
from src.providers.errors import ProviderAuthError, ProviderTimeout, ProviderUnavailable


class OpenAIEmbeddingProvider(EmbeddingProvider):
    provider_id = "openai_embedding"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model = settings.model or "text-embedding-3-large"
        self.timeout_s = settings.timeout_s
        self.max_retries = max(0, settings.max_retries)
        self.retry_backoff_s = max(0.0, settings.retry_backoff_s)

        api_key = settings.api_key
        if not api_key and settings.api_key_env:
            api_key = os.getenv(settings.api_key_env)
        if not api_key:
            raise ProviderAuthError("OpenAI API key is not configured")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("openai package is not installed") from exc

        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.embeddings.create(model=self.model, input=texts, timeout=self.timeout_s)
                data: list[Any] = sorted(response.data, key=lambda row: row.index)
                return [list(row.embedding) for row in data]
            except Exception as exc:  # pragma: no cover - network/provider behavior
                last_error = exc
                if "timeout" in str(exc).lower():
                    mapped = ProviderTimeout(f"OpenAI embedding timeout: {exc}")
                else:
                    mapped = ProviderUnavailable(f"OpenAI embedding failure: {exc}")
                if attempt >= self.max_retries:
                    raise mapped from exc
                time.sleep(self.retry_backoff_s * (attempt + 1))

        raise ProviderUnavailable(f"OpenAI embedding failed after retries: {last_error}")

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            available=True,
            capabilities=["embed", "embed_batch"],
            details={"model": self.model, "timeout_s": self.timeout_s},
        )
