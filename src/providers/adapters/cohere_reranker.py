from __future__ import annotations

import os
import time

from src.providers.base import ProviderHealth, RerankCandidate, RerankResult, RerankerProvider
from src.providers.config import ProviderSettings
from src.providers.errors import ProviderAuthError, ProviderTimeout, ProviderUnavailable


class CohereRerankerProvider(RerankerProvider):
    provider_id = "cohere_reranker"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model = settings.model or "rerank-v3.5"
        self.timeout_s = settings.timeout_s
        self.max_retries = max(0, settings.max_retries)
        self.retry_backoff_s = max(0.0, settings.retry_backoff_s)

        api_key = settings.api_key
        if not api_key and settings.api_key_env:
            api_key = os.getenv(settings.api_key_env)
        if not api_key:
            raise ProviderAuthError("Cohere API key is not configured")

        try:
            import cohere  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("cohere package is not installed") from exc

        self._client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not candidates:
            return []
        docs = [candidate.text for candidate in candidates]
        top_n = top_n or min(len(candidates), self.settings.top_n)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.rerank(
                    model=self.model,
                    query=query,
                    documents=docs,
                    top_n=top_n,
                    request_options={"timeout_in_seconds": self.timeout_s},
                )
                out: list[RerankResult] = []
                for row in response.results:
                    idx = int(row.index)
                    out.append(
                        RerankResult(
                            candidate_id=candidates[idx].candidate_id,
                            score=float(row.relevance_score),
                        )
                    )
                return out
            except Exception as exc:  # pragma: no cover - network/provider behavior
                last_error = exc
                if "timeout" in str(exc).lower():
                    mapped = ProviderTimeout(f"Cohere rerank timeout: {exc}")
                else:
                    mapped = ProviderUnavailable(f"Cohere rerank failure: {exc}")
                if attempt >= self.max_retries:
                    raise mapped from exc
                time.sleep(self.retry_backoff_s * (attempt + 1))

        raise ProviderUnavailable(f"Cohere reranker failed after retries: {last_error}")

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            available=True,
            capabilities=["rerank"],
            details={"model": self.model, "timeout_s": self.timeout_s},
        )
