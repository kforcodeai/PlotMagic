from __future__ import annotations

from src.providers.base import ProviderHealth, RerankCandidate, RerankResult, RerankerProvider
from src.providers.config import ProviderSettings


class IdentityRerankerProvider(RerankerProvider):
    provider_id = "identity_reranker"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        ranked = sorted(candidates, key=lambda candidate: candidate.base_score, reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return [RerankResult(candidate_id=item.candidate_id, score=item.base_score) for item in ranked]

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            available=True,
            capabilities=["rerank"],
            details={"mode": "identity"},
        )


class NoRerankerProvider(IdentityRerankerProvider):
    provider_id = "no_reranker"
