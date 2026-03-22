from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from src.providers import (
    FeatureFlags,
    ProviderAuthError,
    ProviderFactory,
    ProviderSettings,
    ProvidersConfig,
    ProviderTimeout,
    RerankCandidate,
    build_default_registry,
)
from src.providers.adapters.cohere_reranker import CohereRerankerProvider
from src.providers.adapters.hash_embedding import HashEmbeddingProvider
from src.providers.adapters.identity_reranker import IdentityRerankerProvider, NoRerankerProvider
from src.providers.adapters.no_llm import NoLLMProvider
from src.providers.adapters.openai_embedding import OpenAIEmbeddingProvider
from src.providers.adapters.openai_responses_llm import OpenAIResponsesLLMProvider


def _providers_config(embedding_provider: str, reranker_provider: str, llm_provider: str = "no_llm") -> ProvidersConfig:
    return ProvidersConfig(
        embedding_provider_id=embedding_provider,
        reranker_provider_id=reranker_provider,
        embedding_settings={
            "hash_embedding": ProviderSettings(provider_id="hash_embedding", dim=64),
            "openai_embedding": ProviderSettings(provider_id="openai_embedding"),
        },
        reranker_settings={
            "no_reranker": ProviderSettings(provider_id="no_reranker"),
            "identity_reranker": ProviderSettings(provider_id="identity_reranker"),
            "cohere_reranker": ProviderSettings(provider_id="cohere_reranker"),
        },
        llm_provider_id=llm_provider,
        llm_settings={
            "no_llm": ProviderSettings(provider_id="no_llm"),
            "openai_responses_llm": ProviderSettings(provider_id="openai_responses_llm"),
        },
        feature_flags=FeatureFlags(rerank_enabled=True, rerank_top_n=20, llm_enabled=True),
    )


def test_hash_embedding_contract() -> None:
    provider = HashEmbeddingProvider(ProviderSettings(provider_id="hash_embedding", dim=64))
    vector = provider.embed("coverage and FAR")
    vectors = provider.embed_batch(["coverage and FAR", "setback"])
    health = provider.health()

    assert len(vector) == 64
    assert len(vectors) == 2
    assert all(len(item) == 64 for item in vectors)
    assert health.available is True
    assert "embed_batch" in health.capabilities


def test_identity_and_no_reranker_contract() -> None:
    candidates = [
        RerankCandidate(candidate_id="a", text="doc a", base_score=0.3),
        RerankCandidate(candidate_id="b", text="doc b", base_score=0.9),
        RerankCandidate(candidate_id="c", text="doc c", base_score=0.6),
    ]
    identity = IdentityRerankerProvider(ProviderSettings(provider_id="identity_reranker"))
    no_reranker = NoRerankerProvider(ProviderSettings(provider_id="no_reranker"))

    ranked_identity = identity.rerank(query="q", candidates=candidates, top_n=2)
    ranked_no_reranker = no_reranker.rerank(query="q", candidates=candidates, top_n=2)

    assert [row.candidate_id for row in ranked_identity] == ["b", "c"]
    assert [row.candidate_id for row in ranked_no_reranker] == ["b", "c"]


def test_provider_factory_falls_back_to_local_defaults() -> None:
    registry = build_default_registry()
    config = _providers_config(
        embedding_provider="openai_embedding",
        reranker_provider="cohere_reranker",
        llm_provider="openai_responses_llm",
    )
    factory = ProviderFactory(registry=registry, config=config)

    embedding = factory.create_embedding_provider()
    reranker = factory.create_reranker_provider()
    llm = factory.create_llm_provider()

    assert embedding.provider_id == "hash_embedding"
    assert reranker.provider_id == "no_reranker"
    assert llm.provider_id == "no_llm"
    assert len(factory.diagnostics) == 3


def test_openai_embedding_requires_api_key() -> None:
    with pytest.raises(ProviderAuthError):
        OpenAIEmbeddingProvider(ProviderSettings(provider_id="openai_embedding"))


def test_cohere_reranker_requires_api_key() -> None:
    with pytest.raises(ProviderAuthError):
        CohereRerankerProvider(ProviderSettings(provider_id="cohere_reranker"))


def test_no_llm_contract() -> None:
    provider = NoLLMProvider(ProviderSettings(provider_id="no_llm"))
    out = provider.generate_structured(
        task="compliance_brief",
        payload={
            "draft": {
                "verdict": "depends",
                "short_summary": "summary",
                "applicable_rules": [{"claim_id": "c1", "text": "Rule applies.", "citation_ids": ["r1"]}],
                "conditions_and_exceptions": [],
                "required_actions": [],
                "risk_flags": ["risk"],
                "clarifications_needed": ["clarify"],
            }
        },
        json_schema={},
    )
    assert out["verdict"] == "depends"
    assert out["applicable_rules"][0]["citation_ids"] == ["r1"]


@dataclass
class _FakeOpenAIEmbeddingRow:
    index: int
    embedding: list[float]


def test_openai_embedding_adapter_with_mocked_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEmbeddings:
        def create(self, model: str, input: list[str], timeout: float) -> object:
            return type(
                "Response",
                (),
                {"data": [_FakeOpenAIEmbeddingRow(index=1, embedding=[0.2]), _FakeOpenAIEmbeddingRow(index=0, embedding=[0.1])]},
            )()

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.embeddings = _FakeEmbeddings()

    fake_module = ModuleType("openai")
    fake_module.OpenAI = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    provider = OpenAIEmbeddingProvider(
        ProviderSettings(
            provider_id="openai_embedding",
            api_key="test-key",
            model="fake-model",
            timeout_s=0.1,
            max_retries=0,
        )
    )

    vectors = provider.embed_batch(["a", "b"])
    assert vectors == [[0.1], [0.2]]


def test_openai_embedding_timeout_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEmbeddings:
        def create(self, model: str, input: list[str], timeout: float) -> object:
            raise RuntimeError("request timeout")

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.embeddings = _FakeEmbeddings()

    fake_module = ModuleType("openai")
    fake_module.OpenAI = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    provider = OpenAIEmbeddingProvider(
        ProviderSettings(
            provider_id="openai_embedding",
            api_key="test-key",
            timeout_s=0.1,
            max_retries=0,
        )
    )
    with pytest.raises(ProviderTimeout):
        provider.embed_batch(["query"])


def test_cohere_reranker_adapter_with_mocked_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        def __init__(self, index: int, relevance_score: float) -> None:
            self.index = index
            self.relevance_score = relevance_score

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def rerank(
            self,
            *,
            model: str,
            query: str,
            documents: list[str],
            top_n: int,
            request_options: dict[str, float],
        ) -> object:
            return type(
                "Response",
                (),
                {"results": [_FakeResult(index=1, relevance_score=0.9), _FakeResult(index=0, relevance_score=0.2)]},
            )()

    fake_module = ModuleType("cohere")
    fake_module.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", fake_module)

    provider = CohereRerankerProvider(
        ProviderSettings(
            provider_id="cohere_reranker",
            api_key="test-key",
            model="fake-model",
            timeout_s=0.1,
            max_retries=0,
        )
    )
    candidates = [
        RerankCandidate(candidate_id="a", text="doc a", base_score=0.1),
        RerankCandidate(candidate_id="b", text="doc b", base_score=0.2),
    ]
    out = provider.rerank(query="q", candidates=candidates, top_n=2)
    assert [row.candidate_id for row in out] == ["b", "a"]


def test_cohere_reranker_timeout_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def rerank(
            self,
            *,
            model: str,
            query: str,
            documents: list[str],
            top_n: int,
            request_options: dict[str, float],
        ) -> object:
            raise RuntimeError("provider timeout")

    fake_module = ModuleType("cohere")
    fake_module.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cohere", fake_module)

    provider = CohereRerankerProvider(
        ProviderSettings(
            provider_id="cohere_reranker",
            api_key="test-key",
            timeout_s=0.1,
            max_retries=0,
        )
    )
    with pytest.raises(ProviderTimeout):
        provider.rerank(query="q", candidates=[RerankCandidate(candidate_id="a", text="a")], top_n=1)


def test_openai_responses_requires_api_key() -> None:
    with pytest.raises(ProviderAuthError):
        OpenAIResponsesLLMProvider(ProviderSettings(provider_id="openai_responses_llm"))


def test_openai_responses_adapter_with_mocked_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponses:
        def create(self, **kwargs) -> object:
            captured.update(kwargs)
            return type("Response", (), {"output_text": '{"verdict":"depends","short_summary":"ok"}'})()

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.responses = _FakeResponses()

    fake_module = ModuleType("openai")
    fake_module.OpenAI = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    provider = OpenAIResponsesLLMProvider(
        ProviderSettings(
            provider_id="openai_responses_llm",
            api_key="test-key",
            model="fake-model",
            timeout_s=0.1,
            max_retries=0,
        )
    )
    out = provider.generate_structured(
        task="compliance_brief",
        payload={"draft": {}},
        json_schema={"type": "object", "properties": {"verdict": {"type": "string"}}},
    )
    assert out["verdict"] == "depends"
    schema = captured["text"]["format"]["schema"]  # type: ignore[index]
    assert isinstance(schema, dict)
    assert schema.get("additionalProperties") is False


def test_openai_responses_timeout_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponses:
        def create(self, **kwargs) -> object:
            raise RuntimeError("request timeout")

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.responses = _FakeResponses()

    fake_module = ModuleType("openai")
    fake_module.OpenAI = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    provider = OpenAIResponsesLLMProvider(
        ProviderSettings(
            provider_id="openai_responses_llm",
            api_key="test-key",
            timeout_s=0.1,
            max_retries=0,
        )
    )
    with pytest.raises(ProviderTimeout):
        provider.generate_structured(task="compliance_brief", payload={"draft": {}}, json_schema={"type": "object"})


def test_provider_sdk_imports_are_adapter_only() -> None:
    root = Path(__file__).resolve().parents[1]
    allowed = {
        "src/providers/adapters/openai_embedding.py",
        "src/providers/adapters/openai_responses_llm.py",
        "src/providers/adapters/openai_llm_reranker.py",
        "src/providers/adapters/cohere_reranker.py",
    }

    offenders: list[str] = []
    for path in sorted((root / "src").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        content = path.read_text(encoding="utf-8")
        if "import openai" in content or "from openai import" in content:
            if rel not in allowed:
                offenders.append(rel)
        if "import cohere" in content or "from cohere import" in content:
            if rel not in allowed:
                offenders.append(rel)

    assert offenders == []
