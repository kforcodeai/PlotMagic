from __future__ import annotations

from pathlib import Path

from src.indexing import HashEmbeddingProvider, InMemoryVectorStore, LexicalIndex, StructuredStore
from src.ingestion.pipeline import IngestionPipeline
from src.models import QueryFact
from src.providers import ProviderHealth, ProviderTimeout, RerankCandidate, RerankerProvider
from src.retrieval import ApplicabilityEngine, HybridRetriever, QueryPlanner


def test_hybrid_retrieval_builds_evidence_matrix(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")

    structured = StructuredStore(tmp_path / "structured.db")
    structured.upsert_documents(docs)

    lexical = LexicalIndex()
    lexical.build(docs)

    vector = InMemoryVectorStore(HashEmbeddingProvider())
    vector.upsert_clauses([clause for doc in docs for clause in doc.clause_nodes])

    planner = QueryPlanner()
    plan = planner.plan("What is the FAR and coverage for Group A1 residential buildings?")
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="municipality",
        occupancies=["A1"],
        topics=plan.topics,
    )
    candidates = ApplicabilityEngine().select_candidates(docs, fact).selected
    retriever = HybridRetriever(vector, lexical, structured, docs)
    result = retriever.retrieve(
        query="What is the FAR and coverage for Group A1 residential buildings?",
        fact=fact,
        plan=plan,
        candidate_docs=candidates,
        top_k=8,
    )
    assert result.evidence_matrix.items
    assert result.retrieved_documents


class _TimeoutReranker(RerankerProvider):
    provider_id = "timeout_reranker"

    def rerank(self, query: str, candidates: list[RerankCandidate], top_n: int | None = None) -> list:
        raise ProviderTimeout("simulated timeout")

    def health(self) -> ProviderHealth:
        return ProviderHealth(provider_id=self.provider_id, available=False)


def test_hybrid_retrieval_degrades_when_reranker_fails(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")

    structured = StructuredStore(tmp_path / "structured.db")
    structured.upsert_documents(docs)

    lexical = LexicalIndex()
    lexical.build(docs)

    vector = InMemoryVectorStore(HashEmbeddingProvider())
    vector.upsert_clauses([clause for doc in docs for clause in doc.clause_nodes])

    planner = QueryPlanner()
    plan = planner.plan("What are the parking requirements for apartment buildings?")
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="municipality",
        occupancies=["A1"],
        topics=plan.topics,
    )
    candidates = ApplicabilityEngine().select_candidates(docs, fact).selected
    retriever = HybridRetriever(
        vector_store=vector,
        lexical_index=lexical,
        structured_store=structured,
        all_docs=docs,
        reranker_provider=_TimeoutReranker(),
        rerank_top_n=20,
    )
    result = retriever.retrieve(
        query="What are the parking requirements for apartment buildings?",
        fact=fact,
        plan=plan,
        candidate_docs=candidates,
        top_k=8,
    )
    assert result.retrieved_documents
    assert result.latency_ms.get("reranker_used") == 1.0
    assert result.latency_ms.get("reranker_degraded") == 1.0
    assert result.latency_ms.get("reranker_failures") == 1.0


def test_vector_only_mode_disables_lexical_structured_and_reranker(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")

    structured = StructuredStore(tmp_path / "structured.db")
    structured.upsert_documents(docs)

    lexical = LexicalIndex()
    lexical.build(docs)

    vector = InMemoryVectorStore(HashEmbeddingProvider())
    vector.upsert_clauses([clause for doc in docs for clause in doc.clause_nodes])

    planner = QueryPlanner()
    plan = planner.plan("What are the parking requirements for apartment buildings?")
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="municipality",
        occupancies=["A1"],
        topics=plan.topics,
    )
    candidates = ApplicabilityEngine().select_candidates(docs, fact).selected
    retriever = HybridRetriever(
        vector_store=vector,
        lexical_index=lexical,
        structured_store=structured,
        all_docs=docs,
        reranker_provider=_TimeoutReranker(),
        rerank_top_n=20,
    )
    result = retriever.retrieve(
        query="What are the parking requirements for apartment buildings?",
        fact=fact,
        plan=plan,
        candidate_docs=candidates,
        top_k=8,
        retrieval_mode="vector_only",
    )
    assert result.retrieved_documents
    assert result.latency_ms.get("reranker_used") == 0.0
    assert result.latency_ms.get("lexical_hits") == 0.0
    assert result.latency_ms.get("structured_hits") == 0.0
    assert result.diagnostics.get("retrieval_mode") == "vector_only"


def test_hybrid_no_reranker_mode_keeps_hybrid_signals_without_reranking(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")

    structured = StructuredStore(tmp_path / "structured.db")
    structured.upsert_documents(docs)

    lexical = LexicalIndex()
    lexical.build(docs)

    vector = InMemoryVectorStore(HashEmbeddingProvider())
    vector.upsert_clauses([clause for doc in docs for clause in doc.clause_nodes])

    planner = QueryPlanner()
    plan = planner.plan("What are the parking requirements for apartment buildings?")
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="municipality",
        occupancies=["A1"],
        topics=plan.topics,
    )
    candidates = ApplicabilityEngine().select_candidates(docs, fact).selected
    retriever = HybridRetriever(
        vector_store=vector,
        lexical_index=lexical,
        structured_store=structured,
        all_docs=docs,
        reranker_provider=_TimeoutReranker(),
        rerank_top_n=20,
    )
    result = retriever.retrieve(
        query="What are the parking requirements for apartment buildings?",
        fact=fact,
        plan=plan,
        candidate_docs=candidates,
        top_k=8,
        retrieval_mode="hybrid_no_reranker",
    )
    assert result.retrieved_documents
    assert result.latency_ms.get("reranker_used") == 0.0
    assert result.latency_ms.get("lexical_query_variants", 0.0) > 0.0
    assert result.diagnostics.get("retrieval_mode") == "hybrid_no_reranker"


def test_lexical_only_mode_uses_bm25_without_vector_signal(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")

    structured = StructuredStore(tmp_path / "structured.db")
    structured.upsert_documents(docs)

    lexical = LexicalIndex()
    lexical.build(docs)

    vector = InMemoryVectorStore(HashEmbeddingProvider())
    vector.upsert_clauses([clause for doc in docs for clause in doc.clause_nodes])

    planner = QueryPlanner()
    plan = planner.plan("What are the parking requirements for apartment buildings?")
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="municipality",
        occupancies=["A1"],
        topics=plan.topics,
    )
    candidates = ApplicabilityEngine().select_candidates(docs, fact).selected
    retriever = HybridRetriever(
        vector_store=vector,
        lexical_index=lexical,
        structured_store=structured,
        all_docs=docs,
        reranker_provider=_TimeoutReranker(),
        rerank_top_n=20,
    )
    result = retriever.retrieve(
        query="What are the parking requirements for apartment buildings?",
        fact=fact,
        plan=plan,
        candidate_docs=candidates,
        top_k=8,
        retrieval_mode="lexical_only_bm25",
    )
    assert result.retrieved_documents
    assert result.latency_ms.get("vector_hits") == 0.0
    assert result.latency_ms.get("lexical_hits", 0.0) > 0.0
    assert result.latency_ms.get("structured_hits") == 0.0
    assert result.diagnostics.get("fallback_stage") == "vector_disabled"


def test_hybrid_graph_reranker_mode_emits_graph_diagnostics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")

    structured = StructuredStore(tmp_path / "structured.db")
    structured.upsert_documents(docs)

    lexical = LexicalIndex()
    lexical.build(docs)

    vector = InMemoryVectorStore(HashEmbeddingProvider())
    vector.upsert_clauses([clause for doc in docs for clause in doc.clause_nodes])

    planner = QueryPlanner()
    plan = planner.plan("Within 100 metres of defence land, what consultation and timeline applies?")
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="municipality",
        occupancies=["A1"],
        topics=plan.topics,
    )
    candidates = ApplicabilityEngine().select_candidates(docs, fact).selected
    retriever = HybridRetriever(
        vector_store=vector,
        lexical_index=lexical,
        structured_store=structured,
        all_docs=docs,
        reranker_provider=_TimeoutReranker(),
        rerank_top_n=20,
    )
    result = retriever.retrieve(
        query="Within 100 metres of defence land, what consultation and timeline applies?",
        fact=fact,
        plan=plan,
        candidate_docs=candidates,
        top_k=8,
        retrieval_mode="hybrid_graph_reranker",
    )
    assert result.retrieved_documents
    assert result.diagnostics.get("graph_expansion_used") is True
    assert "graph_ref_rules_count" in result.diagnostics
    assert "graph_entity_terms_count" in result.diagnostics
