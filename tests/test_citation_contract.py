from __future__ import annotations

from pathlib import Path

from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_builder import CitationBuilder
from src.ingestion.pipeline import IngestionPipeline
from src.models import QueryFact
from src.retrieval.evidence import EvidenceMatrix, RetrievalEvidence
from src.retrieval.query_planner import QueryPlan


def test_citation_payload_contains_anchor_and_source() -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")
    doc = docs[0]
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id="test-claim",
                topic="general",
                chunk_id=doc.document_id,
                document_id=doc.document_id,
                text=doc.full_text[:120],
                scores={"rrf_score": 0.4},
                source="hybrid",
                has_sufficient_support=True,
            )
        ]
    )
    citations = CitationBuilder().build(matrix, {doc.document_id: doc})
    assert citations
    citation = citations[0]
    assert citation.anchor_id
    assert citation.source_file == doc.source_file
    assert citation.document_id == doc.document_id
    assert citation.display_citation.startswith(doc.ruleset_id)


def test_answer_generator_uses_api_deep_links() -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")
    doc = docs[0]
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id="test-claim",
                topic="general",
                chunk_id=doc.document_id,
                document_id=doc.document_id,
                text=doc.full_text[:120],
                scores={"rrf_score": 0.4},
                source="hybrid",
                has_sufficient_support=True,
            )
        ]
    )
    response = AnswerGenerator().generate(
        query="test query",
        fact=QueryFact(state="kerala", jurisdiction_type="municipality"),
        plan=QueryPlan(query_type="informational", topics=[]),
        evidence_matrix=matrix,
        docs=[doc],
    )
    assert response.citations
    deep_link = response.citations[0].source_url
    assert deep_link == f"/rules/{doc.document_id}/source#{doc.anchor_id}"


def test_kpbr_uses_stable_generated_anchor() -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "panchayat")
    doc = docs[0]
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id="test-claim-kpbr",
                topic="general",
                chunk_id=doc.document_id,
                document_id=doc.document_id,
                text=doc.full_text[:120],
                scores={"rrf_score": 0.4},
                source="hybrid",
                has_sufficient_support=True,
            )
        ]
    )
    citations = CitationBuilder().build(matrix, {doc.document_id: doc})
    assert citations
    assert citations[0].anchor_id.startswith("kpbr-ch")
