from __future__ import annotations

from src.agentic import EvidenceJudge
from src.retrieval.evidence import EvidenceMatrix, RetrievalEvidence
from src.retrieval.query_planner import QueryPlan


def test_evidence_judge_marks_missing_topics() -> None:
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id="permit-doc-1",
                topic="permit",
                chunk_id="doc-1",
                document_id="doc-1",
                text="permit text",
                has_sufficient_support=True,
            )
        ]
    )
    result = EvidenceJudge().evaluate(
        plan=QueryPlan(query_type="procedural", topics=["permit", "fire_safety"]),
        evidence_matrix=matrix,
    )
    assert result.sufficient is False
    assert result.missing_topics == ["fire_safety"]


def test_evidence_judge_accepts_topicless_supported_claims() -> None:
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id="general-doc-1",
                topic="general",
                chunk_id="doc-1",
                document_id="doc-1",
                text="general text",
                has_sufficient_support=True,
            )
        ]
    )
    result = EvidenceJudge().evaluate(
        plan=QueryPlan(query_type="informational", topics=[]),
        evidence_matrix=matrix,
    )
    assert result.sufficient is True


def test_evidence_judge_detects_contradictions() -> None:
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id="permit-doc-1",
                topic="permit",
                chunk_id="doc-1",
                document_id="doc-1",
                text="Permit not necessary for this class of works.",
                has_sufficient_support=True,
                scores={"rrf_score": 0.20},
            ),
            RetrievalEvidence(
                claim_id="permit-doc-2",
                topic="permit",
                chunk_id="doc-2",
                document_id="doc-2",
                text="Building permit shall be obtained before commencement.",
                has_sufficient_support=True,
                scores={"rrf_score": 0.21},
            ),
        ]
    )
    result = EvidenceJudge().evaluate(
        plan=QueryPlan(query_type="procedural", topics=["permit"]),
        evidence_matrix=matrix,
    )
    assert result.sufficient is False
    assert result.conflicting_claim_ids


def test_evidence_judge_allows_exception_with_general_when_query_requests_both() -> None:
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id="exemption-doc-1",
                topic="exemption",
                chunk_id="doc-1",
                document_id="doc-1",
                text="Certain provisions shall not apply for works under this chapter.",
                has_sufficient_support=True,
                scores={"rrf_score": 0.24},
            ),
            RetrievalEvidence(
                claim_id="open_space-doc-2",
                topic="open_space",
                chunk_id="doc-2",
                document_id="doc-2",
                text="Street alignment restrictions shall also apply simultaneously to all buildings.",
                has_sufficient_support=True,
                scores={"rrf_score": 0.23},
            ),
        ]
    )
    plan = QueryPlan(
        query_type="informational",
        topics=["exemption", "open_space"],
        mandatory_components=["exception", "still apply"],
    )
    result = EvidenceJudge().evaluate(plan=plan, evidence_matrix=matrix)
    assert result.conflicting_claim_ids == []
    assert result.can_answer_partially is True
