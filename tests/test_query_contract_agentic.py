from __future__ import annotations

from pathlib import Path

from src.api.service import ComplianceEngine
from src.models.schemas import QueryRequest


def test_query_backward_compatible_fields_present() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = ComplianceEngine(root=root)
    engine.ingest(state="kerala", jurisdiction_type="panchayat")

    response = engine.query(
        QueryRequest(
            query="What is the permit validity period in KPBR?",
            state="kerala",
            jurisdiction_type="panchayat",
            panchayat_category="Category-II",
            top_k=8,
        )
    )
    assert isinstance(response.answer_sections, list)
    assert isinstance(response.citations, list)
    assert response.verdict is not None
    assert response.final_answer is not None
    assert response.final_answer.verdict == response.verdict
    if response.verdict != "insufficient_evidence":
        assert response.final_answer.short_summary.strip()
    assert response.agent_trace == []


def test_query_debug_trace_toggle() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = ComplianceEngine(root=root)
    engine.ingest(state="kerala", jurisdiction_type="panchayat")

    response = engine.query(
        QueryRequest(
            query="What is the permit validity period in KPBR?",
            state="kerala",
            jurisdiction_type="panchayat",
            panchayat_category="Category-II",
            top_k=8,
            debug_trace=True,
        )
    )
    assert response.agent_trace
