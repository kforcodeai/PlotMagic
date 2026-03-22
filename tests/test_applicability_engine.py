from __future__ import annotations

from datetime import date

from src.models import ClauseNode, ClauseType, QueryFact, RuleDocument
from src.retrieval.applicability_engine import ApplicabilityEngine


def _doc(
    doc_id: str,
    is_generic: bool,
    occupancy_groups: list[str],
    topic_tags: list[str],
    conditions: dict[str, float] | None = None,
    panchayat_category: str | None = None,
) -> RuleDocument:
    clause = ClauseNode(
        clause_id=f"{doc_id}-clause",
        clause_type=ClauseType.RULE,
        state="kerala",
        jurisdiction_type="panchayat",
        ruleset_id="KPBR_2011",
        ruleset_version="2011",
        chapter_number=5,
        chapter_title="Occupancy",
        rule_number="35",
        rule_title="Coverage and FAR",
        sub_rule_path="",
        display_citation="Rule 35",
        source_file="mock",
        anchor_id=f"{doc_id}-anchor",
        raw_text="mock text",
        normalized_text="mock text",
        is_generic=is_generic,
        occupancy_groups=occupancy_groups,
        topic_tags=topic_tags,
    )
    return RuleDocument(
        document_id=doc_id,
        state="kerala",
        jurisdiction_type="panchayat",
        ruleset_id="KPBR_2011",
        ruleset_version="2011",
        issuing_authority="LSGD",
        effective_from=date(2011, 2, 14),
        effective_to=None,
        source_file="mock",
        chapter_number=5,
        chapter_title="Occupancy",
        rule_number="35",
        rule_title="Coverage and FAR",
        full_text="mock text",
        anchor_id=f"{doc_id}-anchor",
        occupancy_groups=occupancy_groups,
        is_generic=is_generic,
        panchayat_category=panchayat_category,
        conditions=conditions or {},
        numeric_values={},
        provisos=[],
        notes=[],
        tables=[],
        cross_references=[],
        amendments=[],
        clause_nodes=[clause],
    )


def test_applicability_includes_generic_and_specific() -> None:
    docs = [
        _doc("generic-far", is_generic=True, occupancy_groups=[], topic_tags=["far"]),
        _doc("specific-far-a2", is_generic=False, occupancy_groups=["A2"], topic_tags=["far"]),
        _doc("specific-parking-a2", is_generic=False, occupancy_groups=["A2"], topic_tags=["parking"]),
    ]
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="panchayat",
        occupancies=["A2"],
        topics=["far"],
        panchayat_category=None,
    )
    result = ApplicabilityEngine().select_candidates(docs, fact)
    ids = {doc.document_id for doc in result.selected}
    assert "generic-far" in ids
    assert "specific-far-a2" in ids
    assert "specific-parking-a2" not in ids


def test_applicability_filters_category_and_conditions() -> None:
    docs = [
        _doc(
            "cat1-height",
            is_generic=False,
            occupancy_groups=["A2"],
            topic_tags=["height"],
            conditions={"height_m": 12.0},
            panchayat_category="Category-I",
        ),
        _doc(
            "cat2-height",
            is_generic=False,
            occupancy_groups=["A2"],
            topic_tags=["height"],
            conditions={"height_m": 8.0},
            panchayat_category="Category-II",
        ),
    ]
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="panchayat",
        occupancies=["A2"],
        topics=["height"],
        panchayat_category="Category-I",
        height_m=10.0,
    )
    result = ApplicabilityEngine().select_candidates(docs, fact)
    ids = {doc.document_id for doc in result.selected}
    assert "cat1-height" not in ids
    assert "cat2-height" not in ids


def test_applicability_no_topic_acts_as_wildcard() -> None:
    docs = [
        _doc("doc-a2-far", is_generic=False, occupancy_groups=["A2"], topic_tags=["far"]),
        _doc("doc-a2-parking", is_generic=False, occupancy_groups=["A2"], topic_tags=["parking"]),
    ]
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="panchayat",
        occupancies=["A2"],
        topics=[],
    )
    result = ApplicabilityEngine().select_candidates(docs, fact)
    ids = {doc.document_id for doc in result.selected}
    assert ids == {"doc-a2-far", "doc-a2-parking"}


def test_procedural_queries_ignore_occupancy_requirement() -> None:
    docs = [
        _doc("permit-doc", is_generic=False, occupancy_groups=["A2"], topic_tags=["permit"]),
        _doc("regular-doc", is_generic=False, occupancy_groups=["B"], topic_tags=["regularisation"]),
    ]
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="panchayat",
        occupancies=[],
        topics=["permit"],
        query_intent="procedural",
    )
    result = ApplicabilityEngine().select_candidates(docs, fact)
    ids = {doc.document_id for doc in result.selected}
    assert "permit-doc" in ids


def test_procedural_queries_bypass_topic_tag_gating() -> None:
    docs = [
        _doc("permit-tagged", is_generic=False, occupancy_groups=["A2"], topic_tags=["permit"]),
        _doc("mis-tagged-procedural", is_generic=False, occupancy_groups=["A2"], topic_tags=["height"]),
    ]
    fact = QueryFact(
        state="kerala",
        jurisdiction_type="panchayat",
        occupancies=[],
        topics=["permit"],
        query_intent="procedural",
    )
    result = ApplicabilityEngine().select_candidates(docs, fact)
    ids = {doc.document_id for doc in result.selected}
    assert "permit-tagged" in ids
    assert "mis-tagged-procedural" in ids
