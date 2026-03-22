from __future__ import annotations

from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline
from src.models import ClauseType


def test_ingestion_pipeline_parses_kerala_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline = IngestionPipeline(root / "config" / "states.yaml")
    docs, stats = pipeline.ingest_state("kerala")
    assert docs
    assert stats.parsed_rules > 0
    assert stats.parse_quality_score is not None
    assert any(doc.rule_number == "30" for doc in docs if doc.jurisdiction_type == "municipality")


def test_kpbr_parser_shape_and_appendix_separation() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline = IngestionPipeline(root / "config" / "states.yaml")
    docs, _stats = pipeline.ingest_state("kerala", "panchayat")

    numeric_rules = {int(doc.rule_number) for doc in docs if doc.rule_number.isdigit()}
    appendix_docs = [doc for doc in docs if doc.rule_number.startswith("APP-")]

    assert numeric_rules == set(range(1, 153))
    assert appendix_docs
    assert all(doc.clause_nodes and doc.clause_nodes[0].clause_type == ClauseType.APPENDIX for doc in appendix_docs)


def test_kpbr_clause_nodes_inherit_panchayat_category() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline = IngestionPipeline(root / "config" / "states.yaml")
    docs, _stats = pipeline.ingest_state("kerala", "panchayat")

    docs_with_category = [doc for doc in docs if doc.panchayat_category in {"Category-I", "Category-II", "both"}]
    assert docs_with_category
    for doc in docs_with_category[:50]:
        for clause in doc.clause_nodes:
            assert clause.panchayat_category == doc.panchayat_category
