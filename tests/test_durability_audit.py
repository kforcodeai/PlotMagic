from __future__ import annotations

from pathlib import Path

import yaml

from src.indexing.structured_store import StructuredStore
from src.ingestion.parsers.generic_legal_parser import GenericLegalParser
from src.ingestion.pipeline import IngestionPipeline


def test_structured_store_upsert_is_idempotent_for_secondary_tables(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")

    store = StructuredStore(tmp_path / "structured.db")
    store.upsert_documents(docs)
    first_table_cells = int(store.conn.execute("SELECT COUNT(*) FROM table_cells").fetchone()[0])
    first_cross_refs = int(store.conn.execute("SELECT COUNT(*) FROM cross_references").fetchone()[0])

    store.upsert_documents(docs)
    second_table_cells = int(store.conn.execute("SELECT COUNT(*) FROM table_cells").fetchone()[0])
    second_cross_refs = int(store.conn.execute("SELECT COUNT(*) FROM cross_references").fetchone()[0])

    assert second_table_cells == first_table_cells
    assert second_cross_refs == first_cross_refs


def test_generic_legal_parser_extracts_sections_tables_and_cross_refs(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text(
        "\n".join(
            [
                "Section 1. Permit requirement",
                "Any building permit shall be obtained as per Rule 2.",
                "",
                "| Type | Limit |",
                "| ---- | ----- |",
                "| Residential | 2 floors |",
                "",
                "Section 2. Exemption",
                "Small structures are exempted.",
            ]
        ),
        encoding="utf-8",
    )
    parser = GenericLegalParser(
        state="test_state",
        jurisdiction_type="test_jurisdiction",
        ruleset_id="TEST_RULESET",
        ruleset_version="v1",
        issuing_authority="Test Authority",
    )
    docs = parser.parse_file(source)

    assert len(docs) == 2
    assert docs[0].rule_number == "1"
    assert docs[1].rule_number == "2"
    assert docs[0].tables
    assert any(ref.target_ref.lower().startswith("rule 2") for ref in docs[0].cross_references)


def test_pipeline_recursively_ingests_supported_files_with_generic_parser(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    nested = source_dir / "nested"
    nested.mkdir(parents=True)
    (source_dir / "regulation.md").write_text("Section 1. Scope\nThis applies to all entities.", encoding="utf-8")
    (nested / "annex.txt").write_text("Clause 2: Filing deadline is 30 days.", encoding="utf-8")

    cfg = {
        "states": {
            "test_state": {
                "jurisdictions": {
                    "compliance": {
                        "ruleset_id": "TEST_RULESET",
                        "ruleset_version": "v1",
                        "parser_class": "GenericLegalParser",
                        "source_path": str(source_dir),
                    }
                }
            }
        }
    }
    config_path = tmp_path / "states.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    docs, stats = IngestionPipeline(config_path).ingest_state("test_state", "compliance")
    assert docs
    assert stats.parsed_files == 2
    assert stats.failed_files == 0
