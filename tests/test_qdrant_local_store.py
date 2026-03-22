from __future__ import annotations

from pathlib import Path

import pytest

from src.indexing import HashEmbeddingProvider
from src.indexing.qdrant_local_store import QdrantLocalVectorStore
from src.models import ClauseNode, ClauseType


pytest.importorskip("qdrant_client")


def _clause(clause_id: str, text: str) -> ClauseNode:
    return ClauseNode(
        clause_id=clause_id,
        clause_type=ClauseType.RULE,
        state="kerala",
        jurisdiction_type="panchayat",
        ruleset_id="KPBR_2011",
        ruleset_version="2011",
        chapter_number=1,
        chapter_title="General",
        rule_number="1",
        rule_title="Rule 1",
        sub_rule_path="",
        display_citation="Rule 1",
        source_file="data/kerala/kpbr_panchayat_rule.md",
        anchor_id=f"{clause_id}-anchor",
        raw_text=text,
        normalized_text=text,
        panchayat_category="Category-II",
    )


def test_qdrant_local_store_reuses_cache_and_deletes_stale_points(tmp_path: Path) -> None:
    store = QdrantLocalVectorStore(
        db_path=tmp_path / "qdrant_db",
        embedding_provider=HashEmbeddingProvider(dim=64),
        collection_name="test-collection",
        recreate_collection=True,
    )

    clauses = [_clause("c1", "first clause text"), _clause("c2", "second clause text")]
    store.upsert_clauses(clauses)
    assert store.last_upsert_stats.embedded_count == 2
    assert store.last_upsert_stats.cached_count == 0
    assert store.point_count() == 2

    store.upsert_clauses(clauses)
    assert store.last_upsert_stats.embedded_count == 0
    assert store.last_upsert_stats.cached_count == 2
    assert store.point_count() == 2

    store.upsert_clauses([clauses[0]])
    assert store.last_upsert_stats.deleted_count == 1
    assert store.point_count() == 1
