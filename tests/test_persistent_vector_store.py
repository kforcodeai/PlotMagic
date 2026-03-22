from __future__ import annotations

from pathlib import Path

from src.indexing import HashEmbeddingProvider, PersistentLocalVectorStore
from src.ingestion.pipeline import IngestionPipeline


def test_persistent_vector_store_reuses_cached_embeddings(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")
    clauses = [clause for doc in docs for clause in doc.clause_nodes]
    assert clauses

    db_path = tmp_path / "vector_index.db"
    store = PersistentLocalVectorStore(db_path=db_path, embedding_provider=HashEmbeddingProvider())
    store.upsert_clauses(clauses)
    assert store.last_upsert_stats.total_clauses == len(clauses)
    assert store.last_upsert_stats.embedded_count == len(clauses)
    assert store.last_upsert_stats.cached_count == 0

    warm_store = PersistentLocalVectorStore(db_path=db_path, embedding_provider=HashEmbeddingProvider())
    warm_store.upsert_clauses(clauses)
    assert warm_store.last_upsert_stats.total_clauses == len(clauses)
    assert warm_store.last_upsert_stats.embedded_count == 0
    assert warm_store.last_upsert_stats.cached_count == len(clauses)

    hits = warm_store.search(
        query="coverage and far residential",
        payload_filter={"state": "kerala", "jurisdiction_type": "municipality"},
        limit=5,
    )
    assert hits


def test_persistent_vector_store_reembeds_only_changed_clauses(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")
    clauses = [clause for doc in docs for clause in doc.clause_nodes]
    assert clauses

    db_path = tmp_path / "vector_index.db"
    store = PersistentLocalVectorStore(db_path=db_path, embedding_provider=HashEmbeddingProvider())
    store.upsert_clauses(clauses)

    clauses[0].normalized_text = f"{clauses[0].normalized_text} updated token"
    store.upsert_clauses(clauses)
    assert store.last_upsert_stats.total_clauses == len(clauses)
    assert store.last_upsert_stats.embedded_count == 1
    assert store.last_upsert_stats.cached_count == len(clauses) - 1
