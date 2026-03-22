from __future__ import annotations

from pathlib import Path

from src.indexing import HashEmbeddingProvider, InMemoryVectorStore, LexicalIndex, StructuredStore
from src.ingestion.pipeline import IngestionPipeline


def test_indexing_triad_builds_and_queries(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    docs, _stats = IngestionPipeline(root / "config" / "states.yaml").ingest_state("kerala", "municipality")
    assert docs

    # Structured index
    structured = StructuredStore(tmp_path / "structured.db")
    structured.upsert_documents(docs)
    rows = structured.search_rules(state="kerala", jurisdiction_type="municipality", topic_like="occupancy", limit=5)
    assert rows

    # Lexical index
    lexical = LexicalIndex()
    lexical.build(docs)
    lexical_hits = lexical.search("coverage floor area ratio table 2", limit=5)
    assert lexical_hits

    # Vector index
    vector = InMemoryVectorStore(HashEmbeddingProvider())
    vector.upsert_clauses([clause for doc in docs for clause in doc.clause_nodes])
    vector_hits = vector.search(
        "coverage and far for residential",
        payload_filter={"state": "kerala", "jurisdiction_type": "municipality"},
        limit=5,
    )
    assert vector_hits

