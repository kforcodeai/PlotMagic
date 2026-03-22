from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from src.indexing.embeddings import EmbeddingProvider
from src.models import ClauseNode

from .vector_store import (
    VectorHit,
    VectorUpsertStats,
    clause_payload,
    clause_text,
    cosine_similarity,
    matches_filter,
)

_SQLITE_IN_CLAUSE_BATCH_SIZE = 500
_EMBED_BATCH_SIZE = 64


@dataclass(slots=True)
class _ClauseRecord:
    clause_id: str
    text: str
    payload: dict[str, Any]
    text_hash: str


class PersistentLocalVectorStore:
    """
    Local persistent vector backend with incremental embedding cache.
    Cache key uses: clause_id + embedding_model + text_hash.
    """

    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider) -> None:
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.embedding_model = self._embedding_model_key(embedding_provider)
        self.last_upsert_stats = VectorUpsertStats()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        self.vectors: dict[str, list[float]] = {}
        self.payloads: dict[str, dict[str, Any]] = {}
        self.texts: dict[str, str] = {}

        self._init_schema()
        self._load_vectors_from_disk()

    def upsert_clauses(self, clauses: list[ClauseNode]) -> None:
        records: dict[str, _ClauseRecord] = {}
        for clause in clauses:
            text = clause_text(clause).strip()
            if not text:
                continue
            payload = clause_payload(clause)
            records[clause.clause_id] = _ClauseRecord(
                clause_id=clause.clause_id,
                text=text,
                payload=payload,
                text_hash=self._text_hash(text),
            )

        clause_ids = list(records.keys())
        existing_hashes = self._existing_hashes(clause_ids)

        to_embed: list[_ClauseRecord] = []
        cached: list[_ClauseRecord] = []
        for clause_id in clause_ids:
            record = records[clause_id]
            if existing_hashes.get(clause_id) == record.text_hash:
                cached.append(record)
            else:
                to_embed.append(record)

        embedded_vectors: dict[str, list[float]] = {}
        for batch in self._batched(to_embed, _EMBED_BATCH_SIZE):
            vectors = self.embedding_provider.embed_document_batch([item.text for item in batch])
            for item, vector in zip(batch, vectors):
                embedded_vectors[item.clause_id] = vector

        with self._lock:
            cur = self.conn.cursor()
            for item in to_embed:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO clause_vectors (
                        clause_id, embedding_model, text_hash, vector_json, payload_json, text
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.clause_id,
                        self.embedding_model,
                        item.text_hash,
                        json.dumps(embedded_vectors[item.clause_id]),
                        json.dumps(item.payload),
                        item.text,
                    ),
                )
            for item in cached:
                cur.execute(
                    """
                    UPDATE clause_vectors
                    SET payload_json = ?, text = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE clause_id = ? AND embedding_model = ?
                    """,
                    (
                        json.dumps(item.payload),
                        item.text,
                        item.clause_id,
                        self.embedding_model,
                    ),
                )
            deleted_count = self._prune_stale_rows(cur, active_clause_ids=clause_ids)
            self.conn.commit()

        self._load_vectors_from_disk()
        self.last_upsert_stats = VectorUpsertStats(
            total_clauses=len(clause_ids),
            cached_count=len(cached),
            embedded_count=len(to_embed),
            deleted_count=max(0, deleted_count),
            upsert_count=len(to_embed) + len(cached),
            embedding_calls_made=len(to_embed),
        )

    def search(
        self,
        query: str,
        payload_filter: dict[str, Any],
        limit: int = 20,
        allowed_clause_ids: set[str] | None = None,
    ) -> list[VectorHit]:
        query = query.strip()
        if not query:
            return []
        query_vec = self.embedding_provider.embed_query(query)
        candidates: list[VectorHit] = []
        for clause_id, vector in self.vectors.items():
            if allowed_clause_ids is not None and clause_id not in allowed_clause_ids:
                continue
            payload = self.payloads.get(clause_id, {})
            if not matches_filter(payload, payload_filter):
                continue
            score = cosine_similarity(query_vec, vector)
            if score <= 0:
                continue
            candidates.append(
                VectorHit(
                    clause_id=clause_id,
                    score=score,
                    payload=payload,
                    text=self.texts.get(clause_id, ""),
                )
            )
        return sorted(candidates, key=lambda hit: hit.score, reverse=True)[:limit]

    def point_count(self) -> int:
        return len(self.vectors)

    def _init_schema(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS clause_vectors (
                    clause_id TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (clause_id, embedding_model)
                );

                CREATE INDEX IF NOT EXISTS idx_clause_vectors_model
                ON clause_vectors(embedding_model);
                """
            )
            self.conn.commit()

    def _load_vectors_from_disk(self) -> None:
        vectors: dict[str, list[float]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        texts: dict[str, str] = {}

        with self._lock:
            rows = self.conn.execute(
                """
                SELECT clause_id, vector_json, payload_json, text
                FROM clause_vectors
                WHERE embedding_model = ?
                """,
                (self.embedding_model,),
            ).fetchall()

        for row in rows:
            clause_id = str(row["clause_id"])
            try:
                vectors[clause_id] = [float(value) for value in json.loads(row["vector_json"])]
                payloads[clause_id] = dict(json.loads(row["payload_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            texts[clause_id] = str(row["text"] or "")

        self.vectors = vectors
        self.payloads = payloads
        self.texts = texts

    def _existing_hashes(self, clause_ids: list[str]) -> dict[str, str]:
        if not clause_ids:
            return {}
        existing: dict[str, str] = {}
        with self._lock:
            for batch in self._batched(clause_ids, _SQLITE_IN_CLAUSE_BATCH_SIZE):
                placeholders = ",".join(["?"] * len(batch))
                rows = self.conn.execute(
                    f"""
                    SELECT clause_id, text_hash
                    FROM clause_vectors
                    WHERE embedding_model = ? AND clause_id IN ({placeholders})
                    """,
                    [self.embedding_model, *batch],
                ).fetchall()
                for row in rows:
                    existing[str(row["clause_id"])] = str(row["text_hash"])
        return existing

    def _prune_stale_rows(self, cur: sqlite3.Cursor, *, active_clause_ids: list[str]) -> int:
        if not active_clause_ids:
            cur.execute("DELETE FROM clause_vectors WHERE embedding_model = ?", (self.embedding_model,))
            return int(cur.rowcount or 0)

        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _active_clause_ids (clause_id TEXT PRIMARY KEY)")
        cur.execute("DELETE FROM _active_clause_ids")
        cur.executemany(
            "INSERT OR IGNORE INTO _active_clause_ids(clause_id) VALUES (?)",
            [(clause_id,) for clause_id in active_clause_ids],
        )
        cur.execute(
            """
            DELETE FROM clause_vectors
            WHERE embedding_model = ?
              AND clause_id NOT IN (SELECT clause_id FROM _active_clause_ids)
            """,
            (self.embedding_model,),
        )
        deleted = int(cur.rowcount or 0)
        cur.execute("DELETE FROM _active_clause_ids")
        return deleted

    def _embedding_model_key(self, provider: EmbeddingProvider) -> str:
        provider_id = str(getattr(provider, "provider_id", provider.__class__.__name__))
        settings = getattr(provider, "settings", None)
        model = getattr(provider, "model", None) or getattr(settings, "model", None)
        dim = getattr(provider, "dim", None) or getattr(settings, "dim", None)

        parts = [provider_id]
        if model:
            parts.append(str(model))
        if dim:
            parts.append(f"dim={dim}")
        return "|".join(parts)

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _batched(self, values: list[Any], batch_size: int) -> list[list[Any]]:
        return [values[idx : idx + batch_size] for idx in range(0, len(values), batch_size)]
