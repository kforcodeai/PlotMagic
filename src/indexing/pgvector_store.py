from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from src.indexing.embeddings import EmbeddingProvider
from src.models import ClauseNode

from .vector_store import (
    VectorHit,
    VectorUpsertStats,
    clause_payload,
    clause_text,
    matches_filter,
)


@dataclass(slots=True)
class _ClauseRecord:
    clause_id: str
    text: str
    payload: dict[str, Any]
    text_hash: str


class PgVectorStore:
    """
    PostgreSQL + pgvector backend.

    This backend is intended for production deployments where retrieval quality,
    traceability, and index persistence are required.
    """

    def __init__(
        self,
        *,
        dsn: str,
        embedding_provider: EmbeddingProvider,
        table_name: str = "clause_vectors",
    ) -> None:
        self.dsn = dsn
        self.embedding_provider = embedding_provider
        self.embedding_model = self._embedding_model_key(embedding_provider)
        self.table_name = table_name
        self.last_upsert_stats = VectorUpsertStats()

        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("psycopg is required for pgvector backend") from exc

        self._psycopg = psycopg
        self.conn = psycopg.connect(self.dsn)
        self.conn.autocommit = False

        self._init_schema()

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
        if to_embed:
            vectors = self.embedding_provider.embed_document_batch([item.text for item in to_embed])
            for item, vector in zip(to_embed, vectors):
                embedded_vectors[item.clause_id] = vector

        with self.conn.cursor() as cur:
            for item in to_embed:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (
                        clause_id, embedding_model, text_hash, vector, payload_json, text
                    ) VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s)
                    ON CONFLICT (clause_id, embedding_model)
                    DO UPDATE SET
                        text_hash = EXCLUDED.text_hash,
                        vector = EXCLUDED.vector,
                        payload_json = EXCLUDED.payload_json,
                        text = EXCLUDED.text,
                        updated_at = NOW()
                    """,
                    (
                        item.clause_id,
                        self.embedding_model,
                        item.text_hash,
                        self._vector_literal(embedded_vectors[item.clause_id]),
                        json.dumps(item.payload),
                        item.text,
                    ),
                )
            for item in cached:
                cur.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET payload_json = %s::jsonb,
                        text = %s,
                        updated_at = NOW()
                    WHERE clause_id = %s AND embedding_model = %s
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
        search_limit = max(limit * 8, 200)

        clause_ids_arg: list[str] | None = sorted(allowed_clause_ids) if allowed_clause_ids is not None else None
        with self.conn.cursor() as cur:
            if clause_ids_arg is None:
                cur.execute(
                    f"""
                    SELECT clause_id, payload_json, text,
                           1 - (vector <=> %s::vector) AS score
                    FROM {self.table_name}
                    WHERE embedding_model = %s
                    ORDER BY vector <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        self._vector_literal(query_vec),
                        self.embedding_model,
                        self._vector_literal(query_vec),
                        search_limit,
                    ),
                )
            else:
                cur.execute(
                    f"""
                    SELECT clause_id, payload_json, text,
                           1 - (vector <=> %s::vector) AS score
                    FROM {self.table_name}
                    WHERE embedding_model = %s
                      AND clause_id = ANY(%s)
                    ORDER BY vector <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        self._vector_literal(query_vec),
                        self.embedding_model,
                        clause_ids_arg,
                        self._vector_literal(query_vec),
                        search_limit,
                    ),
                )
            rows = cur.fetchall()

        hits: list[VectorHit] = []
        for clause_id, payload_raw, text, score in rows:
            payload = payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
            if not matches_filter(payload, payload_filter):
                continue
            score_f = float(score or 0.0)
            if score_f <= 0:
                continue
            hits.append(
                VectorHit(
                    clause_id=str(clause_id),
                    score=score_f,
                    payload=payload,
                    text=str(text or ""),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def point_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE embedding_model = %s",
                (self.embedding_model,),
            )
            value = cur.fetchone()
        return int(value[0] if value else 0)

    def _init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    clause_id TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector VECTOR,
                    payload_json JSONB NOT NULL,
                    text TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (clause_id, embedding_model)
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_model
                ON {self.table_name}(embedding_model)
                """
            )
        self.conn.commit()

    def _existing_hashes(self, clause_ids: list[str]) -> dict[str, str]:
        if not clause_ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT clause_id, text_hash
                FROM {self.table_name}
                WHERE embedding_model = %s
                  AND clause_id = ANY(%s)
                """,
                (self.embedding_model, clause_ids),
            )
            rows = cur.fetchall()
        return {str(clause_id): str(text_hash) for clause_id, text_hash in rows}

    def _prune_stale_rows(self, cur: Any, *, active_clause_ids: list[str]) -> int:
        if not active_clause_ids:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE embedding_model = %s",
                (self.embedding_model,),
            )
            return int(cur.rowcount or 0)

        cur.execute(
            f"""
            DELETE FROM {self.table_name}
            WHERE embedding_model = %s
              AND NOT (clause_id = ANY(%s))
            """,
            (self.embedding_model, active_clause_ids),
        )
        return int(cur.rowcount or 0)

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

    def _vector_literal(self, vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
