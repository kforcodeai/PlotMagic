from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import uuid

from src.indexing.embeddings import EmbeddingProvider
from src.models import ClauseNode

from .vector_store import (
    VectorHit,
    VectorUpsertStats,
    clause_payload,
    clause_text,
    matches_filter,
)

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]
    qmodels = None  # type: ignore[assignment]


_UPSERT_BATCH_SIZE = 128
_EMBED_BATCH_SIZE = 64
_RETRIEVE_BATCH_SIZE = 256
_SCROLL_BATCH_SIZE = 512


@dataclass(slots=True)
class _ClauseRecord:
    clause_id: str
    text: str
    payload: dict[str, Any]
    text_hash: str


class QdrantLocalVectorStore:
    """
    Disk-backed Qdrant local store with incremental embedding reuse.
    """

    def __init__(
        self,
        *,
        db_path: Path,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "plotmagic_clauses",
        recreate_collection: bool = False,
    ) -> None:
        if QdrantClient is None or qmodels is None:
            raise RuntimeError("qdrant-client is required for qdrant_local backend")

        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self.embedding_model = self._embedding_model_key(embedding_provider)
        self.vector_dim: int | None = self._provider_dim(embedding_provider)
        self.last_upsert_stats = VectorUpsertStats()

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.db_path))

        if recreate_collection and self.collection_exists():
            self.client.delete_collection(self.collection_name)

        if self.collection_exists():
            existing_dim = self._collection_vector_dim()
            if self.vector_dim is None:
                self.vector_dim = existing_dim
            elif existing_dim != self.vector_dim:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' has vector size {existing_dim}, "
                    f"expected {self.vector_dim} for provider '{self.embedding_model}'."
                )
        elif self.vector_dim is not None:
            self._ensure_collection(self.vector_dim)

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

        embedding_calls = 0
        points_to_upsert: list[Any] = []
        for batch in self._batched(to_embed, _EMBED_BATCH_SIZE):
            vectors = self.embedding_provider.embed_document_batch([item.text for item in batch])
            embedding_calls += len(batch)
            for item, vector in zip(batch, vectors):
                if self.vector_dim is None:
                    self.vector_dim = len(vector)
                    self._ensure_collection(self.vector_dim)
                if len(vector) != self.vector_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for {item.clause_id}: got {len(vector)}, expected {self.vector_dim}"
                    )
                points_to_upsert.append(
                    qmodels.PointStruct(
                        id=self._point_id(item.clause_id),
                        vector=vector,
                        payload=self._point_payload(item),
                    )
                )

        for batch in self._batched(points_to_upsert, _UPSERT_BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        for item in cached:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=self._point_payload(item),
                points=[self._point_id(item.clause_id)],
                wait=True,
            )

        deleted_count = 0
        if self.collection_exists():
            existing_ids = set(self._all_clause_ids())
            stale_ids = sorted(existing_ids.difference(set(clause_ids)))
            if stale_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qmodels.PointIdsList(points=[self._point_id(clause_id) for clause_id in stale_ids]),
                    wait=True,
                )
                deleted_count = len(stale_ids)

        self.last_upsert_stats = VectorUpsertStats(
            total_clauses=len(clause_ids),
            cached_count=len(cached),
            embedded_count=len(to_embed),
            deleted_count=deleted_count,
            upsert_count=len(points_to_upsert) + len(cached),
            embedding_calls_made=embedding_calls,
        )

    def search(
        self,
        query: str,
        payload_filter: dict[str, Any],
        limit: int = 20,
        allowed_clause_ids: set[str] | None = None,
    ) -> list[VectorHit]:
        query = query.strip()
        if not query or not self.collection_exists():
            return []

        query_vec = self.embedding_provider.embed_query(query)
        if self.vector_dim is not None and len(query_vec) != self.vector_dim:
            return []

        query_filter = self._qdrant_filter(payload_filter)
        search_limit = max(limit * 8, 100) if allowed_clause_ids is not None else limit
        if hasattr(self.client, "search"):
            rows = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                query_filter=query_filter,
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
            )
        else:
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                query_filter=query_filter,
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
            )
            rows = list(getattr(query_response, "points", []))

        hits: list[VectorHit] = []
        for row in rows:
            payload = dict(row.payload or {})
            clause_id = str(payload.get("clause_id") or row.id)
            if allowed_clause_ids is not None and clause_id not in allowed_clause_ids:
                continue
            if not matches_filter(payload, payload_filter):
                continue
            score = float(row.score or 0.0)
            if score <= 0.0:
                continue
            hits.append(
                VectorHit(
                    clause_id=clause_id,
                    score=score,
                    payload=payload,
                    text=str(payload.get("_text", "")),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def point_count(self) -> int:
        if not self.collection_exists():
            return 0
        info = self.client.get_collection(self.collection_name)
        return int(getattr(info, "points_count", 0) or 0)

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(collection_name=self.collection_name))

    def _ensure_collection(self, vector_dim: int) -> None:
        if self.collection_exists():
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=vector_dim, distance=qmodels.Distance.COSINE),
        )

    def _existing_hashes(self, clause_ids: list[str]) -> dict[str, str]:
        if not clause_ids or not self.collection_exists():
            return {}
        existing: dict[str, str] = {}
        for batch in self._batched(clause_ids, _RETRIEVE_BATCH_SIZE):
            rows = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._point_id(clause_id) for clause_id in batch],
                with_payload=True,
                with_vectors=False,
            )
            for row in rows:
                payload = dict(row.payload or {})
                text_hash = payload.get("_text_hash")
                if isinstance(text_hash, str):
                    clause_id = str(payload.get("clause_id") or row.id)
                    existing[clause_id] = text_hash
        return existing

    def _all_clause_ids(self) -> list[str]:
        if not self.collection_exists():
            return []
        ids: list[str] = []
        offset = None
        while True:
            rows, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=_SCROLL_BATCH_SIZE,
                offset=offset,
            )
            for row in rows:
                payload = dict(row.payload or {})
                ids.append(str(payload.get("clause_id") or row.id))
            if offset is None:
                break
        return ids

    def _collection_vector_dim(self) -> int:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, qmodels.VectorParams):
            return int(vectors.size)
        if isinstance(vectors, dict):
            for value in vectors.values():
                return int(value.size)
        raise ValueError(f"Unable to resolve vector size for Qdrant collection '{self.collection_name}'")

    def _point_payload(self, item: _ClauseRecord) -> dict[str, Any]:
        return {
            **item.payload,
            "_text": item.text,
            "_text_hash": item.text_hash,
            "_embedding_model": self.embedding_model,
        }

    def _qdrant_filter(self, payload_filter: dict[str, Any]) -> Any | None:
        must: list[Any] = []
        for key, expected in payload_filter.items():
            if expected is None:
                continue
            if key == "occupancy_groups":
                # Occupancy filter is evaluated client-side to preserve generic-clause semantics.
                continue
            if key == "panchayat_category":
                # Category semantics need fallback behavior for generic/unscoped clauses.
                continue
            if isinstance(expected, list):
                if not expected:
                    continue
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=expected)))
            else:
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=expected)))
        if not must:
            return None
        return qmodels.Filter(must=must)

    def _embedding_model_key(self, provider: EmbeddingProvider) -> str:
        provider_id = str(getattr(provider, "provider_id", provider.__class__.__name__))
        settings = getattr(provider, "settings", None)
        model = getattr(provider, "model", None) or getattr(settings, "model", None)
        dim = getattr(provider, "dim", None)

        parts = [provider_id]
        if model:
            parts.append(str(model))
        if dim:
            parts.append(f"dim={dim}")
        return "|".join(parts)

    def _provider_dim(self, provider: EmbeddingProvider) -> int | None:
        dim = getattr(provider, "dim", None)
        if isinstance(dim, int) and dim > 0:
            return dim
        return None

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _point_id(self, clause_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"plotmagic:{clause_id}"))

    def _batched(self, values: list[Any], batch_size: int) -> list[list[Any]]:
        return [values[idx : idx + batch_size] for idx in range(0, len(values), batch_size)]
