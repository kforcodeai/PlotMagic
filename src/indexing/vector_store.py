from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol

from src.indexing.embeddings import EmbeddingProvider
from src.models import ClauseNode


@dataclass(slots=True)
class VectorHit:
    clause_id: str
    score: float
    payload: dict[str, Any]
    text: str


@dataclass(slots=True)
class VectorUpsertStats:
    total_clauses: int = 0
    cached_count: int = 0
    embedded_count: int = 0
    deleted_count: int = 0
    upsert_count: int = 0
    embedding_calls_made: int = 0


class VectorStore(Protocol):
    last_upsert_stats: VectorUpsertStats

    def upsert_clauses(self, clauses: list[ClauseNode]) -> None:
        raise NotImplementedError

    def search(
        self,
        query: str,
        payload_filter: dict[str, Any],
        limit: int = 20,
        allowed_clause_ids: set[str] | None = None,
    ) -> list[VectorHit]:
        raise NotImplementedError

    def point_count(self) -> int:
        raise NotImplementedError


def clause_payload(clause: ClauseNode) -> dict[str, Any]:
    return {
        "clause_id": clause.clause_id,
        "state": clause.state,
        "jurisdiction_type": clause.jurisdiction_type,
        "ruleset_id": clause.ruleset_id,
        "ruleset_version": clause.ruleset_version,
        "chapter_number": clause.chapter_number,
        "chapter_title": clause.chapter_title,
        "rule_number": clause.rule_number,
        "rule_title": clause.rule_title,
        "display_citation": clause.display_citation,
        "sub_rule_path": clause.sub_rule_path,
        "occupancy_groups": clause.occupancy_groups,
        "is_generic": clause.is_generic,
        "topic_tags": clause.topic_tags,
        "clause_type": clause.clause_type.value,
        "anchor_id": clause.anchor_id,
        "panchayat_category": clause.panchayat_category,
        "source_file": clause.source_file,
    }


def clause_text(clause: ClauseNode) -> str:
    context = [
        f"[{clause.ruleset_id}] Chapter {clause.chapter_number}: {clause.chapter_title}",
        f"Rule {clause.rule_number}: {clause.rule_title}",
        f"Citation: {clause.display_citation}",
    ]
    body = clause.normalized_text or clause.raw_text
    return "\n".join([item for item in [*context, body] if item]).strip()


def matches_filter(payload: dict[str, Any], payload_filter: dict[str, Any]) -> bool:
    category_policy = os.getenv("PLOTMAGIC_CATEGORY_FILTER_POLICY", "soft").strip().lower() or "soft"
    for key, expected in payload_filter.items():
        if expected is None:
            continue
        value = payload.get(key)
        if key == "panchayat_category":
            if category_policy == "strict":
                if value in {None, ""}:
                    return False
                if str(value).lower() == "both":
                    continue
                if str(value).lower() != str(expected).lower():
                    return False
                continue
            # Soft category matching: missing/universal clause category is eligible.
            if value in {None, "", "both"}:
                continue
            if str(value).lower() != str(expected).lower():
                return False
            continue
        if isinstance(expected, list):
            if key == "occupancy_groups":
                is_generic = payload.get("is_generic", False)
                if is_generic:
                    continue
                if not set(expected).intersection(set(value or [])):
                    return False
            else:
                if value not in expected:
                    return False
        else:
            if value != expected:
                return False
    return True


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.vectors: dict[str, list[float]] = {}
        self.payloads: dict[str, dict[str, Any]] = {}
        self.texts: dict[str, str] = {}
        self.last_upsert_stats = VectorUpsertStats()

    def upsert_clauses(self, clauses: list[ClauseNode]) -> None:
        clause_items: list[tuple[str, str, dict[str, Any]]] = []
        for clause in clauses:
            clause_items.append((clause.clause_id, clause_text(clause), clause_payload(clause)))

        vectors = self.embedding_provider.embed_document_batch([item[1] for item in clause_items])
        for (clause_id, text, payload), vector in zip(clause_items, vectors):
            self.vectors[clause_id] = vector
            self.texts[clause_id] = text
            self.payloads[clause_id] = payload
        self.last_upsert_stats = VectorUpsertStats(
            total_clauses=len(clause_items),
            cached_count=0,
            embedded_count=len(clause_items),
            deleted_count=0,
            upsert_count=len(clause_items),
            embedding_calls_made=len(clause_items),
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
            payload = self.payloads[clause_id]
            if not self._matches_filter(payload, payload_filter):
                continue
            score = self._cosine(query_vec, vector)
            if score <= 0:
                continue
            candidates.append(VectorHit(clause_id=clause_id, score=score, payload=payload, text=self.texts[clause_id]))
        return sorted(candidates, key=lambda hit: hit.score, reverse=True)[:limit]

    def point_count(self) -> int:
        return len(self.vectors)

    def _matches_filter(self, payload: dict[str, Any], payload_filter: dict[str, Any]) -> bool:
        return matches_filter(payload, payload_filter)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        return cosine_similarity(a, b)
