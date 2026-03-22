from __future__ import annotations

from dataclasses import dataclass, field
import os
from time import perf_counter
import re
from typing import Any

from src.indexing.lexical_index import LexicalHit, LexicalIndex
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import VectorHit, VectorStore
from src.models import ClauseType, QueryFact, RuleDocument
from src.policy import RetrievalPolicy
from src.providers import ProviderError, RerankCandidate, RerankerProvider
from src.retrieval.cross_ref_resolver import CrossReferenceResolver
from src.retrieval.evidence import EvidenceMatrix, RetrievalEvidence
from src.retrieval.query_planner import QueryPlan
_GENERIC_TOPIC_HINTS: dict[str, set[str]] = {
    "permit": {"permit", "license", "approval", "application"},
    "exemption": {"exempt", "exception", "waiver", "not required"},
    "penalty": {"penalty", "fine", "offence", "offense"},
    "appeal": {"appeal", "review", "revision"},
    "timeline": {"within", "days", "months", "years", "deadline"},
}
_NUMERIC_QUERY_HINTS = {
    "value",
    "values",
    "minimum",
    "maximum",
    "distance",
    "clearance",
    "metre",
    "meter",
    "days",
    "years",
    "fee",
    "fees",
    "percent",
}

_TOPIC_FAMILY_EXCLUSIONS: dict[str, set[str]] = {
    "street_road": {"parking"},
    "height": {"parking"},
    "open_space": {"parking"},
}


@dataclass(slots=True)
class HybridRetrievalResult:
    evidence_matrix: EvidenceMatrix
    retrieved_documents: list[RuleDocument]
    latency_ms: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        structured_store: StructuredStore,
        all_docs: list[RuleDocument],
        reranker_provider: RerankerProvider | None = None,
        rerank_top_n: int = 80,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self.policy = policy or RetrievalPolicy()
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.structured_store = structured_store
        self.reranker_provider = reranker_provider
        self.rerank_top_n = rerank_top_n
        self.doc_by_id = {doc.document_id: doc for doc in all_docs}
        self.doc_to_clause_ids = {doc.document_id: [clause.clause_id for clause in doc.clause_nodes] for doc in all_docs}
        self.rule_to_doc_ids: dict[str, set[str]] = {}
        for doc in all_docs:
            raw_rule = str(doc.rule_number or "").strip()
            m = re.search(r"\d+", raw_rule)
            if not m:
                continue
            rule = m.group(0)
            self.rule_to_doc_ids.setdefault(rule, set()).add(doc.document_id)
        self.clause_to_doc_id = {
            clause.clause_id: doc.document_id for doc in all_docs for clause in doc.clause_nodes
        }
        self.clause_text = {clause.clause_id: clause.normalized_text for doc in all_docs for clause in doc.clause_nodes}
        self.cross_ref_resolver = CrossReferenceResolver(all_docs)
        self.fallback_policy = (
            os.getenv("PLOTMAGIC_RETRIEVAL_FALLBACK_POLICY", "relax_category_then_occupancy").strip().lower()
            or "relax_category_then_occupancy"
        )

    def retrieve(
        self,
        query: str,
        fact: QueryFact,
        plan: QueryPlan,
        candidate_docs: list[RuleDocument],
        top_k: int = 15,
        retrieval_mode: str = "hybrid",
    ) -> HybridRetrievalResult:
        start_all = perf_counter()
        candidate_ids = {doc.document_id for doc in candidate_docs}
        mode = retrieval_mode.strip().lower()
        use_vector_only = mode == "vector_only"
        use_lexical_only = mode == "lexical_only_bm25"
        use_graph_expansion = mode in {"hybrid_graph_reranker", "hybrid_graph_no_reranker", "agentic_dynamic"}
        use_vector = not use_lexical_only
        use_lexical = not use_vector_only
        use_structured = not (use_vector_only or use_lexical_only)
        disable_reranker = mode in {
            "vector_only",
            "hybrid_no_reranker",
            "hybrid_graph_no_reranker",
            "lexical_only_bm25",
        }
        if not candidate_ids:
            return HybridRetrievalResult(
                evidence_matrix=EvidenceMatrix(),
                retrieved_documents=[],
                latency_ms={"total_retrieval_ms": (perf_counter() - start_all) * 1000},
                diagnostics={
                    "retrieval_mode": mode,
                    "fallback_stage": "insufficient_evidence",
                    "vector_hits": 0,
                    "lexical_hits": 0,
                    "structured_hits": 0,
                    "candidate_count": 0,
                    "top_rules_cited": [],
                },
            )
        latency_ms: dict[str, float] = {}
        latency_ms["reranker_used"] = 1.0 if (self.reranker_provider and not disable_reranker) else 0.0

        pool_factor = max(1.0, float(getattr(self.policy, "candidate_pool_factor", 4.0)))
        if self.reranker_provider and not disable_reranker:
            pool_k = max(int(top_k * pool_factor), self.rerank_top_n)
        else:
            # Decouple retrieval candidate pool from reranker budget when reranker is disabled.
            pool_k = max(int(top_k * pool_factor), top_k + 8)
        lexical_hits: list[LexicalHit] = []
        lexical_queries: list[tuple[str, float]] = []
        if use_lexical:
            start = perf_counter()
            lexical_queries = self._expand_lexical_queries(query=query, plan=plan)
            lexical_hits = self._search_lexical_merged(lexical_queries, candidate_ids=candidate_ids, pool_k=pool_k)
            latency_ms["lexical_query_variants"] = float(len(lexical_queries))
            latency_ms["lexical_search_ms"] = (perf_counter() - start) * 1000
        else:
            latency_ms["lexical_query_variants"] = 0.0
            latency_ms["lexical_search_ms"] = 0.0

        start = perf_counter()
        allowed_clause_ids = {
            clause_id for doc_id in candidate_ids for clause_id in self.doc_to_clause_ids.get(doc_id, [])
        }
        if use_vector:
            vector_hits, fallback_stage = self._vector_search_with_fallback(
                query=query,
                fact=fact,
                limit=pool_k,
                allowed_clause_ids=allowed_clause_ids,
            )
            vector_hits = [hit for hit in vector_hits if self.clause_to_doc_id.get(hit.clause_id) in candidate_ids]
        else:
            vector_hits = []
            fallback_stage = "vector_disabled"
        latency_ms["vector_search_ms"] = (perf_counter() - start) * 1000

        start = perf_counter()
        structured_doc_ids: list[str] = []
        topic_likes: list[str] = []
        if use_structured:
            topic_likes = [topic for topic in plan.topics if topic != "general"][: self.policy.max_topic_likes]
            if not topic_likes:
                topic_likes = self._fallback_topic_likes(query)
            can_use_structured = bool(fact.state and fact.jurisdiction_type)
            if can_use_structured and self.policy.structured_rrf_weight > 0.0 and topic_likes:
                per_topic_limit = max(self.policy.per_topic_structured_limit_floor, pool_k // max(1, len(topic_likes)))
                seen_structured: set[str] = set()
                for topic_like in topic_likes:
                    structured_rows = self.structured_store.search_rules(
                        state=fact.state or "",
                        jurisdiction_type=fact.jurisdiction_type or "",
                        ruleset_id=None,
                        occupancy=fact.occupancies,
                        topic_like=topic_like,
                        limit=per_topic_limit,
                    )
                    for row in structured_rows:
                        doc_id = row["document_id"]
                        if doc_id not in candidate_ids or doc_id in seen_structured:
                            continue
                        structured_doc_ids.append(doc_id)
                        seen_structured.add(doc_id)
                latency_ms["structured_topic_queries"] = float(len(topic_likes))
        else:
            latency_ms["structured_topic_queries"] = 0.0
        latency_ms["structured_search_ms"] = (perf_counter() - start) * 1000
        latency_ms["vector_hits"] = float(len(vector_hits))
        latency_ms["lexical_hits"] = float(len(lexical_hits))
        latency_ms["structured_hits"] = float(len(structured_doc_ids))

        start = perf_counter()
        ranked = self._rrf_merge(
            vector_hits,
            lexical_hits,
            structured_doc_ids,
            top_k=pool_k,
            vector_weight=0.0 if use_lexical_only else self.policy.vector_rrf_weight,
            lexical_weight=0.0 if use_vector_only else self.policy.lexical_rrf_weight,
            structured_weight=self.policy.structured_rrf_weight if use_structured else 0.0,
            k=self.policy.rrf_k,
        )
        ranked_score_map = {doc_id: score for doc_id, score in ranked}
        merged_docs = [self.doc_by_id[item[0]] for item in ranked if item[0] in self.doc_by_id]
        merged_doc_ids = {doc.document_id for doc in merged_docs}

        seed_count = min(len(merged_docs), max(4, top_k))
        expanded = self.cross_ref_resolver.expand(merged_docs[:seed_count], depth=1)
        expanded_scored = 0
        for ref in expanded:
            if ref.target_document_id not in self.doc_by_id:
                continue
            source_score = ranked_score_map.get(ref.source_document_id, 0.0)
            inferred_score = source_score * (self.policy.cross_ref_score_decay**max(ref.depth, 1))
            if inferred_score > ranked_score_map.get(ref.target_document_id, 0.0):
                ranked_score_map[ref.target_document_id] = inferred_score
                expanded_scored += 1
            if ref.target_document_id not in merged_doc_ids:
                merged_docs.append(self.doc_by_id[ref.target_document_id])
                merged_doc_ids.add(ref.target_document_id)
        ranked_with_expansion = sorted(ranked_score_map.items(), key=lambda item: item[1], reverse=True)[
            : pool_k + max(8, len(expanded))
        ]
        latency_ms["cross_ref_seed_docs"] = float(seed_count)
        latency_ms["cross_ref_edges"] = float(len(expanded))
        latency_ms["cross_ref_augmented_docs"] = float(expanded_scored)
        latency_ms["cross_ref_expansion_ms"] = (perf_counter() - start) * 1000

        reranked_docs, ranked_for_evidence = self._apply_reranking(
            query=query,
            docs=merged_docs,
            ranked=ranked_with_expansion,
            top_k=top_k,
            latency_ms=latency_ms,
            enable_reranker=not disable_reranker,
        )
        graph_diag = {
            "graph_expansion_used": False,
            "graph_ref_rules_count": 0,
            "graph_entity_terms_count": 0,
            "graph_expanded_doc_count": 0,
        }
        if use_graph_expansion:
            start = perf_counter()
            reranked_docs, ranked_for_evidence, graph_diag = self._graph_expand(
                query=query,
                plan=plan,
                docs=reranked_docs,
                ranked=ranked_for_evidence,
                candidate_ids=candidate_ids,
                top_k=top_k,
                enable_reranker=not disable_reranker,
                latency_ms=latency_ms,
            )
            latency_ms["graph_expansion_ms"] = (perf_counter() - start) * 1000
        evidence = self._build_evidence_matrix(query=query, plan=plan, docs=reranked_docs, ranked=ranked_for_evidence)
        latency_ms["total_retrieval_ms"] = (perf_counter() - start_all) * 1000
        diagnostics = {
            "retrieval_mode": mode,
            "fallback_stage": fallback_stage,
            "vector_hits": len(vector_hits),
            "lexical_hits": len(lexical_hits),
            "structured_hits": len(structured_doc_ids),
            "candidate_count": len(candidate_ids),
            "top_rules_cited": self._top_rules_cited(reranked_docs),
            **graph_diag,
        }
        return HybridRetrievalResult(
            evidence_matrix=evidence,
            retrieved_documents=reranked_docs,
            latency_ms=latency_ms,
            diagnostics=diagnostics,
        )

    def _expand_lexical_queries(self, *, query: str, plan: QueryPlan) -> list[tuple[str, float]]:
        queries: list[tuple[str, float]] = [(query, 1.0)]
        seen_variants = {query.strip().lower()}

        for topic in plan.topics[:4]:
            topic_phrase = topic.replace("_", " ").strip()
            if not topic_phrase:
                continue
            variant = f"{topic_phrase} requirements {query}"
            key = variant.lower()
            if key in seen_variants:
                continue
            seen_variants.add(key)
            queries.append((variant, 0.7))

        if plan.mentioned_rule_numbers:
            rules_phrase = " ".join([f"rule {rule}" for rule in plan.mentioned_rule_numbers[:4]])
            variant = f"{query} {rules_phrase}".strip()
            key = variant.lower()
            if key not in seen_variants:
                queries.append((variant, 0.8))
                seen_variants.add(key)

        for sub_query in plan.sub_queries[:4]:
            text = str(sub_query.text).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen_variants:
                continue
            seen_variants.add(key)
            queries.append((text, 0.75))

        return queries

    _GRAPH_ENTITY_PATTERNS = [
        re.compile(r"\bdefence(?:\s+establishment)?\b", flags=re.IGNORECASE),
        re.compile(r"\brailway(?:\s+authority|\s+boundary|\s+track)?\b", flags=re.IGNORECASE),
        re.compile(r"\bsecurity\s+zone\b", flags=re.IGNORECASE),
        re.compile(r"\bdistrict\s+collector\b", flags=re.IGNORECASE),
        re.compile(r"\btechnical\s+expert\s+committee\b", flags=re.IGNORECASE),
        re.compile(r"\bappendix\s+[a-z0-9]+\b", flags=re.IGNORECASE),
        re.compile(r"\btable\s+[a-z0-9.]+\b", flags=re.IGNORECASE),
    ]

    def _graph_expand(
        self,
        *,
        query: str,
        plan: QueryPlan,
        docs: list[RuleDocument],
        ranked: list[tuple[str, float]],
        candidate_ids: set[str],
        top_k: int,
        enable_reranker: bool,
        latency_ms: dict[str, float],
    ) -> tuple[list[RuleDocument], list[tuple[str, float]], dict[str, Any]]:
        if not docs:
            return docs, ranked, {
                "graph_expansion_used": False,
                "graph_ref_rules_count": 0,
                "graph_entity_terms_count": 0,
                "graph_expanded_doc_count": 0,
            }

        ranked_score_map = {doc_id: score for doc_id, score in ranked}
        seed_docs = docs[: min(len(docs), max(4, top_k))]
        ref_rules = self._extract_rule_refs(seed_docs, plan=plan)
        entity_terms = self._extract_graph_entities(query=query, docs=seed_docs)
        expanded_ids: set[str] = set()

        for rule in ref_rules:
            for doc_id in self.rule_to_doc_ids.get(rule, set()):
                if doc_id in candidate_ids:
                    expanded_ids.add(doc_id)

        for term in entity_terms[:6]:
            for hit in self.lexical_index.search(term, limit=max(top_k, 8)):
                if hit.document_id in candidate_ids:
                    expanded_ids.add(hit.document_id)

        existing_ids = {doc.document_id for doc in docs}
        new_ids = [doc_id for doc_id in expanded_ids if doc_id not in existing_ids and doc_id in self.doc_by_id]
        if not new_ids:
            return docs, ranked, {
                "graph_expansion_used": True,
                "graph_ref_rules_count": len(ref_rules),
                "graph_entity_terms_count": len(entity_terms),
                "graph_expanded_doc_count": 0,
            }

        base_decay = 0.90
        best_seed_score = 0.0
        for doc_id in existing_ids:
            best_seed_score = max(best_seed_score, ranked_score_map.get(doc_id, 0.0))
        infer_score = best_seed_score * base_decay if best_seed_score > 0 else 0.01
        expanded_docs = list(docs)
        for idx, doc_id in enumerate(new_ids, start=1):
            expanded_docs.append(self.doc_by_id[doc_id])
            ranked_score_map[doc_id] = max(ranked_score_map.get(doc_id, 0.0), infer_score / float(idx + 1))

        ranked_with_expansion = sorted(ranked_score_map.items(), key=lambda item: item[1], reverse=True)[
            : max(top_k * 3, len(ranked_score_map))
        ]

        reranked_docs, reranked_scores = self._apply_reranking(
            query=query,
            docs=expanded_docs,
            ranked=ranked_with_expansion,
            top_k=top_k,
            latency_ms=latency_ms,
            enable_reranker=enable_reranker,
        )
        return reranked_docs, reranked_scores, {
            "graph_expansion_used": True,
            "graph_ref_rules_count": len(ref_rules),
            "graph_entity_terms_count": len(entity_terms),
            "graph_expanded_doc_count": len(new_ids),
        }

    def _extract_rule_refs(self, docs: list[RuleDocument], *, plan: QueryPlan) -> list[str]:
        refs: set[str] = set()
        for value in plan.mentioned_rule_numbers:
            m = re.search(r"\d+", str(value))
            if m:
                refs.add(m.group(0))
        for doc in docs:
            for cref in doc.cross_references:
                if cref.target_type != "rule":
                    continue
                m = re.search(r"\d+", str(cref.target_ref))
                if m:
                    refs.add(m.group(0))
            text = doc.full_text[:2500]
            for m in re.finditer(r"\brule\s+(\d{1,3}[A-Za-z]?)\b", text, flags=re.IGNORECASE):
                value = re.search(r"\d+", m.group(1))
                if value:
                    refs.add(value.group(0))
        return sorted(refs)

    def _extract_graph_entities(self, *, query: str, docs: list[RuleDocument]) -> list[str]:
        corpus = "\n".join([query] + [doc.rule_title for doc in docs] + [doc.full_text[:1500] for doc in docs])
        found: list[str] = []
        seen: set[str] = set()
        for pattern in self._GRAPH_ENTITY_PATTERNS:
            for match in pattern.finditer(corpus):
                term = re.sub(r"\s+", " ", match.group(0).strip().lower())
                if not term or term in seen:
                    continue
                seen.add(term)
                found.append(term)
        return found

    def _vector_search_with_fallback(
        self,
        *,
        query: str,
        fact: QueryFact,
        limit: int,
        allowed_clause_ids: set[str],
    ) -> tuple[list[VectorHit], str]:
        policy = self.fallback_policy
        if policy not in {"strict_only", "relax_category", "relax_category_then_occupancy"}:
            policy = "relax_category_then_occupancy"
        strict_filter = self._payload_filter(fact)
        hits = self.vector_store.search(
            query,
            payload_filter=strict_filter,
            limit=limit,
            allowed_clause_ids=allowed_clause_ids,
        )
        if hits:
            return hits, "strict"
        if policy == "strict_only":
            return [], "strict_only_no_hit"

        if fact.panchayat_category:
            relaxed_category = dict(strict_filter)
            relaxed_category["panchayat_category"] = None
            hits = self.vector_store.search(
                query,
                payload_filter=relaxed_category,
                limit=limit,
                allowed_clause_ids=allowed_clause_ids,
            )
            if hits:
                return hits, "relax_category"
        else:
            relaxed_category = strict_filter

        if policy == "relax_category":
            return [], "relax_category_no_hit"
        if fact.occupancies:
            relaxed_occupancy = dict(relaxed_category)
            relaxed_occupancy["occupancy_groups"] = None
            hits = self.vector_store.search(
                query,
                payload_filter=relaxed_occupancy,
                limit=limit,
                allowed_clause_ids=allowed_clause_ids,
            )
            if hits:
                return hits, "relax_occupancy"

        return [], "insufficient_evidence"

    def _search_lexical_merged(
        self,
        queries: list[tuple[str, float]],
        *,
        candidate_ids: set[str],
        pool_k: int,
    ) -> list[LexicalHit]:
        if not queries:
            return []

        merged_scores: dict[str, float] = {}
        snippets: dict[str, str] = {}
        for variant, weight in queries:
            hits = self.lexical_index.search(variant, limit=pool_k)
            for hit in hits:
                if hit.document_id not in candidate_ids:
                    continue
                merged_scores[hit.document_id] = merged_scores.get(hit.document_id, 0.0) + (hit.score * weight)
                if hit.document_id not in snippets:
                    snippets[hit.document_id] = hit.snippet

        ranked = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)[:pool_k]
        return [
            LexicalHit(document_id=doc_id, score=score, snippet=snippets.get(doc_id, ""))
            for doc_id, score in ranked
        ]

    def _payload_filter(self, fact: QueryFact) -> dict[str, object]:
        return {
            "state": fact.state,
            "jurisdiction_type": fact.jurisdiction_type,
            "occupancy_groups": fact.occupancies or None,
            "panchayat_category": fact.panchayat_category,
        }

    def _rrf_merge(
        self,
        vector_hits: list[VectorHit],
        lexical_hits: list[LexicalHit],
        structured_doc_ids: list[str],
        top_k: int,
        vector_weight: float = 1.0,
        lexical_weight: float = 1.0,
        structured_weight: float = 1.0,
        k: int = 60,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for rank, hit in enumerate(vector_hits, start=1):
            doc_id = self.clause_to_doc_id.get(hit.clause_id)
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + (vector_weight / (k + rank))
        for rank, hit in enumerate(lexical_hits, start=1):
            scores[hit.document_id] = scores.get(hit.document_id, 0.0) + (lexical_weight / (k + rank))
        for rank, doc_id in enumerate(structured_doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (structured_weight / (k + rank))
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def _build_rerank_candidates(
        self,
        *,
        docs: list[RuleDocument],
        ranked_score_map: dict[str, float],
        query: str,
    ) -> list[RerankCandidate]:
        query_terms = self._tokenize_text(query)
        out: list[RerankCandidate] = []
        for doc in docs:
            excerpt, _ = self._select_clause_excerpt(doc=doc, topic="general", query_terms=query_terms)
            text = excerpt.strip() if excerpt.strip() else doc.full_text[:1200]
            if len(text) > 1600:
                text = text[:1600]
            out.append(
                RerankCandidate(
                    candidate_id=doc.document_id,
                    text=text,
                    base_score=ranked_score_map.get(doc.document_id, 0.0),
                )
            )
        return out

    def _apply_reranking(
        self,
        query: str,
        docs: list[RuleDocument],
        ranked: list[tuple[str, float]],
        top_k: int,
        latency_ms: dict[str, float],
        enable_reranker: bool,
    ) -> tuple[list[RuleDocument], list[tuple[str, float]]]:
        if not docs:
            return [], []

        ranked_score_map = {doc_id: score for doc_id, score in ranked}
        unique_docs: dict[str, RuleDocument] = {doc.document_id: doc for doc in docs}
        base_ranked = sorted(unique_docs.values(), key=lambda doc: ranked_score_map.get(doc.document_id, 0.0), reverse=True)
        evidence_doc_budget = min(len(base_ranked), max(top_k * 2, 14))

        if not self.reranker_provider or not enable_reranker:
            selected = base_ranked[:evidence_doc_budget]
            return selected, [(doc.document_id, ranked_score_map.get(doc.document_id, 0.0)) for doc in selected]

        candidates = self._build_rerank_candidates(
            docs=base_ranked,
            ranked_score_map=ranked_score_map,
            query=query,
        )
        top_n = min(len(candidates), max(evidence_doc_budget, self.rerank_top_n))

        start = perf_counter()
        try:
            reranked = self.reranker_provider.rerank(query=query, candidates=candidates, top_n=top_n)
            latency_ms["reranker_ms"] = (perf_counter() - start) * 1000
            latency_ms["reranker_degraded"] = 0.0
            rerank_scores = {item.candidate_id: item.score for item in reranked}
            rerank_order = sorted(
                [doc for doc in base_ranked if doc.document_id in rerank_scores],
                key=lambda doc: rerank_scores[doc.document_id],
                reverse=True,
            )
            selected = rerank_order[:evidence_doc_budget]
            ranked_for_evidence = [(doc.document_id, rerank_scores[doc.document_id]) for doc in selected]
            return selected, ranked_for_evidence
        except ProviderError:
            latency_ms["reranker_ms"] = (perf_counter() - start) * 1000
            latency_ms["reranker_degraded"] = 1.0
            latency_ms["reranker_failures"] = latency_ms.get("reranker_failures", 0.0) + 1.0
            selected = base_ranked[:evidence_doc_budget]
            return selected, [(doc.document_id, ranked_score_map.get(doc.document_id, 0.0)) for doc in selected]

    def _build_evidence_matrix(
        self,
        query: str,
        plan: QueryPlan,
        docs: list[RuleDocument],
        ranked: list[tuple[str, float]],
    ) -> EvidenceMatrix:
        topic_list = plan.topics or ["general"]
        scores_map = {doc_id: score for doc_id, score in ranked}
        evidence = EvidenceMatrix()
        query_terms = self._tokenize_text(query)
        seen_doc_excerpt_keys: set[tuple[str, str]] = set()
        doc_claim_counts: dict[str, int] = {}
        max_claims_per_doc = max(1, int(getattr(self.policy, "max_claims_per_doc", 2)))
        for topic in topic_list:
            topic_docs: list[tuple[RuleDocument, float, float, str, float]] = []
            fallback_docs: list[tuple[RuleDocument, float, float, str, float]] = []
            for doc in docs:
                score = scores_map.get(doc.document_id, 0.0)
                if score <= 0:
                    continue
                doc_topics = {tag for node in doc.clause_nodes for tag in node.topic_tags}
                if not self._topic_family_guard(topic=topic, doc_topics=doc_topics):
                    continue
                excerpt, excerpt_relevance = self._select_clause_excerpt(doc=doc, topic=topic, query_terms=query_terms)
                if not excerpt:
                    continue
                title_relevance = self._doc_query_relevance(doc=doc, query_terms=query_terms)
                relevance = max(title_relevance, excerpt_relevance)
                if relevance < self.policy.min_query_relevance:
                    continue
                topic_match_boost = 1.0 if (topic == "general" or topic in doc_topics) else 0.0
                row = (doc, score, relevance, excerpt, topic_match_boost)
                if topic == "general" or topic in doc_topics:
                    topic_docs.append(row)
                else:
                    fallback_docs.append(row)

            selected_docs = topic_docs
            if not selected_docs:
                selected_docs = fallback_docs
            selected_docs = sorted(
                selected_docs,
                key=lambda item: (
                    item[1]
                    + (self.policy.query_relevance_weight * item[2])
                    + (self.policy.topic_match_weight * item[4])
                ),
                reverse=True,
            )[: self.policy.default_evidence_docs_per_topic]

            if len(selected_docs) < self.policy.topic_min_docs and fallback_docs:
                missing = self.policy.topic_min_docs - len(selected_docs)
                extra = [row for row in fallback_docs if row not in selected_docs][:missing]
                selected_docs = [*selected_docs, *extra]

            for doc, score, relevance, text, topic_match_boost in selected_docs:
                excerpt_key = self._dedupe_key(text)
                doc_excerpt_key = (doc.document_id, excerpt_key)
                if doc_excerpt_key in seen_doc_excerpt_keys:
                    continue
                if doc_claim_counts.get(doc.document_id, 0) >= max_claims_per_doc and topic != "general":
                    continue
                claim_id = f"{topic}-{doc.document_id}"
                evidence_score = (
                    score
                    + (self.policy.query_relevance_weight * relevance)
                    + (self.policy.topic_match_weight * topic_match_boost)
                )
                evidence.add(
                    RetrievalEvidence(
                        claim_id=claim_id,
                        topic=topic,
                        chunk_id=doc.document_id,
                        document_id=doc.document_id,
                        text=text,
                        scores={
                            "rrf_score": evidence_score,
                            "retrieval_score": score,
                            "query_relevance": relevance,
                        },
                        source="hybrid",
                        has_sufficient_support=evidence_score >= self.policy.min_evidence_score,
                    )
                )
                seen_doc_excerpt_keys.add(doc_excerpt_key)
                doc_claim_counts[doc.document_id] = doc_claim_counts.get(doc.document_id, 0) + 1
        return evidence

    def _select_clause_excerpt(self, *, doc: RuleDocument, topic: str, query_terms: set[str]) -> tuple[str, float]:
        scored_sentences: list[tuple[float, int, str]] = []
        topic_hints = self._topic_hint_terms(topic)
        numeric_query = bool(query_terms.intersection(_NUMERIC_QUERY_HINTS))

        for clause in doc.clause_nodes:
            clause_text = (clause.normalized_text or clause.raw_text or "").strip()
            if len(clause_text) < 40:
                continue
            if clause.clause_type in {ClauseType.SUB_RULE, ClauseType.PROVISO}:
                scored_sentences.append((0.6, len(scored_sentences), clause_text[:500]))
            # For table clauses, include compact table text as a single chunk
            if clause.table_data and clause.table_data.raw_text:
                table_compact = self._compact_table_text(clause.table_data.raw_text)
                if len(table_compact) >= 30:
                    table_terms = self._tokenize_text(table_compact)
                    table_overlap = len(query_terms.intersection(table_terms))
                    table_hint_overlap = len(topic_hints.intersection(table_terms))
                    if table_overlap == 0 and table_hint_overlap == 0:
                        continue
                    table_score = float(table_overlap) + (0.75 * float(table_hint_overlap))
                    if re.search(r"\b\d+(?:\.\d+)?\b", table_compact):
                        table_score += 0.5
                    if numeric_query:
                        table_score += 0.4
                    if topic != "general" and topic in set(clause.topic_tags):
                        table_score += 1.25
                    if table_score > 0:
                        scored_sentences.append((table_score, len(scored_sentences), table_compact[:900]))
            for sentence in self._split_sentences(clause_text):
                sentence_terms = self._tokenize_text(sentence)
                overlap = len(query_terms.intersection(sentence_terms))
                hint_overlap = len(topic_hints.intersection(sentence_terms))
                topic_match = topic != "general" and topic in set(clause.topic_tags)
                if overlap == 0 and hint_overlap == 0 and not topic_match:
                    continue
                score = float(overlap) + (0.75 * float(hint_overlap))
                if topic_match:
                    score += 1.25
                lowered = sentence.lower()
                has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", sentence))
                if has_number:
                    score += 0.25
                has_time = any(token in lowered for token in ["day", "days", "month", "months", "year", "years"])
                if has_number and has_time:
                    score += 0.5  # Numeric timelines are high-value in legal text
                if any(
                    token in lowered
                    for token in ["shall", "must", "within", "before", "after", "provided that", "required"]
                ):
                    score += 0.2
                if score > 0:
                    scored_sentences.append((score, len(scored_sentences), sentence))

        if not scored_sentences:
            fallback = re.sub(r"\s+", " ", doc.full_text).strip()
            return fallback[:1800], 0.0

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, int, str]] = []
        seen_keys: set[str] = set()
        for score, order, sentence in scored_sentences:
            key = self._dedupe_key(sentence)
            if not key:
                continue
            # Strip common legal prefixes and clause numbers for near-identical matches
            core_key = re.sub(
                r"^(provided\s+(that\s+|also\s+that\s+)?|subject\s+to\s+|in\s+case\s+|if\s+)",
                "",
                re.sub(r"^\(?\d+[a-z]?\)?\s*", "", key),
            ).strip()[:140]
            if key in seen_keys or core_key in seen_keys:
                continue
            selected.append((score, order, sentence))
            seen_keys.add(key)
            seen_keys.add(core_key)
            if len(selected) >= self.policy.max_excerpt_sentences:
                break

        if not selected:
            fallback = re.sub(r"\s+", " ", doc.full_text).strip()
            return fallback[:1800], 0.0

        excerpt = re.sub(r"\s+", " ", " ".join(sentence for _score, _order, sentence in selected)).strip()
        avg_score = sum(score for score, _order, _sentence in selected) / float(len(selected))
        denom = float(max(1, len(query_terms)))
        relevance = min(1.0, avg_score / denom)
        return excerpt[: self.policy.max_excerpt_chars], relevance

    _STOPWORDS = {
        "the", "and", "for", "are", "was", "were", "been", "being",
        "this", "that", "these", "those", "with", "from", "into",
        "has", "have", "had", "not", "but", "also", "any", "all",
        "such", "than", "will", "may", "shall", "can", "per",
        "which", "where", "when", "who", "how", "what",
    }

    def _tokenize_text(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {token for token in tokens if len(token) >= 3 and token not in self._STOPWORDS}

    def _doc_query_relevance(self, *, doc: RuleDocument, query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        title_terms = self._tokenize_text(doc.rule_title)
        chapter_terms = self._tokenize_text(doc.chapter_title)
        overlap = len(query_terms.intersection(title_terms))
        overlap += 0.5 * len(query_terms.intersection(chapter_terms))
        return min(1.0, float(overlap) / float(len(query_terms)))

    def _split_sentences(self, text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return []
        # Split on punctuation and sub-rule markers to avoid giant sentence blobs.
        marked = re.sub(r"(\(\d+[A-Za-z]?\))", r"\n\1 ", compact)
        parts = re.split(r"(?<=[.;:])\s+|\n+", marked)
        sentences: list[str] = []
        for part in parts:
            cleaned = part.strip(" -_")
            if len(cleaned) < 30:
                continue
            sentences.append(cleaned)
        return sentences

    def _topic_hint_terms(self, topic: str) -> set[str]:
        explicit = _GENERIC_TOPIC_HINTS.get(topic, set())
        if explicit:
            return explicit
        return self._tokenize_text(topic.replace("_", " "))

    def _fallback_topic_likes(self, query: str) -> list[str]:
        tokens = [token for token in re.findall(r"[a-z0-9]{4,}", query.lower()) if token]
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= self.policy.max_topic_likes:
                break
        return deduped

    def _compact_table_text(self, raw_text: str) -> str:
        lines = [re.sub(r"\s+", " ", line).strip(" -|`") for line in raw_text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return ""
        header = lines[:4]
        numeric_rows = [line for line in lines[4:] if re.search(r"\b\d+(?:\.\d+)?\b", line)]
        selected = header + numeric_rows[:8]
        return " ".join(selected)

    def _dedupe_key(self, sentence: str) -> str:
        lowered = sentence.lower()
        lowered = re.sub(r"^\(?\d+[a-z]?\)?\s*", "", lowered)
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
        return lowered[:220]

    def _topic_family_guard(self, *, topic: str, doc_topics: set[str]) -> bool:
        if topic == "general":
            return True
        excluded = _TOPIC_FAMILY_EXCLUSIONS.get(topic, set())
        if not excluded:
            return True
        if topic in doc_topics:
            return True
        if doc_topics.intersection(excluded):
            return False
        return True

    def _top_rules_cited(self, docs: list[RuleDocument], limit: int = 5) -> list[str]:
        cited: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            rule = str(doc.rule_number or "").strip()
            if not rule:
                continue
            if rule in seen:
                continue
            seen.add(rule)
            cited.append(rule)
            if len(cited) >= limit:
                break
        return cited
