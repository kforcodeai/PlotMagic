from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.models import QueryFact
from src.models.schemas import QueryRequest


RULE_LINE_PATTERN = re.compile(r"^\s*(?:##\s*)?_?(\d{1,3})\.(?!\d)")
LINE_RANGE_PATTERN = re.compile(r":(\d+)-(\d+)$")
RULE_REF_PATTERN = re.compile(r"\brule\s*(\d{1,3})\b", flags=re.IGNORECASE)
CATEGORY_II_PATTERN = re.compile(r"category[\s-]*ii\b", flags=re.IGNORECASE)
CATEGORY_I_PATTERN = re.compile(r"category[\s-]*i\b", flags=re.IGNORECASE)


@dataclass(slots=True)
class RetrievalRun:
    status: str
    retrieved_doc_ids: list[str]
    latency_ms: float
    candidate_count: int
    occupancy_skipped: bool
    reranker_used: bool
    reranker_degraded: bool
    reranker_failures: int
    query_intent: str | None
    fallback_stage: str | None = None
    lexical_hits: int = 0
    vector_hits: int = 0
    structured_hits: int = 0


@dataclass(slots=True)
class QueryMetrics:
    query_id: str
    status: str
    doc_hit_at_k: float
    doc_recall_at_k: float
    mrr: float
    chunk_recall_at_k: float
    relevant_doc_count: int
    retrieved_doc_count: int
    candidate_doc_count: int
    unresolved_reason: str | None
    latency_ms: float
    query_intent: str | None
    fallback_stage: str | None
    reranker_used: bool
    reranker_degraded: bool
    reranker_failures: int
    occupancy_skipped: bool
    lexical_hits: int
    vector_hits: int
    structured_hits: int
    missing_chunk_citations: list[str]


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("\u00a0", " ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    idx = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[idx]


def parse_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def build_line_to_rule(source_path: Path) -> dict[int, int | None]:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    line_to_rule: dict[int, int | None] = {}

    current_rule: int | None = None
    max_seen_rule = 0

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip().lower()
        if stripped.startswith("## appendix") or stripped.startswith("### appendix"):
            current_rule = None

        m = RULE_LINE_PATTERN.match(line)
        if m:
            value = int(m.group(1))
            if 1 <= value <= 999 and value > max_seen_rule:
                current_rule = value
                max_seen_rule = value

        line_to_rule[lineno] = current_rule

    return line_to_rule


def extract_line_range(citation: str) -> tuple[int, int] | None:
    m = LINE_RANGE_PATTERN.search(citation)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def infer_category(query: str, default_category: str | None, *, use_default: bool = True) -> str | None:
    if CATEGORY_II_PATTERN.search(query):
        return "Category-II"
    if CATEGORY_I_PATTERN.search(query):
        return "Category-I"
    return default_category if use_default else None


def build_docs_lookup(engine: ComplianceEngine, state: str, jurisdiction: str) -> tuple[dict[int, set[str]], dict[str, str]]:
    docs = [doc for doc in engine.state.docs if doc.state == state and doc.jurisdiction_type == jurisdiction]

    by_rule: dict[int, set[str]] = {}
    norm_text: dict[str, str] = {}

    for doc in docs:
        norm_text[doc.document_id] = normalize_text(doc.full_text)
        if doc.rule_number:
            m = re.search(r"\d+", doc.rule_number)
            if m:
                rule_num = int(m.group(0))
                by_rule.setdefault(rule_num, set()).add(doc.document_id)

    return by_rule, norm_text


def chunk_relevant_doc_ids(
    chunk: dict[str, Any],
    line_to_rule: dict[int, int | None],
    docs_by_rule: dict[int, set[str]],
    doc_text_norm: dict[str, str],
) -> set[str]:
    relevant: set[str] = set()

    citation = chunk.get("citation", "")
    line_range = extract_line_range(citation)
    if line_range:
        start, _end = line_range
        mapped_rule = line_to_rule.get(start)
        if mapped_rule is not None:
            relevant.update(docs_by_rule.get(mapped_rule, set()))

    snippet = chunk.get("snippet", "")
    for m in RULE_REF_PATTERN.finditer(snippet):
        relevant.update(docs_by_rule.get(int(m.group(1)), set()))

    snippet_norm = normalize_text(snippet)
    if snippet_norm and len(snippet_norm) >= 40:
        for doc_id, text_norm in doc_text_norm.items():
            if snippet_norm in text_norm:
                relevant.add(doc_id)

    if not relevant and snippet_norm:
        tokens = [tok for tok in snippet_norm.split() if len(tok) >= 5]
        tokens = tokens[:20]
        if len(tokens) >= 3:
            threshold = max(3, len(tokens) // 3)
            for doc_id, text_norm in doc_text_norm.items():
                hits = sum(1 for token in tokens if token in text_norm)
                if hits >= threshold:
                    relevant.add(doc_id)

    return relevant


def run_retrieval(
    engine: ComplianceEngine,
    request: QueryRequest,
    category_filter: str | None,
) -> RetrievalRun:
    if not engine.state.docs:
        engine.ingest(state=request.state or "kerala", jurisdiction_type=request.jurisdiction_type)

    inferred = engine._infer_scope_hints(request.query)
    requested_state = (request.state or "kerala").lower()
    scope = engine._scope_resolver(requested_state).resolve(
        location=request.location,
        state_hint=request.state or inferred["state_hint"],
        jurisdiction_hint=request.jurisdiction_type or inferred["jurisdiction_hint"],
        panchayat_category_hint=request.panchayat_category or inferred["panchayat_category_hint"],
    )
    if not scope.resolved:
        return RetrievalRun(
            status="scope_unresolved",
            retrieved_doc_ids=[],
            latency_ms=0.0,
            candidate_count=0,
            occupancy_skipped=False,
            reranker_used=False,
            reranker_degraded=False,
            reranker_failures=0,
            query_intent=None,
            fallback_stage="scope_unresolved",
        )

    plan = engine.query_planner.plan(request.query)

    occupancy = engine._occupancy_resolver(scope.state or requested_state).resolve(
        state=scope.state or requested_state,
        building_description=request.query,
        explicit_occupancy=request.explicit_occupancy,
    )
    occupancy_skipped = False
    if engine._can_skip_occupancy(plan=plan, query=request.query):
        occupancy = type(occupancy)(resolved=True, candidates=occupancy.candidates, selected=[])
        occupancy_skipped = True
    elif not occupancy.resolved:
        return RetrievalRun(
            status="occupancy_unresolved",
            retrieved_doc_ids=[],
            latency_ms=0.0,
            candidate_count=0,
            occupancy_skipped=occupancy_skipped,
            reranker_used=False,
            reranker_degraded=False,
            reranker_failures=0,
            query_intent=plan.query_type,
            fallback_stage="occupancy_unresolved",
        )

    fact = QueryFact(
        state=scope.state,
        location_text=request.location,
        jurisdiction_type=scope.jurisdiction_type,
        panchayat_category=category_filter,
        occupancies=occupancy.selected,
        topics=plan.topics,
        query_date=request.query_date,
        mentioned_rules=plan.mentioned_rule_numbers,
        query_intent=plan.query_type,
    )
    fact = engine.fact_extractor.extract(request.query, seed=fact)

    docs_in_scope = [
        doc
        for doc in engine.state.docs
        if doc.state == scope.state and doc.jurisdiction_type == scope.jurisdiction_type and doc.ruleset_id == scope.ruleset_id
    ]
    candidates = engine.applicability_engine.select_candidates(docs_in_scope, fact).selected

    if not engine.state.hybrid_retriever:
        engine.state.hybrid_retriever = engine._build_hybrid_retriever()

    effective_top_k = max(request.top_k, getattr(plan, "suggested_top_k", request.top_k))
    retrieval = engine.state.hybrid_retriever.retrieve(
        query=request.query,
        fact=fact,
        plan=plan,
        candidate_docs=candidates,
        top_k=effective_top_k,
        retrieval_mode=request.retrieval_mode,
    )
    reranker_used = bool(retrieval.latency_ms.get("reranker_used", 0.0))
    reranker_degraded = bool(retrieval.latency_ms.get("reranker_degraded", 0.0))
    reranker_failures = int(retrieval.latency_ms.get("reranker_failures", 0.0))
    return RetrievalRun(
        status="ok",
        retrieved_doc_ids=[doc.document_id for doc in retrieval.retrieved_documents],
        latency_ms=retrieval.latency_ms.get("total_retrieval_ms", 0.0),
        candidate_count=len(candidates),
        occupancy_skipped=occupancy_skipped,
        reranker_used=reranker_used,
        reranker_degraded=reranker_degraded,
        reranker_failures=reranker_failures,
        query_intent=plan.query_type,
        fallback_stage=str(retrieval.diagnostics.get("fallback_stage")) if retrieval.diagnostics else None,
        lexical_hits=int(retrieval.latency_ms.get("lexical_hits", 0.0)),
        vector_hits=int(retrieval.latency_ms.get("vector_hits", 0.0)),
        structured_hits=int(retrieval.latency_ms.get("structured_hits", 0.0)),
    )


def compute_mrr(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    if not relevant_doc_ids:
        return 0.0
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / float(idx)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval with ground-truth snippet dataset.")
    parser.add_argument(
        "--dataset",
        default="evaluation/kpbr/kpbr_multihop_retrieval_dataset.jsonl",
        help="Path to dataset JSON/JSONL.",
    )
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="panchayat", choices=["panchayat", "municipality"])
    parser.add_argument("--category", default="Category-II", help="Panchayat category override.")
    parser.add_argument("--location", default=None)
    parser.add_argument(
        "--top-k",
        default=None,
        type=int,
        help="Retrieval top-k. Defaults to 15 for panchayat and 20 for municipality when omitted.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/kpbr/kpbr_multihop_retrieval_benchmark.json",
        help="Output path for benchmark result JSON.",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid_no_reranker",
        choices=[
            "vector_only",
            "hybrid_no_reranker",
            "lexical_only_bm25",
            "hybrid_reranker",
            "hybrid_graph_reranker",
            "hybrid_graph_no_reranker",
            "agentic_dynamic",
        ],
        help="Retrieval mode used for all queries in this benchmark run.",
    )
    parser.add_argument(
        "--rerank-provider",
        default=None,
        help="Optional reranker provider id override for this run (for example: openai_llm_reranker).",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Optional reranker model override for this run (for example: gpt-4.1-mini).",
    )
    args = parser.parse_args()
    top_k = args.top_k if args.top_k is not None else (15 if args.jurisdiction == "panchayat" else 20)

    if args.rerank_provider:
        os.environ["PLOTMAGIC_RERANK_PROVIDER"] = str(args.rerank_provider)
    if args.rerank_model:
        os.environ["PLOTMAGIC_RERANK_MODEL"] = str(args.rerank_model)

    dataset_path = (ROOT / args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    records = parse_dataset(dataset_path)
    if not records:
        raise SystemExit("Dataset is empty.")

    first_source = records[0]["source_document"]
    source_path = Path(first_source)
    if not source_path.is_absolute():
        source_path = (ROOT / first_source).resolve()
    if not source_path.exists():
        raise SystemExit(f"Source document not found: {source_path}")

    line_to_rule = build_line_to_rule(source_path)

    engine = ComplianceEngine(root=ROOT)
    ingest_start = perf_counter()
    engine.ingest(state=args.state, jurisdiction_type=args.jurisdiction)
    ingest_ms = (perf_counter() - ingest_start) * 1000
    vector_stats = getattr(engine.vector_store, "last_upsert_stats", None)
    vector_total = int(getattr(vector_stats, "total_clauses", 0) or 0)
    vector_cached = int(getattr(vector_stats, "cached_count", 0) or 0)
    vector_embedded = int(getattr(vector_stats, "embedded_count", 0) or 0)
    vector_deleted = int(getattr(vector_stats, "deleted_count", 0) or 0)
    vector_cache_hit_rate = (vector_cached / float(vector_total)) if vector_total else 0.0
    vector_db_path = getattr(engine.vector_store, "db_path", None)

    docs_by_rule, doc_text_norm = build_docs_lookup(engine, state=args.state, jurisdiction=args.jurisdiction)

    per_query: list[QueryMetrics] = []
    latencies: list[float] = []

    scope_unresolved = 0
    occupancy_unresolved = 0
    mappable_queries = 0
    total_chunks = 0
    mappable_chunks = 0
    candidate_nonzero = 0
    candidate_nonzero_procedural = 0
    procedural_queries = 0
    reranker_used_queries = 0
    reranker_degraded_queries = 0
    reranker_failure_count = 0
    occupancy_skipped_count = 0
    lexical_active = 0
    vector_active = 0
    structured_active = 0

    for item in records:
        query_id = item["id"]
        query_text = item["query"]
        scope_category = infer_category(query_text, args.category, use_default=True)
        category_filter = infer_category(query_text, None, use_default=False)

        request = QueryRequest(
            query=query_text,
            state=args.state,
            location=args.location,
            jurisdiction_type=args.jurisdiction,
            panchayat_category=scope_category,
            top_k=top_k,
            retrieval_mode=args.retrieval_mode,
        )

        run = run_retrieval(engine, request, category_filter=category_filter)
        retrieved_ids = run.retrieved_doc_ids[: top_k]
        retrieved_set = set(retrieved_ids)

        if run.status == "scope_unresolved":
            scope_unresolved += 1
        elif run.status == "occupancy_unresolved":
            occupancy_unresolved += 1
        else:
            latencies.append(run.latency_ms)
            if run.candidate_count > 0:
                candidate_nonzero += 1
            if run.occupancy_skipped:
                occupancy_skipped_count += 1
            if run.reranker_used:
                reranker_used_queries += 1
            if run.reranker_degraded:
                reranker_degraded_queries += 1
            reranker_failure_count += run.reranker_failures
            if run.lexical_hits > 0:
                lexical_active += 1
            if run.vector_hits > 0:
                vector_active += 1
            if run.structured_hits > 0:
                structured_active += 1

        if run.query_intent == "procedural":
            procedural_queries += 1
            if run.status == "ok" and run.candidate_count > 0:
                candidate_nonzero_procedural += 1

        chunk_doc_sets: list[set[str]] = []
        missing_chunk_citations: list[str] = []
        for chunk in item.get("ground_truth_chunks", []):
            total_chunks += 1
            ids = chunk_relevant_doc_ids(chunk, line_to_rule, docs_by_rule, doc_text_norm)
            if ids:
                mappable_chunks += 1
            else:
                missing_chunk_citations.append(chunk.get("citation", ""))
            chunk_doc_sets.append(ids)

        relevant_doc_ids: set[str] = set().union(*chunk_doc_sets) if chunk_doc_sets else set()
        if relevant_doc_ids:
            mappable_queries += 1

        doc_hit = 1.0 if relevant_doc_ids and bool(retrieved_set.intersection(relevant_doc_ids)) else 0.0
        doc_recall = (
            len(retrieved_set.intersection(relevant_doc_ids)) / float(len(relevant_doc_ids)) if relevant_doc_ids else 0.0
        )
        mrr = compute_mrr(retrieved_ids, relevant_doc_ids)

        retrieved_chunk_count = 0
        for idx, chunk in enumerate(item.get("ground_truth_chunks", [])):
            chunk_ids = chunk_doc_sets[idx]
            if chunk_ids and retrieved_set.intersection(chunk_ids):
                retrieved_chunk_count += 1
                continue
            if not chunk_ids and run.status == "ok":
                snippet_norm = normalize_text(chunk.get("snippet", ""))
                if snippet_norm:
                    for doc_id in retrieved_ids:
                        if snippet_norm in doc_text_norm.get(doc_id, ""):
                            retrieved_chunk_count += 1
                            break

        chunk_total = len(item.get("ground_truth_chunks", []))
        chunk_recall = (retrieved_chunk_count / float(chunk_total)) if chunk_total else 0.0

        per_query.append(
            QueryMetrics(
                query_id=query_id,
                status=run.status,
                doc_hit_at_k=doc_hit,
                doc_recall_at_k=doc_recall,
                mrr=mrr,
                chunk_recall_at_k=chunk_recall,
                relevant_doc_count=len(relevant_doc_ids),
                retrieved_doc_count=len(retrieved_ids),
                candidate_doc_count=run.candidate_count,
                unresolved_reason=None if run.status == "ok" else run.status,
                latency_ms=run.latency_ms,
                query_intent=run.query_intent,
                fallback_stage=run.fallback_stage,
                reranker_used=run.reranker_used,
                reranker_degraded=run.reranker_degraded,
                reranker_failures=run.reranker_failures,
                occupancy_skipped=run.occupancy_skipped,
                lexical_hits=run.lexical_hits,
                vector_hits=run.vector_hits,
                structured_hits=run.structured_hits,
                missing_chunk_citations=missing_chunk_citations,
            )
        )

    retrieval_ok_count = len(records) - scope_unresolved - occupancy_unresolved
    scope_resolved_count = len(records) - scope_unresolved
    occupancy_resolution_rate = (
        (scope_resolved_count - occupancy_unresolved) / float(scope_resolved_count) if scope_resolved_count else 0.0
    )
    occupancy_unresolved_pct = (occupancy_unresolved / float(scope_resolved_count)) if scope_resolved_count else 0.0
    candidate_nonzero_rate = candidate_nonzero / float(retrieval_ok_count) if retrieval_ok_count else 0.0
    procedural_candidate_nonzero_rate = (
        candidate_nonzero_procedural / float(procedural_queries) if procedural_queries else 0.0
    )
    reranker_coverage_rate = reranker_used_queries / float(retrieval_ok_count) if retrieval_ok_count else 0.0
    reranker_degraded_rate = (
        reranker_degraded_queries / float(reranker_used_queries) if reranker_used_queries else 0.0
    )
    occupancy_skipped_rate = occupancy_skipped_count / float(retrieval_ok_count) if retrieval_ok_count else 0.0
    lexical_active_rate = lexical_active / float(retrieval_ok_count) if retrieval_ok_count else 0.0
    vector_active_rate = vector_active / float(retrieval_ok_count) if retrieval_ok_count else 0.0
    structured_active_rate = structured_active / float(retrieval_ok_count) if retrieval_ok_count else 0.0

    doc_hit_at_k = statistics.mean(q.doc_hit_at_k for q in per_query) if per_query else 0.0
    doc_recall_at_k = statistics.mean(q.doc_recall_at_k for q in per_query) if per_query else 0.0
    chunk_recall_at_k = statistics.mean(q.chunk_recall_at_k for q in per_query) if per_query else 0.0
    mrr = statistics.mean(q.mrr for q in per_query) if per_query else 0.0

    summary = {
        "dataset": str(dataset_path),
        "query_count": len(records),
        "top_k": top_k,
        "retrieval_mode": args.retrieval_mode,
        "rerank_provider": getattr(engine.reranker_provider, "provider_id", None),
        "rerank_model": getattr(engine.reranker_provider, "model", None),
        "ingest_ms": ingest_ms,
        "vector_backend": type(engine.vector_store).__name__,
        "vector_db_path": str(vector_db_path) if vector_db_path else None,
        "vector_index_clause_count": len(getattr(engine.vector_store, "vectors", {})),
        "vector_total_clauses": vector_total,
        "vector_cached_count": vector_cached,
        "vector_embedded_count": vector_embedded,
        "vector_deleted_count": vector_deleted,
        "vector_cache_hit_rate": vector_cache_hit_rate,
        "scope_unresolved_count": scope_unresolved,
        "occupancy_unresolved_count": occupancy_unresolved,
        "retrieval_ok_count": retrieval_ok_count,
        "occupancy_resolution_rate": occupancy_resolution_rate,
        "occupancy_unresolved_pct": occupancy_unresolved_pct,
        "candidate_nonzero_count": candidate_nonzero,
        "candidate_nonzero_rate": candidate_nonzero_rate,
        "procedural_query_count": procedural_queries,
        "procedural_candidate_nonzero_rate": procedural_candidate_nonzero_rate,
        "reranker_coverage_count": reranker_used_queries,
        "reranker_coverage_rate": reranker_coverage_rate,
        "reranker_degraded_count": reranker_degraded_queries,
        "reranker_degraded_rate": reranker_degraded_rate,
        "reranker_failure_count": reranker_failure_count,
        "occupancy_skipped_count": occupancy_skipped_count,
        "occupancy_skipped_rate": occupancy_skipped_rate,
        "lexical_active_rate": lexical_active_rate,
        "vector_active_rate": vector_active_rate,
        "structured_active_rate": structured_active_rate,
        "ground_truth_mappable_queries": mappable_queries,
        "ground_truth_mappable_chunks": mappable_chunks,
        "ground_truth_total_chunks": total_chunks,
        "doc_hit_at_k": doc_hit_at_k,
        "doc_recall_at_k": doc_recall_at_k,
        "chunk_recall_at_k": chunk_recall_at_k,
        "mrr": mrr,
        "latency_ms_p50": percentile(latencies, 50.0),
        "latency_ms_p95": percentile(latencies, 95.0),
        "latency_ms_avg": statistics.mean(latencies) if latencies else 0.0,
    }
    summary["target_gates"] = {
        "doc_hit_at_20_target": 0.80,
        "chunk_recall_at_20_target": 0.70,
        "occupancy_unresolved_max": 0.05,
        "doc_hit_at_20_pass": (doc_hit_at_k >= 0.80) if top_k == 20 else None,
        "chunk_recall_at_20_pass": (chunk_recall_at_k >= 0.70) if top_k == 20 else None,
        "occupancy_unresolved_pass": occupancy_unresolved_pct <= 0.05,
    }
    fallback_stage_counts: dict[str, int] = {}
    for row in per_query:
        stage = str(row.fallback_stage or "unknown")
        fallback_stage_counts[stage] = fallback_stage_counts.get(stage, 0) + 1
    top_blocking_queries = sorted(
        [asdict(row) for row in per_query],
        key=lambda item: (float(item.get("chunk_recall_at_k", 0.0)), float(item.get("mrr", 0.0))),
    )[:10]
    summary["fallback_stage_counts"] = fallback_stage_counts
    summary["top_blocking_queries"] = [
        {
            "query_id": row["query_id"],
            "status": row["status"],
            "chunk_recall_at_k": row["chunk_recall_at_k"],
            "doc_recall_at_k": row["doc_recall_at_k"],
            "mrr": row["mrr"],
            "candidate_doc_count": row["candidate_doc_count"],
            "fallback_stage": row.get("fallback_stage"),
            "vector_hits": row["vector_hits"],
            "lexical_hits": row["lexical_hits"],
            "structured_hits": row["structured_hits"],
        }
        for row in top_blocking_queries
    ]

    result = {
        "summary": summary,
        "per_query": [asdict(row) for row in per_query],
    }

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote detailed result: {out_path}")


if __name__ == "__main__":
    main()
