from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.indexing.index_manifest import (
    build_manifest,
    compute_sources_hash,
    deterministic_collection_name,
    load_manifest,
    manifest_matches,
    save_manifest,
)
from src.ingestion.cleaners import CLEANING_VERSION
from src.ingestion.parsers.kpbr_markdown_parser import PARSER_VERSION as KPBR_PARSER_VERSION
from src.ingestion.pipeline import IngestionPipeline
from src.models.schemas import QueryRequest
from src.providers import load_providers_config

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "at",
    "by",
    "with",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "as",
    "it",
    "its",
    "if",
    "then",
    "than",
    "within",
    "under",
    "into",
    "per",
    "can",
    "may",
    "shall",
    "should",
    "must",
    "not",
    "no",
    "yes",
    "has",
    "have",
    "had",
    "when",
    "where",
    "before",
    "after",
    "also",
    "any",
    "all",
    "such",
    "but",
    "only",
    "each",
    "both",
    "other",
    "does",
    "do",
}

HARD_GATES = {
    "citation_groundedness_mean": 1.0,
    "contract_violation_rate": 0.0,
    "vector_hits_zero_count": 0,
}

IMPROVEMENT_GATES = {
    "token_set_f1_mean": 0.10,
    "token_set_recall_mean": 0.10,
    "numeric_token_recall_mean": 0.15,
    "mandatory_component_completeness_mean": 0.10,
}

WEIGHTED_WINNER_METRICS = {
    "mandatory_component_completeness_mean": 0.40,
    "token_set_recall_mean": 0.25,
    "token_set_f1_mean": 0.20,
    "numeric_token_recall_mean": 0.15,
}


@contextmanager
def index_build_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def normalize_tokens(text: str) -> list[str]:
    lowered = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    # Split compound terms: "front/rear/side" → "front rear side", "built-up" → "built up"
    lowered = re.sub(r"[/-]", " ", lowered)
    cleaned = re.sub(r"[^a-z0-9%.()]+", " ", lowered)
    tokens: list[str] = []
    for token in cleaned.split():
        if not token or token in _STOPWORDS:
            continue
        # Strip surrounding parentheses and trailing punctuation.
        token = token.strip("().,;:")
        if not token:
            continue
        # Split "1.5m" → "1.5" + "m", "300sq.m" → "300" + "sq.m", "100m" → "100" + "m".
        num_unit = re.match(r"^(\d+(?:\.\d+)?)(m|cm|mm|km|sq\.?m|metre|meter|day|month|year|%)$", token)
        if num_unit:
            tokens.append(num_unit.group(1))
            unit = num_unit.group(2)
            if unit not in _STOPWORDS:
                tokens.append(unit)
            continue
        # Normalize common suffix forms for better token overlap.
        if token.endswith("ied") and len(token) > 5:
            token = token[:-3] + "y"  # e.g., "complied" → "comply"
        elif token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"  # e.g., "boundaries" → "boundary"
        elif token.endswith("tted") and len(token) > 6:
            token = token[:-3]  # e.g., "permitted" → "permit"
        elif token.endswith("nced") and len(token) > 6:
            token = token[:-1]  # e.g., "commenced" → "commence" (keep the e)
        elif token.endswith("ed") and len(token) > 4 and not token.endswith("eed"):
            base = token[:-2]
            # Handle doubled consonant before -ed: "transferred" → "transfer", "occurred" → "occur"
            if len(base) > 3 and base[-1] == base[-2]:
                token = base[:-1]
            # If removing -ed leaves a base that likely had a trailing 'e', restore it.
            # Only for consonants where base+e is the natural form (issue, produce, approve, etc.)
            elif base and base[-1] in "csuvz" and len(base) > 2:
                token = base + "e"  # e.g., "issued" → "issue", "produced" → "produce"
            else:
                token = base  # e.g., "obtained" → "obtain", "granted" → "grant"
        elif token.endswith("es") and len(token) > 4 and token[-3] in "shx":
            token = token[:-2]  # e.g., "charges" → "charg", "processes" → "process"
        elif token.endswith("s") and len(token) > 3 and token[-2] not in "su":
            token = token[:-1]  # e.g., "metres" → "metre", "days" → "day"
        if token and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def set_f1(pred: str, ref: str) -> float:
    pred_set = set(normalize_tokens(pred))
    ref_set = set(normalize_tokens(ref))
    if not pred_set or not ref_set:
        return 0.0
    overlap = len(pred_set.intersection(ref_set))
    if overlap == 0:
        return 0.0
    precision = overlap / float(len(pred_set))
    recall = overlap / float(len(ref_set))
    return (2.0 * precision * recall) / (precision + recall)


def set_recall(pred: str, ref: str) -> float:
    pred_set = set(normalize_tokens(pred))
    ref_set = set(normalize_tokens(ref))
    if not ref_set:
        return 0.0
    return len(pred_set.intersection(ref_set)) / float(len(ref_set))


def numeric_recall(pred: str, ref: str) -> float:
    pred_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", pred))
    ref_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", ref))
    if not ref_numbers:
        return 0.0
    return len(pred_numbers.intersection(ref_numbers)) / float(len(ref_numbers))


def response_to_text(payload: dict[str, Any]) -> str:
    """Extract the predicted answer text for evaluation.

    Uses short_summary as the primary answer (matching reference answer style).
    The summary should be comprehensive — items provide citation backing, not additional text.
    """
    final_answer = payload.get("final_answer") or {}
    if isinstance(final_answer, dict):
        short_summary = str(final_answer.get("short_summary", "")).strip()
        if short_summary:
            return short_summary

        # Fallback: concatenate all item texts if no summary.
        lines: list[str] = []
        for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
            section = final_answer.get(section_name, [])
            if isinstance(section, list):
                for item in section:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    if text:
                        lines.append(text)
        if lines:
            return "\n".join(lines)

    sections = payload.get("answer_sections", [])
    lines: list[str] = []
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            for rule in section.get("rules", []):
                if not isinstance(rule, dict):
                    continue
                excerpt = str(rule.get("excerpt", "")).strip()
                if excerpt:
                    lines.append(excerpt)
    return "\n".join(lines)


def retrieval_oracle_text(payload: dict[str, Any], max_items: int = 30) -> str:
    """Gather all evidence text for computing retrieval ceiling.

    Uses a generous max_items to measure the true ceiling of what retrieval provides.
    """
    evidence = payload.get("evidence_matrix", [])
    if not isinstance(evidence, list):
        return ""
    texts: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        texts.append(text)
        if len(texts) >= max_items:
            break
    return "\n".join(texts)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0 if numerator <= 0.0 else 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _classify_bottleneck(
    *,
    retrieval_ceiling_recall: float,
    predicted_recall: float,
    synthesis_efficiency_recall: float,
    retrieval_ceiling_numeric_recall: float,
    predicted_numeric_recall: float,
) -> str:
    retrieval_gap = max(0.0, 1.0 - retrieval_ceiling_recall)
    synthesis_gap = max(0.0, retrieval_ceiling_recall - predicted_recall)
    numeric_gap = max(0.0, retrieval_ceiling_numeric_recall - predicted_numeric_recall)

    retrieval_limited = retrieval_ceiling_recall < 0.55 or retrieval_gap > 0.45
    synthesis_limited = synthesis_efficiency_recall < 0.75 or synthesis_gap > 0.20

    if retrieval_limited and synthesis_limited:
        return "mixed"
    if retrieval_limited:
        return "retrieval"
    if synthesis_limited or numeric_gap > 0.20:
        return "synthesis"
    return "balanced"


def _bottleneck_counts(per_query: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "retrieval": 0,
        "synthesis": 0,
        "mixed": 0,
        "balanced": 0,
    }
    for item in per_query:
        label = str(item.get("bottleneck_class", "")).strip().lower()
        if label in counts:
            counts[label] += 1
    return counts


def extract_mandatory_components(question: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", question.strip())
    if not normalized:
        return []
    parts = re.split(r"\?|;|,(?:\s+and\s+)?|\band\b|\bor\b", normalized, flags=re.IGNORECASE)
    components: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = part.strip(" .:")
        if len(cleaned.split()) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        components.append(cleaned)
    return components[:8] if components else [normalized]


def component_completeness(predicted: str, components: list[str]) -> float:
    if not components:
        return 1.0
    pred_l = predicted.lower()
    covered = 0
    for component in components:
        tokens = [tok for tok in re.findall(r"[a-z0-9]{4,}", component.lower()) if tok]
        if not tokens:
            continue
        overlap = sum(1 for token in tokens[:10] if token in pred_l)
        min_required = max(1, min(3, len(tokens) // 2))
        if overlap >= min_required:
            covered += 1
    return covered / float(len(components))


def contradiction_rate(payload: dict[str, Any]) -> float:
    grounding = payload.get("grounding") or {}
    supported = int(grounding.get("supported_claim_count", 0) or 0)
    unsupported = int(grounding.get("unsupported_claim_count", 0) or 0)
    conflicting = int(grounding.get("conflicting_claim_count", 0) or 0)
    denom = max(1, supported + unsupported + conflicting)
    return conflicting / float(denom)


def citation_groundedness(payload: dict[str, Any]) -> float:
    final_answer = payload.get("final_answer") or {}
    claim_citations = payload.get("claim_citations") or {}
    if not isinstance(final_answer, dict) or not isinstance(claim_citations, dict):
        return 0.0

    checks = 0
    passed = 0
    for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
        section = final_answer.get(section_name, [])
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            checks += 1
            claim_id = str(item.get("claim_id", "")).strip()
            citations = item.get("citation_ids", [])
            if not claim_id or not isinstance(citations, list) or not citations:
                continue
            allowed = set(claim_citations.get(claim_id, []))
            if allowed and all(str(citation) in allowed for citation in citations):
                passed += 1
    if checks == 0:
        return 0.0
    return passed / float(checks)


def contract_violations(payload: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    verdict = payload.get("verdict")
    final_answer = payload.get("final_answer") or {}
    claim_citations = payload.get("claim_citations") or {}

    if isinstance(final_answer, dict):
        final_verdict = final_answer.get("verdict")
        if final_verdict is not None and verdict is not None and final_verdict != verdict:
            violations.append("verdict_mismatch")
        summary = str(final_answer.get("short_summary", "")).strip()
        if verdict != "insufficient_evidence" and not summary:
            violations.append("empty_summary")

        for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
            section = final_answer.get(section_name, [])
            if not isinstance(section, list):
                continue
            for item in section:
                if not isinstance(item, dict):
                    violations.append("invalid_item_type")
                    continue
                claim_id = str(item.get("claim_id", "")).strip()
                citations = item.get("citation_ids", [])
                if not claim_id:
                    violations.append("missing_claim_id")
                    continue
                if not isinstance(citations, list) or not citations:
                    violations.append("missing_citation_ids")
                    continue
                allowed = set((claim_citations or {}).get(claim_id, []))
                if not allowed or any(str(citation) not in allowed for citation in citations):
                    violations.append("uncited_or_unmapped_claim")

    deduped: list[str] = []
    seen: set[str] = set()
    for issue in violations:
        if issue in seen:
            continue
        seen.add(issue)
        deduped.append(issue)
    return deduped


def baseline_diff(current: dict[str, Any], baseline_summary: dict[str, Any] | None) -> dict[str, float]:
    if not baseline_summary:
        return {}
    deltas: dict[str, float] = {}
    comparable_fields = [
        "token_set_f1_mean",
        "token_set_recall_mean",
        "numeric_token_recall_mean",
        "semantic_accuracy_proxy_mean",
        "mandatory_component_completeness_mean",
        "citation_groundedness_mean",
        "contradiction_rate_mean",
        "contract_violation_rate",
    ]
    for field in comparable_fields:
        cur = current.get(field)
        base = baseline_summary.get(field)
        if isinstance(cur, (int, float)) and isinstance(base, (int, float)):
            deltas[field] = float(cur) - float(base)
    return deltas


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Dataset is empty or invalid JSON list.")
    return rows


def _embedding_model_for(provider_id: str) -> str:
    cfg = load_providers_config(ROOT / "config" / "providers.yaml")
    settings = cfg.embedding_settings[provider_id]
    return str(settings.model or "")


def _corpus_profile(state: str, jurisdiction: str) -> dict[str, Any]:
    pipeline = IngestionPipeline(ROOT / "config" / "states.yaml")
    state_cfg = pipeline.config["states"][state]
    jurisdiction_cfg = state_cfg["jurisdictions"][jurisdiction]
    source_root = pipeline._resolve_path(Path(str(jurisdiction_cfg["source_path"])))
    source_paths = pipeline._resolve_source_paths(
        source_root,
        source_glob=jurisdiction_cfg.get("source_glob"),
        source_format=jurisdiction_cfg.get("source_format"),
    )
    parser_class = str(jurisdiction_cfg.get("parser_class", ""))
    parser_version = KPBR_PARSER_VERSION if parser_class == "KPBRMarkdownParser" else parser_class
    return {
        "ruleset_id": str(jurisdiction_cfg["ruleset_id"]),
        "parser_version": parser_version,
        "source_paths": source_paths,
    }


def _runtime_env(
    *,
    vector_backend: str,
    vector_db_path: Path,
    embedding_provider: str,
    rerank_provider: str | None,
    rerank_model: str | None,
    collection_name: str,
    recreate_collection: bool,
) -> dict[str, str]:
    pool_factor = os.getenv("PLOTMAGIC_RETRIEVAL_POOL_FACTOR", "").strip()
    fallback_policy = os.getenv("PLOTMAGIC_RETRIEVAL_FALLBACK_POLICY", "").strip()
    category_policy = os.getenv("PLOTMAGIC_CATEGORY_FILTER_POLICY", "").strip()
    min_evidence = os.getenv("PLOTMAGIC_RETRIEVAL_MIN_EVIDENCE_SCORE", "").strip()
    env = {
        "PLOTMAGIC_VECTOR_BACKEND": vector_backend,
        "PLOTMAGIC_VECTOR_DB_PATH": str(vector_db_path),
        "PLOTMAGIC_EMBEDDING_PROVIDER": embedding_provider,
        "PLOTMAGIC_QDRANT_COLLECTION": collection_name,
        "PLOTMAGIC_QDRANT_RECREATE_COLLECTION": "true" if recreate_collection else "false",
        "PLOTMAGIC_RETRIEVAL_POOL_FACTOR": pool_factor,
        "PLOTMAGIC_RETRIEVAL_FALLBACK_POLICY": fallback_policy,
        "PLOTMAGIC_CATEGORY_FILTER_POLICY": category_policy,
        "PLOTMAGIC_RETRIEVAL_MIN_EVIDENCE_SCORE": min_evidence,
    }
    if rerank_provider:
        env["PLOTMAGIC_RERANK_PROVIDER"] = rerank_provider
    if rerank_model:
        env["PLOTMAGIC_RERANK_MODEL"] = rerank_model
    return env


def _apply_env(env_vars: dict[str, str]) -> None:
    for key, value in env_vars.items():
        os.environ[key] = value


def _strict_provider_check(engine: ComplianceEngine, required_embedding_provider: str) -> None:
    actual_embedding_provider = str(getattr(engine.embedding_provider, "provider_id", ""))
    if actual_embedding_provider != required_embedding_provider:
        raise SystemExit(
            "Strict provider check failed: expected embedding provider "
            f"'{required_embedding_provider}', got '{actual_embedding_provider}'."
        )
    degraded = [
        message
        for message in engine.provider_diagnostics
        if "Embedding provider" in message and "falling back" in message
    ]
    if degraded:
        raise SystemExit(
            "Strict provider check failed: embedding provider fell back during runtime. "
            + " | ".join(degraded)
        )


def _extract_retrieval_diagnostics(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    pass1 = None
    pass2 = None
    for event in events:
        if event.get("step") == "tool.retrieve.pass1" and event.get("status") == "ok":
            pass1 = event.get("details") or {}
        elif event.get("step") == "tool.retrieve.pass2" and event.get("status") == "ok":
            pass2 = event.get("details") or {}
    if not isinstance(pass1, dict):
        return None

    diagnostics = {
        "vector_hits": int(pass1.get("vector_hits", 0) or 0),
        "lexical_hits": int(pass1.get("lexical_hits", 0) or 0),
        "structured_hits": int(pass1.get("structured_hits", 0) or 0),
        "candidate_count": int(pass1.get("candidate_count", 0) or 0),
        "fallback_stage": pass1.get("fallback_stage"),
        "top_rules_cited": pass1.get("top_rules_cited", []),
    }
    if isinstance(pass2, dict):
        diagnostics["pass2_vector_hits"] = int(pass2.get("vector_hits", 0) or 0)
        diagnostics["pass2_lexical_hits"] = int(pass2.get("lexical_hits", 0) or 0)
        diagnostics["pass2_structured_hits"] = int(pass2.get("structured_hits", 0) or 0)
        diagnostics["pass2_fallback_stage"] = pass2.get("fallback_stage")
    return diagnostics


def _numeric_heavy_question(expected_answer: str) -> bool:
    return len(re.findall(r"\b\d+(?:\.\d+)?\b", expected_answer)) >= 2


def _index_snapshot(
    engine: ComplianceEngine,
    *,
    ingest_ms: float,
    reused_vs_rebuilt: str,
) -> dict[str, Any]:
    stats = getattr(engine.vector_store, "last_upsert_stats", None)
    total = int(getattr(stats, "total_clauses", 0) or 0)
    cached = int(getattr(stats, "cached_count", 0) or 0)
    embedded = int(getattr(stats, "embedded_count", 0) or 0)
    deleted = int(getattr(stats, "deleted_count", 0) or 0)
    upsert_count = int(getattr(stats, "upsert_count", embedded + cached) or 0)
    embedding_calls = int(getattr(stats, "embedding_calls_made", embedded) or 0)
    point_count = 0
    if hasattr(engine.vector_store, "point_count"):
        try:
            point_count = int(engine.vector_store.point_count())
        except Exception:
            point_count = 0

    return {
        "ingest_ms": ingest_ms,
        "vector_backend": type(engine.vector_store).__name__,
        "vector_db_path": str(getattr(engine.vector_store, "db_path", "")) or None,
        "point_count": point_count,
        "reused_vs_rebuilt": reused_vs_rebuilt,
        "vector_total_clauses": total,
        "vector_cached_count": cached,
        "vector_embedded_count": embedded,
        "vector_deleted_count": deleted,
        "upsert_count": upsert_count,
        "embedding_calls_made": embedding_calls,
    }


def _run_mode(
    *,
    engine: ComplianceEngine,
    mode: str,
    rows: list[dict[str, Any]],
    state: str,
    jurisdiction: str,
    category: str,
    top_k: int,
    run_id: str,
    phase: str,
    dataset_path: Path,
    baseline_summary: dict[str, Any] | None,
    index_snapshot: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    retrieval_trace: list[dict[str, Any]] = []
    all_contract_violations = 0
    missing_diagnostics: list[str] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        expected = str(row.get("answer", "")).strip()
        if not question or not expected:
            continue

        events: list[dict[str, Any]] = []
        request = QueryRequest(
            query=question,
            state=state,
            jurisdiction_type=jurisdiction,
            panchayat_category=category,
            top_k=top_k,
            retrieval_mode=mode,
            debug_trace=True,
        )
        response = engine.query(request, event_sink=events.append)
        response_payload = response.model_dump()
        predicted = response_to_text(response_payload)
        components = extract_mandatory_components(question)
        completeness = component_completeness(predicted, components)
        contradictions = contradiction_rate(response_payload)
        groundedness = citation_groundedness(response_payload)
        violations = contract_violations(response_payload)
        verdict = str(response_payload.get("verdict") or "")
        abstention_correct = 1.0 if verdict != "insufficient_evidence" else 0.0
        all_contract_violations += 1 if violations else 0

        retrieval_diag = _extract_retrieval_diagnostics(events)
        if retrieval_diag is None:
            missing_diagnostics.append(question)
            retrieval_diag = {
                "vector_hits": 0,
                "lexical_hits": 0,
                "structured_hits": 0,
                "candidate_count": 0,
                "fallback_stage": "missing",
                "top_rules_cited": [],
            }

        token_set_f1 = set_f1(predicted, expected)
        token_set_recall = set_recall(predicted, expected)
        numeric_token_recall = numeric_recall(predicted, expected)
        oracle_text = retrieval_oracle_text(response_payload)
        retrieval_ceiling_token_set_f1 = set_f1(oracle_text, expected)
        retrieval_ceiling_token_set_recall = set_recall(oracle_text, expected)
        retrieval_ceiling_numeric_token_recall = numeric_recall(oracle_text, expected)
        synthesis_efficiency_f1 = _safe_ratio(token_set_f1, retrieval_ceiling_token_set_f1)
        synthesis_efficiency_recall = _safe_ratio(token_set_recall, retrieval_ceiling_token_set_recall)
        synthesis_efficiency_numeric_recall = _safe_ratio(
            numeric_token_recall,
            retrieval_ceiling_numeric_token_recall,
        )
        bottleneck_class = _classify_bottleneck(
            retrieval_ceiling_recall=retrieval_ceiling_token_set_recall,
            predicted_recall=token_set_recall,
            synthesis_efficiency_recall=synthesis_efficiency_recall,
            retrieval_ceiling_numeric_recall=retrieval_ceiling_numeric_token_recall,
            predicted_numeric_recall=numeric_token_recall,
        )

        per_query.append(
            {
                "question": question,
                "mandatory_components": components,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "token_set_f1": token_set_f1,
                "token_set_recall": token_set_recall,
                "numeric_token_recall": numeric_token_recall,
                "retrieval_ceiling_token_set_f1": retrieval_ceiling_token_set_f1,
                "retrieval_ceiling_token_set_recall": retrieval_ceiling_token_set_recall,
                "retrieval_ceiling_numeric_token_recall": retrieval_ceiling_numeric_token_recall,
                "synthesis_efficiency_f1": synthesis_efficiency_f1,
                "synthesis_efficiency_recall": synthesis_efficiency_recall,
                "synthesis_efficiency_numeric_recall": synthesis_efficiency_numeric_recall,
                "retrieval_shortfall_recall": max(0.0, 1.0 - retrieval_ceiling_token_set_recall),
                "generation_shortfall_recall": max(0.0, retrieval_ceiling_token_set_recall - token_set_recall),
                "bottleneck_class": bottleneck_class,
                "semantic_accuracy_proxy": token_set_f1,
                "mandatory_component_completeness": completeness,
                "contradiction_rate": contradictions,
                "citation_groundedness": groundedness,
                "abstention_correctness": abstention_correct,
                "contract_violations": violations,
                "predicted_length_chars": len(predicted),
                "citation_count": len(response_payload.get("citations", [])),
                "verdict": response_payload.get("verdict"),
                "numeric_heavy": _numeric_heavy_question(expected),
                **retrieval_diag,
            }
        )
        retrieval_trace.append(
            {
                "question": question,
                "retrieval_mode": mode,
                **retrieval_diag,
            }
        )

    if missing_diagnostics:
        raise SystemExit(
            "Retrieval diagnostics incomplete. Missing pass-1 retrieval traces for: "
            + "; ".join(missing_diagnostics)
        )

    if not per_query:
        raise SystemExit("No valid question-answer rows found in dataset.")

    summary = {
        "schema_version": "legal_benchmark.v3",
        "run_id": run_id,
        "phase": phase,
        "retrieval_mode": mode,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "query_count": len(per_query),
        "top_k": top_k,
        **index_snapshot,
        "token_set_f1_mean": statistics.mean(item["token_set_f1"] for item in per_query),
        "token_set_f1_min": min(item["token_set_f1"] for item in per_query),
        "token_set_recall_mean": statistics.mean(item["token_set_recall"] for item in per_query),
        "numeric_token_recall_mean": statistics.mean(item["numeric_token_recall"] for item in per_query),
        "retrieval_ceiling_token_set_f1_mean": statistics.mean(
            item["retrieval_ceiling_token_set_f1"] for item in per_query
        ),
        "retrieval_ceiling_token_set_recall_mean": statistics.mean(
            item["retrieval_ceiling_token_set_recall"] for item in per_query
        ),
        "retrieval_ceiling_numeric_token_recall_mean": statistics.mean(
            item["retrieval_ceiling_numeric_token_recall"] for item in per_query
        ),
        "synthesis_efficiency_f1_mean": statistics.mean(item["synthesis_efficiency_f1"] for item in per_query),
        "synthesis_efficiency_recall_mean": statistics.mean(item["synthesis_efficiency_recall"] for item in per_query),
        "synthesis_efficiency_numeric_recall_mean": statistics.mean(
            item["synthesis_efficiency_numeric_recall"] for item in per_query
        ),
        "retrieval_shortfall_recall_mean": statistics.mean(item["retrieval_shortfall_recall"] for item in per_query),
        "generation_shortfall_recall_mean": statistics.mean(item["generation_shortfall_recall"] for item in per_query),
        "semantic_accuracy_proxy_mean": statistics.mean(item["semantic_accuracy_proxy"] for item in per_query),
        "mandatory_component_completeness_mean": statistics.mean(
            item["mandatory_component_completeness"] for item in per_query
        ),
        "contradiction_rate_mean": statistics.mean(item["contradiction_rate"] for item in per_query),
        "citation_groundedness_mean": statistics.mean(item["citation_groundedness"] for item in per_query),
        "abstention_correctness_mean": statistics.mean(item["abstention_correctness"] for item in per_query),
        "predicted_length_chars_mean": statistics.mean(item["predicted_length_chars"] for item in per_query),
        "citation_count_mean": statistics.mean(item["citation_count"] for item in per_query),
        "contract_violation_rate": all_contract_violations / float(len(per_query)),
        "vector_hits_zero_count": sum(1 for item in per_query if int(item["vector_hits"]) <= 0),
        "lexical_hits_zero_count": sum(1 for item in per_query if int(item["lexical_hits"]) <= 0),
        "bottleneck_counts": _bottleneck_counts(per_query),
    }

    payload = {
        "summary": summary,
        "baseline_diff": baseline_diff(summary, baseline_summary),
        "per_query": per_query,
    }

    dataset_stem = dataset_path.stem
    answer_path = output_dir / f"{dataset_stem}_answer_benchmark_{phase}_{mode}_{run_id}.json"
    retrieval_path = output_dir / f"{dataset_stem}_retrieval_trace_{phase}_{mode}_{run_id}.json"
    answer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    retrieval_path.write_text(json.dumps({"summary": summary, "queries": retrieval_trace}, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "per_query": per_query,
        "baseline_diff": payload["baseline_diff"],
        "answer_artifact": str(answer_path),
        "retrieval_artifact": str(retrieval_path),
    }


def _evaluate_mode_gates(
    *,
    mode: str,
    summary: dict[str, Any],
    per_query: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    failures: list[str] = []
    gate_results: dict[str, bool] = {}

    effective_hard_gates = dict(HARD_GATES)
    if mode == "lexical_only_bm25":
        effective_hard_gates.pop("vector_hits_zero_count", None)
        effective_hard_gates["lexical_hits_zero_count"] = 0
    for metric, expected in effective_hard_gates.items():
        value = summary.get(metric)
        passed = bool(value == expected)
        gate_results[f"hard::{metric}"] = passed
        if not passed:
            failures.append(f"hard gate failed: {metric} expected {expected}, got {value}")

    baseline_available = isinstance(baseline_summary, dict)
    if not baseline_available:
        for metric in IMPROVEMENT_GATES:
            gate_results[f"improvement::{metric}"] = False
            failures.append(f"improvement gate failed: baseline summary missing for {metric}")
    else:
        for metric, threshold in IMPROVEMENT_GATES.items():
            cur = float(summary.get(metric, 0.0) or 0.0)
            base = float(baseline_summary.get(metric, 0.0) or 0.0)
            delta = cur - base
            passed = delta >= threshold
            gate_results[f"improvement::{metric}"] = passed
            if not passed:
                failures.append(
                    f"improvement gate failed: {metric} delta {delta:.4f} < {threshold:.4f}"
                )

    per_query_failures: list[dict[str, Any]] = []
    for item in per_query:
        question = str(item.get("question", ""))
        completeness = float(item.get("mandatory_component_completeness", 0.0) or 0.0)
        if completeness < 0.50:
            per_query_failures.append(
                {
                    "question": question,
                    "reason": "mandatory_component_completeness < 0.50",
                    "value": completeness,
                }
            )
        if bool(item.get("numeric_heavy")):
            numeric_score = float(item.get("numeric_token_recall", 0.0) or 0.0)
            if numeric_score < 0.50:
                per_query_failures.append(
                    {
                        "question": question,
                        "reason": "numeric-heavy query numeric_token_recall < 0.50",
                        "value": numeric_score,
                    }
                )

    gate_results["per_query::coverage"] = len(per_query_failures) == 0
    if per_query_failures:
        failures.append(f"per-query gates failed for {len(per_query_failures)} query checks")

    weighted_score = 0.0
    for metric, weight in WEIGHTED_WINNER_METRICS.items():
        weighted_score += float(summary.get(metric, 0.0) or 0.0) * weight

    return {
        "passed_all": len(failures) == 0,
        "gate_results": gate_results,
        "failures": failures,
        "per_query_failures": per_query_failures,
        "weighted_score": weighted_score,
    }


def _compare_modes(mode_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not {"vector_only", "hybrid_no_reranker"}.issubset(set(mode_payloads.keys())):
        return []

    left_rows = {row["question"]: row for row in mode_payloads["vector_only"]["per_query"]}
    right_rows = {row["question"]: row for row in mode_payloads["hybrid_no_reranker"]["per_query"]}
    shared_questions = sorted(set(left_rows.keys()).intersection(set(right_rows.keys())))

    deltas: list[dict[str, Any]] = []
    for question in shared_questions:
        left = left_rows[question]
        right = right_rows[question]
        deltas.append(
            {
                "question": question,
                "token_set_f1_delta_vector_minus_hybrid": float(left["token_set_f1"]) - float(right["token_set_f1"]),
                "token_set_recall_delta_vector_minus_hybrid": float(left["token_set_recall"])
                - float(right["token_set_recall"]),
                "numeric_recall_delta_vector_minus_hybrid": float(left["numeric_token_recall"])
                - float(right["numeric_token_recall"]),
                "completeness_delta_vector_minus_hybrid": float(left["mandatory_component_completeness"])
                - float(right["mandatory_component_completeness"]),
            }
        )
    return deltas


def _winner_decision(
    *,
    mode_payloads: dict[str, dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    gate_results: dict[str, Any] = {}
    passing_modes: list[str] = []
    for mode, payload in mode_payloads.items():
        evaluation = _evaluate_mode_gates(
            mode=mode,
            summary=payload["summary"],
            per_query=payload["per_query"],
            baseline_summary=baseline_summary,
        )
        gate_results[mode] = evaluation
        if evaluation["passed_all"]:
            passing_modes.append(mode)

    selected_mode = "no_winner"
    if len(passing_modes) == 1:
        selected_mode = passing_modes[0]
    elif len(passing_modes) > 1:
        selected_mode = max(
            passing_modes,
            key=lambda mode: float(gate_results[mode].get("weighted_score", 0.0)),
        )

    return {
        "selected_mode": selected_mode,
        "gate_results": gate_results,
        "per_query_failures": {
            mode: result.get("per_query_failures", []) for mode, result in gate_results.items()
        },
    }


def _run_phase(
    *,
    phase: str,
    rows: list[dict[str, Any]],
    dataset_path: Path,
    baseline_summary: dict[str, Any] | None,
    state: str,
    jurisdiction: str,
    category: str,
    top_k: int,
    output_dir: Path,
    run_id: str,
    vector_backend: str,
    vector_db_path: Path,
    embedding_provider: str,
    rerank_provider: str | None,
    rerank_model: str | None,
    strict_providers: bool,
    selected_modes: list[str],
    manifest_path: Path,
    manifest_expected: dict[str, Any],
    lock_path: Path,
    force_rebuild: bool,
    require_manifest_consistency: bool,
) -> dict[str, Any]:
    existing_manifest = load_manifest(manifest_path)
    manifest_ok = manifest_matches(existing_manifest, manifest_expected)
    if require_manifest_consistency and not manifest_ok:
        raise SystemExit(
            f"Index manifest missing or inconsistent before {phase} run. "
            f"Expected keys: {sorted(manifest_expected.keys())}"
        )

    should_rebuild = force_rebuild or not manifest_ok
    env_vars = _runtime_env(
        vector_backend=vector_backend,
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
        rerank_provider=rerank_provider,
        rerank_model=rerank_model,
        collection_name=str(manifest_expected["collection_name"]),
        recreate_collection=should_rebuild,
    )
    _apply_env(env_vars)

    with index_build_lock(lock_path):
        engine = ComplianceEngine(ROOT)
        if strict_providers:
            _strict_provider_check(engine, required_embedding_provider=embedding_provider)

        ingest_start = perf_counter()
        ingest_result = engine.ingest(state=state, jurisdiction_type=jurisdiction)
        ingest_ms = (perf_counter() - ingest_start) * 1000
        reused_vs_rebuilt = "rebuilt" if should_rebuild else "reused"
        index_snapshot = _index_snapshot(engine, ingest_ms=ingest_ms, reused_vs_rebuilt=reused_vs_rebuilt)

        vector_dim = int(getattr(engine.vector_store, "vector_dim", 0) or 0)
        manifest_payload = build_manifest(
            state=state,
            jurisdiction=jurisdiction,
            ruleset_id=str(manifest_expected["ruleset_id"]),
            source_hash=str(manifest_expected["source_hash"]),
            parser_version=str(manifest_expected["parser_version"]),
            cleaning_version=str(manifest_expected["cleaning_version"]),
            embedding_provider=str(manifest_expected["embedding_provider"]),
            embedding_model=str(manifest_expected["embedding_model"]),
            vector_dim=vector_dim,
            clause_count=int(index_snapshot.get("vector_total_clauses", 0) or 0),
            collection_name=str(manifest_expected["collection_name"]),
        )
        save_manifest(manifest_path, manifest_payload)

        post_manifest = load_manifest(manifest_path)
        if not manifest_matches(post_manifest, manifest_expected):
            raise SystemExit("Index manifest is inconsistent after ingestion/index build.")

        quality_report_path = output_dir / f"qna_panchayat_ingestion_quality_{phase}_{run_id}.json"
        quality_report = {
            "phase": phase,
            "run_id": run_id,
            "ingestion_result": ingest_result.model_dump(),
            "index_snapshot": index_snapshot,
            "manifest": manifest_payload,
        }
        quality_report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")

    mode_payloads: dict[str, dict[str, Any]] = {}
    for mode in selected_modes:
        mode_payloads[mode] = _run_mode(
            engine=engine,
            mode=mode,
            rows=rows,
            state=state,
            jurisdiction=jurisdiction,
            category=category,
            top_k=top_k,
            run_id=run_id,
            phase=phase,
            dataset_path=dataset_path,
            baseline_summary=baseline_summary,
            index_snapshot=index_snapshot,
            output_dir=output_dir,
        )

        runtime_snapshot = {
            "run_id": run_id,
            "phase": phase,
            "retrieval_mode": mode,
            "provider_health": {
                key: asdict(value) for key, value in engine.provider_health.items()
            },
            "provider_diagnostics": list(engine.provider_diagnostics),
            "runtime_env": env_vars,
            "manifest": manifest_payload,
            "index_snapshot": index_snapshot,
        }
        runtime_path = output_dir / f"qna_panchayat_runtime_provider_snapshot_{phase}_{mode}_{run_id}.json"
        index_path = output_dir / f"qna_panchayat_index_snapshot_{phase}_{mode}_{run_id}.json"
        runtime_path.write_text(json.dumps(runtime_snapshot, indent=2), encoding="utf-8")
        index_path.write_text(json.dumps(index_snapshot, indent=2), encoding="utf-8")
        mode_payloads[mode]["runtime_artifact"] = str(runtime_path)
        mode_payloads[mode]["index_artifact"] = str(index_path)
        mode_payloads[mode]["ingestion_quality_artifact"] = str(quality_report_path)

    return {
        "phase": phase,
        "mode_payloads": mode_payloads,
        "manifest": load_manifest(manifest_path) or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KPBR QnA benchmark with disk-backed Qdrant index.")
    parser.add_argument("--dataset", default="evaluation/kpbr/qna_panchayat.json")
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="panchayat")
    parser.add_argument("--category", default="Category-II")
    parser.add_argument("--top-k", default=12, type=int)
    parser.add_argument("--baseline", default="evaluation/kpbr/qna_panchayat_answer_benchmark_final.json")
    parser.add_argument("--output-dir", default="evaluation/kpbr")
    parser.add_argument("--run-id", default=None)

    parser.add_argument("--vector-backend", default="qdrant_local", choices=["qdrant_local", "sqlite"])
    parser.add_argument("--vector-db-path", default=".cache/qdrant_kpbr")
    parser.add_argument("--embedding-provider", default="openai_embedding", choices=["openai_embedding"])
    parser.add_argument(
        "--rerank-provider",
        default=None,
        help="Optional reranker provider id (for example: cohere_reranker). If omitted, no explicit override is set.",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Optional reranker model override for the selected provider.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=[
            "vector_only",
            "hybrid_no_reranker",
            "lexical_only_bm25",
            "hybrid_reranker",
            "hybrid_graph_reranker",
            "hybrid_graph_no_reranker",
            "agentic_dynamic",
        ],
        default=None,
    )
    parser.add_argument("--skip-cold", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--reuse-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-index", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-providers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--schema-version", default="v1")
    args = parser.parse_args()

    dataset_path = (ROOT / args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")
    rows = _load_dataset(dataset_path)

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary: dict[str, Any] | None = None
    if args.baseline:
        baseline_path = (ROOT / args.baseline).resolve()
        if baseline_path.exists():
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            if isinstance(baseline_payload, dict):
                baseline_summary = baseline_payload.get("summary")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    vector_db_path = (ROOT / args.vector_db_path).resolve()
    if args.vector_backend == "sqlite":
        vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        vector_db_path.mkdir(parents=True, exist_ok=True)
    rerank_provider = (args.rerank_provider or os.getenv("PLOTMAGIC_RERANK_PROVIDER", "").strip()) or None
    rerank_model = (args.rerank_model or os.getenv("PLOTMAGIC_RERANK_MODEL", "").strip()) or None

    corpus = _corpus_profile(state=args.state, jurisdiction=args.jurisdiction)
    embedding_model = _embedding_model_for(args.embedding_provider)
    collection_name = deterministic_collection_name(
        state=args.state,
        jurisdiction=args.jurisdiction,
        ruleset_id=corpus["ruleset_id"],
        embedding_model=embedding_model,
        schema_version=args.schema_version,
    )
    source_hash = compute_sources_hash(corpus["source_paths"])

    if args.vector_backend == "sqlite":
        manifest_dir = vector_db_path.parent
    else:
        manifest_dir = vector_db_path
    manifest_path = manifest_dir / "manifest.json"
    lock_path = manifest_dir / "index.lock"
    manifest_expected = {
        "state": args.state,
        "jurisdiction": args.jurisdiction,
        "ruleset_id": corpus["ruleset_id"],
        "source_hash": source_hash,
        "parser_version": corpus["parser_version"],
        "cleaning_version": CLEANING_VERSION,
        "embedding_provider": args.embedding_provider,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
    }

    if args.retrieval_mode:
        selected_modes = [args.retrieval_mode]
    else:
        selected_modes = ["vector_only", "hybrid_no_reranker"]
        if rerank_provider and rerank_provider != "no_reranker":
            selected_modes.append("hybrid_reranker")

    cold = {"phase": "cold", "mode_payloads": {}, "manifest": load_manifest(manifest_path) or {}}
    if not args.skip_cold:
        cold = _run_phase(
            phase="cold",
            rows=rows,
            dataset_path=dataset_path,
            baseline_summary=baseline_summary,
            state=args.state,
            jurisdiction=args.jurisdiction,
            category=args.category,
            top_k=args.top_k,
            output_dir=output_dir,
            run_id=run_id,
            vector_backend=args.vector_backend,
            vector_db_path=vector_db_path,
            embedding_provider=args.embedding_provider,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            strict_providers=args.strict_providers,
            selected_modes=selected_modes,
            manifest_path=manifest_path,
            manifest_expected=manifest_expected,
            lock_path=lock_path,
            force_rebuild=True,
            require_manifest_consistency=False,
        )

    warm_force_rebuild = bool(args.rebuild_index or (not args.reuse_index))
    warm = _run_phase(
        phase="warm",
        rows=rows,
        dataset_path=dataset_path,
        baseline_summary=baseline_summary,
        state=args.state,
        jurisdiction=args.jurisdiction,
        category=args.category,
        top_k=args.top_k,
        output_dir=output_dir,
        run_id=run_id,
        vector_backend=args.vector_backend,
        vector_db_path=vector_db_path,
        embedding_provider=args.embedding_provider,
        rerank_provider=rerank_provider,
        rerank_model=rerank_model,
        strict_providers=args.strict_providers,
        selected_modes=selected_modes,
        manifest_path=manifest_path,
        manifest_expected=manifest_expected,
        lock_path=lock_path,
        force_rebuild=warm_force_rebuild,
        require_manifest_consistency=not warm_force_rebuild,
    )

    winner_source = warm["mode_payloads"]
    winner_report = _winner_decision(mode_payloads=winner_source, baseline_summary=baseline_summary)
    comparison_payload = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cold": cold["mode_payloads"],
        "warm": warm["mode_payloads"],
        "per_query_metric_deltas": _compare_modes(winner_source),
        "winner": winner_report,
    }

    comparison_path = output_dir / f"qna_panchayat_mode_comparison_{run_id}.json"
    comparison_path.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

    winner_path = output_dir / "winner_report.json"
    winner_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "selected_mode": winner_report["selected_mode"],
                "gate_results": winner_report["gate_results"],
                "per_query_failures": winner_report["per_query_failures"],
                "comparison_artifact": str(comparison_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "selected_mode": winner_report["selected_mode"],
                "comparison_artifact": str(comparison_path),
                "winner_report": str(winner_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
