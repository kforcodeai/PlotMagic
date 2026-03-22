from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import threading
from typing import Any

from src.agentic.compliance_brief_composer import ComplianceBriefComposer
from src.api.service import ComplianceEngine
from src.models.schemas import GroundingReportPayload, QueryRequest
from src.providers import ProviderFactory, build_default_registry, load_providers_config


REQUIRED_DATASET_KEYS = {
    "id",
    "query",
    "query_plan",
    "ground_truth_chunks",
    "final_answer",
    "answer_evidence",
}

RULE_LINE_PATTERN = re.compile(r"^\s*(?:##\s*)?_?(\d{1,3})\.(?!\d)")
LINE_RANGE_PATTERN = re.compile(r"^(?P<source>.+):(?P<start>\d+)-(?P<end>\d+)$")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")

RETRIEVAL_SCORE_WEIGHTS = {
    "hop_coverage_at_k": 0.30,
    "rule_recall_at_k": 0.25,
    "snippet_recall_at_k": 0.20,
    "mrr": 0.15,
    "one_minus_zero_hit_rate": 0.10,
}

SYNTHESIS_SCORE_WEIGHTS = {
    "token_set_f1_mean": 0.35,
    "token_set_recall_mean": 0.25,
    "numeric_token_recall_mean": 0.25,
    "mandatory_component_completeness_mean": 0.15,
}

LEGAL_GATES = {
    "citation_groundedness_mean": 1.0,
    "contract_violation_rate": 0.0,
    "hop_coverage_at_k_mean": 0.90,
    "zero_hit_rate": 0.0,
}
LLM_COMPOSE_TIMEOUT_S = float(os.getenv("PLOTMAGIC_LLM_COMPOSE_TIMEOUT_S", "6.0"))

STOPWORDS = {
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
}


class MultiHopEvalError(ValueError):
    pass


@dataclass(slots=True)
class CitationRef:
    source_file: str
    line_start: int
    line_end: int

    @property
    def canonical(self) -> str:
        return f"{self.source_file}:{self.line_start}-{self.line_end}"


@dataclass(slots=True)
class GroundTruthChunk:
    citation: CitationRef
    snippet: str
    normalized_snippet: str
    snippet_tokens: set[str]


@dataclass(slots=True)
class HopSpec:
    hop: int
    sub_query: str
    query_tokens: set[str]


@dataclass(slots=True)
class MultiHopRecord:
    record_id: str
    query: str
    final_answer: str
    query_type: str
    source_document: str
    ground_truth_chunks: list[GroundTruthChunk]
    answer_evidence: list[CitationRef]
    query_plan: list[HopSpec]


@dataclass(slots=True)
class GoldLabels:
    record_id: str
    gold_rules: set[str]
    chunk_canonicals: list[str]
    chunk_tokens: list[set[str]]
    chunk_snippets: list[str]
    hop_targets: dict[int, set[int]]


@dataclass(slots=True)
class RetrievalStrategy:
    name: str
    retrieval_mode: str
    lexical_algorithm: str
    reranker_provider: str
    agentic_llm_provider: str
    fallback_policy: str
    category_policy: str
    top_k: int
    candidate_pool_factor: float
    min_evidence_score: float
    primary_signal: str


@dataclass(slots=True)
class SynthesisStrategy:
    name: str
    llm_provider_id: str


@dataclass(slots=True)
class RetrievalQueryArtifact:
    query_id: str
    query: str
    expected_answer: str
    retrieved_texts: list[str]
    top_rules_cited: list[str]
    vector_hits: int
    lexical_hits: int
    structured_hits: int
    fallback_stage: str
    zero_hit: bool
    rule_hit_at_k: float
    rule_recall_at_k: float
    hop_coverage_at_k: float
    snippet_recall_at_k: float
    mrr: float
    retrieved_context_extractive_f1: float
    claim_citations: dict[str, list[str]]
    claim_texts: dict[str, str]
    claim_scores: dict[str, float]
    unresolved: list[str]
    grounding: dict[str, Any]
    verdict: str
    strategy_name: str


@dataclass(slots=True)
class EndToEndQueryMetrics:
    query_id: str
    query: str
    retrieval_strategy: str
    synthesis_strategy: str
    token_set_f1: float
    token_set_recall: float
    numeric_token_recall: float
    mandatory_component_completeness: float
    citation_groundedness: float
    contract_violation: float
    oracle_context_f1: float
    retrieved_context_extractive_f1: float
    retrieval_gap: float
    synthesis_gap: float
    bottleneck_bucket: str
    zero_hit: bool
    hop_coverage_at_k: float
    fallback_stage: str
    vector_hits: int
    lexical_hits: int
    structured_hits: int


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _stable_sort(rows: list[dict[str, Any]], score_key: str, secondary_keys: list[str] | None = None) -> list[dict[str, Any]]:
    extra = secondary_keys or []
    return sorted(
        rows,
        key=lambda row: tuple([-(float(row.get(score_key, 0.0) or 0.0))] + [str(row.get(key, "")) for key in extra]),
    )


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    cleaned = re.sub(r"[^a-z0-9%./()-]+", " ", lowered)
    return " ".join(cleaned.split())


def normalize_tokens(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token and token not in STOPWORDS}


def set_f1(pred: str, ref: str) -> float:
    pred_set = normalize_tokens(pred)
    ref_set = normalize_tokens(ref)
    if not pred_set or not ref_set:
        return 0.0
    overlap = len(pred_set.intersection(ref_set))
    if overlap <= 0:
        return 0.0
    precision = overlap / float(len(pred_set))
    recall = overlap / float(len(ref_set))
    return (2.0 * precision * recall) / (precision + recall)


def set_recall(pred: str, ref: str) -> float:
    pred_set = normalize_tokens(pred)
    ref_set = normalize_tokens(ref)
    if not ref_set:
        return 0.0
    return len(pred_set.intersection(ref_set)) / float(len(ref_set))


def numeric_recall(pred: str, ref: str) -> float:
    pred_numbers = set(NUMBER_PATTERN.findall(pred))
    ref_numbers = set(NUMBER_PATTERN.findall(ref))
    if not ref_numbers:
        return 0.0
    return len(pred_numbers.intersection(ref_numbers)) / float(len(ref_numbers))


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


def parse_citation_ref(citation: str) -> CitationRef:
    text = str(citation).strip()
    m = LINE_RANGE_PATTERN.match(text)
    if not m:
        raise MultiHopEvalError(f"Malformed citation: {citation}")
    source_file = m.group("source").strip()
    line_start = int(m.group("start"))
    line_end = int(m.group("end"))
    if line_start <= 0 or line_end < line_start:
        raise MultiHopEvalError(f"Invalid citation line range: {citation}")
    return CitationRef(source_file=source_file, line_start=line_start, line_end=line_end)


def _parse_query_plan(raw_plan: Any, *, record_id: str) -> list[HopSpec]:
    if not isinstance(raw_plan, list) or not raw_plan:
        raise MultiHopEvalError(f"{record_id}: query_plan must be non-empty list")
    hops: list[HopSpec] = []
    for idx, item in enumerate(raw_plan):
        if not isinstance(item, dict):
            raise MultiHopEvalError(f"{record_id}: query_plan item #{idx} is not an object")
        hop = int(item.get("hop", idx + 1))
        sub_query = str(item.get("sub_query", "")).strip()
        if not sub_query:
            raise MultiHopEvalError(f"{record_id}: query_plan hop {hop} missing sub_query")
        hops.append(HopSpec(hop=hop, sub_query=sub_query, query_tokens=normalize_tokens(sub_query)))
    return hops


def load_multihop_dataset(dataset_path: Path) -> list[MultiHopRecord]:
    rows: list[dict[str, Any]] = []
    text = dataset_path.read_text(encoding="utf-8")
    if dataset_path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise MultiHopEvalError("Dataset must be a JSON list or JSONL stream")
        rows = payload

    records: list[MultiHopRecord] = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MultiHopEvalError(f"Row #{idx} is not an object")
        missing = REQUIRED_DATASET_KEYS.difference(set(row.keys()))
        if missing:
            raise MultiHopEvalError(f"Row #{idx} missing required keys: {sorted(missing)}")

        record_id = str(row["id"]).strip()
        if not record_id:
            raise MultiHopEvalError(f"Row #{idx} has empty id")
        if record_id in seen_ids:
            raise MultiHopEvalError(f"Duplicate record id: {record_id}")
        seen_ids.add(record_id)

        query = str(row["query"]).strip()
        final_answer = str(row["final_answer"]).strip()
        if not query or not final_answer:
            raise MultiHopEvalError(f"{record_id}: query/final_answer cannot be empty")

        raw_chunks = row["ground_truth_chunks"]
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise MultiHopEvalError(f"{record_id}: ground_truth_chunks must be non-empty list")
        chunks: list[GroundTruthChunk] = []
        for cidx, item in enumerate(raw_chunks):
            if not isinstance(item, dict):
                raise MultiHopEvalError(f"{record_id}: chunk #{cidx} is not an object")
            if "citation" not in item or "snippet" not in item:
                raise MultiHopEvalError(f"{record_id}: chunk #{cidx} missing citation/snippet")
            citation = parse_citation_ref(str(item["citation"]))
            snippet = str(item["snippet"]).strip()
            if not snippet:
                raise MultiHopEvalError(f"{record_id}: chunk #{cidx} has empty snippet")
            chunks.append(
                GroundTruthChunk(
                    citation=citation,
                    snippet=snippet,
                    normalized_snippet=normalize_text(snippet),
                    snippet_tokens=normalize_tokens(snippet),
                )
            )

        raw_answer_evidence = row["answer_evidence"]
        if not isinstance(raw_answer_evidence, list) or not raw_answer_evidence:
            raise MultiHopEvalError(f"{record_id}: answer_evidence must be non-empty list")
        answer_evidence = [parse_citation_ref(str(item)) for item in raw_answer_evidence]

        query_plan = _parse_query_plan(row["query_plan"], record_id=record_id)
        records.append(
            MultiHopRecord(
                record_id=record_id,
                query=query,
                final_answer=final_answer,
                query_type=str(row.get("query_type", "multi_hop_complex")),
                source_document=str(row.get("source_document", "")).strip(),
                ground_truth_chunks=chunks,
                answer_evidence=answer_evidence,
                query_plan=query_plan,
            )
        )

    if not records:
        raise MultiHopEvalError("Dataset has no records")
    return records


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
            if 1 <= value <= 300 and value > max_seen_rule:
                current_rule = value
                max_seen_rule = value
        line_to_rule[lineno] = current_rule
    return line_to_rule


def map_rule_number_for_ref(citation: CitationRef, line_to_rule: dict[int, int | None]) -> str | None:
    value = line_to_rule.get(citation.line_start)
    if value is None:
        return None
    return str(value)


def build_gold_labels(records: list[MultiHopRecord], root: Path) -> dict[str, GoldLabels]:
    source_rule_maps: dict[str, dict[int, int | None]] = {}
    labels: dict[str, GoldLabels] = {}
    for record in records:
        source = record.source_document
        if not source:
            source = record.ground_truth_chunks[0].citation.source_file
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = (root / source_path).resolve()
        source_key = str(source_path)
        if source_key not in source_rule_maps:
            if not source_path.exists():
                raise MultiHopEvalError(f"Source file not found for record {record.record_id}: {source_path}")
            source_rule_maps[source_key] = build_line_to_rule(source_path)
        rule_map = source_rule_maps[source_key]

        gold_rules: set[str] = set()
        chunk_canonicals: list[str] = []
        chunk_tokens: list[set[str]] = []
        chunk_snippets: list[str] = []
        for chunk in record.ground_truth_chunks:
            gold_rule = map_rule_number_for_ref(chunk.citation, rule_map)
            if gold_rule:
                gold_rules.add(gold_rule)
            chunk_canonicals.append(chunk.citation.canonical)
            chunk_tokens.append(set(chunk.snippet_tokens))
            chunk_snippets.append(chunk.snippet)

        if not chunk_canonicals:
            raise MultiHopEvalError(f"{record.record_id}: no gold chunks")

        hop_targets: dict[int, set[int]] = {}
        for hop in record.query_plan:
            overlaps: list[int] = []
            for cidx, token_set in enumerate(chunk_tokens):
                overlap = len(hop.query_tokens.intersection(token_set))
                overlaps.append(overlap)
            max_overlap = max(overlaps) if overlaps else 0
            if max_overlap <= 0:
                hop_targets[hop.hop] = set(range(len(chunk_tokens)))
            else:
                hop_targets[hop.hop] = {idx for idx, score in enumerate(overlaps) if score == max_overlap}

        labels[record.record_id] = GoldLabels(
            record_id=record.record_id,
            gold_rules=gold_rules,
            chunk_canonicals=chunk_canonicals,
            chunk_tokens=chunk_tokens,
            chunk_snippets=chunk_snippets,
            hop_targets=hop_targets,
        )
    return labels


def _similarity_recall(gold_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not gold_tokens:
        return 0.0
    return len(gold_tokens.intersection(candidate_tokens)) / float(len(gold_tokens))


def match_gold_chunks(gold: GoldLabels, retrieved_texts: list[str], threshold: float = 0.60) -> set[int]:
    candidate_norm = [normalize_text(text) for text in retrieved_texts if str(text).strip()]
    candidate_tokens = [set(item.split()) for item in candidate_norm if item]
    matched: set[int] = set()
    for idx, gold_snippet in enumerate(gold.chunk_snippets):
        gold_norm = normalize_text(gold_snippet)
        gold_tokens = set(gold_norm.split())
        if not gold_norm:
            continue
        for ctext, ctokens in zip(candidate_norm, candidate_tokens):
            if len(gold_norm) >= 30 and gold_norm in ctext:
                matched.add(idx)
                break
            if _similarity_recall(gold_tokens, ctokens) >= threshold:
                matched.add(idx)
                break
    return matched


def compute_retrieval_quality(
    *,
    gold: GoldLabels,
    retrieved_texts: list[str],
    ranked_rules: list[str],
    zero_hit: bool,
) -> dict[str, float]:
    matched_chunks = match_gold_chunks(gold, retrieved_texts)

    hit = 0.0
    if gold.gold_rules:
        if any(rule in gold.gold_rules for rule in ranked_rules):
            hit = 1.0
    else:
        hit = 1.0 if matched_chunks else 0.0

    if gold.gold_rules:
        unique_ranked = set(ranked_rules)
        rule_recall = len(unique_ranked.intersection(gold.gold_rules)) / float(len(gold.gold_rules))
    else:
        rule_recall = 0.0

    mrr = 0.0
    if gold.gold_rules:
        for idx, rule in enumerate(ranked_rules, start=1):
            if rule in gold.gold_rules:
                mrr = 1.0 / float(idx)
                break

    snippet_recall = len(matched_chunks) / float(len(gold.chunk_canonicals))
    hop_covered = 0
    for hop_targets in gold.hop_targets.values():
        if matched_chunks.intersection(hop_targets):
            hop_covered += 1
    hop_coverage = hop_covered / float(max(1, len(gold.hop_targets)))
    return {
        "rule_hit_at_k": hit,
        "rule_recall_at_k": rule_recall,
        "hop_coverage_at_k": hop_coverage,
        "snippet_recall_at_k": snippet_recall,
        "mrr": mrr,
        "zero_hit": 1.0 if zero_hit else 0.0,
    }


def _response_to_text(payload: dict[str, Any]) -> str:
    final_answer = payload.get("final_answer") or {}
    if isinstance(final_answer, dict):
        lines: list[str] = []
        summary = str(final_answer.get("short_summary", "")).strip()
        if summary:
            lines.append(summary)
        for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
            section = final_answer.get(section_name, [])
            if not isinstance(section, list):
                continue
            for item in section:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if text:
                    lines.append(text)
        if lines:
            return "\n".join(lines)
    return ""


def _brief_to_text(brief: Any) -> str:
    lines: list[str] = []
    summary = str(getattr(brief, "short_summary", "")).strip()
    if summary:
        lines.append(summary)
    for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
        section = getattr(brief, section_name, [])
        if not isinstance(section, list):
            continue
        for item in section:
            text = str(getattr(item, "text", "")).strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def _extract_retrieval_diagnostics(events: list[dict[str, Any]]) -> dict[str, Any]:
    passes: list[dict[str, Any]] = []
    for event in events:
        if event.get("status") != "ok":
            continue
        step = str(event.get("step") or "")
        if not step.startswith("tool.retrieve.pass"):
            continue
        details = event.get("details") or {}
        if isinstance(details, dict):
            passes.append(details)
    if not passes:
        return {
            "vector_hits": 0,
            "lexical_hits": 0,
            "structured_hits": 0,
            "candidate_count": 0,
            "fallback_stage": "missing",
            "top_rules_cited": [],
            "iteration_count": 0,
        }
    last = passes[-1]
    top_rules: list[str] = []
    for item in last.get("top_rules_cited", []):
        text = str(item).strip()
        if text:
            top_rules.append(text)
    return {
        "vector_hits": int(sum(float(row.get("vector_hits", 0) or 0.0) for row in passes)),
        "lexical_hits": int(sum(float(row.get("lexical_hits", 0) or 0.0) for row in passes)),
        "structured_hits": int(sum(float(row.get("structured_hits", 0) or 0.0) for row in passes)),
        "candidate_count": int(max(float(row.get("candidate_count", 0) or 0.0) for row in passes)),
        "fallback_stage": str(last.get("fallback_stage") or ""),
        "top_rules_cited": top_rules,
        "iteration_count": len(passes),
    }


def _extract_retrieved_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for item in payload.get("evidence_matrix", []):
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if text:
                texts.append(text)
    for citation in payload.get("citations", []):
        if isinstance(citation, dict):
            quote = str(citation.get("quote_excerpt", "")).strip()
            if quote:
                texts.append(quote)
    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        norm = normalize_text(text)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(text)
    return deduped


def _extract_ranked_rules(payload: dict[str, Any], top_rules: list[str]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for value in top_rules:
        if value and value not in seen:
            seen.add(value)
            ranked.append(value)
    for citation in payload.get("citations", []):
        if not isinstance(citation, dict):
            continue
        raw = str(citation.get("rule_number", "")).strip()
        if not raw:
            continue
        m = re.search(r"\d+", raw)
        if not m:
            continue
        value = m.group(0)
        if value in seen:
            continue
        seen.add(value)
        ranked.append(value)
    return ranked


def _extract_claim_context(payload: dict[str, Any]) -> dict[str, Any]:
    claim_citations_raw = payload.get("claim_citations") or {}
    claim_citations: dict[str, list[str]] = {}
    if isinstance(claim_citations_raw, dict):
        for claim_id, ids in claim_citations_raw.items():
            if not isinstance(ids, list):
                continue
            claim_citations[str(claim_id)] = [str(item) for item in ids if str(item).strip()]

    claim_texts: dict[str, str] = {}
    claim_scores: dict[str, float] = {}
    for item in payload.get("evidence_matrix", []):
        if not isinstance(item, dict):
            continue
        claim_id = str(item.get("claim_id", "")).strip()
        text = str(item.get("text", "")).strip()
        if claim_id and text and claim_id not in claim_texts:
            claim_texts[claim_id] = text
        if claim_id:
            score = float((item.get("scores") or {}).get("rrf_score", 0.0) or 0.0)
            if score > claim_scores.get(claim_id, 0.0):
                claim_scores[claim_id] = score

    grounding = payload.get("grounding")
    grounding_payload = (
        grounding
        if isinstance(grounding, dict)
        else {
            "supported_claim_count": 0,
            "unsupported_claim_count": 0,
            "conflicting_claim_count": 0,
            "missing_topics": [],
            "abstained": False,
        }
    )
    return {
        "claim_citations": claim_citations,
        "claim_texts": claim_texts,
        "claim_scores": claim_scores,
        "unresolved": [str(item) for item in payload.get("unresolved", [])],
        "grounding": grounding_payload,
        "verdict": str(payload.get("verdict") or "depends"),
    }


def _brief_citation_groundedness(brief: Any, claim_citations: dict[str, list[str]]) -> float:
    checks = 0
    passed = 0
    for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
        section = getattr(brief, section_name, [])
        if not isinstance(section, list):
            continue
        for item in section:
            checks += 1
            claim_id = str(getattr(item, "claim_id", "")).strip()
            citations = list(getattr(item, "citation_ids", []))
            if not claim_id or not citations:
                continue
            allowed = set(claim_citations.get(claim_id, []))
            if allowed and all(str(citation) in allowed for citation in citations):
                passed += 1
    if checks == 0:
        return 0.0
    return passed / float(checks)


def _brief_contract_violation(
    brief: Any,
    *,
    orchestrated_verdict: str,
    claim_citations: dict[str, list[str]],
) -> float:
    if str(getattr(brief, "verdict", "")) != orchestrated_verdict:
        return 1.0
    summary = str(getattr(brief, "short_summary", "")).strip()
    if orchestrated_verdict != "insufficient_evidence" and not summary:
        return 1.0
    for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
        section = getattr(brief, section_name, [])
        if not isinstance(section, list):
            return 1.0
        for item in section:
            claim_id = str(getattr(item, "claim_id", "")).strip()
            citations = list(getattr(item, "citation_ids", []))
            if not claim_id or not citations:
                return 1.0
            allowed = set(claim_citations.get(claim_id, []))
            if not allowed or any(str(citation) not in allowed for citation in citations):
                return 1.0
    return 0.0


def _retrieval_composite(summary: dict[str, float]) -> float:
    return (
        RETRIEVAL_SCORE_WEIGHTS["hop_coverage_at_k"] * summary.get("hop_coverage_at_k_mean", 0.0)
        + RETRIEVAL_SCORE_WEIGHTS["rule_recall_at_k"] * summary.get("rule_recall_at_k_mean", 0.0)
        + RETRIEVAL_SCORE_WEIGHTS["snippet_recall_at_k"] * summary.get("snippet_recall_at_k_mean", 0.0)
        + RETRIEVAL_SCORE_WEIGHTS["mrr"] * summary.get("mrr_mean", 0.0)
        + RETRIEVAL_SCORE_WEIGHTS["one_minus_zero_hit_rate"] * (1.0 - summary.get("zero_hit_rate", 1.0))
    )


def _synthesis_composite(summary: dict[str, float]) -> float:
    return (
        SYNTHESIS_SCORE_WEIGHTS["token_set_f1_mean"] * summary.get("token_set_f1_mean", 0.0)
        + SYNTHESIS_SCORE_WEIGHTS["token_set_recall_mean"] * summary.get("token_set_recall_mean", 0.0)
        + SYNTHESIS_SCORE_WEIGHTS["numeric_token_recall_mean"] * summary.get("numeric_token_recall_mean", 0.0)
        + SYNTHESIS_SCORE_WEIGHTS["mandatory_component_completeness_mean"]
        * summary.get("mandatory_component_completeness_mean", 0.0)
    )


def classify_bottleneck(oracle_context_f1: float, retrieved_context_extractive_f1: float, end_to_end_f1: float) -> str:
    retrieval_gap = max(0.0, oracle_context_f1 - retrieved_context_extractive_f1)
    synthesis_gap = max(0.0, retrieved_context_extractive_f1 - end_to_end_f1)
    if retrieval_gap >= 0.05 and retrieval_gap >= (synthesis_gap * 1.20):
        return "retrieval_bottleneck"
    if synthesis_gap >= 0.05 and synthesis_gap >= (retrieval_gap * 1.20):
        return "synthesis_bottleneck"
    return "mixed"


def build_retrieval_strategies(tier: str, default_top_k: int) -> list[RetrievalStrategy]:
    core = [
        RetrievalStrategy(
            name="vector_only",
            retrieval_mode="vector_only",
            lexical_algorithm="bm25",
            reranker_provider="no_reranker",
            agentic_llm_provider="no_llm",
            fallback_policy="relax_category_then_occupancy",
            category_policy="soft",
            top_k=default_top_k,
            candidate_pool_factor=4.0,
            min_evidence_score=0.05,
            primary_signal="vector",
        ),
        RetrievalStrategy(
            name="hybrid_no_reranker",
            retrieval_mode="hybrid_no_reranker",
            lexical_algorithm="bm25",
            reranker_provider="no_reranker",
            agentic_llm_provider="no_llm",
            fallback_policy="relax_category_then_occupancy",
            category_policy="soft",
            top_k=default_top_k,
            candidate_pool_factor=4.0,
            min_evidence_score=0.05,
            primary_signal="vector",
        ),
        RetrievalStrategy(
            name="lexical_only_bm25",
            retrieval_mode="lexical_only_bm25",
            lexical_algorithm="bm25",
            reranker_provider="no_reranker",
            agentic_llm_provider="no_llm",
            fallback_policy="strict_only",
            category_policy="soft",
            top_k=default_top_k,
            candidate_pool_factor=4.0,
            min_evidence_score=0.05,
            primary_signal="lexical",
        ),
        RetrievalStrategy(
            name="tfidf_only",
            retrieval_mode="lexical_only_bm25",
            lexical_algorithm="tfidf",
            reranker_provider="no_reranker",
            agentic_llm_provider="no_llm",
            fallback_policy="strict_only",
            category_policy="soft",
            top_k=default_top_k,
            candidate_pool_factor=4.0,
            min_evidence_score=0.05,
            primary_signal="lexical",
        ),
        RetrievalStrategy(
            name="hybrid_dense_bm25",
            retrieval_mode="hybrid_no_reranker",
            lexical_algorithm="bm25",
            reranker_provider="no_reranker",
            agentic_llm_provider="no_llm",
            fallback_policy="relax_category_then_occupancy",
            category_policy="soft",
            top_k=default_top_k,
            candidate_pool_factor=6.0,
            min_evidence_score=0.05,
            primary_signal="vector",
        ),
        RetrievalStrategy(
            name="structure_hybrid_graph_llm_rerank",
            retrieval_mode="hybrid_graph_reranker",
            lexical_algorithm="bm25",
            reranker_provider="openai_llm_reranker",
            agentic_llm_provider="no_llm",
            fallback_policy="relax_category_then_occupancy",
            category_policy="soft",
            top_k=default_top_k,
            candidate_pool_factor=6.0,
            min_evidence_score=0.05,
            primary_signal="vector",
        ),
        RetrievalStrategy(
            name="agentic_dynamic_llm",
            retrieval_mode="agentic_dynamic",
            lexical_algorithm="bm25",
            reranker_provider="openai_llm_reranker",
            agentic_llm_provider="openai_responses_llm",
            fallback_policy="relax_category_then_occupancy",
            category_policy="soft",
            top_k=default_top_k,
            candidate_pool_factor=6.0,
            min_evidence_score=0.05,
            primary_signal="vector",
        ),
    ]
    if tier == "core":
        return core

    exhaustive: list[RetrievalStrategy] = []
    fallback_variants = ["strict_only", "relax_category", "relax_category_then_occupancy"]
    category_variants = ["strict", "soft"]
    top_k_variants = [8, 12, 16, 20]
    for item in core:
        for fallback in fallback_variants:
            for category in category_variants:
                for top_k in top_k_variants:
                    name = f"{item.name}__k{top_k}__cat-{category}__fb-{fallback}"
                    exhaustive.append(
                        RetrievalStrategy(
                            name=name,
                            retrieval_mode=item.retrieval_mode,
                            lexical_algorithm=item.lexical_algorithm,
                            reranker_provider=item.reranker_provider,
                            agentic_llm_provider=item.agentic_llm_provider,
                            fallback_policy=fallback,
                            category_policy=category,
                            top_k=top_k,
                            candidate_pool_factor=item.candidate_pool_factor,
                            min_evidence_score=item.min_evidence_score,
                            primary_signal=item.primary_signal,
                        )
                    )
    return exhaustive


def build_synthesis_strategies() -> list[SynthesisStrategy]:
    return [
        SynthesisStrategy(name="deterministic_draft_only", llm_provider_id="no_llm"),
        SynthesisStrategy(name="llm_structured", llm_provider_id="openai_responses_llm"),
    ]


def _sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


@contextmanager
def temporary_env(overrides: dict[str, str]):
    original: dict[str, str | None] = {}
    for key, value in overrides.items():
        original[key] = os.getenv(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in original.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _strict_provider_check(
    engine: ComplianceEngine,
    *,
    required_embedding_provider: str,
) -> None:
    actual_embedding_provider = str(getattr(engine.embedding_provider, "provider_id", ""))
    if actual_embedding_provider != required_embedding_provider:
        raise MultiHopEvalError(
            "Strict provider check failed: expected embedding provider "
            f"'{required_embedding_provider}', got '{actual_embedding_provider}'."
        )
    degraded = [
        message
        for message in engine.provider_diagnostics
        if "Embedding provider" in message and "falling back" in message
    ]
    if degraded:
        raise MultiHopEvalError(
            "Strict provider check failed: embedding provider fell back during runtime. "
            + " | ".join(degraded)
        )


def _build_composer(
    *,
    root: Path,
    llm_provider_id: str,
    strict_providers: bool,
) -> ComplianceBriefComposer:
    with temporary_env({"PLOTMAGIC_LLM_PROVIDER": llm_provider_id}):
        cfg = load_providers_config(root / "config" / "providers.yaml")
        registry = build_default_registry()
        factory = ProviderFactory(registry, cfg)
        llm_provider = factory.create_llm_provider()
        if strict_providers and llm_provider_id == "openai_responses_llm":
            if str(getattr(llm_provider, "provider_id", "")) != "openai_responses_llm":
                raise MultiHopEvalError(
                    "Strict provider check failed: LLM strategy requested openai_responses_llm "
                    f"but got '{getattr(llm_provider, 'provider_id', '')}'."
                )
            degraded = [
                message
                for message in factory.diagnostics
                if "LLM provider" in message and "falling back" in message
            ]
            if degraded:
                raise MultiHopEvalError(
                    "Strict provider check failed: LLM provider fell back during runtime. "
                    + " | ".join(degraded)
                )
    return ComplianceBriefComposer(llm_provider=llm_provider)


def _compose_answer(
    composer: ComplianceBriefComposer,
    *,
    query: str,
    verdict: str,
    claim_citations: dict[str, list[str]],
    claim_texts: dict[str, str],
    claim_scores: dict[str, float],
    unresolved: list[str],
    grounding_payload: dict[str, Any],
) -> Any:
    grounding = GroundingReportPayload.model_validate(grounding_payload)
    return composer.compose(
        query=query,
        verdict=verdict,
        claim_citations=claim_citations,
        claim_texts=claim_texts,
        claim_scores=claim_scores,
        unresolved=unresolved,
        grounding=grounding,
    )


def _compose_answer_with_timeout(
    composer: ComplianceBriefComposer,
    *,
    fallback_composer: ComplianceBriefComposer,
    timeout_s: float,
    query: str,
    verdict: str,
    claim_citations: dict[str, list[str]],
    claim_texts: dict[str, str],
    claim_scores: dict[str, float],
    unresolved: list[str],
    grounding_payload: dict[str, Any],
) -> Any:
    result_box: dict[str, Any] = {}
    error_box: dict[str, Exception] = {}

    def _run_compose() -> None:
        try:
            result_box["brief"] = _compose_answer(
                composer,
                query=query,
                verdict=verdict,
                claim_citations=claim_citations,
                claim_texts=claim_texts,
                claim_scores=claim_scores,
                unresolved=unresolved,
                grounding_payload=grounding_payload,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            error_box["error"] = exc

    worker = threading.Thread(target=_run_compose, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive() or "error" in error_box:
        return _compose_answer(
            fallback_composer,
            query=query,
            verdict=verdict,
            claim_citations=claim_citations,
            claim_texts=claim_texts,
            claim_scores=claim_scores,
            unresolved=unresolved,
            grounding_payload=grounding_payload,
        )
    brief = result_box.get("brief")
    if brief is None:
        return _compose_answer(
            fallback_composer,
            query=query,
            verdict=verdict,
            claim_citations=claim_citations,
            claim_texts=claim_texts,
            claim_scores=claim_scores,
            unresolved=unresolved,
            grounding_payload=grounding_payload,
        )
    return brief


def _oracle_claims(record: MultiHopRecord) -> tuple[dict[str, list[str]], dict[str, str], dict[str, float]]:
    claim_citations: dict[str, list[str]] = {}
    claim_texts: dict[str, str] = {}
    claim_scores: dict[str, float] = {}
    for idx, chunk in enumerate(record.ground_truth_chunks):
        claim_id = f"oracle_{idx + 1}"
        claim_citations[claim_id] = [chunk.citation.canonical]
        claim_texts[claim_id] = chunk.snippet
        claim_scores[claim_id] = 1.0 - (0.01 * idx)
    return claim_citations, claim_texts, claim_scores


def _trim_claim_context(
    *,
    claim_citations: dict[str, list[str]],
    claim_texts: dict[str, str],
    claim_scores: dict[str, float],
    max_claims: int = 18,
    max_chars_per_claim: int = 650,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, float]]:
    if not claim_texts:
        return {}, {}, {}
    ranked_ids = sorted(
        claim_texts.keys(),
        key=lambda claim_id: (
            -float(claim_scores.get(claim_id, 0.0)),
            claim_id,
        ),
    )
    keep_ids: list[str] = []
    for claim_id in ranked_ids:
        text = str(claim_texts.get(claim_id, "")).strip()
        citations = [str(item) for item in claim_citations.get(claim_id, []) if str(item).strip()]
        if not text or not citations:
            continue
        keep_ids.append(claim_id)
        if len(keep_ids) >= max_claims:
            break
    trimmed_citations: dict[str, list[str]] = {}
    trimmed_texts: dict[str, str] = {}
    trimmed_scores: dict[str, float] = {}
    for claim_id in keep_ids:
        text = " ".join(str(claim_texts.get(claim_id, "")).split())
        trimmed_texts[claim_id] = text[:max_chars_per_claim]
        trimmed_citations[claim_id] = [
            str(item)
            for item in claim_citations.get(claim_id, [])[:4]
            if str(item).strip()
        ]
        trimmed_scores[claim_id] = float(claim_scores.get(claim_id, 0.0))
    return trimmed_citations, trimmed_texts, trimmed_scores


def _base_env(
    *,
    vector_backend: str,
    vector_db_path: Path,
    embedding_provider: str,
) -> dict[str, str]:
    return {
        "PLOTMAGIC_VECTOR_BACKEND": vector_backend,
        "PLOTMAGIC_VECTOR_DB_PATH": str(vector_db_path),
        "PLOTMAGIC_EMBEDDING_PROVIDER": embedding_provider,
        "PLOTMAGIC_RERANK_PROVIDER": "no_reranker",
    }


def _strategy_env(strategy: RetrievalStrategy) -> dict[str, str]:
    return {
        "PLOTMAGIC_LEXICAL_ALGORITHM": strategy.lexical_algorithm,
        "PLOTMAGIC_RERANK_PROVIDER": strategy.reranker_provider,
        "PLOTMAGIC_LLM_PROVIDER": strategy.agentic_llm_provider,
        "PLOTMAGIC_RETRIEVAL_FALLBACK_POLICY": strategy.fallback_policy,
        "PLOTMAGIC_CATEGORY_FILTER_POLICY": strategy.category_policy,
        "PLOTMAGIC_RETRIEVAL_POOL_FACTOR": str(strategy.candidate_pool_factor),
        "PLOTMAGIC_RETRIEVAL_MIN_EVIDENCE_SCORE": str(strategy.min_evidence_score),
    }


def run_multihop_evaluation(
    *,
    root: Path,
    dataset_path: Path,
    output_dir: Path,
    run_id: str,
    tier: str,
    state: str,
    jurisdiction: str,
    category: str,
    top_k: int,
    strict_providers: bool,
    vector_backend: str,
    vector_db_path: Path,
    embedding_provider: str,
) -> dict[str, Any]:
    records = load_multihop_dataset(dataset_path)
    gold_labels = build_gold_labels(records, root)
    retrieval_strategies = build_retrieval_strategies(tier=tier, default_top_k=top_k)
    synthesis_strategies = build_synthesis_strategies()

    run_output_dir = output_dir / f"multihop20_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "multihop_eval.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "tier": tier,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_path(dataset_path),
        "query_count": len(records),
        "state": state,
        "jurisdiction": jurisdiction,
        "category": category,
        "strict_providers": strict_providers,
        "vector_backend": vector_backend,
        "vector_db_path": str(vector_db_path),
        "embedding_provider": embedding_provider,
        "retrieval_strategies": [asdict(item) for item in retrieval_strategies],
        "synthesis_strategies": [asdict(item) for item in synthesis_strategies],
    }
    (run_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    retrieval_results: dict[str, dict[str, RetrievalQueryArtifact]] = {}
    retrieval_summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    base_env = _base_env(
        vector_backend=vector_backend,
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
    )

    engine_env = {
        **base_env,
        "PLOTMAGIC_LLM_PROVIDER": "no_llm",
    }
    with temporary_env(engine_env):
        engine = ComplianceEngine(root)
        if strict_providers:
            _strict_provider_check(engine, required_embedding_provider=embedding_provider)
        engine.ingest(state=state, jurisdiction_type=jurisdiction)

        for strategy in retrieval_strategies:
            with temporary_env(_strategy_env(strategy)):
                providers_cfg = load_providers_config(root / "config" / "providers.yaml")
                provider_factory = ProviderFactory(build_default_registry(), providers_cfg)
                reranker_provider = provider_factory.create_reranker_provider()
                retrieval_llm_provider = provider_factory.create_llm_provider()
                if strict_providers:
                    actual_reranker = str(getattr(reranker_provider, "provider_id", ""))
                    expected_reranker = str(strategy.reranker_provider)
                    if actual_reranker != expected_reranker:
                        raise MultiHopEvalError(
                            "Strict provider check failed: expected reranker provider "
                            f"'{expected_reranker}', got '{actual_reranker}'."
                        )
                    degraded = [
                        message
                        for message in provider_factory.diagnostics
                        if "Reranker provider" in message and "falling back" in message
                    ]
                    if degraded:
                        raise MultiHopEvalError(
                            "Strict provider check failed: reranker provider fell back during runtime. "
                            + " | ".join(degraded)
                        )
                    actual_llm = str(getattr(retrieval_llm_provider, "provider_id", ""))
                    expected_llm = str(strategy.agentic_llm_provider)
                    if actual_llm != expected_llm:
                        raise MultiHopEvalError(
                            "Strict provider check failed: expected retrieval-control LLM provider "
                            f"'{expected_llm}', got '{actual_llm}'."
                        )
                    llm_degraded = [
                        message
                        for message in provider_factory.diagnostics
                        if "LLM provider" in message and "falling back" in message
                    ]
                    if llm_degraded:
                        raise MultiHopEvalError(
                            "Strict provider check failed: retrieval-control LLM provider fell back during runtime. "
                            + " | ".join(llm_degraded)
                        )
                engine.providers_config = providers_cfg
                engine.reranker_provider = reranker_provider
                engine.provider_health["reranker"] = reranker_provider.health()
                engine.agentic_orchestrator.retrieval_controller_llm = retrieval_llm_provider
                if hasattr(engine.lexical_index, "set_algorithm"):
                    engine.lexical_index.set_algorithm(strategy.lexical_algorithm)
                engine.state.hybrid_retriever = engine._build_hybrid_retriever()

                query_runs: dict[str, RetrievalQueryArtifact] = {}
                per_query_rows: list[dict[str, Any]] = []
                for record in records:
                    events: list[dict[str, Any]] = []
                    request = QueryRequest(
                        query=record.query,
                        state=state,
                        jurisdiction_type=jurisdiction,
                        panchayat_category=category,
                        top_k=strategy.top_k,
                        retrieval_mode=strategy.retrieval_mode,  # type: ignore[arg-type]
                        debug_trace=True,
                    )
                    response = engine.query(request, event_sink=events.append)
                    payload = response.model_dump()
                    diagnostics = _extract_retrieval_diagnostics(events)
                    retrieved_texts = _extract_retrieved_texts(payload)
                    ranked_rules = _extract_ranked_rules(payload, diagnostics["top_rules_cited"])

                    primary_hits = diagnostics["vector_hits"] if strategy.primary_signal == "vector" else diagnostics["lexical_hits"]
                    zero_hit = int(primary_hits) <= 0
                    gold = gold_labels[record.record_id]
                    retrieval_quality = compute_retrieval_quality(
                        gold=gold,
                        retrieved_texts=retrieved_texts,
                        ranked_rules=ranked_rules,
                        zero_hit=zero_hit,
                    )
                    claim_ctx = _extract_claim_context(payload)
                    extractive_context = "\n".join(retrieved_texts[:8])
                    extracted_f1 = set_f1(extractive_context, record.final_answer)

                    artifact = RetrievalQueryArtifact(
                        query_id=record.record_id,
                        query=record.query,
                        expected_answer=record.final_answer,
                        retrieved_texts=retrieved_texts,
                        top_rules_cited=ranked_rules,
                        vector_hits=int(diagnostics["vector_hits"]),
                        lexical_hits=int(diagnostics["lexical_hits"]),
                        structured_hits=int(diagnostics["structured_hits"]),
                        fallback_stage=str(diagnostics["fallback_stage"]),
                        zero_hit=zero_hit,
                        rule_hit_at_k=float(retrieval_quality["rule_hit_at_k"]),
                        rule_recall_at_k=float(retrieval_quality["rule_recall_at_k"]),
                        hop_coverage_at_k=float(retrieval_quality["hop_coverage_at_k"]),
                        snippet_recall_at_k=float(retrieval_quality["snippet_recall_at_k"]),
                        mrr=float(retrieval_quality["mrr"]),
                        retrieved_context_extractive_f1=extracted_f1,
                        claim_citations=claim_ctx["claim_citations"],
                        claim_texts=claim_ctx["claim_texts"],
                        claim_scores=claim_ctx["claim_scores"],
                        unresolved=claim_ctx["unresolved"],
                        grounding=claim_ctx["grounding"],
                        verdict=claim_ctx["verdict"],
                        strategy_name=strategy.name,
                    )
                    query_runs[record.record_id] = artifact

                    per_query_rows.append(
                        {
                            "query_id": record.record_id,
                            "rule_hit_at_k": artifact.rule_hit_at_k,
                            "rule_recall_at_k": artifact.rule_recall_at_k,
                            "hop_coverage_at_k": artifact.hop_coverage_at_k,
                            "snippet_recall_at_k": artifact.snippet_recall_at_k,
                            "mrr": artifact.mrr,
                            "zero_hit": 1.0 if artifact.zero_hit else 0.0,
                        }
                    )
                traces.append(
                    {
                        "trace_type": "retrieval",
                        "retrieval_strategy": strategy.name,
                        "query_id": record.record_id,
                            "rule_hit_at_k": artifact.rule_hit_at_k,
                            "rule_recall_at_k": artifact.rule_recall_at_k,
                            "hop_coverage_at_k": artifact.hop_coverage_at_k,
                            "snippet_recall_at_k": artifact.snippet_recall_at_k,
                            "mrr": artifact.mrr,
                            "zero_hit": artifact.zero_hit,
                        "vector_hits": artifact.vector_hits,
                        "lexical_hits": artifact.lexical_hits,
                        "structured_hits": artifact.structured_hits,
                        "fallback_stage": artifact.fallback_stage,
                        "reranker_provider": strategy.reranker_provider,
                    }
                )

                summary = {
                    "retrieval_strategy": strategy.name,
                    "query_count": len(per_query_rows),
                    "rule_hit_at_k_mean": _safe_mean([item["rule_hit_at_k"] for item in per_query_rows]),
                    "rule_recall_at_k_mean": _safe_mean([item["rule_recall_at_k"] for item in per_query_rows]),
                    "hop_coverage_at_k_mean": _safe_mean([item["hop_coverage_at_k"] for item in per_query_rows]),
                    "snippet_recall_at_k_mean": _safe_mean([item["snippet_recall_at_k"] for item in per_query_rows]),
                    "mrr_mean": _safe_mean([item["mrr"] for item in per_query_rows]),
                    "zero_hit_rate": _safe_mean([item["zero_hit"] for item in per_query_rows]),
                }
                summary["retrieval_composite_score"] = _retrieval_composite(summary)
                retrieval_summaries.append(summary)
                retrieval_results[strategy.name] = query_runs

    retrieval_leaderboard = _stable_sort(retrieval_summaries, "retrieval_composite_score", ["retrieval_strategy"])
    (run_output_dir / "retrieval_leaderboard.json").write_text(
        json.dumps({"rows": retrieval_leaderboard}, indent=2),
        encoding="utf-8",
    )

    oracle_synthesis_rows: dict[str, list[dict[str, Any]]] = {}
    oracle_per_query_f1: dict[str, dict[str, float]] = {}
    synthesis_leaderboard_rows: list[dict[str, Any]] = []
    composers: dict[str, ComplianceBriefComposer] = {}
    deterministic_composer = _build_composer(
        root=root,
        llm_provider_id="no_llm",
        strict_providers=False,
    )
    for strategy in synthesis_strategies:
        composers[strategy.name] = _build_composer(
            root=root,
            llm_provider_id=strategy.llm_provider_id,
            strict_providers=strict_providers,
        )

        rows: list[dict[str, Any]] = []
        per_query: dict[str, float] = {}
        for record in records:
            claim_citations, claim_texts, claim_scores = _oracle_claims(record)
            brief = _compose_answer_with_timeout(
                composers[strategy.name],
                fallback_composer=deterministic_composer,
                timeout_s=LLM_COMPOSE_TIMEOUT_S,
                query=record.query,
                verdict="depends",
                claim_citations=claim_citations,
                claim_texts=claim_texts,
                claim_scores=claim_scores,
                unresolved=[],
                grounding_payload={
                    "supported_claim_count": len(claim_texts),
                    "unsupported_claim_count": 0,
                    "conflicting_claim_count": 0,
                    "missing_topics": [],
                    "abstained": False,
                },
            )
            predicted = _brief_to_text(brief)
            row = {
                "query_id": record.record_id,
                "token_set_f1": set_f1(predicted, record.final_answer),
                "token_set_recall": set_recall(predicted, record.final_answer),
                "numeric_token_recall": numeric_recall(predicted, record.final_answer),
                "mandatory_component_completeness": component_completeness(
                    predicted,
                    extract_mandatory_components(record.query),
                ),
                "citation_groundedness": _brief_citation_groundedness(brief, claim_citations),
                "contract_violation": _brief_contract_violation(
                    brief,
                    orchestrated_verdict="depends",
                    claim_citations=claim_citations,
                ),
            }
            rows.append(row)
            per_query[record.record_id] = float(row["token_set_f1"])
            traces.append(
                {
                    "trace_type": "oracle_synthesis",
                    "synthesis_strategy": strategy.name,
                    "query_id": record.record_id,
                    **row,
                }
            )

        oracle_synthesis_rows[strategy.name] = rows
        oracle_per_query_f1[strategy.name] = per_query
        summary = {
            "synthesis_strategy": strategy.name,
            "query_count": len(rows),
            "token_set_f1_mean": _safe_mean([item["token_set_f1"] for item in rows]),
            "token_set_recall_mean": _safe_mean([item["token_set_recall"] for item in rows]),
            "numeric_token_recall_mean": _safe_mean([item["numeric_token_recall"] for item in rows]),
            "mandatory_component_completeness_mean": _safe_mean(
                [item["mandatory_component_completeness"] for item in rows]
            ),
            "citation_groundedness_mean": _safe_mean([item["citation_groundedness"] for item in rows]),
            "contract_violation_rate": _safe_mean([item["contract_violation"] for item in rows]),
        }
        summary["synthesis_composite_score"] = _synthesis_composite(summary)
        synthesis_leaderboard_rows.append(summary)

    synthesis_leaderboard = _stable_sort(
        synthesis_leaderboard_rows,
        "synthesis_composite_score",
        ["synthesis_strategy"],
    )
    (run_output_dir / "synthesis_leaderboard.json").write_text(
        json.dumps({"rows": synthesis_leaderboard}, indent=2),
        encoding="utf-8",
    )

    end_to_end_rows: list[dict[str, Any]] = []
    end_to_end_per_query: dict[str, list[EndToEndQueryMetrics]] = {}
    for retrieval_strategy in retrieval_strategies:
        query_artifacts = retrieval_results[retrieval_strategy.name]
        retrieval_summary = next(
            (row for row in retrieval_leaderboard if row["retrieval_strategy"] == retrieval_strategy.name),
            None,
        )
        retrieval_composite = float((retrieval_summary or {}).get("retrieval_composite_score", 0.0))
        for synthesis_strategy in synthesis_strategies:
            per_query_metrics: list[EndToEndQueryMetrics] = []
            composer = composers[synthesis_strategy.name]
            for record in records:
                artifact = query_artifacts[record.record_id]
                trimmed_claim_citations, trimmed_claim_texts, trimmed_claim_scores = _trim_claim_context(
                    claim_citations=artifact.claim_citations,
                    claim_texts=artifact.claim_texts,
                    claim_scores=artifact.claim_scores,
                )
                brief = _compose_answer_with_timeout(
                    composer,
                    fallback_composer=deterministic_composer,
                    timeout_s=LLM_COMPOSE_TIMEOUT_S,
                    query=record.query,
                    verdict=artifact.verdict,
                    claim_citations=trimmed_claim_citations,
                    claim_texts=trimmed_claim_texts,
                    claim_scores=trimmed_claim_scores,
                    unresolved=artifact.unresolved[:8],
                    grounding_payload=artifact.grounding,
                )
                predicted = _brief_to_text(brief)
                f1 = set_f1(predicted, record.final_answer)
                recall = set_recall(predicted, record.final_answer)
                num_recall = numeric_recall(predicted, record.final_answer)
                completeness = component_completeness(predicted, extract_mandatory_components(record.query))
                groundedness = _brief_citation_groundedness(brief, trimmed_claim_citations)
                contract_violation = _brief_contract_violation(
                    brief,
                    orchestrated_verdict=artifact.verdict,
                    claim_citations=trimmed_claim_citations,
                )
                oracle_f1 = float(oracle_per_query_f1[synthesis_strategy.name].get(record.record_id, 0.0))
                retrieval_gap = max(0.0, oracle_f1 - artifact.retrieved_context_extractive_f1)
                synthesis_gap = max(0.0, artifact.retrieved_context_extractive_f1 - f1)
                bucket = classify_bottleneck(
                    oracle_context_f1=oracle_f1,
                    retrieved_context_extractive_f1=artifact.retrieved_context_extractive_f1,
                    end_to_end_f1=f1,
                )
                metric = EndToEndQueryMetrics(
                    query_id=record.record_id,
                    query=record.query,
                    retrieval_strategy=retrieval_strategy.name,
                    synthesis_strategy=synthesis_strategy.name,
                    token_set_f1=f1,
                    token_set_recall=recall,
                    numeric_token_recall=num_recall,
                    mandatory_component_completeness=completeness,
                    citation_groundedness=groundedness,
                    contract_violation=contract_violation,
                    oracle_context_f1=oracle_f1,
                    retrieved_context_extractive_f1=artifact.retrieved_context_extractive_f1,
                    retrieval_gap=retrieval_gap,
                    synthesis_gap=synthesis_gap,
                    bottleneck_bucket=bucket,
                    zero_hit=artifact.zero_hit,
                    hop_coverage_at_k=artifact.hop_coverage_at_k,
                    fallback_stage=artifact.fallback_stage,
                    vector_hits=artifact.vector_hits,
                    lexical_hits=artifact.lexical_hits,
                    structured_hits=artifact.structured_hits,
                )
                per_query_metrics.append(metric)
                traces.append(
                    {
                        "trace_type": "end_to_end",
                        **asdict(metric),
                    }
                )

            end_to_end_per_query[f"{retrieval_strategy.name}::{synthesis_strategy.name}"] = per_query_metrics
            summary = {
                "retrieval_strategy": retrieval_strategy.name,
                "synthesis_strategy": synthesis_strategy.name,
                "query_count": len(per_query_metrics),
                "token_set_f1_mean": _safe_mean([item.token_set_f1 for item in per_query_metrics]),
                "token_set_recall_mean": _safe_mean([item.token_set_recall for item in per_query_metrics]),
                "numeric_token_recall_mean": _safe_mean(
                    [item.numeric_token_recall for item in per_query_metrics]
                ),
                "mandatory_component_completeness_mean": _safe_mean(
                    [item.mandatory_component_completeness for item in per_query_metrics]
                ),
                "citation_groundedness_mean": _safe_mean([item.citation_groundedness for item in per_query_metrics]),
                "contract_violation_rate": _safe_mean([item.contract_violation for item in per_query_metrics]),
                "hop_coverage_at_k_mean": _safe_mean([item.hop_coverage_at_k for item in per_query_metrics]),
                "zero_hit_rate": _safe_mean([1.0 if item.zero_hit else 0.0 for item in per_query_metrics]),
                "retrieval_gap_mean": _safe_mean([item.retrieval_gap for item in per_query_metrics]),
                "synthesis_gap_mean": _safe_mean([item.synthesis_gap for item in per_query_metrics]),
            }
            synthesis_score = _synthesis_composite(summary)
            summary["end_to_end_score"] = (0.70 * synthesis_score) + (0.30 * retrieval_composite)
            failures: list[str] = []
            if summary["citation_groundedness_mean"] != LEGAL_GATES["citation_groundedness_mean"]:
                failures.append(
                    "citation_groundedness_mean gate failed"
                )
            if summary["contract_violation_rate"] != LEGAL_GATES["contract_violation_rate"]:
                failures.append(
                    "contract_violation_rate gate failed"
                )
            if summary["hop_coverage_at_k_mean"] < LEGAL_GATES["hop_coverage_at_k_mean"]:
                failures.append("hop_coverage_at_k gate failed")
            if summary["zero_hit_rate"] != LEGAL_GATES["zero_hit_rate"]:
                failures.append("zero_hit_rate gate failed")
            summary["passed_legal_gates"] = len(failures) == 0
            summary["gate_failures"] = failures
            end_to_end_rows.append(summary)

    end_to_end_leaderboard = sorted(
        end_to_end_rows,
        key=lambda row: (
            not bool(row.get("passed_legal_gates", False)),
            -(float(row.get("end_to_end_score", 0.0) or 0.0)),
            str(row.get("retrieval_strategy", "")),
            str(row.get("synthesis_strategy", "")),
        ),
    )
    (run_output_dir / "end_to_end_leaderboard.json").write_text(
        json.dumps({"rows": end_to_end_leaderboard}, indent=2),
        encoding="utf-8",
    )

    for line in traces:
        pass
    with (run_output_dir / "per_query_traces.jsonl").open("w", encoding="utf-8") as handle:
        for line in traces:
            handle.write(json.dumps(line) + "\n")

    best_retrieval = retrieval_leaderboard[0] if retrieval_leaderboard else {}
    worst_retrieval = retrieval_leaderboard[-1] if retrieval_leaderboard else {}
    best_synthesis = synthesis_leaderboard[0] if synthesis_leaderboard else {}
    worst_synthesis = synthesis_leaderboard[-1] if synthesis_leaderboard else {}
    gated = [row for row in end_to_end_leaderboard if bool(row.get("passed_legal_gates"))]
    best_end_to_end = gated[0] if gated else None
    best_f1_row = max(end_to_end_leaderboard, key=lambda row: float(row.get("token_set_f1_mean", 0.0) or 0.0))
    worst_f1_row = min(end_to_end_leaderboard, key=lambda row: float(row.get("token_set_f1_mean", 0.0) or 0.0))

    best_key = f"{best_f1_row['retrieval_strategy']}::{best_f1_row['synthesis_strategy']}"
    best_per_query = end_to_end_per_query.get(best_key, [])
    bucket_counter = Counter(item.bottleneck_bucket for item in best_per_query)
    retrieval_gap_values = [item.retrieval_gap for item in best_per_query]
    synthesis_gap_values = [item.synthesis_gap for item in best_per_query]

    rca_report = {
        "reference_strategy": {
            "retrieval_strategy": best_f1_row["retrieval_strategy"],
            "synthesis_strategy": best_f1_row["synthesis_strategy"],
            "token_set_f1_mean": best_f1_row["token_set_f1_mean"],
            "token_set_recall_mean": best_f1_row["token_set_recall_mean"],
            "numeric_token_recall_mean": best_f1_row["numeric_token_recall_mean"],
        },
        "gap_summary": {
            "retrieval_gap_mean": _safe_mean(retrieval_gap_values),
            "synthesis_gap_mean": _safe_mean(synthesis_gap_values),
        },
        "bucket_counts": dict(bucket_counter),
        "top_blocking_queries": [
            asdict(item)
            for item in sorted(best_per_query, key=lambda row: row.token_set_f1)[:5]
        ],
        "all_strategy_gap_overview": [
            {
                "retrieval_strategy": row["retrieval_strategy"],
                "synthesis_strategy": row["synthesis_strategy"],
                "retrieval_gap_mean": row["retrieval_gap_mean"],
                "synthesis_gap_mean": row["synthesis_gap_mean"],
                "token_set_f1_mean": row["token_set_f1_mean"],
            }
            for row in end_to_end_leaderboard
        ],
    }
    (run_output_dir / "rca_report.json").write_text(json.dumps(rca_report, indent=2), encoding="utf-8")

    bucket_order = sorted(
        bucket_counter.keys(),
        key=lambda key: (
            -int(bucket_counter.get(key, 0)),
            -(
                _safe_mean([item.retrieval_gap for item in best_per_query if item.bottleneck_bucket == key])
                + _safe_mean([item.synthesis_gap for item in best_per_query if item.bottleneck_bucket == key])
            ),
            key,
        ),
    )
    intervention_map = {
        "retrieval_bottleneck": "Increase hop-aware retrieval recall: expand top-k, relax filters where safe, and strengthen hop sub-query expansion.",
        "synthesis_bottleneck": "Constrain synthesis to preserve numeric/legal spans from evidence and force component-complete section coverage.",
        "mixed": "Apply joint retrieval+synthesis tuning with query-level fallback diagnostics and stricter evidence-to-claim mapping.",
    }
    interventions: list[dict[str, Any]] = []
    for bucket in bucket_order[:3]:
        rows = [item for item in best_per_query if item.bottleneck_bucket == bucket]
        if not rows:
            continue
        if bucket == "retrieval_bottleneck":
            estimated = _safe_mean([item.retrieval_gap for item in rows])
        elif bucket == "synthesis_bottleneck":
            estimated = _safe_mean([item.synthesis_gap for item in rows])
        else:
            estimated = _safe_mean([item.retrieval_gap + item.synthesis_gap for item in rows])
        interventions.append(
            {
                "bucket": bucket,
                "query_count": len(rows),
                "average_gap": estimated,
                "estimated_f1_uplift": estimated,
                "recommended_intervention": intervention_map.get(bucket, "Investigate bucket-specific failure traces."),
            }
        )
    best_f1_plan = {
        "reference_strategy": {
            "retrieval_strategy": best_f1_row["retrieval_strategy"],
            "synthesis_strategy": best_f1_row["synthesis_strategy"],
            "token_set_f1_mean": best_f1_row["token_set_f1_mean"],
        },
        "top_interventions": interventions,
        "summary": (
            f"Best F1 strategy is {best_f1_row['retrieval_strategy']} + {best_f1_row['synthesis_strategy']}. "
            f"Dominant blocker buckets: {dict(bucket_counter)}."
        ),
    }
    (run_output_dir / "best_f1_plan.json").write_text(json.dumps(best_f1_plan, indent=2), encoding="utf-8")

    winner_report = {
        "run_id": run_id,
        "best_retrieval_strategy": best_retrieval.get("retrieval_strategy", "n/a"),
        "worst_retrieval_strategy": worst_retrieval.get("retrieval_strategy", "n/a"),
        "best_synthesis_strategy": best_synthesis.get("synthesis_strategy", "n/a"),
        "worst_synthesis_strategy": worst_synthesis.get("synthesis_strategy", "n/a"),
        "best_end_to_end_strategy_or_no_winner": (
            {
                "retrieval_strategy": best_end_to_end["retrieval_strategy"],
                "synthesis_strategy": best_end_to_end["synthesis_strategy"],
                "end_to_end_score": best_end_to_end["end_to_end_score"],
            }
            if best_end_to_end
            else "no_winner"
        ),
        "best_f1_strategy": {
            "retrieval_strategy": best_f1_row["retrieval_strategy"],
            "synthesis_strategy": best_f1_row["synthesis_strategy"],
            "token_set_f1_mean": best_f1_row["token_set_f1_mean"],
        },
        "worst_f1_strategy": {
            "retrieval_strategy": worst_f1_row["retrieval_strategy"],
            "synthesis_strategy": worst_f1_row["synthesis_strategy"],
            "token_set_f1_mean": worst_f1_row["token_set_f1_mean"],
        },
        "best_f1_plan_summary": best_f1_plan["summary"],
        "artifacts": {
            "manifest": str(run_output_dir / "manifest.json"),
            "retrieval_leaderboard": str(run_output_dir / "retrieval_leaderboard.json"),
            "synthesis_leaderboard": str(run_output_dir / "synthesis_leaderboard.json"),
            "end_to_end_leaderboard": str(run_output_dir / "end_to_end_leaderboard.json"),
            "rca_report": str(run_output_dir / "rca_report.json"),
            "best_f1_plan": str(run_output_dir / "best_f1_plan.json"),
            "per_query_traces": str(run_output_dir / "per_query_traces.jsonl"),
        },
    }
    (run_output_dir / "winner_report.json").write_text(json.dumps(winner_report, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "output_dir": str(run_output_dir),
        "winner_report": winner_report,
    }
