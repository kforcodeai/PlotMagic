from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.models.schemas import QueryRequest

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
}


def _normalize_tokens(text: str) -> set[str]:
    lowered = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    cleaned = re.sub(r"[^a-z0-9%./()-]+", " ", lowered)
    return {token for token in cleaned.split() if token and token not in _STOPWORDS}


def _set_recall(pred: str, ref: str) -> float:
    pred_set = _normalize_tokens(pred)
    ref_set = _normalize_tokens(ref)
    if not ref_set:
        return 0.0
    return len(pred_set.intersection(ref_set)) / float(len(ref_set))


def _set_f1(pred: str, ref: str) -> float:
    pred_set = _normalize_tokens(pred)
    ref_set = _normalize_tokens(ref)
    if not pred_set or not ref_set:
        return 0.0
    overlap = len(pred_set.intersection(ref_set))
    if overlap <= 0:
        return 0.0
    precision = overlap / float(len(pred_set))
    recall = overlap / float(len(ref_set))
    return (2.0 * precision * recall) / (precision + recall)


def _numeric_recall(pred: str, ref: str) -> float:
    pred_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", pred))
    ref_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", ref))
    if not ref_numbers:
        return 0.0
    return len(pred_numbers.intersection(ref_numbers)) / float(len(ref_numbers))


def _safe_mean(values: list[float]) -> float:
    return (sum(values) / float(len(values))) if values else 0.0


def _best_overlap_score(gold_snippet: str, retrieved_chunks: list[str]) -> float:
    if not retrieved_chunks:
        return 0.0
    gold_tokens = _normalize_tokens(gold_snippet)
    if not gold_tokens:
        return 0.0
    best = 0.0
    for chunk in retrieved_chunks:
        chunk_tokens = _normalize_tokens(chunk)
        if not chunk_tokens:
            continue
        overlap = len(gold_tokens.intersection(chunk_tokens))
        score = overlap / float(len(gold_tokens))
        if score > best:
            best = score
    return best


def _snippet_recall(gold_snippets: list[str], retrieved_chunks: list[str], threshold: float = 0.55) -> tuple[float, int]:
    if not gold_snippets:
        return 0.0, 0
    matched = 0
    for snippet in gold_snippets:
        if _best_overlap_score(snippet, retrieved_chunks) >= threshold:
            matched += 1
    return matched / float(len(gold_snippets)), matched


def _final_answer_text(payload: dict[str, Any]) -> str:
    final_answer = payload.get("final_answer") or {}
    if not isinstance(final_answer, dict):
        return ""
    lines: list[str] = []
    short_summary = str(final_answer.get("short_summary", "")).strip()
    if short_summary:
        lines.append(short_summary)
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
    return "\n".join(lines)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_baseline_summary(path: Path | None) -> dict[str, float]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    out: dict[str, float] = {}
    if isinstance(summary, dict):
        for key in [
            "mean_snippet_recall",
            "mean_retrieval_ceiling_recall",
            "mean_answer_f1_vs_gold",
            "mean_answer_recall_vs_gold",
            "mean_answer_numeric_recall_vs_gold",
            "mean_retrieval_numeric_recall_vs_gold",
        ]:
            value = summary.get(key)
            if isinstance(value, (int, float)):
                out[key] = float(value)
    return out


def _provider_health_to_jsonable(provider_health: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in provider_health.items():
        if hasattr(value, "provider_id"):
            out[key] = {
                "provider_id": getattr(value, "provider_id", ""),
                "available": bool(getattr(value, "available", False)),
                "capabilities": list(getattr(value, "capabilities", [])),
                "details": dict(getattr(value, "details", {}) or {}),
            }
        else:
            out[key] = str(value)
    return out


def run(args: argparse.Namespace) -> Path:
    dataset_path = (ROOT / args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    rows = _load_dataset(dataset_path)
    if not rows:
        raise SystemExit(f"No rows in dataset: {dataset_path}")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"kmbr_decomposed_benchmark_{run_id}.json"

    engine = ComplianceEngine(ROOT)
    if args.strict_providers:
        embedding_health = engine.provider_health.get("embedding")
        provider_id = getattr(embedding_health, "provider_id", "")
        available = bool(getattr(embedding_health, "available", False))
        if provider_id != args.embedding_provider or not available:
            raise SystemExit(
                "Strict provider check failed: "
                f"expected embedding provider '{args.embedding_provider}' available, got '{provider_id}' available={available}."
            )
    engine.ingest(state=args.state, jurisdiction_type=args.jurisdiction)

    per_query: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row.get("id", "")).strip()
        query = str(row.get("query", "")).strip()
        gold_chunks = row.get("ground_truth_chunks", [])
        if not query_id or not query or not isinstance(gold_chunks, list):
            continue
        gold_snippets = [str(item.get("snippet", "")).strip() for item in gold_chunks if isinstance(item, dict)]
        gold_snippets = [item for item in gold_snippets if item]
        gold_text = "\n".join(gold_snippets)

        request = QueryRequest(
            query=query,
            state=args.state,
            jurisdiction_type=args.jurisdiction,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            debug_trace=False,
        )
        payload = engine.query(request).model_dump()

        evidence = payload.get("evidence_matrix", [])
        retrieved_chunks = []
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if text:
                    retrieved_chunks.append(text)
                if len(retrieved_chunks) >= args.max_chunks:
                    break
        retrieved_text = "\n".join(retrieved_chunks)
        final_text = _final_answer_text(payload)
        grounding = payload.get("grounding") or {}
        snippet_recall, matched_chunks = _snippet_recall(gold_snippets=gold_snippets, retrieved_chunks=retrieved_chunks)

        per_query.append(
            {
                "id": query_id,
                "query": query,
                "verdict": payload.get("verdict"),
                "judge_sufficient": bool(grounding.get("sufficient")),
                "judge_partial": bool(grounding.get("partial")),
                "judge_abstained": bool(grounding.get("abstained")),
                "judge_support_ratio": float(grounding.get("support_ratio", 0.0) or 0.0),
                "judge_missing_components": list(grounding.get("missing_topics") or []),
                "snippet_recall_at_k": snippet_recall,
                "matched_gold_snippets": matched_chunks,
                "gold_snippets_count": len(gold_snippets),
                "retrieval_ceiling_recall": _set_recall(retrieved_text, gold_text),
                "retrieval_numeric_recall": _numeric_recall(retrieved_text, gold_text),
                "answer_f1_vs_gold": _set_f1(final_text, gold_text),
                "answer_recall_vs_gold": _set_recall(final_text, gold_text),
                "answer_numeric_recall_vs_gold": _numeric_recall(final_text, gold_text),
            }
        )

    summary = {
        "query_count": len(per_query),
        "retrieval_coverage_mean": _safe_mean([item["snippet_recall_at_k"] for item in per_query]),
        "retrieval_ceiling_recall_mean": _safe_mean([item["retrieval_ceiling_recall"] for item in per_query]),
        "judge_pass_rate": _safe_mean([0.0 if item["judge_abstained"] else 1.0 for item in per_query]),
        "judge_sufficient_rate": _safe_mean([1.0 if item["judge_sufficient"] else 0.0 for item in per_query]),
        "judge_partial_rate": _safe_mean([1.0 if item["judge_partial"] else 0.0 for item in per_query]),
        "answer_f1_mean": _safe_mean([item["answer_f1_vs_gold"] for item in per_query]),
        "answer_recall_mean": _safe_mean([item["answer_recall_vs_gold"] for item in per_query]),
        "answer_numeric_recall_mean": _safe_mean([item["answer_numeric_recall_vs_gold"] for item in per_query]),
        "retrieval_numeric_recall_mean": _safe_mean([item["retrieval_numeric_recall"] for item in per_query]),
    }

    baseline = _load_baseline_summary((ROOT / args.baseline_diagnostics).resolve() if args.baseline_diagnostics else None)
    deltas: dict[str, float] = {}
    if baseline:
        key_map = {
            "retrieval_coverage_mean": "mean_snippet_recall",
            "retrieval_ceiling_recall_mean": "mean_retrieval_ceiling_recall",
            "answer_f1_mean": "mean_answer_f1_vs_gold",
            "answer_recall_mean": "mean_answer_recall_vs_gold",
            "answer_numeric_recall_mean": "mean_answer_numeric_recall_vs_gold",
            "retrieval_numeric_recall_mean": "mean_retrieval_numeric_recall_vs_gold",
        }
        for current_key, baseline_key in key_map.items():
            if baseline_key in baseline:
                deltas[current_key] = float(summary[current_key]) - float(baseline[baseline_key])

    provider_health_json = _provider_health_to_jsonable(engine.provider_health)

    artifact = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "state": args.state,
        "jurisdiction": args.jurisdiction,
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "max_chunks": args.max_chunks,
        "provider_health": provider_health_json,
        "summary": summary,
        "deltas_vs_baseline": deltas,
        "per_query": per_query,
    }
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "output": str(output_path)}, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Decomposed KMBR benchmark (retrieval coverage + judge pass + synthesis quality).")
    parser.add_argument("--dataset", default="evaluation/kmbr/kmbr_multihop_retrieval_dataset.jsonl")
    parser.add_argument("--output-dir", default="evaluation/kmbr")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="municipality")
    parser.add_argument("--retrieval-mode", default="lexical_only_bm25")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--baseline-diagnostics", default="evaluation/kmbr/kmbr_winner_20260309_diagnostics.json")
    parser.add_argument("--embedding-provider", default="openai_embedding")
    parser.add_argument("--strict-providers", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
