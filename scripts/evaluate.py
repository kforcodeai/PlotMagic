from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.evaluation import EvaluationMetrics, ReleaseGateEvaluator
from src.evaluation.metrics import accuracy, precision, recall_at_k
from src.models.schemas import QueryRequest

_MAXIMIZE_METRICS = {
    "semantic_accuracy_proxy",
    "mandatory_component_completeness",
    "citation_groundedness",
    "abstention_correctness",
    "numeric_correctness",
}
_MINIMIZE_METRICS = {
    "contradiction_rate",
    "contract_violation_rate",
    "unsupported_claim_rate",
}
_HIGH_RISK_CATEGORY_HINTS = {
    "numeric",
    "table",
    "timeline",
    "deemed",
    "permit",
    "exempt",
    "setback",
    "height",
    "distance",
    "flood",
    "crz",
    "safety",
}
_LEGACY_PROVIDER_ENV = {
    "PLOTMAGIC_EMBEDDING_PROVIDER": "hash_embedding",
    "PLOTMAGIC_RERANK_PROVIDER": "no_reranker",
    "PLOTMAGIC_LLM_PROVIDER": "no_llm",
    "PLOTMAGIC_VECTOR_BACKEND": "sqlite",
}


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Iterator[None]:
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


def _infer_location(query: str) -> str | None:
    lowered = query.lower()
    for token in ["thrissur", "thiruvananthapuram", "kochi", "kozhikode", "anthikkad", "adimali"]:
        if token in lowered:
            return token
    return None


def _is_high_risk(item: dict[str, Any]) -> bool:
    category = str(item.get("category", "")).lower()
    if any(token in category for token in _HIGH_RISK_CATEGORY_HINTS):
        return True
    query = str(item.get("query", "")).lower()
    return any(token in query for token in _HIGH_RISK_CATEGORY_HINTS)


def _evaluate_once(
    *,
    state: str,
    gold_file: Path,
) -> dict[str, Any]:
    engine = ComplianceEngine(root=ROOT)
    engine.ingest(state=state)

    gold = json.loads(gold_file.read_text(encoding="utf-8"))
    jurisdiction_total = 0
    jurisdiction_correct = 0
    occupancy_total = 0
    occupancy_correct = 0
    recall_total = 0
    recall_hit = 0
    citation_claims_total = 0
    citation_supported = 0
    numeric_total = 0
    numeric_correct = 0
    unsupported_claims = 0
    contradiction_rates: list[float] = []
    citation_grounded_checks = 0
    citation_grounded_pass = 0
    mandatory_component_scores: list[float] = []
    abstention_checks = 0
    abstention_correct = 0
    contract_violations = 0

    high_risk_total = 0
    high_risk_supported = 0
    high_risk_grounded_checks = 0
    high_risk_grounded_pass = 0
    high_risk_abstention_checks = 0
    high_risk_abstention_correct = 0

    for item in gold:
        request = QueryRequest(
            query=item["query"],
            state=state,
            location=item.get("request_location", _infer_location(item["query"])),
            jurisdiction_type=item.get("request_jurisdiction_type"),
            panchayat_category=item.get("request_panchayat_category"),
        )
        response = engine.query(request)
        expected_jurisdiction = item.get("expected_jurisdiction_type")
        is_high_risk = _is_high_risk(item)

        if expected_jurisdiction:
            jurisdiction_total += 1
            if f"::{expected_jurisdiction}" in response.jurisdiction:
                jurisdiction_correct += 1

        if "expected_occupancy" in item:
            occupancy_total += 1
            expected = set(item["expected_occupancy"])
            if expected.intersection(set(response.occupancy_groups)):
                occupancy_correct += 1

        expected_topics = item.get("expected_topics", [])
        if expected_topics:
            recall_total += len(expected_topics)
            supported = {section["topic"] for section in response.answer_sections if section.get("rules")}
            hit_count = len(set(expected_topics).intersection(supported))
            recall_hit += hit_count
            if is_high_risk:
                high_risk_total += len(expected_topics)
                high_risk_supported += hit_count

        citation_claims_total += len(response.evidence_matrix)
        citation_supported += len([e for e in response.evidence_matrix if e.citations or response.citations])

        if item.get("category") == "numeric_table":
            numeric_total += 1
            if response.citations:
                numeric_correct += 1

        unsupported_claims += len(response.unresolved)
        grounding = response.grounding
        if grounding is not None:
            denom = max(
                1,
                grounding.supported_claim_count + grounding.unsupported_claim_count + grounding.conflicting_claim_count,
            )
            contradiction_rates.append(grounding.conflicting_claim_count / float(denom))

            missing_count = len(grounding.missing_topics)
            topic_count = len(item.get("expected_topics", [])) or 1
            mandatory_score = max(0.0, 1.0 - (missing_count / float(topic_count)))
            mandatory_component_scores.append(mandatory_score)

            abstention_checks += 1
            if not grounding.abstained:
                abstention_correct += 1

            if is_high_risk:
                high_risk_abstention_checks += 1
                if not grounding.abstained:
                    high_risk_abstention_correct += 1
        else:
            mandatory_component_scores.append(0.0)

        if response.final_answer is not None:
            if response.verdict != response.final_answer.verdict:
                contract_violations += 1
            if response.verdict != "insufficient_evidence" and not response.final_answer.short_summary.strip():
                contract_violations += 1

            for section in [
                response.final_answer.applicable_rules,
                response.final_answer.conditions_and_exceptions,
                response.final_answer.required_actions,
            ]:
                for claim in section:
                    citation_grounded_checks += 1
                    mapped = set(response.claim_citations.get(claim.claim_id, []))
                    if mapped and claim.citation_ids and set(claim.citation_ids).issubset(mapped):
                        citation_grounded_pass += 1
                    if is_high_risk:
                        high_risk_grounded_checks += 1
                        if mapped and claim.citation_ids and set(claim.citation_ids).issubset(mapped):
                            high_risk_grounded_pass += 1

    metrics = EvaluationMetrics(
        jurisdiction_accuracy=accuracy(jurisdiction_correct, jurisdiction_total),
        occupancy_accuracy=accuracy(occupancy_correct, occupancy_total),
        clause_recall_at_20=recall_at_k(recall_hit, recall_total),
        citation_precision=precision(citation_supported, citation_claims_total),
        numeric_correctness=accuracy(numeric_correct, numeric_total),
        unsupported_claim_rate=unsupported_claims / max(1, citation_claims_total),
        semantic_accuracy_proxy=recall_at_k(recall_hit, recall_total),
        mandatory_component_completeness=sum(mandatory_component_scores) / max(1, len(mandatory_component_scores)),
        contradiction_rate=sum(contradiction_rates) / max(1, len(contradiction_rates)),
        citation_groundedness=precision(citation_grounded_pass, citation_grounded_checks),
        abstention_correctness=accuracy(abstention_correct, abstention_checks),
        contract_violation_rate=contract_violations / max(1, len(gold)),
    )
    gates = ReleaseGateEvaluator(ROOT / "config" / "evaluation" / "metrics.yaml").evaluate(metrics)
    high_risk = {
        "topic_recall_proxy": recall_at_k(high_risk_supported, high_risk_total),
        "citation_groundedness": precision(high_risk_grounded_pass, high_risk_grounded_checks),
        "abstention_correctness": accuracy(high_risk_abstention_correct, high_risk_abstention_checks),
    }
    return {
        "metrics": asdict(metrics),
        "gate_passed": gates.passed,
        "failures": gates.failures,
        "high_risk": high_risk,
        "query_count": len(gold),
    }


def _extract_metric_view(payload: dict[str, Any]) -> dict[str, float]:
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return {key: float(value) for key, value in payload["metrics"].items() if isinstance(value, (int, float))}
    if "summary" in payload and isinstance(payload["summary"], dict):
        summary = payload["summary"]
        return {key: float(value) for key, value in summary.items() if isinstance(value, (int, float))}
    return {}


def _baseline_regressions(
    *,
    current: dict[str, float],
    baseline: dict[str, float],
    tolerance: float,
) -> tuple[dict[str, float], list[str]]:
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for key in sorted(set(current.keys()).intersection(set(baseline.keys()))):
        cur = float(current[key])
        base = float(baseline[key])
        delta = cur - base
        deltas[key] = delta
        if key in _MAXIMIZE_METRICS and delta < -abs(tolerance):
            regressions.append(f"{key} regressed by {delta:.4f}")
        if key in _MINIMIZE_METRICS and delta > abs(tolerance):
            regressions.append(f"{key} regressed by {delta:.4f}")
    return deltas, regressions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PlotMagic evaluation harness.")
    parser.add_argument("--state", default="kerala")
    parser.add_argument(
        "--gold-file",
        default="tests/evaluation/gold_queries_kerala.json",
        help="Path to gold query dataset",
    )
    parser.add_argument(
        "--output",
        default="evaluation/latest_eval.json",
        help="Artifact output path",
    )
    parser.add_argument(
        "--baseline-artifact",
        default=None,
        help="Optional prior evaluation artifact for regression detection.",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.005,
        help="Allowed metric delta before flagging a regression.",
    )
    parser.add_argument(
        "--shadow-legacy",
        action="store_true",
        help="Run a legacy-provider shadow pass and report new-vs-legacy deltas.",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit non-zero when any gate/regression/shadow check fails.",
    )
    args = parser.parse_args()

    gold_file = (ROOT / args.gold_file).resolve()
    if not gold_file.exists():
        raise SystemExit(f"Gold dataset not found: {gold_file}")

    current = _evaluate_once(state=args.state, gold_file=gold_file)
    current_metrics = _extract_metric_view(current)

    baseline_payload: dict[str, Any] | None = None
    baseline_deltas: dict[str, float] = {}
    baseline_regressions: list[str] = []
    if args.baseline_artifact:
        baseline_path = (ROOT / args.baseline_artifact).resolve()
        if baseline_path.exists():
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_metrics = _extract_metric_view(baseline_payload)
            baseline_deltas, baseline_regressions = _baseline_regressions(
                current=current_metrics,
                baseline=baseline_metrics,
                tolerance=args.regression_tolerance,
            )

    shadow: dict[str, Any] | None = None
    if args.shadow_legacy:
        with temporary_env(_LEGACY_PROVIDER_ENV):
            legacy = _evaluate_once(state=args.state, gold_file=gold_file)
        legacy_metrics = _extract_metric_view(legacy)
        shadow_deltas, shadow_regressions = _baseline_regressions(
            current=current_metrics,
            baseline=legacy_metrics,
            tolerance=args.regression_tolerance,
        )
        shadow = {
            "legacy_metrics": legacy_metrics,
            "delta_vs_legacy": shadow_deltas,
            "regressions_vs_legacy": shadow_regressions,
            "shadow_passed": not shadow_regressions,
        }

    final_passed = bool(current.get("gate_passed", False))
    if baseline_regressions:
        final_passed = False
    if shadow and not shadow.get("shadow_passed", False):
        final_passed = False

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": args.state,
        "gold_file": str(gold_file),
        "query_count": current.get("query_count", 0),
        "current": current,
        "baseline": {
            "artifact": args.baseline_artifact,
            "deltas": baseline_deltas,
            "regressions": baseline_regressions,
        },
        "shadow": shadow,
        "final_gate_passed": final_passed,
    }

    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    if args.fail_on_gate and not final_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
