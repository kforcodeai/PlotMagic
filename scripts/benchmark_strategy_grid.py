from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WEIGHTED_WINNER_METRICS = {
    "mandatory_component_completeness_mean": 0.40,
    "token_set_recall_mean": 0.25,
    "token_set_f1_mean": 0.20,
    "numeric_token_recall_mean": 0.15,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _csv_values(raw: str, *, cast) -> list[Any]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return [cast(item) for item in items]


def _strategy_id(strategy: dict[str, Any], idx: int) -> str:
    mode = str(strategy["retrieval_mode"]).replace("_", "-")
    top_k = strategy["top_k"]
    pool = str(strategy["candidate_pool_factor"]).replace(".", "p")
    category = strategy["category_filter_policy"]
    fallback = strategy["fallback_policy"].replace("_", "-")
    min_ev = str(strategy["min_evidence_score"]).replace(".", "p")
    return f"s{idx:03d}-{mode}-k{top_k}-pf{pool}-cat-{category}-fb-{fallback}-ev{min_ev}"


def _run_single_strategy(
    *,
    strategy: dict[str, Any],
    run_id: str,
    strict: bool,
    skip_cold: bool,
    output_dir: Path,
    dataset: str,
    baseline: str,
    state: str,
    jurisdiction: str,
    category: str,
    vector_db_path: str,
) -> None:
    env = os.environ.copy()
    env["PLOTMAGIC_RETRIEVAL_POOL_FACTOR"] = str(strategy["candidate_pool_factor"])
    env["PLOTMAGIC_CATEGORY_FILTER_POLICY"] = str(strategy["category_filter_policy"])
    env["PLOTMAGIC_RETRIEVAL_FALLBACK_POLICY"] = str(strategy["fallback_policy"])
    env["PLOTMAGIC_RETRIEVAL_MIN_EVIDENCE_SCORE"] = str(strategy["min_evidence_score"])

    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "benchmark_qna_answers.py"),
        "--dataset",
        dataset,
        "--baseline",
        baseline,
        "--state",
        state,
        "--jurisdiction",
        jurisdiction,
        "--category",
        category,
        "--top-k",
        str(strategy["top_k"]),
        "--retrieval-mode",
        str(strategy["retrieval_mode"]),
        "--vector-backend",
        "qdrant_local",
        "--vector-db-path",
        vector_db_path,
        "--embedding-provider",
        "openai_embedding",
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
    ]
    if strict:
        cmd.append("--strict-providers")
    else:
        cmd.append("--no-strict-providers")
    if skip_cold:
        cmd.append("--skip-cold")
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Strategy run failed ({run_id}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _gate_eval(summary: dict[str, Any], per_query: list[dict[str, Any]], baseline_summary: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    mode = str(summary.get("retrieval_mode", ""))
    citation_groundedness = _safe_float(summary.get("citation_groundedness_mean"), 0.0)
    contract_violation_rate = _safe_float(summary.get("contract_violation_rate"), 1.0)
    vector_hits_zero = int(summary.get("vector_hits_zero_count", 0) or 0)
    lexical_hits_zero = int(summary.get("lexical_hits_zero_count", 0) or 0)

    if citation_groundedness != 1.0:
        failures.append(f"hard gate: citation_groundedness_mean={citation_groundedness} != 1.0")
    if contract_violation_rate != 0.0:
        failures.append(f"hard gate: contract_violation_rate={contract_violation_rate} != 0.0")
    if mode == "lexical_only_bm25":
        if lexical_hits_zero != 0:
            failures.append(f"hard gate: lexical_hits_zero_count={lexical_hits_zero} != 0")
    else:
        if vector_hits_zero != 0:
            failures.append(f"hard gate: vector_hits_zero_count={vector_hits_zero} != 0")

    improvements = {
        "token_set_f1_mean": 0.10,
        "token_set_recall_mean": 0.10,
        "numeric_token_recall_mean": 0.15,
        "mandatory_component_completeness_mean": 0.10,
    }
    for metric, threshold in improvements.items():
        cur = float(summary.get(metric, 0.0) or 0.0)
        base = float(baseline_summary.get(metric, 0.0) or 0.0)
        if (cur - base) < threshold:
            failures.append(f"improvement gate: {metric} delta {(cur-base):.4f} < {threshold:.4f}")

    per_query_failures: list[dict[str, Any]] = []
    for row in per_query:
        comp = float(row.get("mandatory_component_completeness", 0.0) or 0.0)
        if comp < 0.50:
            per_query_failures.append({"question": row.get("question"), "reason": "mandatory_component_completeness < 0.50"})
        if bool(row.get("numeric_heavy")):
            num = float(row.get("numeric_token_recall", 0.0) or 0.0)
            if num < 0.50:
                per_query_failures.append({"question": row.get("question"), "reason": "numeric-heavy numeric_token_recall < 0.50"})
    if per_query_failures:
        failures.append(f"per-query gates failed: {len(per_query_failures)}")

    weighted = 0.0
    for metric, weight in WEIGHTED_WINNER_METRICS.items():
        weighted += float(summary.get(metric, 0.0) or 0.0) * weight

    return {
        "passed_all": not failures,
        "failures": failures,
        "per_query_failures": per_query_failures,
        "weighted_score": weighted,
    }


def _dominant_failed_gates(results: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in results:
        for failure in row["gates"]["failures"]:
            counter[str(failure)] += 1
    return [{"failure": failure, "count": count} for failure, count in counter.most_common(limit)]


def _top_blocking_queries(per_query: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = sorted(per_query, key=lambda item: _safe_float(item.get("token_set_f1", 0.0)))
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        out.append(
            {
                "question": row.get("question"),
                "token_set_f1": _safe_float(row.get("token_set_f1")),
                "token_set_recall": _safe_float(row.get("token_set_recall")),
                "numeric_token_recall": _safe_float(row.get("numeric_token_recall")),
                "retrieval_ceiling_token_set_recall": _safe_float(row.get("retrieval_ceiling_token_set_recall")),
                "retrieval_ceiling_numeric_token_recall": _safe_float(
                    row.get("retrieval_ceiling_numeric_token_recall")
                ),
                "synthesis_efficiency_recall": _safe_float(row.get("synthesis_efficiency_recall")),
                "synthesis_efficiency_numeric_recall": _safe_float(
                    row.get("synthesis_efficiency_numeric_recall")
                ),
                "bottleneck_class": row.get("bottleneck_class"),
                "fallback_stage": row.get("fallback_stage"),
                "vector_hits": int(row.get("vector_hits", 0) or 0),
                "lexical_hits": int(row.get("lexical_hits", 0) or 0),
            }
        )
    return out


def _rca_blockers(best_f1_row: dict[str, Any], all_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = best_f1_row["summary"]
    per_query = best_f1_row["per_query"]
    ceiling_recall = _safe_float(summary.get("retrieval_ceiling_token_set_recall_mean"))
    answer_recall = _safe_float(summary.get("token_set_recall_mean"))
    ceiling_f1 = _safe_float(summary.get("retrieval_ceiling_token_set_f1_mean"))
    answer_f1 = _safe_float(summary.get("token_set_f1_mean"))
    ceiling_numeric = _safe_float(summary.get("retrieval_ceiling_numeric_token_recall_mean"))
    answer_numeric = _safe_float(summary.get("numeric_token_recall_mean"))

    bottlenecks = summary.get("bottleneck_counts")
    if not isinstance(bottlenecks, dict):
        counts = Counter(str(row.get("bottleneck_class", "unknown")) for row in per_query)
        bottlenecks = {key: int(value) for key, value in counts.items()}

    return {
        "reference_strategy_id": best_f1_row["strategy_id"],
        "reference_run_id": best_f1_row["run_id"],
        "reference_strategy": best_f1_row["strategy"],
        "reference_mode": summary.get("retrieval_mode"),
        "reference_metrics": {
            "token_set_f1_mean": answer_f1,
            "token_set_recall_mean": answer_recall,
            "numeric_token_recall_mean": answer_numeric,
            "retrieval_ceiling_token_set_f1_mean": ceiling_f1,
            "retrieval_ceiling_token_set_recall_mean": ceiling_recall,
            "retrieval_ceiling_numeric_token_recall_mean": ceiling_numeric,
            "synthesis_efficiency_recall_mean": _safe_float(summary.get("synthesis_efficiency_recall_mean")),
            "synthesis_efficiency_numeric_recall_mean": _safe_float(
                summary.get("synthesis_efficiency_numeric_recall_mean")
            ),
        },
        "mean_gap_diagnosis": {
            "retrieval_ceiling_minus_answer_recall": max(0.0, ceiling_recall - answer_recall),
            "retrieval_ceiling_minus_answer_f1": max(0.0, ceiling_f1 - answer_f1),
            "retrieval_ceiling_minus_answer_numeric_recall": max(0.0, ceiling_numeric - answer_numeric),
        },
        "bottleneck_distribution": bottlenecks,
        "top_blocking_queries": _top_blocking_queries(per_query),
        "dominant_failed_gates_across_grid": _dominant_failed_gates(all_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict benchmark across retrieval strategy permutations.")
    parser.add_argument("--dataset", default="evaluation/kpbr/qna_panchayat.json")
    parser.add_argument("--baseline", default="evaluation/kpbr/qna_panchayat_answer_benchmark_final.json")
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="panchayat")
    parser.add_argument("--category", default="Category-II")
    parser.add_argument("--vector-db-path", default=".cache/qdrant_kpbr")
    parser.add_argument("--output-dir", default="evaluation/kpbr")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--strict-providers", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--retrieval-modes", default="vector_only,hybrid_no_reranker,lexical_only_bm25,hybrid_graph_reranker")
    parser.add_argument("--top-k-values", default="8,12,16,20")
    parser.add_argument("--candidate-pool-factors", default="2,4")
    parser.add_argument("--category-filter-policies", default="strict,soft")
    parser.add_argument("--fallback-policies", default="strict_only,relax_category,relax_category_then_occupancy")
    parser.add_argument("--min-evidence-scores", default="0.05,0.08,0.12")
    parser.add_argument("--max-strategies", type=int, default=0, help="Optional cap for dry-runs/debug.")
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    baseline_path = (ROOT / args.baseline).resolve()
    if not baseline_path.exists():
        raise SystemExit(f"Baseline artifact not found: {baseline_path}")
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_summary = baseline_payload.get("summary") if isinstance(baseline_payload, dict) else None
    if not isinstance(baseline_summary, dict):
        raise SystemExit("Baseline summary missing/invalid in baseline artifact.")

    retrieval_modes = _csv_values(args.retrieval_modes, cast=str)
    top_k_values = _csv_values(args.top_k_values, cast=int)
    pool_factors = _csv_values(args.candidate_pool_factors, cast=float)
    category_policies = _csv_values(args.category_filter_policies, cast=str)
    fallback_policies = _csv_values(args.fallback_policies, cast=str)
    min_evidence_scores = _csv_values(args.min_evidence_scores, cast=float)

    strategies: list[dict[str, Any]] = []
    for retrieval_mode, top_k, pool_factor, category_policy, fallback_policy, min_evidence in itertools.product(
        retrieval_modes,
        top_k_values,
        pool_factors,
        category_policies,
        fallback_policies,
        min_evidence_scores,
    ):
        if retrieval_mode == "lexical_only_bm25" and fallback_policy != "strict_only":
            # Fallback policy is vector-specific; keep lexical permutations non-redundant.
            continue
        strategies.append(
            {
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "candidate_pool_factor": pool_factor,
                "category_filter_policy": category_policy,
                "fallback_policy": fallback_policy,
                "min_evidence_score": min_evidence,
            }
        )
    if args.max_strategies > 0:
        strategies = strategies[: args.max_strategies]

    if not strategies:
        raise SystemExit("No strategies to run after applying permutation filters.")

    results: list[dict[str, Any]] = []
    first = True
    for idx, strategy in enumerate(strategies, start=1):
        sid = _strategy_id(strategy, idx)
        strategy_run_id = f"{run_id}_{sid}"
        _run_single_strategy(
            strategy=strategy,
            run_id=strategy_run_id,
            strict=args.strict_providers,
            skip_cold=not first,
            output_dir=output_dir,
            dataset=args.dataset,
            baseline=args.baseline,
            state=args.state,
            jurisdiction=args.jurisdiction,
            category=args.category,
            vector_db_path=args.vector_db_path,
        )
        first = False

        warm_answer = output_dir / f"qna_panchayat_answer_benchmark_warm_{strategy['retrieval_mode']}_{strategy_run_id}.json"
        if not warm_answer.exists():
            raise RuntimeError(f"Expected warm artifact missing: {warm_answer}")
        payload = json.loads(warm_answer.read_text(encoding="utf-8"))
        summary = payload["summary"]
        per_query = payload["per_query"]
        gates = _gate_eval(summary, per_query, baseline_summary)
        results.append(
            {
                "strategy_id": sid,
                "run_id": strategy_run_id,
                "strategy": strategy,
                "summary": summary,
                "per_query": per_query,
                "gates": gates,
                "answer_artifact": str(warm_answer),
                "retrieval_artifact": str(
                    output_dir / f"qna_panchayat_retrieval_trace_warm_{strategy['retrieval_mode']}_{strategy_run_id}.json"
                ),
            }
        )

    passing = [row for row in results if row["gates"]["passed_all"]]
    if len(passing) == 1:
        selected = passing[0]
    elif len(passing) > 1:
        selected = sorted(passing, key=lambda item: item["gates"]["weighted_score"], reverse=True)[0]
    else:
        selected = None

    ranked = sorted(results, key=lambda item: item["gates"]["weighted_score"], reverse=True)
    best_f1 = max(results, key=lambda item: _safe_float(item["summary"].get("token_set_f1_mean", 0.0)))
    best_weighted = ranked[0] if ranked else None
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_providers": args.strict_providers,
        "total_strategies_evaluated": len(results),
        "selected_strategy_id": selected["strategy_id"] if selected else "no_winner",
        "selected_strategy": selected["strategy"] if selected else None,
        "selected_run_id": selected["run_id"] if selected else None,
        "best_weighted_strategy_id": best_weighted["strategy_id"] if best_weighted else None,
        "best_weighted_run_id": best_weighted["run_id"] if best_weighted else None,
        "best_f1_strategy_id": best_f1["strategy_id"],
        "best_f1_run_id": best_f1["run_id"],
        "best_f1_score": _safe_float(best_f1["summary"].get("token_set_f1_mean", 0.0)),
        "top_10_ranked_strategies": [
            {
                "strategy_id": row["strategy_id"],
                "run_id": row["run_id"],
                "strategy": row["strategy"],
                "weighted_score": row["gates"]["weighted_score"],
                "passed_all_gates": row["gates"]["passed_all"],
                "failure_count": len(row["gates"]["failures"]),
            }
            for row in ranked[:10]
        ],
        "gate_results_by_strategy": {
            row["strategy_id"]: {
                "run_id": row["run_id"],
                "strategy": row["strategy"],
                "passed_all": row["gates"]["passed_all"],
                "weighted_score": row["gates"]["weighted_score"],
                "failures": row["gates"]["failures"],
                "per_query_failures": row["gates"]["per_query_failures"],
                "artifacts": {
                    "answer": row["answer_artifact"],
                    "retrieval": row["retrieval_artifact"],
                },
            }
            for row in results
        },
        "retrieval_vs_synthesis_rca": _rca_blockers(best_f1, results),
    }

    out_path = output_dir / f"winner_report_grid_{run_id}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "winner_report": str(out_path), "selected_strategy_id": report["selected_strategy_id"]}, indent=2))


if __name__ == "__main__":
    main()
