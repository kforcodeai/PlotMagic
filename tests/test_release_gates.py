from __future__ import annotations

from pathlib import Path

from src.evaluation import EvaluationMetrics, ReleaseGateEvaluator


def test_release_gates_pass_for_high_quality_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluator = ReleaseGateEvaluator(root / "config" / "evaluation" / "metrics.yaml")
    metrics = EvaluationMetrics(
        jurisdiction_accuracy=1.0,
        occupancy_accuracy=0.98,
        clause_recall_at_20=0.96,
        citation_precision=1.0,
        numeric_correctness=1.0,
        unsupported_claim_rate=0.0,
    )
    result = evaluator.evaluate(metrics)
    assert result.passed is True
    assert result.failures == []

