from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.evaluation.metrics import EvaluationMetrics


@dataclass(slots=True)
class GateResult:
    passed: bool
    failures: list[str]


class ReleaseGateEvaluator:
    def __init__(self, metrics_config_path: Path) -> None:
        cfg = yaml.safe_load(metrics_config_path.read_text(encoding="utf-8"))
        self.thresholds = cfg["thresholds"]

    def evaluate(self, metrics: EvaluationMetrics) -> GateResult:
        failures: list[str] = []
        if metrics.jurisdiction_accuracy < self.thresholds["jurisdiction_accuracy_min"]:
            failures.append("jurisdiction_accuracy below threshold")
        if metrics.occupancy_accuracy < self.thresholds["occupancy_accuracy_min"]:
            failures.append("occupancy_accuracy below threshold")
        if metrics.clause_recall_at_20 < self.thresholds["clause_recall_at_20_min"]:
            failures.append("clause_recall_at_20 below threshold")
        if metrics.citation_precision < self.thresholds["citation_precision_min"]:
            failures.append("citation_precision below threshold")
        if metrics.numeric_correctness < self.thresholds["numeric_correctness_min"]:
            failures.append("numeric_correctness below threshold")
        if metrics.unsupported_claim_rate > self.thresholds["unsupported_claim_rate_max"]:
            failures.append("unsupported_claim_rate above threshold")
        if metrics.semantic_accuracy_proxy < self.thresholds.get("semantic_accuracy_proxy_min", 0.0):
            failures.append("semantic_accuracy_proxy below threshold")
        if metrics.mandatory_component_completeness < self.thresholds.get("mandatory_component_completeness_min", 0.0):
            failures.append("mandatory_component_completeness below threshold")
        if metrics.contradiction_rate > self.thresholds.get("contradiction_rate_max", 1.0):
            failures.append("contradiction_rate above threshold")
        if metrics.citation_groundedness < self.thresholds.get("citation_groundedness_min", 0.0):
            failures.append("citation_groundedness below threshold")
        if metrics.abstention_correctness < self.thresholds.get("abstention_correctness_min", 0.0):
            failures.append("abstention_correctness below threshold")
        if metrics.contract_violation_rate > self.thresholds.get("contract_violation_rate_max", 1.0):
            failures.append("contract_violation_rate above threshold")
        return GateResult(passed=not failures, failures=failures)
