from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvaluationMetrics:
    jurisdiction_accuracy: float
    occupancy_accuracy: float
    clause_recall_at_20: float
    citation_precision: float
    numeric_correctness: float
    unsupported_claim_rate: float
    semantic_accuracy_proxy: float = 1.0
    mandatory_component_completeness: float = 1.0
    contradiction_rate: float = 0.0
    citation_groundedness: float = 1.0
    abstention_correctness: float = 1.0
    contract_violation_rate: float = 0.0


def accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return correct / total


def recall_at_k(retrieved_relevant: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return retrieved_relevant / total_relevant


def precision(correct_supported_claims: int, total_claims: int) -> float:
    if total_claims <= 0:
        return 0.0
    return correct_supported_claims / total_claims
