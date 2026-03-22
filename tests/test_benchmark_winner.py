from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "benchmark_qna_answers.py"
    spec = importlib.util.spec_from_file_location("benchmark_qna_answers", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_winner_decision_selects_only_mode_passing_all_gates() -> None:
    module = _load_module()
    baseline = {
        "token_set_f1_mean": 0.5,
        "token_set_recall_mean": 0.5,
        "numeric_token_recall_mean": 0.5,
        "mandatory_component_completeness_mean": 0.5,
    }
    mode_payloads = {
        "vector_only": {
            "summary": {
                "citation_groundedness_mean": 1.0,
                "contract_violation_rate": 0.0,
                "vector_hits_zero_count": 0,
                "token_set_f1_mean": 0.7,
                "token_set_recall_mean": 0.7,
                "numeric_token_recall_mean": 0.7,
                "mandatory_component_completeness_mean": 0.7,
            },
            "per_query": [
                {
                    "question": "q1",
                    "mandatory_component_completeness": 0.8,
                    "numeric_heavy": True,
                    "numeric_token_recall": 0.8,
                }
            ],
        },
        "hybrid_no_reranker": {
            "summary": {
                "citation_groundedness_mean": 1.0,
                "contract_violation_rate": 0.2,
                "vector_hits_zero_count": 0,
                "token_set_f1_mean": 0.8,
                "token_set_recall_mean": 0.8,
                "numeric_token_recall_mean": 0.8,
                "mandatory_component_completeness_mean": 0.8,
            },
            "per_query": [
                {
                    "question": "q1",
                    "mandatory_component_completeness": 0.8,
                    "numeric_heavy": True,
                    "numeric_token_recall": 0.8,
                }
            ],
        },
    }
    winner = module._winner_decision(mode_payloads=mode_payloads, baseline_summary=baseline)
    assert winner["selected_mode"] == "vector_only"


def test_winner_decision_returns_no_winner_when_both_fail() -> None:
    module = _load_module()
    baseline = {
        "token_set_f1_mean": 0.9,
        "token_set_recall_mean": 0.9,
        "numeric_token_recall_mean": 0.9,
        "mandatory_component_completeness_mean": 0.9,
    }
    mode_payloads = {
        "vector_only": {
            "summary": {
                "citation_groundedness_mean": 0.0,
                "contract_violation_rate": 0.5,
                "vector_hits_zero_count": 3,
                "token_set_f1_mean": 0.2,
                "token_set_recall_mean": 0.2,
                "numeric_token_recall_mean": 0.2,
                "mandatory_component_completeness_mean": 0.2,
            },
            "per_query": [{"question": "q1", "mandatory_component_completeness": 0.2, "numeric_heavy": True, "numeric_token_recall": 0.2}],
        },
        "hybrid_no_reranker": {
            "summary": {
                "citation_groundedness_mean": 0.0,
                "contract_violation_rate": 0.5,
                "vector_hits_zero_count": 2,
                "token_set_f1_mean": 0.3,
                "token_set_recall_mean": 0.3,
                "numeric_token_recall_mean": 0.3,
                "mandatory_component_completeness_mean": 0.3,
            },
            "per_query": [{"question": "q1", "mandatory_component_completeness": 0.2, "numeric_heavy": True, "numeric_token_recall": 0.2}],
        },
    }
    winner = module._winner_decision(mode_payloads=mode_payloads, baseline_summary=baseline)
    assert winner["selected_mode"] == "no_winner"
