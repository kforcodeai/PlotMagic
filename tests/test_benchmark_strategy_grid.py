from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "benchmark_strategy_grid.py"
    spec = importlib.util.spec_from_file_location("benchmark_strategy_grid", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gate_eval_treats_zero_contract_violation_as_pass() -> None:
    module = _load_module()
    summary = {
        "retrieval_mode": "lexical_only_bm25",
        "citation_groundedness_mean": 1.0,
        "contract_violation_rate": 0.0,
        "lexical_hits_zero_count": 0,
        "token_set_f1_mean": 0.30,
        "token_set_recall_mean": 0.30,
        "numeric_token_recall_mean": 0.30,
        "mandatory_component_completeness_mean": 0.30,
    }
    per_query = [
        {
            "question": "q1",
            "mandatory_component_completeness": 0.9,
            "numeric_heavy": True,
            "numeric_token_recall": 0.9,
        }
    ]
    baseline = {
        "token_set_f1_mean": 0.0,
        "token_set_recall_mean": 0.0,
        "numeric_token_recall_mean": 0.0,
        "mandatory_component_completeness_mean": 0.0,
    }
    result = module._gate_eval(summary, per_query, baseline)
    assert result["passed_all"] is True
    assert all("contract_violation_rate" not in item for item in result["failures"])


def test_gate_eval_fails_when_contract_violation_is_non_zero() -> None:
    module = _load_module()
    summary = {
        "retrieval_mode": "lexical_only_bm25",
        "citation_groundedness_mean": 1.0,
        "contract_violation_rate": 0.5,
        "lexical_hits_zero_count": 0,
        "token_set_f1_mean": 0.30,
        "token_set_recall_mean": 0.30,
        "numeric_token_recall_mean": 0.30,
        "mandatory_component_completeness_mean": 0.30,
    }
    per_query = [
        {
            "question": "q1",
            "mandatory_component_completeness": 0.9,
            "numeric_heavy": False,
            "numeric_token_recall": 0.9,
        }
    ]
    baseline = {
        "token_set_f1_mean": 0.0,
        "token_set_recall_mean": 0.0,
        "numeric_token_recall_mean": 0.0,
        "mandatory_component_completeness_mean": 0.0,
    }
    result = module._gate_eval(summary, per_query, baseline)
    assert result["passed_all"] is False
    assert any("contract_violation_rate" in item for item in result["failures"])
