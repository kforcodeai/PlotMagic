from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.multihop_eval import (
    GoldLabels,
    classify_bottleneck,
    compute_retrieval_quality,
    load_multihop_dataset,
    map_rule_number_for_ref,
    parse_citation_ref,
    build_line_to_rule,
    _stable_sort,
)
from src.evaluation.multihop_eval import build_retrieval_strategies


def test_dataset_parser_rejects_malformed_citation(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "x1",
                "query": "q",
                "query_plan": [{"hop": 1, "sub_query": "sub"}],
                "ground_truth_chunks": [{"citation": "malformed", "snippet": "text"}],
                "final_answer": "a",
                "answer_evidence": ["data/file.md:1-2"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_multihop_dataset(dataset)


def test_line_range_to_rule_mapping(tmp_path: Path) -> None:
    source = tmp_path / "rules.md"
    source.write_text(
        "\n".join(
            [
                "## 1. Preliminary",
                "Text 1",
                "## 2. Applicability",
                "Text 2",
                "## Appendix A",
                "Appendix text",
            ]
        ),
        encoding="utf-8",
    )
    mapping = build_line_to_rule(source)
    ref_rule_1 = parse_citation_ref(f"{source}:1-2")
    ref_rule_2 = parse_citation_ref(f"{source}:3-4")
    ref_appendix = parse_citation_ref(f"{source}:6-6")
    assert map_rule_number_for_ref(ref_rule_1, mapping) == "1"
    assert map_rule_number_for_ref(ref_rule_2, mapping) == "2"
    assert map_rule_number_for_ref(ref_appendix, mapping) is None


def test_hop_and_snippet_recall_computation() -> None:
    gold = GoldLabels(
        record_id="r1",
        gold_rules={"5", "6"},
        chunk_canonicals=["doc.md:10-12", "doc.md:20-22"],
        chunk_tokens=[{"consult", "defence", "30"}, {"railway", "objection", "30"}],
        chunk_snippets=[
            "consult defence authority within 30 days",
            "railway objection received within 30 days",
        ],
        hop_targets={1: {0}, 2: {1}},
    )
    metrics = compute_retrieval_quality(
        gold=gold,
        retrieved_texts=[
            "Secretary shall consult defence authority within 30 days",
            "Railway objection is to be considered within 30 days",
        ],
        ranked_rules=["6", "9", "5"],
        zero_hit=False,
    )
    assert metrics["rule_hit_at_k"] == 1.0
    assert metrics["rule_recall_at_k"] == 1.0
    assert metrics["hop_coverage_at_k"] == 1.0
    assert metrics["snippet_recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_stable_sort_tie_breaks_by_secondary_keys() -> None:
    rows = [
        {"name": "b", "score": 0.9},
        {"name": "a", "score": 0.9},
    ]
    sorted_rows = _stable_sort(rows, "score", ["name"])
    assert [row["name"] for row in sorted_rows] == ["a", "b"]


def test_rca_bucket_assignment() -> None:
    assert classify_bottleneck(oracle_context_f1=0.8, retrieved_context_extractive_f1=0.5, end_to_end_f1=0.48) == "retrieval_bottleneck"
    assert classify_bottleneck(oracle_context_f1=0.8, retrieved_context_extractive_f1=0.78, end_to_end_f1=0.5) == "synthesis_bottleneck"
    assert classify_bottleneck(oracle_context_f1=0.8, retrieved_context_extractive_f1=0.6, end_to_end_f1=0.41) == "mixed"


def test_core_strategy_grid_includes_new_graph_and_agentic_strategies() -> None:
    names = [item.name for item in build_retrieval_strategies(tier="core", default_top_k=12)]
    assert "structure_hybrid_graph_llm_rerank" in names
    assert "agentic_dynamic_llm" in names
