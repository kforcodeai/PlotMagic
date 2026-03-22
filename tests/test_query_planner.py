from __future__ import annotations

from src.retrieval.query_planner import QueryPlanner


def test_query_planner_keeps_topics_empty_when_not_detected() -> None:
    plan = QueryPlanner().plan("Explain the overall compliance flow in Kerala panchayat.")
    assert plan.topics == []


def test_query_planner_detects_solar_assisted_as_environment_topic() -> None:
    plan = QueryPlanner().plan(
        "When does solar-assisted heating and lighting become mandatory and what is verified before occupancy certificate?"
    )
    assert "environment" in plan.topics


def test_query_planner_ignores_non_rule_tokens_in_rule_mentions() -> None:
    plan = QueryPlanner().plan("Does rule apply to huts, and does Rule 133 apply to this case?")
    assert "apply" not in [item.lower() for item in plan.mentioned_rule_numbers]
    assert "133" in plan.mentioned_rule_numbers
