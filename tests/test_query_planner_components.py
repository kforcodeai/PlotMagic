from __future__ import annotations

from src.retrieval.query_planner import QueryPlanner


def test_query_planner_canonical_components_and_dedup() -> None:
    planner = QueryPlanner()
    plan = planner.plan(
        "If deemed approval is delayed, what timeline applies and what exception/provided further clause controls it?"
    )
    assert "deemed_approval" in plan.mandatory_components
    assert "timeline" in plan.mandatory_components
    # Canonical key should not be duplicated with slug/phrase variants.
    assert plan.mandatory_components.count("deemed_approval") == 1


def test_query_planner_detects_distance_authority_numeric_slots() -> None:
    planner = QueryPlanner()
    plan = planner.plan(
        "Within 100m of defence property and 30m of railway boundary, what authority and timeline apply?"
    )
    assert "distance" in plan.mandatory_components
    assert "authority" in plan.mandatory_components
    assert any(item.startswith("numeric:100") for item in plan.mandatory_components)
    assert any(item.startswith("numeric:30") for item in plan.mandatory_components)
