from __future__ import annotations

from pathlib import Path

from src.api.service import ComplianceEngine


def test_generic_query_can_skip_occupancy() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = ComplianceEngine(root=root)
    query = "How is maximum building height computed from street width and setback?"
    plan = engine.query_planner.plan(query)
    assert engine._can_skip_occupancy(plan=plan, query=query) is True


def test_project_specific_query_requires_occupancy() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = ComplianceEngine(root=root)
    query = "Can I build a house for my plot in Kerala panchayat?"
    plan = engine.query_planner.plan(query)
    assert engine._can_skip_occupancy(plan=plan, query=query) is False
