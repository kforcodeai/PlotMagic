from __future__ import annotations

from pathlib import Path

from src.retrieval.occupancy_resolver import OccupancyResolver


def test_occupancy_resolver_classifies_hotel_as_a2() -> None:
    root = Path(__file__).resolve().parents[1]
    resolver = OccupancyResolver(root / "config" / "states.yaml")
    result = resolver.resolve(state="kerala", building_description="A 3-storey hotel with 40 rooms")
    assert result.resolved is True
    assert result.selected == ["A2"]


def test_occupancy_resolver_handles_ambiguity() -> None:
    root = Path(__file__).resolve().parents[1]
    resolver = OccupancyResolver(root / "config" / "states.yaml")
    result = resolver.resolve(state="kerala", building_description="A residential and hotel mixed building")
    assert result.resolved is False
    assert result.clarification_questions

