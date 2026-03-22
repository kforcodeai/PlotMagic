from __future__ import annotations

from pathlib import Path

from src.retrieval.scope_resolver import ScopeResolver


def test_scope_resolver_maps_thrissur_to_municipality() -> None:
    root = Path(__file__).resolve().parents[1]
    resolver = ScopeResolver(
        root / "config" / "states.yaml",
        root / "config" / "local_bodies" / "kerala_local_bodies.yaml",
    )
    result = resolver.resolve(location="Thrissur", state_hint="kerala")
    assert result.resolved is True
    assert result.jurisdiction_type == "municipality"


def test_scope_resolver_asks_clarification_for_unknown_location() -> None:
    root = Path(__file__).resolve().parents[1]
    resolver = ScopeResolver(
        root / "config" / "states.yaml",
        root / "config" / "local_bodies" / "kerala_local_bodies.yaml",
    )
    result = resolver.resolve(location="unknown place", state_hint="kerala")
    assert result.resolved is False
    assert result.clarification_questions

