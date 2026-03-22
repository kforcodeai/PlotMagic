from __future__ import annotations

from pathlib import Path

from src.statepack import StatePackFactory


def test_statepack_factory_loads_kerala_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    pack = StatePackFactory(root).load("kerala")
    assert pack.state_code == "kerala"
    assert "municipality" in pack.parser_registry
    assert pack.validate() == []

