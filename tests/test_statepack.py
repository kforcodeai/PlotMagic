from __future__ import annotations

from pathlib import Path

from src.statepack import kerala_statepack


def test_kerala_statepack_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    pack = kerala_statepack(root)
    warnings = pack.validate()
    assert warnings == []

