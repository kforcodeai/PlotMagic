from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.api.service import ComplianceEngine


@lru_cache(maxsize=1)
def get_engine() -> ComplianceEngine:
    root = Path(__file__).resolve().parents[2]
    return ComplianceEngine(root=root)

