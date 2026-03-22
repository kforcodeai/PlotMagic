from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    level: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

