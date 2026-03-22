from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalEvidence:
    claim_id: str
    topic: str
    chunk_id: str
    document_id: str
    text: str
    scores: dict[str, float] = field(default_factory=dict)
    source: str = ""
    has_sufficient_support: bool = False


@dataclass(slots=True)
class EvidenceMatrix:
    items: list[RetrievalEvidence] = field(default_factory=list)

    def add(self, item: RetrievalEvidence) -> None:
        self.items.append(item)

    def for_topic(self, topic: str) -> list[RetrievalEvidence]:
        return [item for item in self.items if item.topic == topic]

    def supported_topics(self) -> set[str]:
        return {item.topic for item in self.items if item.has_sufficient_support}

