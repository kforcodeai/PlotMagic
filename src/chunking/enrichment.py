from __future__ import annotations

import re

from src.chunking.chunks import Chunk

_GENERIC_LEGAL_TOPICS: dict[str, tuple[str, ...]] = {
    "permit": ("permit", "license", "approval", "authorisation", "authorization"),
    "fees": ("fee", "fees", "charge", "charges", "penalty amount"),
    "penalty": ("penalty", "fine", "punish", "offence", "offense"),
    "appeal": ("appeal", "review", "revision"),
    "timeline": ("within", "days", "months", "years", "deadline"),
    "definition": ("means", "defined as", "definition"),
    "exemption": ("exempt", "exception", "not required", "waiver"),
}


class MetadataEnricher:
    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            text_l = chunk.text.lower()
            tags = set(chunk.metadata.get("topic_tags", []))
            for topic, keywords in _GENERIC_LEGAL_TOPICS.items():
                if any(keyword in text_l for keyword in keywords):
                    tags.add(topic)
            if re.search(r"\b(?:shall|must|required)\b", text_l):
                tags.add("obligation")
            if re.search(r"\b(?:may|can|permitted)\b", text_l):
                tags.add("permission")
            chunk.metadata["topic_tags"] = sorted(tags)
        return chunks
