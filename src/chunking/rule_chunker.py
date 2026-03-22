from __future__ import annotations

import re

from src.chunking.chunks import Chunk
from src.models import RuleDocument


class RuleChunker:
    def chunk(self, doc: RuleDocument, max_chars: int = 3500, sentence_overlap: int = 1) -> list[Chunk]:
        prefix = self._doc_context(doc)
        chunks: list[Chunk] = []
        clause_nodes = doc.clause_nodes or []

        if not clause_nodes:
            return self._split_text_fallback(doc, prefix, max_chars=max_chars, sentence_overlap=sentence_overlap)

        for clause in clause_nodes:
            clause_text = (clause.normalized_text or clause.raw_text or "").strip()
            if not clause_text:
                continue
            clause_prefix = f"{prefix}\nCitation: {clause.display_citation}"
            metadata = {
                "is_generic": doc.is_generic,
                "occupancy_groups": doc.occupancy_groups,
                "clause_id": clause.clause_id,
                "clause_type": clause.clause_type.value,
                "anchor_id": clause.anchor_id,
                "topic_tags": clause.topic_tags,
            }
            full = f"{clause_prefix}\n{clause_text}".strip()
            if len(full) <= max_chars:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}-{self._slug(clause.clause_id)}",
                        document_id=doc.document_id,
                        level=clause.clause_type.value,
                        text=full,
                        metadata=metadata,
                    )
                )
                continue
            parts = self._split_by_sentences(clause_text, max_chars=max_chars - len(clause_prefix), overlap=sentence_overlap)
            for index, part in enumerate(parts, start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}-{self._slug(clause.clause_id)}-p{index}",
                        document_id=doc.document_id,
                        level=clause.clause_type.value,
                        text=f"{clause_prefix}\n{part}".strip(),
                        metadata={**metadata, "part": index},
                    )
                )
        return chunks

    def _split_text_fallback(self, doc: RuleDocument, prefix: str, *, max_chars: int, sentence_overlap: int) -> list[Chunk]:
        parts = self._split_by_sentences(doc.full_text, max_chars=max_chars - len(prefix), overlap=sentence_overlap)
        if not parts:
            return []
        chunks: list[Chunk] = []
        for index, part in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}-p{index}",
                    document_id=doc.document_id,
                    level="rule",
                    text=f"{prefix}\n{part}".strip(),
                    metadata={"is_generic": doc.is_generic, "occupancy_groups": doc.occupancy_groups, "part": index},
                )
            )
        return chunks

    def _split_by_sentences(self, text: str, *, max_chars: int, overlap: int) -> list[str]:
        max_chars = max(500, max_chars)
        sentences = [item.strip() for item in re.split(r"(?<=[.;:])\s+|\n+", text) if item.strip()]
        if not sentences:
            compact = re.sub(r"\s+", " ", text).strip()
            return [compact] if compact else []

        chunks: list[str] = []
        buffer: list[str] = []
        for sentence in sentences:
            candidate = " ".join(buffer + [sentence]).strip()
            if len(candidate) <= max_chars:
                buffer.append(sentence)
                continue
            if buffer:
                chunks.append(" ".join(buffer).strip())
                buffer = buffer[-overlap:] if overlap > 0 else []
            if len(sentence) > max_chars:
                chunks.extend(self._hard_split(sentence, max_chars=max_chars))
                buffer = []
            else:
                buffer.append(sentence)
        if buffer:
            chunks.append(" ".join(buffer).strip())
        return chunks

    def _hard_split(self, text: str, *, max_chars: int) -> list[str]:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return []
        return [compact[index : index + max_chars] for index in range(0, len(compact), max_chars)]

    def _doc_context(self, doc: RuleDocument) -> str:
        return (
            f"[{doc.ruleset_id}] Chapter {doc.chapter_number}: {doc.chapter_title} "
            f"> Rule {doc.rule_number}: {doc.rule_title}"
        )

    def _slug(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "x"
