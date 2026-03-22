from __future__ import annotations

from typing import Iterable

from src.models import Citation, RuleDocument
from src.retrieval.evidence import EvidenceMatrix


class CitationBuilder:
    def build(self, evidence: EvidenceMatrix, docs: dict[str, RuleDocument]) -> list[Citation]:
        citations: list[Citation] = []
        for item in evidence.items:
            doc = docs.get(item.document_id)
            if not doc:
                continue
            citations.append(
                Citation(
                    claim_id=item.claim_id,
                    ruleset_id=doc.ruleset_id,
                    ruleset_name=doc.ruleset_id.replace("_", " "),
                    chapter_number=doc.chapter_number,
                    rule_number=doc.rule_number,
                    sub_rule_path="",
                    table_ref=self._table_ref(doc),
                    anchor_id=doc.anchor_id,
                    source_file=doc.source_file,
                    display_citation=f"{doc.ruleset_id} Rule {doc.rule_number}",
                    quote_excerpt=item.text,
                    document_id=doc.document_id,
                )
            )
        return self._dedupe(citations)

    def _table_ref(self, doc: RuleDocument) -> str | None:
        if not doc.tables:
            return None
        return doc.tables[0].caption or doc.tables[0].table_id

    def _dedupe(self, citations: Iterable[Citation]) -> list[Citation]:
        seen: set[tuple[str, str]] = set()
        result: list[Citation] = []
        for item in citations:
            key = (item.claim_id, item.display_citation)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result
