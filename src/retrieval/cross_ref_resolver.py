from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models import RuleDocument


@dataclass(slots=True)
class ExpandedReference:
    source_document_id: str
    source_clause_id: str
    target_document_id: str
    target_ref: str
    depth: int
    path: list[str] = field(default_factory=list)


class CrossReferenceResolver:
    def __init__(self, docs: list[RuleDocument]) -> None:
        self.docs = docs
        self.rule_number_to_docs: dict[str, list[RuleDocument]] = {}
        for doc in docs:
            self.rule_number_to_docs.setdefault(doc.rule_number.lower(), []).append(doc)

    def expand(self, seed_docs: list[RuleDocument], depth: int = 2) -> list[ExpandedReference]:
        if depth < 1:
            return []
        expanded: list[ExpandedReference] = []
        seen: set[tuple[str, str]] = set()

        frontier = [(doc, 1, [doc.document_id]) for doc in seed_docs]
        while frontier:
            current_doc, current_depth, path = frontier.pop(0)
            if current_depth > depth:
                continue

            for ref in current_doc.cross_references:
                target_rule = self._extract_rule_number(ref.target_ref)
                if not target_rule:
                    continue
                for target_doc in self.rule_number_to_docs.get(target_rule.lower(), []):
                    edge = (current_doc.document_id, target_doc.document_id)
                    if edge in seen:
                        continue
                    seen.add(edge)
                    expanded_ref = ExpandedReference(
                        source_document_id=current_doc.document_id,
                        source_clause_id=ref.source_clause_id,
                        target_document_id=target_doc.document_id,
                        target_ref=ref.target_ref,
                        depth=current_depth,
                        path=path + [target_doc.document_id],
                    )
                    expanded.append(expanded_ref)
                    if current_depth < depth:
                        frontier.append((target_doc, current_depth + 1, expanded_ref.path))
        return expanded

    def _extract_rule_number(self, text: str) -> str | None:
        match = re.search(r"\bRule\s+(\d+[A-Za-z]?)\b", text, flags=re.IGNORECASE)
        return match.group(1) if match else None

