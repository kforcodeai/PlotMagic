from __future__ import annotations

from collections import defaultdict

from src.generation.citation_builder import CitationBuilder
from src.models import QueryFact, RuleDocument
from src.models.schemas import AnswerResponse, CitationPayload, EvidenceItem
from src.retrieval.evidence import EvidenceMatrix
from src.retrieval.query_planner import QueryPlan


class AnswerGenerator:
    def __init__(self) -> None:
        self.citation_builder = CitationBuilder()

    def generate(
        self,
        query: str,
        fact: QueryFact,
        plan: QueryPlan,
        evidence_matrix: EvidenceMatrix,
        docs: list[RuleDocument],
        latency_ms: dict[str, float] | None = None,
    ) -> AnswerResponse:
        docs_by_id = {doc.document_id: doc for doc in docs}
        citations = self.citation_builder.build(evidence_matrix, docs_by_id)

        sections = self._compose_sections(plan, evidence_matrix, docs_by_id)
        unresolved = self._compute_unresolved(plan, evidence_matrix)
        if not evidence_matrix.items:
            unresolved = sorted(set([*unresolved, "Insufficient relevant context to answer confidently."]))
        assumptions = self._assumptions(fact)

        evidence_items = [
            EvidenceItem(
                claim_id=item.claim_id,
                text=item.text,
                chunk_id=item.chunk_id,
                scores=item.scores,
                citations=[],
            )
            for item in evidence_matrix.items
        ]

        citation_payloads = [
            CitationPayload(
                claim_id=citation.claim_id,
                ruleset_id=citation.ruleset_id,
                chapter_number=citation.chapter_number,
                rule_number=citation.rule_number,
                sub_rule_path=citation.sub_rule_path,
                table_ref=citation.table_ref,
                anchor_id=citation.anchor_id,
                source_file=citation.source_file,
                display_citation=citation.display_citation,
                quote_excerpt=citation.quote_excerpt,
                document_id=citation.document_id,
                source_url=self._api_source_url(citation.document_id, citation.anchor_id),
            )
            for citation in citations
        ]

        return AnswerResponse(
            jurisdiction=f"{fact.state or 'unknown'}::{fact.jurisdiction_type or 'unknown'}",
            occupancy_groups=fact.occupancies,
            assumptions=assumptions,
            unresolved=unresolved,
            answer_sections=sections,
            evidence_matrix=evidence_items,
            citations=citation_payloads,
            clarifications=[],
            latency_ms=latency_ms or {},
        )

    def _compose_sections(
        self,
        plan: QueryPlan,
        evidence_matrix: EvidenceMatrix,
        docs_by_id: dict[str, RuleDocument],
    ) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for evidence in evidence_matrix.items:
            doc = docs_by_id.get(evidence.document_id)
            if not doc:
                continue
            grouped[evidence.topic].append(
                {
                    "claim_id": evidence.claim_id,
                    "rule": f"{doc.ruleset_id} Rule {doc.rule_number}",
                    "title": doc.rule_title,
                    "excerpt": evidence.text,
                    "anchor": doc.anchor_id,
                }
            )
        sections: list[dict[str, object]] = []
        if not plan.topics:
            rules: list[dict[str, str]] = []
            for items in grouped.values():
                rules.extend(items)
            if rules:
                sections.append({"topic": "general", "rules": rules})
            return sections

        for topic in plan.topics:
            sections.append({"topic": topic, "rules": grouped.get(topic, [])})
        return sections

    def _compute_unresolved(self, plan: QueryPlan, evidence_matrix: EvidenceMatrix) -> list[str]:
        if not plan.topics:
            return []
        unresolved: list[str] = []
        supported_topics = evidence_matrix.supported_topics()
        for topic in plan.topics:
            if topic not in supported_topics:
                unresolved.append(f"Insufficient evidence for topic '{topic}'.")
        return unresolved

    def _assumptions(self, fact: QueryFact) -> list[str]:
        assumptions: list[str] = []
        if not fact.jurisdiction_type:
            assumptions.append("Jurisdiction not explicitly provided.")
        if not fact.occupancies:
            assumptions.append("Occupancy inferred as generic.")
        if fact.height_m is None:
            assumptions.append("Building height not provided.")
        if fact.plot_area_sqm is None:
            assumptions.append("Plot area not provided.")
        return assumptions

    def _api_source_url(self, document_id: str | None, anchor_id: str) -> str | None:
        if not document_id:
            return None
        return f"/rules/{document_id}/source#{anchor_id}"
