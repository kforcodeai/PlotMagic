from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.models import QueryFact, RuleDocument
from src.policy import ApplicabilityPolicy


@dataclass(slots=True)
class CandidateRule:
    document_id: str
    reason: str
    priority: int


@dataclass(slots=True)
class ApplicabilityResult:
    selected: list[RuleDocument] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)


class ApplicabilityEngine:
    """
    Deterministic selection of candidate rules before semantic retrieval.
    """

    def __init__(self, policy: ApplicabilityPolicy | None = None) -> None:
        self.policy = policy or ApplicabilityPolicy()

    def select_candidates(self, docs: list[RuleDocument], fact: QueryFact) -> ApplicabilityResult:
        strict: list[RuleDocument] = []
        relaxed_topic: list[RuleDocument] = []
        generic_fallback: list[RuleDocument] = []
        reasons: dict[str, list[str]] = {}

        procedural_topic_agnostic = self._procedural_topic_agnostic(fact)
        for doc in docs:
            reason_items: list[str] = []

            if fact.state and doc.state != fact.state:
                continue
            if fact.jurisdiction_type and doc.jurisdiction_type != fact.jurisdiction_type:
                continue
            if not self._is_effective(doc, fact.query_date):
                continue

            if not self._category_matches(doc, fact.panchayat_category):
                continue

            topic_ok = self._topic_matches(doc, fact.topics)
            occupancy_ok = self._occupancy_matches(doc, fact.occupancies)
            if fact.query_intent == "procedural" and self.policy.procedural_occupancy_agnostic:
                occupancy_ok = True
            condition_ok = self._conditions_match(doc.conditions, fact)

            if not condition_ok:
                continue

            if doc.is_generic and topic_ok:
                reason_items.append("Tier-3 generic candidate matching topic.")
                generic_fallback.append(doc)
                reasons[doc.document_id] = reason_items
                continue

            if occupancy_ok and (topic_ok or not fact.topics or procedural_topic_agnostic):
                if fact.query_intent == "procedural" and self.policy.procedural_occupancy_agnostic:
                    reason_items.append("Policy allows occupancy-agnostic procedural retrieval.")
                reason_items.append("Tier-1 strict candidate matched occupancy/topic predicates.")
                strict.append(doc)
                reasons[doc.document_id] = reason_items
                continue

            if occupancy_ok and not self.policy.strict_topic_matching:
                reason_items.append("Tier-2 candidate retained with relaxed topic matching.")
                relaxed_topic.append(doc)
                reasons[doc.document_id] = reason_items
                continue

            if self.policy.generic_fallback_enabled and doc.is_generic:
                reason_items.append("Tier-3 generic fallback retained.")
                generic_fallback.append(doc)
                reasons[doc.document_id] = reason_items

        selected = self._select_tier(strict, relaxed_topic, generic_fallback, topics=fact.topics)
        return ApplicabilityResult(selected=selected, reasons=reasons)

    def resolve_conflicts(self, docs: list[RuleDocument], topic: str, mixed_use: bool = False) -> list[RuleDocument]:
        if not docs:
            return []
        sorted_docs = sorted(
            docs,
            key=lambda doc: (
                self._effective_priority(doc),
                self._specificity_priority(doc),
                self._proviso_priority(doc),
            ),
            reverse=True,
        )
        if not mixed_use:
            return sorted_docs
        # Mixed-use: apply "most restrictive" deterministically by topic-specific comparator.
        return [self._most_restrictive(sorted_docs, topic)]

    def _is_effective(self, doc: RuleDocument, query_date: date | None) -> bool:
        if query_date is None:
            return True
        if doc.effective_from and query_date < doc.effective_from:
            return False
        if doc.effective_to and query_date > doc.effective_to:
            return False
        return True

    def _category_matches(self, doc: RuleDocument, category: str | None) -> bool:
        if not category or not doc.panchayat_category:
            return True
        if doc.panchayat_category == "both":
            return True
        return doc.panchayat_category.lower() == category.lower()

    def _topic_matches(self, doc: RuleDocument, topics: list[str]) -> bool:
        if not topics:
            return True
        doc_topics = {tag.lower() for node in doc.clause_nodes for tag in node.topic_tags}
        return bool(doc_topics.intersection({item.lower() for item in topics}))

    def _occupancy_matches(self, doc: RuleDocument, occupancies: list[str]) -> bool:
        if not occupancies:
            return True
        if doc.is_generic:
            return True
        return bool(set(doc.occupancy_groups).intersection(set(occupancies)))

    def _conditions_match(self, conditions: dict[str, float], fact: QueryFact) -> bool:
        if not conditions:
            return True
        for key, value in conditions.items():
            if key == "height_m" and fact.height_m is not None and fact.height_m < value:
                return False
            if key == "area_sqm" and fact.floor_area_sqm is not None and fact.floor_area_sqm < value:
                return False
        return True

    def _effective_priority(self, doc: RuleDocument) -> int:
        return 1 if doc.effective_from else 0

    def _specificity_priority(self, doc: RuleDocument) -> int:
        return 0 if doc.is_generic else 1

    def _proviso_priority(self, doc: RuleDocument) -> int:
        return 1 if doc.provisos else 0

    def _most_restrictive(self, docs: list[RuleDocument], topic: str) -> RuleDocument:
        topic_l = topic.lower()
        if topic_l in {"far", "coverage"}:
            return min(docs, key=lambda doc: doc.numeric_values.get(topic_l, 10_000.0))
        if topic_l in {"setback", "parking"}:
            return max(docs, key=lambda doc: doc.numeric_values.get(topic_l, 0.0))
        return docs[0]

    def _dedupe(self, docs: list[RuleDocument]) -> list[RuleDocument]:
        deduped: dict[str, RuleDocument] = {}
        for doc in docs:
            deduped[doc.document_id] = doc
        return list(deduped.values())

    def _procedural_topic_agnostic(self, fact: QueryFact) -> bool:
        if fact.query_intent != "procedural":
            return False
        if not fact.topics:
            return True
        return all(topic in self.policy.broad_procedural_topics for topic in fact.topics)

    def _select_tier(
        self,
        strict: list[RuleDocument],
        relaxed_topic: list[RuleDocument],
        generic_fallback: list[RuleDocument],
        *,
        topics: list[str],
    ) -> list[RuleDocument]:
        merged: list[RuleDocument] = [*strict, *relaxed_topic]
        if topics:
            merged.extend(doc for doc in generic_fallback if self._topic_matches(doc, topics))
        else:
            merged.extend(generic_fallback)
        if merged:
            return self._dedupe(merged)
        return self._dedupe(generic_fallback)
