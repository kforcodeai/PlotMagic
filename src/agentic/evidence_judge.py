from __future__ import annotations

from dataclasses import dataclass, field
import re

from src.policy import AbstentionPolicy
from src.retrieval.evidence import EvidenceMatrix
from src.retrieval.query_planner import QueryPlan


@dataclass(slots=True)
class EvidenceJudgeResult:
    supported_claim_ids: list[str] = field(default_factory=list)
    unsupported_claim_ids: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    missing_mandatory_components: list[str] = field(default_factory=list)
    conflicting_claim_ids: list[str] = field(default_factory=list)
    sufficient: bool = False
    can_answer_partially: bool = False
    support_ratio: float = 0.0
    insufficiency_reasons: list[str] = field(default_factory=list)


class EvidenceJudge:
    _CONTRADICTION_PAIRS = [
        ("permit not necessary", "permit shall be obtained"),
        ("deemed to have been issued", "shall be issued"),
        ("shall not apply", "shall apply"),
        ("no work shall be commenced", "work may be commenced"),
    ]

    def __init__(self, policy: AbstentionPolicy | None = None) -> None:
        self.policy = policy or AbstentionPolicy()

    def evaluate(self, plan: QueryPlan, evidence_matrix: EvidenceMatrix) -> EvidenceJudgeResult:
        supported_claim_ids: list[str] = []
        partial_supported_claim_ids: list[str] = []
        unsupported_claim_ids: list[str] = []
        for item in evidence_matrix.items:
            score = float(
                item.scores.get(
                    "rrf_score",
                    self.policy.min_evidence_score if item.has_sufficient_support else 0.0,
                )
            )
            if item.has_sufficient_support and score >= self.policy.min_evidence_score:
                supported_claim_ids.append(item.claim_id)
            elif score >= self.policy.partial_min_evidence_score:
                partial_supported_claim_ids.append(item.claim_id)
            else:
                unsupported_claim_ids.append(item.claim_id)
        supported_topics = {
            item.topic
            for item in evidence_matrix.items
            if item.claim_id in (set(supported_claim_ids).union(set(partial_supported_claim_ids)))
        }

        missing_topics: list[str] = []
        if plan.topics:
            for topic in plan.topics:
                if topic not in supported_topics:
                    missing_topics.append(topic)

        supported_ids_set = set(supported_claim_ids).union(set(partial_supported_claim_ids))
        supported_texts = [item.text for item in evidence_matrix.items if item.claim_id in supported_ids_set]
        missing_components = self._missing_mandatory_components(
            plan=plan,
            supported_texts=supported_texts,
            supported_topics=supported_topics,
        )
        if self._missing_numeric_coverage(plan=plan, supported_texts=supported_texts):
            missing_components.append("numeric_coverage")
        missing_components = self._dedupe_components(missing_components)
        conflicting_claim_ids = self._detect_conflicting_claims(plan=plan, evidence_matrix=evidence_matrix)

        total_items = len(evidence_matrix.items)
        support_ratio = (len(supported_claim_ids) / float(total_items)) if total_items else 0.0
        partial_support_ratio = (len(supported_ids_set) / float(total_items)) if total_items else 0.0
        sufficient = bool(supported_claim_ids)
        insufficiency_reasons: list[str] = []
        if plan.topics and missing_topics:
            sufficient = False
            insufficiency_reasons.append("missing_topics")
        if total_items == 0 or support_ratio < self.policy.min_support_ratio:
            sufficient = False
            insufficiency_reasons.append("low_support_ratio")
        if self.policy.conservative_mandatory_components and missing_components:
            sufficient = False
            insufficiency_reasons.append("missing_components")
        if self.policy.contradiction_blocking and conflicting_claim_ids:
            sufficient = False
            insufficiency_reasons.append("conflicting_evidence")

        can_answer_partially = (
            bool(supported_ids_set)
            and partial_support_ratio >= self.policy.partial_support_ratio
            and not bool(conflicting_claim_ids)
        )
        # If contradiction rules do not trigger, preserve partial answering even when
        # one or more mandatory components remain unresolved.
        if sufficient:
            can_answer_partially = True

        return EvidenceJudgeResult(
            supported_claim_ids=sorted(set(supported_claim_ids)),
            unsupported_claim_ids=sorted(set(unsupported_claim_ids)),
            missing_topics=sorted(set(missing_topics)),
            missing_mandatory_components=sorted(set(missing_components)),
            conflicting_claim_ids=sorted(set(conflicting_claim_ids)),
            sufficient=sufficient,
            can_answer_partially=can_answer_partially,
            support_ratio=support_ratio,
            insufficiency_reasons=sorted(set(insufficiency_reasons)),
        )

    def _missing_mandatory_components(
        self,
        *,
        plan: QueryPlan,
        supported_texts: list[str],
        supported_topics: set[str],
    ) -> list[str]:
        if not plan.mandatory_components:
            return []
        if not supported_texts:
            return list(plan.mandatory_components)
        merged = " ".join(supported_texts).lower()
        supported_topics_l = {topic.lower() for topic in supported_topics}
        planned_topics_l = {topic.lower() for topic in plan.topics}
        missing: list[str] = []
        for component in plan.mandatory_components:
            component_l = self._normalize_component(component)
            if component_l in planned_topics_l and component_l in supported_topics_l:
                continue
            if self._component_supported(component=component_l, merged=merged, supported_topics=supported_topics_l):
                continue
            tokens = [tok for tok in re.findall(r"[a-z0-9]{3,}", component_l) if tok]
            overlap = sum(1 for token in tokens[:8] if token in merged)
            min_required = max(1, min(2, len(tokens) // 2))
            if not tokens or overlap < min_required:
                missing.append(component)
        return missing

    def _missing_numeric_coverage(self, *, plan: QueryPlan, supported_texts: list[str]) -> bool:
        numeric_topics = {"height", "open_space", "street_road", "timeline", "fees", "validity_extension"}
        numeric_component_tokens = {"minimum", "maximum", "distance", "clearance", "width", "height", "days", "years", "fee"}
        plan_text = " ".join(plan.mandatory_components).lower()
        numeric_intent = bool(set(plan.topics).intersection(numeric_topics)) or any(
            token in plan_text for token in numeric_component_tokens
        )
        if not numeric_intent:
            return False
        merged = " ".join(supported_texts)
        unique_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", merged))
        return len(unique_numbers) < 2

    def _detect_conflicting_claims(self, *, plan: QueryPlan, evidence_matrix: EvidenceMatrix) -> list[str]:
        conflicting: list[str] = []
        by_topic: dict[str, list[tuple[str, str]]] = {}
        for item in evidence_matrix.items:
            by_topic.setdefault(item.topic, []).append((item.claim_id, item.text.lower()))
        allow_exception_with_general = self._query_allows_exception_and_general(plan)
        for rows in by_topic.values():
            for left_id, left_text in rows:
                for right_id, right_text in rows:
                    if left_id >= right_id:
                        continue
                    if not self._same_scope(left_text=left_text, right_text=right_text):
                        continue
                    for neg_text, pos_text in self._CONTRADICTION_PAIRS:
                        clash = (neg_text in left_text and pos_text in right_text) or (
                            neg_text in right_text and pos_text in left_text
                        )
                        if not clash:
                            continue
                        if allow_exception_with_general and self._is_exception_general_pair(left_text, right_text):
                            continue
                        if clash:
                            conflicting.extend([left_id, right_id])
                            break
        return conflicting

    def _component_supported(self, *, component: str, merged: str, supported_topics: set[str]) -> bool:
        if not component:
            return True
        if component in supported_topics:
            return True
        if component.startswith("numeric:"):
            number = component.split(":", 1)[1]
            return bool(number and number in merged)
        if component == "timeline":
            return bool(re.search(r"\b\d+(?:\.\d+)?\b", merged)) and any(
                token in merged for token in ["day", "days", "month", "months", "year", "years"]
            )
        if component == "distance":
            return bool(re.search(r"\b\d+(?:\.\d+)?\b", merged)) and any(
                token in merged for token in ["metre", "meter", "m ", "cm", "within"]
            )
        if component in {"exception", "exemption"}:
            return any(
                token in merged
                for token in ["provided that", "provided further", "except", "unless", "shall not apply", "not necessary"]
            )
        if component == "authority":
            return any(
                token in merged
                for token in ["secretary", "authority", "officer", "committee", "planner", "council", "railway", "defence", "defense"]
            )
        if component in {"validity_extension", "validity", "renewal"}:
            return any(token in merged for token in ["valid", "validity", "renewal", "extend", "extension"])
        if component in {"fees", "fee"}:
            return any(token in merged for token in ["fee", "fees", "charge", "charges", "%", "percent"])
        if component in {"numeric_thresholds", "numeric_coverage"}:
            return len(set(re.findall(r"\b\d+(?:\.\d+)?\b", merged))) >= 1
        return False

    def _dedupe_components(self, components: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw in components:
            key = self._normalize_component(raw)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _normalize_component(self, value: str) -> str:
        lowered = re.sub(r"\s+", " ", str(value).strip().lower())
        lowered = lowered.replace("-", " ").replace("/", " ")
        lowered = re.sub(r"[^a-z0-9:_ ]+", "", lowered).strip()
        aliases = {
            "deemed approval": "deemed_approval",
            "deemed_approval": "deemed_approval",
            "open space": "open_space",
            "open_space": "open_space",
            "validity extension": "validity_extension",
            "validity_extension": "validity_extension",
            "defence railway": "defence_railway",
            "defense railway": "defence_railway",
            "defence_railway": "defence_railway",
        }
        if lowered in aliases:
            return aliases[lowered]
        if lowered.startswith("numeric:"):
            return lowered
        return lowered.replace(" ", "_")

    def _query_allows_exception_and_general(self, plan: QueryPlan) -> bool:
        normalized_components = {self._normalize_component(item) for item in plan.mandatory_components}
        if "exemption" in plan.topics and len(plan.topics) > 1:
            return True
        if "exception" in normalized_components and normalized_components.intersection(
            {"distance", "open_space", "street_road", "procedure", "numeric_thresholds"}
        ):
            return True
        merged = " ".join(plan.mandatory_components).lower()
        return any(
            token in merged
            for token in [
                "still apply",
                "continue to apply",
                "parallel",
                "jointly",
                "along with",
                "both",
            ]
        )

    def _is_exception_general_pair(self, left_text: str, right_text: str) -> bool:
        exception_markers = [
            "provided that",
            "provided further",
            "except",
            "unless",
            "shall not apply",
            "not necessary",
            "relaxation",
        ]
        left_is_exception = any(token in left_text for token in exception_markers)
        right_is_exception = any(token in right_text for token in exception_markers)
        return left_is_exception != right_is_exception

    def _same_scope(self, *, left_text: str, right_text: str) -> bool:
        rule_refs_left = set(re.findall(r"\brule\s+\d+[a-z]?\b", left_text))
        rule_refs_right = set(re.findall(r"\brule\s+\d+[a-z]?\b", right_text))
        if rule_refs_left and rule_refs_right and not rule_refs_left.intersection(rule_refs_right):
            return False

        scope_terms = [
            "permit",
            "defence",
            "defense",
            "railway",
            "telecommunication",
            "row building",
            "hut",
            "street",
            "road",
            "parking",
            "fire",
            "registration",
            "validity",
            "renewal",
        ]
        left_scope = {term for term in scope_terms if term in left_text}
        right_scope = {term for term in scope_terms if term in right_text}
        if left_scope and right_scope and not left_scope.intersection(right_scope):
            return False
        if not rule_refs_left.intersection(rule_refs_right):
            overlap = self._scope_token_overlap(left_text=left_text, right_text=right_text)
            if overlap < 0.12:
                return False
        return True

    def _scope_token_overlap(self, *, left_text: str, right_text: str) -> float:
        stopwords = {
            "shall",
            "apply",
            "also",
            "provided",
            "that",
            "under",
            "this",
            "rule",
            "rules",
            "building",
            "buildings",
            "case",
            "with",
            "for",
            "and",
            "the",
            "any",
            "all",
            "not",
        }
        left_tokens = {tok for tok in re.findall(r"[a-z0-9]{3,}", left_text) if tok not in stopwords}
        right_tokens = {tok for tok in re.findall(r"[a-z0-9]{3,}", right_text) if tok not in stopwords}
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens.intersection(right_tokens))
        union = len(left_tokens.union(right_tokens))
        return intersection / float(union)
