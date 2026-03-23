from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_TOPIC_PATTERNS: dict[str, list[str]] = {
    "permit": [r"\bpermit\b", r"\blicen[cs]e\b", r"\bapproval\b", r"\bauthori[sz]ation\b"],
    "registration": [r"\bregister(?:ed|ing|ation)?\b", r"\blicensed professional\b"],
    "exemption": [
        r"\bexempt(?:ion)?\b",
        r"\bexception\b",
        r"\bwaiver\b",
        r"\bnot required\b",
        r"\bpermit not necessary\b",
        r"\bprior intimation\b",
        r"\bhut(?:s)?\b",
        r"\bsmall constructions?\b",
        r"\bappendix\s*a2\b",
    ],
    "penalty": [r"\bpenalt(?:y|ies)\b", r"\bfine(?:s)?\b", r"\boffence\b", r"\boffense\b"],
    "appeal": [r"\bappeal(?:s|ed|ing)?\b", r"\brevision\b", r"\breview\b"],
    "fees": [r"\bfee(?:s)?\b", r"\bcharge(?:s|d)?\b", r"\bcost(?:s)?\b"],
    "timeline": [r"\btimeline\b", r"\bwithin\b", r"\bdeadline\b", r"\bday(?:s)?\b", r"\bmonth(?:s)?\b", r"\byear(?:s)?\b"],
    "deemed_approval": [r"\bdeemed\b", r"\bfinal remarks\b", r"\bdelay(?:s|ed)?\b"],
    "validity_extension": [r"\bvalidity\b", r"\bextension\b", r"\brenewal\b"],
    "completion_certificate": [
        r"\bcompletion certificate\b",
        r"\bdevelopment certificate\b",
        r"\boccupancy certificate\b",
        r"\bpartial occupancy\b",
        r"\btolerance\b",
        r"\bdeemed issuance\b",
    ],
    "transfer": [r"\btransfer(?:or|ee|red|ring)?\b", r"\btransferee\b", r"\bmid-project\b"],
    "excavation": [
        r"\bexcavat(?:ion|e)\b",
        r"\bearthwork\b",
        r"\bcutting\b",
        r"\bgeotechnical\b",
        r"\bretaining wall\b",
        r"\bsolatium\b",
        r"\bconcurrence\b",
    ],
    "flood_crz": [r"\bflood(?:able|[-\s]?prone)?\b", r"\berosion\b", r"\bcrz\b", r"\bcoastal regulation\b"],
    "electric_line": [
        r"\boverhead electric\b",
        r"\belectric(?:al)?\s+line\b",
        r"\bline clearance\b",
        r"\bvoltage\b",
    ],
    "open_space": [
        r"\bopen[\s-]?space\b",
        r"\bset[\s-]?back\b",
        r"\bfront yard\b",
        r"\brear yard\b",
        r"\bside yard\b",
        r"\binterior open\b",
        r"\bdistance between blocks\b",
        r"\brecreation(?:al)?\s+space\b",
    ],
    "street_road": [
        r"\bcentre line of (?:the )?road\b",
        r"\bcenter line of (?:the )?road\b",
        r"\bcentral line of (?:the )?road\b",
        r"\bstreet boundary\b",
        r"\bcul[\s-]?de[\s-]?sac\b",
        r"\bnarrow street\b",
    ],
    "defence_railway": [r"\bdefen[cs]e\b", r"\brailway\b"],
    "security_zone": [r"\bsecurity zone\b", r"\bsecurity\s+zone\b"],
    "religious_building": [r"\breligious\b", r"\bworship\b", r"\bcommunal\b"],
    "height": [r"\bheight\b", r"\bstorey\b", r"\bstoreys\b", r"\bsetback\b"],
    "far": [r"\bf\.?a\.?r\.?\b", r"\bfloor area ratio\b", r"\bcoverage\b"],
    "authority": [
        r"\bsecretary\b",
        r"\bcouncil\b",
        r"\bregistering authority\b",
        r"\bdefen[cs]e\b",
        r"\brailway\b",
        r"\bfire force\b",
        r"\bdistrict town planner\b",
        r"\bchief town planner\b",
        r"\bdistrict collector\b",
        r"\bdirector general of police\b",
    ],
    "definition": [r"\bmeans\b", r"\bdefined as\b", r"\bdefinition(?:s)?\b"],
    "safety": [r"\bsafety\b", r"\bhazard(?:ous)?\b", r"\bfire\b", r"\bemergency\b"],
    "environment": [r"\bwater\b", r"\bwaste\b", r"\bdrainage\b", r"\bsolar\b", r"\bpollution\b"],
    "row_building": [r"\brow[\s-]?building(?:s)?\b", r"\brow[\s-]?house(?:s)?\b"],
    "parking": [r"\bparking\b", r"\bgarage\b", r"\bvehicle\b"],
    "ventilation": [r"\bventilat(?:ion|ing|ed|or)?\b", r"\bair[\s-]?flow\b", r"\blighting\b"],
    "sanitation": [r"\bsanit(?:ation|ary)\b", r"\bsewage\b", r"\btoilet\b", r"\bseptic\b"],
    "regularisation": [r"\bregulari[sz](?:ation|e|ed)\b", r"\bunauthori[sz]ed\b"],
    "accessibility": [r"\baccessib(?:ility|le)\b", r"\bdisabled\b", r"\bramp\b", r"\bbarrier[\s-]?free\b"],
    "rainwater": [r"\brainwater\b", r"\bwater[\s-]?harvest\b", r"\bwater[\s-]?storage\b"],
    "small_plot": [r"\bsmall[\s-]?plot\b", r"\bsmall[\s-]?site\b", r"\bnarrow[\s-]?plot\b"],
}

_TOPIC_TO_COMPONENT = {
    "deemed_approval": "deemed_approval",
    "open_space": "open_space",
    "street_road": "street_road",
    "defence_railway": "defence_railway",
    "validity_extension": "validity_extension",
    "completion_certificate": "completion_certificate",
    "flood_crz": "flood_crz",
    "electric_line": "electric_line",
    "authority": "authority",
    "timeline": "timeline",
    "fees": "fees",
    "registration": "registration",
    "security_zone": "security_zone",
    "religious_building": "religious_building",
    "exemption": "exception",
    "far": "numeric_thresholds",
    "height": "numeric_thresholds",
}

_CANONICAL_SLOT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("distance", re.compile(r"\b(?:distance|setback|clearance|within)\b", flags=re.IGNORECASE)),
    ("authority", re.compile(r"\b(?:secretary|council|authority|officer|committee|planner|defen[cs]e|railway)\b", flags=re.IGNORECASE)),
    ("timeline", re.compile(r"\b(?:within|deadline|time(?:line)?|day(?:s)?|month(?:s)?|year(?:s)?)\b", flags=re.IGNORECASE)),
    ("exception", re.compile(r"\b(?:exception|exempt(?:ion)?|waiver|provided that|provided further|unless|shall not apply|permit not necessary|relaxation|relaxations)\b", flags=re.IGNORECASE)),
    ("numeric_thresholds", re.compile(r"\b(?:minimum|maximum|range|threshold|height|width|coverage|far|percentage|percent|sq\.?\s*m|m|met(?:re|er)s?)\b", flags=re.IGNORECASE)),
    ("fees", re.compile(r"\b(?:fee|fees|charge|charges|cost|rate|compounding)\b", flags=re.IGNORECASE)),
    ("validity_extension", re.compile(r"\b(?:validity|renewal|extension|extend)\b", flags=re.IGNORECASE)),
    ("procedure", re.compile(r"\b(?:application|apply|submission|submit|processing|disposed|issue|issued|certificate)\b", flags=re.IGNORECASE)),
    ("registration", re.compile(r"\b(?:register|registered|registration|registering authority)\b", flags=re.IGNORECASE)),
]

_COMPONENT_CANONICAL_ALIASES = {
    "deemed approval": "deemed_approval",
    "deemed_approval": "deemed_approval",
    "open space": "open_space",
    "open_space": "open_space",
    "street road": "street_road",
    "street_road": "street_road",
    "defence railway": "defence_railway",
    "defense railway": "defence_railway",
    "defence_railway": "defence_railway",
    "validity extension": "validity_extension",
    "validity_extension": "validity_extension",
}


@dataclass(slots=True)
class PlannedSubQuery:
    topic: str
    text: str


@dataclass(slots=True)
class QueryPlan:
    query_type: str
    topics: list[str] = field(default_factory=list)
    sub_queries: list[PlannedSubQuery] = field(default_factory=list)
    mentioned_rule_numbers: list[str] = field(default_factory=list)
    mandatory_components: list[str] = field(default_factory=list)
    suggested_top_k: int = 12


class QueryPlanner:
    _RULE_MENTION_STOPWORDS = {
        "apply",
        "applies",
        "applicable",
        "applicability",
        "the",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "to",
        "for",
        "of",
    }
    _DEFAULT_PROCEDURAL_TOKENS = [
        "permit",
        "approval",
        "application",
        "renewal",
        "validity",
        "regularisation",
        "regularization",
        "compounding",
        "appeal",
        "penalty",
        "registration",
        "certificate",
        "notice",
        "form",
        "appendix",
        "schedule",
        "annexure",
        "deadline",
        "deemed",
    ]

    def __init__(
        self,
        topic_patterns: dict[str, list[str]] | None = None,
        procedural_tokens: list[str] | None = None,
    ) -> None:
        self.topic_patterns = topic_patterns or DEFAULT_TOPIC_PATTERNS
        self.procedural_tokens = procedural_tokens or list(self._DEFAULT_PROCEDURAL_TOKENS)

    def plan(self, query: str) -> QueryPlan:
        query_l = query.lower()
        topics = self._extract_topics(query_l)
        raw_rules = re.findall(
            r"\b(?:rule|section|clause|article)\s+([a-z0-9()./\-]+)\b",
            query_l,
            flags=re.IGNORECASE,
        )
        rules = sorted(set(self._normalize_rule_mentions(raw_rules)))
        query_type = self._classify(query_l)
        mandatory_components = self._mandatory_components(query, topics)

        sub_queries = [PlannedSubQuery(topic=topic, text=f"{topic.replace('_', ' ')} requirements {query}") for topic in topics]
        suggested_top_k = self._compute_suggested_top_k(
            topics=topics, rules=rules, query_type=query_type,
        )
        return QueryPlan(
            query_type=query_type,
            topics=topics,
            sub_queries=sub_queries,
            mentioned_rule_numbers=rules,
            mandatory_components=mandatory_components,
            suggested_top_k=suggested_top_k,
        )

    def _extract_topics(self, query_l: str) -> list[str]:
        topics: list[str] = []
        for topic, patterns in self.topic_patterns.items():
            if any(re.search(pattern, query_l, flags=re.IGNORECASE) for pattern in patterns):
                topics.append(topic)
        # "permit" is a broad lexical trigger and often over-expands retrieval when
        # more specific intent topics are already present.
        specific_topics = {
            "exemption",
            "deemed_approval",
            "validity_extension",
            "completion_certificate",
            "transfer",
            "excavation",
            "flood_crz",
            "electric_line",
            "open_space",
            "street_road",
            "defence_railway",
            "height",
        }
        if "permit" in topics and any(topic in specific_topics for topic in topics):
            topics = [topic for topic in topics if topic != "permit"]
        if self._is_multi_subject_query(query_l):
            for item in ["permit", "timeline"]:
                if item not in topics:
                    topics.append(item)
        return topics

    @staticmethod
    def _compute_suggested_top_k(
        *, topics: list[str], rules: list[str], query_type: str,
    ) -> int:
        n_topics = len(topics)
        n_rules = len(rules)
        if n_topics <= 1 and n_rules == 0:
            base = 10
        elif n_topics <= 3 and n_rules <= 2:
            base = 15
        else:
            base = 25
        if query_type == "comparison":
            base = max(base, 20)
        if n_rules >= 3:
            base = max(base, 25)
        return min(base, 50)

    def _classify(self, query_l: str) -> str:
        if self._is_procedural_query(query_l):
            return "procedural"
        if self._is_multi_subject_query(query_l):
            return "comparison"
        if any(token in query_l for token in ["can i", "is it allowed", "permissible", "comply", "compliance"]):
            return "compliance_check"
        if any(token in query_l for token in ["difference", "compare", "versus", "vs"]):
            return "comparison"
        if any(token in query_l for token in ["how many", "calculate", "compute"]):
            return "calculation"
        return "informational"

    def _is_multi_subject_query(self, query_l: str) -> bool:
        return any(
            token in query_l
            for token in [
                "mixed use",
                "mixed-use",
                "multiple use",
                "more than one use",
                "more than one occupancy",
            ]
        )

    def _is_procedural_query(self, query_l: str) -> bool:
        return any(token in query_l for token in self.procedural_tokens)

    def _mandatory_components(self, query: str, topics: list[str]) -> list[str]:
        normalized = re.sub(r"\s+", " ", query.strip())
        if not normalized:
            return []

        components: list[str] = []
        lowered = normalized.lower()

        # Topic-derived canonical slots.
        for topic in topics:
            canonical = _TOPIC_TO_COMPONENT.get(topic, topic.replace(" ", "_").strip().lower())
            if canonical:
                components.append(canonical)

        # Query lexical slots.
        for slot, pattern in _CANONICAL_SLOT_PATTERNS:
            if pattern.search(lowered):
                components.append(slot)

        # Preserve explicit numeric constraints as normalized component keys.
        for match in re.finditer(
            r"\b(\d+(?:\.\d+)?)\s*(days?|months?|years?|m|metres?|meters?|cm|%|sq\.?\s*m)\b",
            lowered,
            flags=re.IGNORECASE,
        ):
            value = match.group(1)
            unit = re.sub(r"\s+", "", match.group(2))
            components.append(f"numeric:{value}{unit}")

        # Stable fallback when no slots are detected.
        if not components:
            fragments = re.split(r"\?|;|,(?:\s+and\s+)?|\band\b|\bor\b", normalized, flags=re.IGNORECASE)
            for fragment in fragments:
                cleaned = fragment.strip(" .:")
                if len(cleaned.split()) < 3:
                    continue
                components.append(self._canonical_component_key(cleaned))
                if len(components) >= 6:
                    break

        deduped: list[str] = []
        seen: set[str] = set()
        for item in components:
            key = self._canonical_component_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped[:10]

    def _canonical_component_key(self, value: str) -> str:
        lowered = re.sub(r"\s+", " ", value.strip().lower())
        if not lowered:
            return ""
        lowered = lowered.replace("-", " ").replace("/", " ")
        lowered = re.sub(r"[^a-z0-9:_ ]+", "", lowered).strip()
        if lowered in _COMPONENT_CANONICAL_ALIASES:
            return _COMPONENT_CANONICAL_ALIASES[lowered]
        if lowered.startswith("numeric:"):
            return lowered
        lowered = lowered.replace(" ", "_")
        if lowered in _COMPONENT_CANONICAL_ALIASES:
            return _COMPONENT_CANONICAL_ALIASES[lowered]
        return lowered

    def _normalize_rule_mentions(self, raw_mentions: list[str]) -> list[str]:
        cleaned_mentions: list[str] = []
        for token in raw_mentions:
            cleaned = token.strip().strip(".,;:()[]{}").lower()
            if not cleaned or cleaned in self._RULE_MENTION_STOPWORDS:
                continue
            if re.fullmatch(r"[ivxlcdm]+", cleaned):
                cleaned_mentions.append(cleaned.upper())
                continue
            if not any(char.isdigit() for char in cleaned):
                continue
            cleaned_mentions.append(cleaned)
        return cleaned_mentions
