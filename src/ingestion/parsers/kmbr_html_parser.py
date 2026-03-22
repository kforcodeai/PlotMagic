from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.cleaners import strip_frontpage_javascript
from src.ingestion.normalizers import normalize_text, parse_effective_date
from src.models import (
    AmendmentAction,
    AmendmentRecord,
    ClauseNode,
    ClauseType,
    CrossReference,
    RuleDocument,
    TableData,
)

from .base import RulesetParser


_ANCHOR_RULE_RE = re.compile(r'<a\s+[^>]*name="(?P<anchor>chapter\d+-\d+)"[^>]*>', flags=re.IGNORECASE)

_RULE_NUM_RE = re.compile(r"^\s*(?P<number>\d+[A-Za-z]?)\.\s*(?P<title>.+?)\s*$")
_RULE_INLINE_RE = re.compile(r"(?<!\()\b(?P<number>\d+[A-Za-z]?)\s*\.\s*(?P<title>[A-Za-z][^\n]{2,220})")

_TITLE_RE = re.compile(r'title="(?P<title>[^"]+)"', flags=re.IGNORECASE)
_SUB_RULE_RE = re.compile(r"^\(\s*([0-9A-Za-z]+)\s*\)\s*(.*)$")

_CROSS_REF_PATTERNS = [
    (r"\bRule\s+\d+[A-Za-z]?\b", "rule"),
    (r"\bTable\s+\d+[A-Za-z\-]*\b", "table"),
    (r"\bAppendix\s+[A-Z]+\b", "appendix"),
    (r"\bSchedule\s+[IVXLC\d]+\b", "schedule"),
]

_TOPIC_KEYWORDS = {
    "setback": ["yard", "setback", "open space", "front yard", "rear yard", "side yard"],
    "coverage": ["coverage"],
    "far": ["f.a.r", "floor area ratio"],
    "parking": ["parking", "loading", "unloading", "access width", "access"],
    "height": ["height", "storey", "storeys", "high rise", "floor", "floors"],
    "row_building": ["row building", "row buildings", "row house", "row houses", "ews housing", "economically weaker section"],
    "fire_safety": ["fire", "emergency exit", "staircase", "travel distance", "exit width", "exit"],
    "ventilation": ["ventilation", "lighting", "air shaft"],
    "sanitation": ["sanitation", "water closet", "latrine", "wash basin"],
    "definitions": ["means", "definition"],
    "permit": ["permit", "approval", "application", "renewal", "validity", "deemed approval"],
    "exemption": ["permit not necessary", "not necessary", "without permit", "prior intimation", "intimation", "objection"],
    "regularisation": ["regularisation", "regularization", "compounding", "unauthorised", "deviation"],
    "excavation": ["excavation", "earthwork", "cutting", "concurrence", "retaining wall", "compensation", "geotechnical"],
    "telecom": ["telecommunication", "tower", "pole structure"],
    "water": ["rainwater", "groundwater", "recharge", "solar assisted", "solar-assisted", "solar"],
    "accessibility": ["disabilit", "wheel chair", "ramp", "special water closet"],
    "heritage": ["heritage", "art and heritage commission"],
    "appeal_penalty": [
        "appeal",
        "penalty",
        "vigilance",
        "illegal construction",
        "illegal",
        "notice",
        "show cause",
        "demolition",
        "police",
    ],
    "registration": ["registering authority", "architect", "engineer", "town planner", "supervisor"],
}

_OCCUPANCY_GROUPS = ["A1", "A2", "B", "C", "D", "E", "F", "G1", "G2", "H", "I(1)", "I(2)"]


class KMBRHTMLParser(RulesetParser):
    def parse_file(self, file_path: Path) -> list[RuleDocument]:
        raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
        html = strip_frontpage_javascript(raw_html)

        chapter_number = self._chapter_number_from_name(file_path.name)
        chapter_title = self._chapter_title_from_html(html, chapter_number)

        rule_docs: list[RuleDocument] = []
        seen_rule_numbers: set[str] = set()
        anchors = self._find_anchor_spans(html)
        for idx, (anchor, anchor_start, anchor_end) in enumerate(anchors):
            end = anchors[idx + 1][1] if idx + 1 < len(anchors) else len(html)
            parsed = self._parse_rule_header(
                html=html,
                anchor_start=anchor_start,
                segment_start=anchor_end,
                segment_end=end,
            )
            if not parsed:
                continue

            rule_number, rule_title = parsed
            if rule_number in seen_rule_numbers:
                continue
            seen_rule_numbers.add(rule_number)
            segment_html = html[anchor_end:end]
            soup = BeautifulSoup(segment_html, "lxml")

            full_text = normalize_text(soup.get_text("\n", strip=True))
            full_text = self._ensure_rule_header(full_text, rule_number, rule_title)
            tables = self._extract_tables(soup, chapter_number, rule_number)
            amendments = self._extract_amendments(segment_html)
            cross_refs = self._extract_cross_refs(full_text, anchor)
            provisos = self._extract_provisos(full_text)
            notes = self._extract_notes(full_text)
            occupancy_groups = self._extract_occupancies(full_text)
            topic_tags = self._extract_topic_tags(full_text)
            conditions = self._extract_conditions(full_text)
            numeric_values = self._extract_numeric_values(full_text)
            is_generic = self._is_generic_rule(full_text, occupancy_groups)

            document_id = f"{self.ruleset_id}-ch{chapter_number}-r{rule_number}"
            rule_citation = f"Chapter {chapter_number}, Rule {rule_number}"
            clause_nodes = self._build_clause_nodes(
                document_id=document_id,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                rule_number=rule_number,
                rule_title=rule_title,
                anchor=anchor,
                full_text=full_text,
                provisos=provisos,
                notes=notes,
                tables=tables,
                occupancy_groups=occupancy_groups,
                topic_tags=topic_tags,
                cross_refs=cross_refs,
                amendments=amendments,
                conditions=conditions,
                numeric_values=numeric_values,
                source_file=str(file_path),
                display_citation=rule_citation,
            )

            rule_docs.append(
                RuleDocument(
                    document_id=document_id,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    issuing_authority=self.issuing_authority or "UNKNOWN",
                    effective_from=self._derive_effective_from(amendments) or self.default_effective_from,
                    effective_to=None,
                    source_file=str(file_path),
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    full_text=full_text,
                    anchor_id=anchor,
                    occupancy_groups=occupancy_groups,
                    is_generic=is_generic,
                    panchayat_category=None,
                    conditions=conditions,
                    numeric_values=numeric_values,
                    provisos=provisos,
                    notes=notes,
                    tables=tables,
                    cross_references=cross_refs,
                    amendments=amendments,
                    clause_nodes=clause_nodes,
                )
            )
        return rule_docs

    def _chapter_number_from_name(self, name: str) -> int:
        match = re.search(r"chapter(\d+)", name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _find_anchor_spans(self, html: str) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        seen: set[str] = set()
        for match in _ANCHOR_RULE_RE.finditer(html):
            anchor = match.group("anchor").lower()
            if anchor in seen:
                continue
            seen.add(anchor)
            spans.append((anchor, match.start(), match.end()))
        return spans

    def _parse_rule_header(
        self,
        html: str,
        anchor_start: int,
        segment_start: int,
        segment_end: int,
    ) -> tuple[str, str] | None:
        forward_window = html[segment_start : min(segment_end, segment_start + 2600)]
        parsed = self._extract_rule_header_from_snippet(forward_window, take_last=False)
        if parsed:
            return parsed

        lookaround_start = max(0, anchor_start - 1200)
        lookaround_end = min(len(html), segment_start + 1200)
        lookaround_window = html[lookaround_start:lookaround_end]
        return self._extract_rule_header_from_snippet(lookaround_window, take_last=True)

    def _extract_rule_header_from_snippet(self, snippet: str, take_last: bool) -> tuple[str, str] | None:
        text = self._snippet_to_text(snippet)
        raw_lines = text.splitlines()
        candidates: list[tuple[str, str]] = []
        for idx, line in enumerate(raw_lines):
            cleaned = normalize_text(line.strip())
            if not cleaned:
                continue
            cleaned = cleaned.strip(" -:;")
            split_match = re.match(r"^(?P<number>\d+[A-Za-z]?)\.$", cleaned)
            if split_match and idx + 1 < len(raw_lines):
                split_title = self._clean_rule_title(raw_lines[idx + 1])
                split_title = self._maybe_extend_title(split_title, raw_lines, idx + 1)
                if self._is_likely_rule_title(split_title):
                    candidates.append((split_match.group("number"), split_title))
                    continue
            direct_match = _RULE_NUM_RE.match(cleaned)
            if direct_match:
                title = self._clean_rule_title(direct_match.group("title"))
                title = self._maybe_extend_title(title, raw_lines, idx)
                if self._is_likely_rule_title(title):
                    candidates.append((direct_match.group("number"), title))
                    continue
            inline_match = _RULE_INLINE_RE.search(cleaned)
            if inline_match:
                title = self._clean_rule_title(inline_match.group("title"))
                title = self._maybe_extend_title(title, raw_lines, idx)
                if self._is_likely_rule_title(title):
                    candidates.append((inline_match.group("number"), title))
        if not candidates:
            return None
        return candidates[-1] if take_last else candidates[0]

    def _snippet_to_text(self, snippet: str) -> str:
        parsed = BeautifulSoup(snippet, "lxml").get_text("\n", strip=True)
        parsed = parsed.replace("\xa0", " ")
        lines = [normalize_text(line) for line in parsed.splitlines()]
        return "\n".join([line for line in lines if line])

    def _clean_rule_title(self, title: str) -> str:
        cleaned = normalize_text(title)
        cleaned = re.split(r"\(\s*1\s*\)", cleaned, maxsplit=1)[0]
        cleaned = re.split(r"(?i)\bProvided(?:\s+further|\s+also)?\s+that\b", cleaned, maxsplit=1)[0]
        cleaned = re.split(r"\.\-\s*", cleaned, maxsplit=1)[0]
        cleaned = re.split(r"\s*-\s*(?=[A-Z])", cleaned, maxsplit=1)[0]
        return cleaned.strip(" -:;.")

    def _maybe_extend_title(self, title: str, raw_lines: list[str], title_index: int) -> str:
        if title_index + 1 >= len(raw_lines):
            return title
        if title.endswith((".", ";", ":")):
            return title
        next_line = normalize_text(raw_lines[title_index + 1].strip(" -:;."))
        if not next_line:
            return title
        if re.match(r"^\(?\d+[A-Za-z]?\)?\b", next_line):
            return title
        if re.match(r"^(Provided|Note|Table|Appendix|Schedule)\b", next_line, flags=re.IGNORECASE):
            return title
        if re.search(r"\b(shall|may|must|is|are|was|were)\b", next_line, flags=re.IGNORECASE):
            return title
        if next_line.endswith(","):
            return title
        if len(next_line.split()) > 6:
            return title
        return normalize_text(f"{title} {next_line}").strip(" -:;.")

    def _is_likely_rule_title(self, title: str) -> bool:
        cleaned = normalize_text(title)
        if len(cleaned) < 4:
            return False
        if cleaned[0].isdigit():
            return False
        if cleaned.startswith("("):
            return False
        if cleaned.lower().startswith(("rule ", "table ")):
            return False
        alpha_count = sum(ch.isalpha() for ch in cleaned)
        return alpha_count >= 3

    def _ensure_rule_header(self, full_text: str, rule_number: str, rule_title: str) -> str:
        heading = normalize_text(f"{rule_number}. {rule_title}")
        if not full_text:
            return heading
        first_line = normalize_text(full_text.splitlines()[0])
        if _RULE_NUM_RE.match(first_line):
            return full_text
        return f"{heading}\n{full_text}"

    def _chapter_title_from_html(self, html: str, chapter_number: int) -> str:
        soup = BeautifulSoup(html, "lxml")
        text_lines = [
            normalize_text(line)
            for line in soup.get_text("\n", strip=True).replace("\xa0", " ").splitlines()
            if normalize_text(line)
        ]
        chapter_indices = [
            idx
            for idx, line in enumerate(text_lines)
            if re.match(r"^CHAPTER\s+(?:[IVXLC]+|\d+|[A-Z]{1,4})\b", line, flags=re.IGNORECASE)
        ]
        for idx in reversed(chapter_indices):
            for candidate in text_lines[idx + 1 : idx + 7]:
                if re.match(r"^\d+[A-Za-z]?\.", candidate):
                    continue
                if candidate.upper() in {"RULES", "NOTIFICATION"}:
                    continue
                if candidate.upper().startswith("CHAPTER "):
                    continue
                cleaned_candidate = re.sub(r"\s+\d+\s*$", "", candidate).strip(" -:")
                if not cleaned_candidate:
                    continue
                if sum(ch.isalpha() for ch in cleaned_candidate) < 4:
                    continue
                return cleaned_candidate

        match = re.search(
            r"CHAPTER\s*(?:[IVXLC]+|\d+|[A-Z]{1,4})\s*<br>\s*(?:<[^>]+>\s*)*(?P<title>[^<]+)",
            html,
            flags=re.IGNORECASE,
        )
        if match:
            cleaned = normalize_text(BeautifulSoup(match.group("title"), "lxml").get_text(" ", strip=True))
            if cleaned and not cleaned.upper().startswith("CHAPTER "):
                return cleaned

        title_match = re.search(r"<title>\s*(?P<title>[^<]+)\s*</title>", html, flags=re.IGNORECASE)
        if title_match:
            title = normalize_text(title_match.group("title"))
            title = re.sub(r"(?i)^CHAPTER\s+[A-Z0-9IVXLC]+\s*", "", title).strip(" -:")
            title = re.sub(r"\s+\d+\s*$", "", title).strip(" -:")
            if title:
                return title

        fallback = re.search(r"OCCUPANCY|PARTS OF BUILDINGS|DEFINITIONS", html, flags=re.IGNORECASE)
        return fallback.group(0).title() if fallback else f"Chapter {chapter_number}"

    def _extract_tables(self, soup: BeautifulSoup, chapter: int, rule: str) -> list[TableData]:
        tables: list[TableData] = []
        for idx, table in enumerate(soup.find_all("table"), start=1):
            rows: list[list[str]] = []
            headers: list[str] = []
            for tr in table.find_all("tr"):
                cells = [normalize_text(td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])]
                if not cells:
                    continue
                rows.append(cells)
            if not rows:
                continue
            headers = rows[0]
            caption = ""
            if headers and headers[0].upper().startswith("TABLE"):
                caption = headers[0]
            elif len(rows) > 1 and rows[1]:
                caption = rows[1][0]
            table_id = f"{self.ruleset_id}-ch{chapter}-r{rule}-t{idx}"
            tables.append(TableData(table_id=table_id, caption=caption, headers=headers, rows=rows[1:], raw_text=str(table)))
        return tables

    def _extract_amendments(self, segment_html: str) -> list[AmendmentRecord]:
        amendments: list[AmendmentRecord] = []
        seen_refs: set[str] = set()
        for title_match in _TITLE_RE.finditer(segment_html):
            title_text = normalize_text(title_match.group("title"))
            if title_text in seen_refs:
                continue
            seen_refs.add(title_text)
            lowered = title_text.lower()
            action = AmendmentAction.OTHER
            if "substituted" in lowered:
                action = AmendmentAction.SUBSTITUTED
            elif "inserted" in lowered:
                action = AmendmentAction.INSERTED
            elif "omitted" in lowered:
                action = AmendmentAction.OMITTED
            elif "added" in lowered:
                action = AmendmentAction.ADDED

            sro_match = re.search(r"SRO\s*No\.?\s*([0-9/]+)", title_text, flags=re.IGNORECASE)
            gzt_match = re.search(r"K\.?G\.?[^,]*", title_text, flags=re.IGNORECASE)
            amendments.append(
                AmendmentRecord(
                    action=action,
                    reference_text=title_text,
                    sro_number=sro_match.group(1) if sro_match else None,
                    gazette_reference=gzt_match.group(0) if gzt_match else None,
                    effective_from=parse_effective_date(title_text),
                )
            )
        return amendments

    def _extract_cross_refs(self, text: str, source_anchor: str) -> list[CrossReference]:
        refs: list[CrossReference] = []
        for pattern, target_type in _CROSS_REF_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                target_ref = normalize_text(match.group(0))
                refs.append(
                    CrossReference(
                        source_clause_id=source_anchor,
                        target_ref=target_ref,
                        target_type=target_type,
                    )
                )
        return refs

    def _extract_provisos(self, text: str) -> list[str]:
        provisos = []
        for item in re.split(r"(?i)\bProvided(?:\s+further|\s+also)?\s+that\b", text):
            cleaned = normalize_text(item)
            if cleaned and cleaned != normalize_text(text):
                provisos.append(cleaned)
        return provisos[1:] if len(provisos) > 1 else []

    def _extract_notes(self, text: str) -> list[str]:
        notes: list[str] = []
        for match in re.finditer(r"(?i)\bNote[s]?\s*[:-]\s*(.+?)(?=(?:\b[A-Z][a-z]+\b\s*[:.-])|$)", text):
            notes.append(normalize_text(match.group(1)))
        return notes

    def _extract_occupancies(self, text: str) -> list[str]:
        matches = []
        for group in _OCCUPANCY_GROUPS:
            if re.search(rf"\b{re.escape(group)}\b", text):
                matches.append(group)
        return sorted(set(matches))

    def _extract_topic_tags(self, text: str) -> list[str]:
        lowered = text.lower()
        tags: list[str] = []
        for topic, keywords in _TOPIC_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                tags.append(topic)
        return sorted(set(tags))

    def _extract_conditions(self, text: str) -> dict[str, float]:
        conditions: dict[str, float] = {}
        height_match = re.search(r"(\d+(?:\.\d+)?)\s*metres?\s+in\s+height", text, flags=re.IGNORECASE)
        if height_match:
            conditions["height_m"] = float(height_match.group(1))
        area_match = re.search(r"(\d+(?:\.\d+)?)\s*sq\.m", text, flags=re.IGNORECASE)
        if area_match:
            conditions["area_sqm"] = float(area_match.group(1))
        return conditions

    def _extract_numeric_values(self, text: str) -> dict[str, float]:
        values: dict[str, float] = {}
        far_match = re.search(r"F\.?A\.?R\.?\s*[:=]?\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if far_match:
            values["far"] = float(far_match.group(1))
        coverage_match = re.search(r"coverage[^0-9]*(\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE)
        if coverage_match:
            values["coverage_pct"] = float(coverage_match.group(1))
        return values

    def _is_generic_rule(self, full_text: str, occupancy_groups: list[str]) -> bool:
        generic_hints = ["all buildings", "every building", "any building"]
        lowered = full_text.lower()
        if occupancy_groups:
            return False
        return any(hint in lowered for hint in generic_hints)

    def _derive_effective_from(self, amendments: list[AmendmentRecord]) -> date | None:
        dates = [item.effective_from for item in amendments if item.effective_from]
        return max(dates) if dates else None

    def _build_clause_nodes(
        self,
        document_id: str,
        chapter_number: int,
        chapter_title: str,
        rule_number: str,
        rule_title: str,
        anchor: str,
        full_text: str,
        provisos: list[str],
        notes: list[str],
        tables: list[TableData],
        occupancy_groups: list[str],
        topic_tags: list[str],
        cross_refs: list[CrossReference],
        amendments: list[AmendmentRecord],
        conditions: dict[str, float],
        numeric_values: dict[str, float],
        source_file: str,
        display_citation: str,
    ) -> list[ClauseNode]:
        nodes: list[ClauseNode] = []
        rule_clause_id = f"{document_id}-rule"
        nodes.append(
            ClauseNode(
                clause_id=rule_clause_id,
                clause_type=ClauseType.RULE,
                state=self.state,
                jurisdiction_type=self.jurisdiction_type,
                ruleset_id=self.ruleset_id,
                ruleset_version=self.ruleset_version,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                rule_number=rule_number,
                rule_title=rule_title,
                sub_rule_path="",
                display_citation=display_citation,
                source_file=source_file,
                anchor_id=anchor,
                raw_text=full_text,
                normalized_text=full_text,
                occupancy_groups=occupancy_groups,
                topic_tags=topic_tags,
                is_generic=not occupancy_groups,
                conditions=conditions,
                numeric_values=numeric_values,
                amendments=amendments,
                cross_references=cross_refs,
            )
        )

        sub_rule_ordinals: dict[str, int] = {}
        for sub_rule_path, sub_rule_text in self._extract_sub_rule_blocks(full_text):
            sub_rule_base = re.sub(r"[^a-z0-9]+", "-", sub_rule_path.lower()).strip("-") or "x"
            ordinal = sub_rule_ordinals.get(sub_rule_base, 0) + 1
            sub_rule_ordinals[sub_rule_base] = ordinal
            sub_rule_id = f"{sub_rule_base}-o{ordinal}"
            sub_rule_topics = self._extract_topic_tags(sub_rule_text) or topic_tags
            nodes.append(
                ClauseNode(
                    clause_id=f"{document_id}-subrule-{sub_rule_id}",
                    clause_type=ClauseType.SUB_RULE,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    sub_rule_path=f"({sub_rule_path})",
                    display_citation=f"{display_citation} ({sub_rule_path})",
                    source_file=source_file,
                    anchor_id=f"{anchor}-subrule-{sub_rule_id}",
                    raw_text=sub_rule_text,
                    normalized_text=normalize_text(sub_rule_text),
                    occupancy_groups=occupancy_groups,
                    topic_tags=sub_rule_topics,
                    is_generic=not occupancy_groups,
                    conditions=self._extract_conditions(sub_rule_text),
                    numeric_values=self._extract_numeric_values(sub_rule_text),
                    cross_references=self._extract_cross_refs(sub_rule_text, anchor),
                    parent_clause_id=rule_clause_id,
                )
            )

        for idx, proviso in enumerate(provisos, start=1):
            nodes.append(
                ClauseNode(
                    clause_id=f"{document_id}-proviso-{idx}",
                    clause_type=ClauseType.PROVISO,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    sub_rule_path=f"proviso-{idx}",
                    display_citation=f"{display_citation} Proviso {idx}",
                    source_file=source_file,
                    anchor_id=f"{anchor}-proviso-{idx}",
                    raw_text=proviso,
                    normalized_text=proviso,
                    occupancy_groups=occupancy_groups,
                    topic_tags=topic_tags,
                    parent_clause_id=rule_clause_id,
                )
            )

        for idx, note in enumerate(notes, start=1):
            nodes.append(
                ClauseNode(
                    clause_id=f"{document_id}-note-{idx}",
                    clause_type=ClauseType.NOTE,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    sub_rule_path=f"note-{idx}",
                    display_citation=f"{display_citation} Note {idx}",
                    source_file=source_file,
                    anchor_id=f"{anchor}-note-{idx}",
                    raw_text=note,
                    normalized_text=note,
                    occupancy_groups=occupancy_groups,
                    topic_tags=topic_tags,
                    parent_clause_id=rule_clause_id,
                )
            )

        for idx, table in enumerate(tables, start=1):
            table_text = self._compact_table_for_embedding(table)
            nodes.append(
                ClauseNode(
                    clause_id=f"{document_id}-table-{idx}",
                    clause_type=ClauseType.TABLE,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    sub_rule_path=f"table-{idx}",
                    display_citation=f"{display_citation} Table {idx}",
                    source_file=source_file,
                    anchor_id=f"{anchor}-table-{idx}",
                    raw_text=table.raw_text,
                    normalized_text=table_text,
                    occupancy_groups=occupancy_groups,
                    topic_tags=topic_tags,
                    table_data=table,
                    parent_clause_id=rule_clause_id,
                )
            )
            for row_idx, row in enumerate(table.rows, start=1):
                row_text = self._table_row_text(table.headers, row)
                if not row_text:
                    continue
                row_clause_id = f"{document_id}-table-{idx}-row-{row_idx}"
                nodes.append(
                    ClauseNode(
                        clause_id=row_clause_id,
                        clause_type=ClauseType.TABLE_ROW,
                        state=self.state,
                        jurisdiction_type=self.jurisdiction_type,
                        ruleset_id=self.ruleset_id,
                        ruleset_version=self.ruleset_version,
                        chapter_number=chapter_number,
                        chapter_title=chapter_title,
                        rule_number=rule_number,
                        rule_title=rule_title,
                        sub_rule_path=f"table-{idx}.row-{row_idx}",
                        display_citation=f"{display_citation} Table {idx} Row {row_idx}",
                        source_file=source_file,
                        anchor_id=f"{anchor}-table-{idx}-row-{row_idx}",
                        raw_text=row_text,
                        normalized_text=normalize_text(row_text),
                        occupancy_groups=occupancy_groups,
                        topic_tags=topic_tags,
                        parent_clause_id=f"{document_id}-table-{idx}",
                    )
                )
        return nodes

    def _compact_table_for_embedding(self, table: TableData) -> str:
        caption = normalize_text(table.caption)
        headers = [normalize_text(item) for item in table.headers if normalize_text(item)]
        rows = [self._table_row_text(headers, row) for row in table.rows]
        rows = [row for row in rows if row]
        compact = [f"Table: {caption}"] if caption else []
        if headers:
            compact.append("Columns: " + "; ".join(headers))
        compact.extend(rows[:40])
        return normalize_text(" ".join(compact))

    def _table_row_text(self, headers: list[str], row: list[str]) -> str:
        parts: list[str] = []
        for idx, cell in enumerate(row):
            cell_norm = normalize_text(cell)
            if not cell_norm:
                continue
            header = headers[idx] if idx < len(headers) else f"value_{idx+1}"
            header_norm = normalize_text(header) or f"value_{idx+1}"
            parts.append(f"{header_norm}={cell_norm}")
        return "; ".join(parts)

    def _extract_sub_rule_blocks(self, text: str) -> list[tuple[str, str]]:
        blocks: list[tuple[str, str]] = []
        current_path: str | None = None
        current_lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if current_path and current_lines:
                    current_lines.append("")
                continue
            match = _SUB_RULE_RE.match(stripped)
            if match:
                if current_path and current_lines:
                    body = normalize_text("\n".join(current_lines))
                    if body:
                        blocks.append((current_path, body))
                current_path = match.group(1)
                initial_text = match.group(2).strip()
                current_lines = [initial_text] if initial_text else []
                continue
            if current_path:
                current_lines.append(stripped)

        if current_path and current_lines:
            body = normalize_text("\n".join(current_lines))
            if body:
                blocks.append((current_path, body))
        return blocks
