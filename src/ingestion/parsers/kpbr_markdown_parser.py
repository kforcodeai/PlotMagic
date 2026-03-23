from __future__ import annotations

import re
from pathlib import Path

from src.ingestion.cleaners import clean_markdown_noise
from src.ingestion.normalizers import normalize_text
from src.models import (
    ClauseNode,
    ClauseType,
    CrossReference,
    RuleDocument,
    TableData,
)

from .base import RulesetParser


_CHAPTER_RE = re.compile(r"^#{1,4}\s*CHAPTER[-\s]*([IVXLC]+|\d+)\b", flags=re.IGNORECASE)
_RULE_RE = re.compile(r"^(?:#{1,3}\s*)?_?(?P<number>\d+[A-Za-z]?)\.(?!\d)\s*(?P<title>.+?)\s*$")
_RULE_ALT_RE = re.compile(r"^(?:#{1,3}\s*)?_?(?P<number>\d+[A-Za-z]?)\._\s*(?P<title>.+?)\s*$")
_TABLE_RE = re.compile(r"^#{0,3}\s*TABLE\s*[-\s]*([0-9A-Za-z.]+)?", flags=re.IGNORECASE)
_SUB_RULE_RE = re.compile(r"^\((\d+[A-Za-z]?|[ivxlcdm]+|[a-z])\)\s*(.*)", flags=re.IGNORECASE)

_TOPIC_KEYWORDS = {
    "setback": ["yard", "setback", "open space", "front yard", "rear yard", "side yard"],
    "coverage": ["coverage"],
    "far": ["f.a.r", "floor area ratio"],
    "parking": ["parking", "loading", "unloading", "access width", "access"],
    "height": ["height", "storey", "storeys", "high rise", "floor", "floors"],
    "row_building": ["row building", "row buildings", "row house", "row houses", "ews housing", "economically weaker section"],
    "fire_safety": ["fire", "emergency exit", "staircase", "travel distance", "exit width", "exit"],
    "definitions": ["means", "definition"],
    "permit": ["permit", "approval", "application", "renewal", "validity"],
    "exemption": [
        "permit not necessary",
        "not necessary",
        "without permit",
        "prior intimation",
        "intimation",
        "objection",
        "huts",
        "certain other constructions",
        "appendix a2",
        "proforma prescribed in appendix a2",
    ],
    "regularisation": ["regularisation", "regularization", "compounding", "unauthorised", "deviation"],
    "excavation": ["excavation", "earthwork", "cutting", "concurrence", "retaining wall", "compensation", "geotechnical"],
    "telecom": ["telecommunication", "tower", "pole structure"],
    "water": ["rainwater", "groundwater", "recharge", "solar assisted", "solar-assisted", "solar"],
    "accessibility": ["disabilit", "wheel chair", "ramp", "special water closet"],
    "heritage": ["heritage", "art and heritage commission"],
    "appeal_penalty": ["appeal", "penalty", "vigilance", "illegal construction", "illegal", "notice", "demolition", "police"],
    "registration": ["registering authority", "architect", "engineer", "town planner", "supervisor"],
    "completion_certificate": [
        "completion certificate", "development certificate", "occupancy certificate",
        "deemed issuance", "tolerance", "partial occupancy",
    ],
    "transfer": ["transfer of plots", "transferor", "transferee", "transfer the permit"],
    "flood_crz": [
        "flood", "floodable", "erosion", "crz", "coastal regulation",
    ],
    "electric_line": [
        "electric", "overhead electric", "voltage", "clearance from overhead",
    ],
    "open_space": [
        "open space", "front yard", "rear yard", "side yard",
        "interior open", "distance between blocks",
    ],
    "street_road": [
        "centre line of road", "center line of road", "central line of road",
        "cul-de-sac", "cul de sac", "street boundary",
    ],
    "defence_railway": [
        "defence", "railway",
    ],
    "security_zone": [
        "security zone", "security zones",
    ],
    "religious_building": [
        "religious", "worship", "communal disturbance", "communal harmony",
    ],
    "deemed_approval": [
        "deemed to have been given", "deemed approval",
    ],
    "validity_extension": [
        "validity", "extension", "renewal", "valid for three years",
        "valid for one year", "valid for two years",
    ],
}

_OCCUPANCY_GROUPS = ["A1", "A2", "B", "C", "D", "E", "F", "G1", "G2", "H", "I(1)", "I(2)"]

_CROSS_REF_PATTERNS = [
    (r"\bRule\s+\d+[A-Za-z]?\b", "rule"),
    (r"\bTable\s+\d+[A-Za-z\-\.]*\b", "table"),
    (r"\bAppendix\s+[A-Z]+\b", "appendix"),
    (r"\bSchedule\s+[IVXLC\d]+\b", "schedule"),
]

_APPENDIX_RE = re.compile(r"^#{1,4}\s*APPENDIX\s*[-–]?\s*([A-Za-z0-9–\\-]+)?", flags=re.IGNORECASE)
PARSER_VERSION = "kpbr_markdown_parser.v2"


class KPBRMarkdownParser(RulesetParser):
    def parse_file(self, file_path: Path) -> list[RuleDocument]:
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        text = clean_markdown_noise(raw)
        lines = [line.rstrip() for line in text.splitlines()]

        chapter_number = 0
        chapter_title = "Preliminary"
        current_rule_number: str | None = None
        current_rule_title = ""
        current_buffer: list[str] = []
        in_appendix = False
        current_appendix_id: str | None = None
        current_appendix_title = ""
        current_appendix_buffer: list[str] = []
        appendix_count = 0
        docs: list[RuleDocument] = []
        seen_rule_numbers_by_chapter: dict[int, set[str]] = {}

        def flush_rule() -> None:
            if not current_rule_number:
                return
            full_text = normalize_text("\n".join(current_buffer))
            if not full_text:
                return

            tables = self._extract_tables(current_buffer, chapter_number, current_rule_number)
            cross_refs = self._extract_cross_refs(full_text, current_rule_number)
            occupancy_groups = self._extract_occupancies(full_text)
            panchayat_category = self._extract_category(full_text)
            is_generic = self._is_generic_rule(full_text, occupancy_groups)
            topic_tags = self._extract_topic_tags(full_text)
            conditions = self._extract_conditions(full_text)
            numeric_values = self._extract_numeric_values(full_text)
            provisos = self._extract_provisos(full_text)
            notes = self._extract_notes(full_text)
            anchor = f"kpbr-ch{chapter_number}-r{current_rule_number}".lower()
            document_id = f"{self.ruleset_id}-ch{chapter_number}-r{current_rule_number}"

            clause_nodes = self._build_clause_nodes(
                document_id=document_id,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                rule_number=current_rule_number,
                rule_title=current_rule_title,
                full_text=full_text,
                source_file=str(file_path),
                anchor=anchor,
                provisos=provisos,
                notes=notes,
                tables=tables,
                occupancy_groups=occupancy_groups,
                topic_tags=topic_tags,
                panchayat_category=panchayat_category,
                conditions=conditions,
                numeric_values=numeric_values,
                cross_refs=cross_refs,
            )

            docs.append(
                RuleDocument(
                    document_id=document_id,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    issuing_authority=self.issuing_authority or "UNKNOWN",
                    effective_from=self.default_effective_from,
                    effective_to=None,
                    source_file=str(file_path),
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=current_rule_number,
                    rule_title=current_rule_title,
                    full_text=full_text,
                    anchor_id=anchor,
                    occupancy_groups=occupancy_groups,
                    is_generic=is_generic,
                    panchayat_category=panchayat_category,
                    conditions=conditions,
                    numeric_values=numeric_values,
                    provisos=provisos,
                    notes=notes,
                    tables=tables,
                    cross_references=cross_refs,
                    clause_nodes=clause_nodes,
                    amendments=[],
                )
            )

        def flush_appendix() -> None:
            if not current_appendix_id:
                return
            full_text = normalize_text("\n".join(current_appendix_buffer))
            if not full_text:
                return

            appendix_rule_number = f"APP-{current_appendix_id}"
            tables = self._extract_tables(current_appendix_buffer, chapter_number, appendix_rule_number)
            cross_refs = self._extract_cross_refs(full_text, appendix_rule_number)
            occupancy_groups = self._extract_occupancies(full_text)
            panchayat_category = self._extract_category(full_text)
            topic_tags = self._extract_topic_tags(full_text)
            conditions = self._extract_conditions(full_text)
            numeric_values = self._extract_numeric_values(full_text)
            provisos = self._extract_provisos(full_text)
            notes = self._extract_notes(full_text)
            anchor = f"kpbr-app-{current_appendix_id}".lower()
            document_id = f"{self.ruleset_id}-app-{current_appendix_id}".lower()

            clause_nodes = self._build_clause_nodes(
                document_id=document_id,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                rule_number=appendix_rule_number,
                rule_title=current_appendix_title or f"Appendix {current_appendix_id}",
                full_text=full_text,
                source_file=str(file_path),
                anchor=anchor,
                provisos=provisos,
                notes=notes,
                tables=tables,
                occupancy_groups=occupancy_groups,
                topic_tags=topic_tags,
                panchayat_category=panchayat_category,
                conditions=conditions,
                numeric_values=numeric_values,
                cross_refs=cross_refs,
                clause_type=ClauseType.APPENDIX,
                display_citation_prefix=f"Appendix {current_appendix_id}",
            )

            docs.append(
                RuleDocument(
                    document_id=document_id,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    issuing_authority=self.issuing_authority or "UNKNOWN",
                    effective_from=self.default_effective_from,
                    effective_to=None,
                    source_file=str(file_path),
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=appendix_rule_number,
                    rule_title=current_appendix_title or f"Appendix {current_appendix_id}",
                    full_text=full_text,
                    anchor_id=anchor,
                    occupancy_groups=occupancy_groups,
                    is_generic=True,
                    panchayat_category=panchayat_category,
                    conditions=conditions,
                    numeric_values=numeric_values,
                    provisos=provisos,
                    notes=notes,
                    tables=tables,
                    cross_references=cross_refs,
                    clause_nodes=clause_nodes,
                    amendments=[],
                )
            )

        for line in lines:
            appendix_match = _APPENDIX_RE.match(line.strip())
            if appendix_match:
                flush_rule()
                current_rule_number = None
                current_buffer = []

                if in_appendix:
                    flush_appendix()
                in_appendix = True
                appendix_count += 1
                current_appendix_id = self._appendix_token(appendix_match.group(1), line, appendix_count)
                current_appendix_title = normalize_text(line.strip("# ").strip())
                current_appendix_buffer = [line]
                continue

            if in_appendix:
                current_appendix_buffer.append(line)
                continue

            chapter_match = _CHAPTER_RE.match(line.strip())
            if chapter_match:
                flush_rule()
                current_rule_number = None
                current_buffer = []
                chapter_number = self._parse_chapter_no(chapter_match.group(1))
                chapter_title = f"Chapter {chapter_match.group(1)}"
                continue

            parsed_header = self._parse_rule_header(line.strip())
            rule_match = parsed_header
            if rule_match:
                # Ignore lines that are clearly table rows.
                if self._is_table_like_line(line):
                    current_buffer.append(line)
                    continue
                if not self._is_likely_rule_header(line.strip(), rule_match.group("title")):
                    current_buffer.append(line)
                    continue
                chapter_seen = seen_rule_numbers_by_chapter.setdefault(chapter_number, set())
                if rule_match.group("number") in chapter_seen:
                    # Duplicate rule numbers are usually conversion artifacts; keep as content.
                    current_buffer.append(line)
                    continue
                flush_rule()
                current_rule_number = rule_match.group("number")
                current_rule_title = normalize_text(rule_match.group("title").strip("_- "))
                current_buffer = [line]
                chapter_seen.add(current_rule_number)
                continue

            if current_rule_number:
                current_buffer.append(line)

        flush_rule()
        flush_appendix()
        return docs

    def _parse_rule_header(self, line: str) -> re.Match[str] | None:
        match = _RULE_RE.match(line)
        if match:
            return match
        return _RULE_ALT_RE.match(line)

    def _is_likely_rule_header(self, line: str, title: str) -> bool:
        stripped = line.strip()
        title_clean = normalize_text(title)
        if len(title_clean) < 6:
            return False
        if stripped.lower().startswith("table"):
            return False
        # Common rule header markers in this corpus.
        if ".-" in stripped or stripped.startswith("_") or stripped.startswith("##"):
            return True
        alpha_count = sum(ch.isalpha() for ch in title_clean)
        ratio = alpha_count / max(1, len(title_clean))
        if ratio < 0.4:
            return False
        # Accept long clause-like headers that end with punctuation.
        return len(title_clean.split()) >= 3 and stripped.endswith((".", ":", ";"))

    def _parse_chapter_no(self, token: str) -> int:
        if token.isdigit():
            return int(token)
        return self._roman_to_int(token.upper())

    def _roman_to_int(self, roman: str) -> int:
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
        total = 0
        prev = 0
        for char in reversed(roman):
            current = values.get(char, 0)
            if current < prev:
                total -= current
            else:
                total += current
                prev = current
        return total

    def _extract_tables(self, lines: list[str], chapter: int, rule: str) -> list[TableData]:
        tables: list[TableData] = []
        current_lines: list[str] = []
        current_id = ""
        for line in lines:
            if _TABLE_RE.match(line.strip()):
                if current_lines and current_id:
                    tables.append(self._make_table(current_id, current_lines))
                current_id = f"{self.ruleset_id}-ch{chapter}-r{rule}-t{len(tables)+1}"
                current_lines = [line]
                continue
            if current_id:
                # Continue capturing until a new rule marker appears.
                if _RULE_RE.match(line.strip()):
                    tables.append(self._make_table(current_id, current_lines))
                    current_lines = []
                    current_id = ""
                else:
                    current_lines.append(line)
        if current_lines and current_id:
            tables.append(self._make_table(current_id, current_lines))
        return tables

    def _make_table(self, table_id: str, lines: list[str]) -> TableData:
        raw_lines = [line.rstrip() for line in lines if line and line.strip()]
        cleaned = [normalize_text(line) for line in raw_lines if normalize_text(line)]
        caption = cleaned[0] if cleaned else "Table"

        markdown_rows = [line for line in raw_lines if "|" in line]
        if len(markdown_rows) >= 2:
            parsed_rows = [self._split_markdown_row(line) for line in markdown_rows]
            parsed_rows = [row for row in parsed_rows if row]
            if len(parsed_rows) >= 2:
                if self._looks_separator_row(parsed_rows[1]):
                    header_row = parsed_rows[0]
                    body_rows = parsed_rows[2:]
                else:
                    width = max(len(row) for row in parsed_rows)
                    header_row = [f"col_{idx+1}" for idx in range(width)]
                    body_rows = parsed_rows
                headers = [normalize_text(item) or f"col_{idx+1}" for idx, item in enumerate(header_row)]
                rows = self._normalize_table_rows(body_rows, width=len(headers))
                if rows:
                    return TableData(
                        table_id=table_id,
                        caption=caption,
                        headers=headers,
                        rows=rows,
                        raw_text="\n".join(lines),
                    )

        plain_headers, plain_rows = self._parse_plain_table(raw_lines[1:])
        if plain_rows:
            return TableData(
                table_id=table_id,
                caption=caption,
                headers=plain_headers,
                rows=plain_rows,
                raw_text="\n".join(lines),
            )

        # Final fallback preserving order as raw single-column table.
        headers = ["raw_row"]
        rows = [[line] for line in cleaned[1:]]
        return TableData(table_id=table_id, caption=caption, headers=headers, rows=rows, raw_text="\n".join(lines))

    def _split_markdown_row(self, line: str) -> list[str]:
        compact = line.strip().strip("|")
        if not compact:
            return []
        return [normalize_text(cell.strip()) for cell in compact.split("|")]

    def _looks_separator_row(self, row: list[str]) -> bool:
        if not row:
            return False
        return all(bool(re.fullmatch(r"[:\- ]{2,}", cell)) for cell in row)

    def _normalize_table_rows(self, rows: list[list[str]], width: int) -> list[list[str]]:
        normalized: list[list[str]] = []
        for row in rows:
            if not row:
                continue
            padded = row[:width] + [""] * max(0, width - len(row))
            normalized.append([normalize_text(cell) for cell in padded])
        return normalized

    def _parse_plain_table(self, lines: list[str]) -> tuple[list[str], list[list[str]]]:
        """
        Parse OCR/codeblock style legal tables that do not use markdown pipes.
        Expected forms include rows like:
        - "1 Upto 300 sq.metres 3.0"
        - "(ii) Two Storeyed NA 2.0 3.6 ..."
        """
        sanitized = [self._sanitize_table_line(line) for line in lines]
        sanitized = [line for line in sanitized if line]
        if not sanitized:
            return [], []

        row_records: list[tuple[list[str], str]] = []
        header_lines: list[str] = []
        seen_row = False
        for line in sanitized:
            row = self._split_plain_row(line)
            if row:
                row_records.append((row, line))
                seen_row = True
                continue
            if not seen_row:
                header_lines.append(line)

        if not row_records:
            return [], []

        width = max(len(record[0]) for record in row_records)
        if width < 2:
            return [], []

        headers = self._infer_plain_headers(header_lines, width)
        rows = self._normalize_table_rows([record[0] for record in row_records], width=width)
        return headers, rows

    def _sanitize_table_line(self, line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return ""
        if stripped == "```":
            return ""
        if re.fullmatch(r"`{3,}", stripped):
            return ""
        if re.fullmatch(r"_?\d+_?", stripped):
            return ""
        normalized = normalize_text(stripped)
        if normalized in {"sl no", "sl no:", "sl.no", "sl.no."}:
            return "Sl No"
        return normalized

    def _split_plain_row(self, line: str) -> list[str]:
        row_prefix = re.match(r"^\s*(\(?\d+[a-z]?\)?|\(?[ivxlcdm]+\)?)\s+(.+)$", line, flags=re.IGNORECASE)
        if not row_prefix:
            return []
        row_id = normalize_text(row_prefix.group(1))
        tail = row_prefix.group(2).strip()
        if not tail:
            return []

        tokens = tail.split()
        numeric_tail: list[str] = []
        while tokens and re.fullmatch(r"\d+(?:\.\d+)?", tokens[-1]):
            numeric_tail.append(tokens.pop())
        numeric_tail.reverse()

        descriptor = " ".join(tokens).strip()
        if not descriptor and not numeric_tail:
            return []

        row = [row_id]
        if descriptor:
            row.append(normalize_text(descriptor))
        row.extend(numeric_tail)
        return row

    def _infer_plain_headers(self, header_lines: list[str], width: int) -> list[str]:
        if width <= 0:
            return []
        headers: list[str] = ["sl_no"]

        merged_headers = " ".join(header_lines).lower()
        if "type of building" in merged_headers:
            headers.append("type_of_building")
        elif "total floor area" in merged_headers:
            headers.append("total_floor_area")
        elif "no. of storeys" in merged_headers or "no of storeys" in merged_headers:
            headers.append("storeys")
        else:
            headers.append("description")

        while len(headers) < width:
            idx = len(headers) - 1
            headers.append(f"value_{idx}")
        return headers[:width]

    def _extract_cross_refs(self, text: str, rule_number: str) -> list[CrossReference]:
        refs: list[CrossReference] = []
        source_clause_id = f"rule-{rule_number}"
        for pattern, target_type in _CROSS_REF_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                refs.append(
                    CrossReference(
                        source_clause_id=source_clause_id,
                        target_ref=normalize_text(match.group(0)),
                        target_type=target_type,
                    )
                )
        return refs

    def _appendix_token(self, token: str | None, line: str, default_index: int) -> str:
        raw = token or ""
        if not raw:
            match = re.search(r"appendix\s*[-–]?\s*([A-Za-z0-9\-]+)", line, flags=re.IGNORECASE)
            if match:
                raw = match.group(1)
        cleaned = re.sub(r"[^A-Za-z0-9]+", "", raw).upper()
        if not cleaned:
            cleaned = str(default_index)
        return cleaned

    def _is_table_like_line(self, line: str) -> bool:
        lowered = line.strip().lower()
        if lowered.startswith("table"):
            return True
        return any(token in lowered for token in ["sl.no", "sl no", "fitments assembly occupancies"])

    def _extract_occupancies(self, text: str) -> list[str]:
        found: list[str] = []
        for occupancy in _OCCUPANCY_GROUPS:
            if re.search(rf"\b{re.escape(occupancy)}\b", text):
                found.append(occupancy)
        return sorted(set(found))

    def _extract_category(self, text: str) -> str | None:
        lowered = text.lower()
        has_category_i = bool(re.search(r"\bcategory[\s-]*i\b", lowered))
        has_category_ii = bool(re.search(r"\bcategory[\s-]*ii\b", lowered))
        if has_category_i and has_category_ii:
            return "both"
        if has_category_i:
            return "Category-I"
        if has_category_ii:
            return "Category-II"
        return None

    def _is_generic_rule(self, text: str, occupancies: list[str]) -> bool:
        if occupancies:
            return False
        lowered = text.lower()
        return any(token in lowered for token in ["all buildings", "every building", "any building"])

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

    def _extract_provisos(self, text: str) -> list[str]:
        chunks = re.split(r"(?i)\bProvided(?:\s+further|\s+also)?\s+that\b", text)
        if len(chunks) <= 1:
            return []
        return [normalize_text(chunk) for chunk in chunks[1:] if normalize_text(chunk)]

    def _extract_notes(self, text: str) -> list[str]:
        notes: list[str] = []
        for match in re.finditer(r"(?i)\bNote[s]?\s*[:-]\s*(.+?)(?=(?:\n|$))", text):
            notes.append(normalize_text(match.group(1)))
        return notes

    def _build_clause_nodes(
        self,
        document_id: str,
        chapter_number: int,
        chapter_title: str,
        rule_number: str,
        rule_title: str,
        full_text: str,
        source_file: str,
        anchor: str,
        provisos: list[str],
        notes: list[str],
        tables: list[TableData],
        occupancy_groups: list[str],
        topic_tags: list[str],
        panchayat_category: str | None,
        conditions: dict[str, float],
        numeric_values: dict[str, float],
        cross_refs: list[CrossReference],
        clause_type: ClauseType = ClauseType.RULE,
        display_citation_prefix: str | None = None,
    ) -> list[ClauseNode]:
        nodes: list[ClauseNode] = []
        rule_clause_id = f"{document_id}-rule"
        display_citation = display_citation_prefix or f"Chapter {chapter_number}, Rule {rule_number}"
        nodes.append(
            ClauseNode(
                clause_id=rule_clause_id,
                clause_type=clause_type,
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
                panchayat_category=panchayat_category,
                topic_tags=topic_tags,
                is_generic=not occupancy_groups,
                conditions=conditions,
                numeric_values=numeric_values,
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
                    panchayat_category=panchayat_category,
                    topic_tags=sub_rule_topics,
                    is_generic=not occupancy_groups,
                    conditions=self._extract_conditions(sub_rule_text),
                    numeric_values=self._extract_numeric_values(sub_rule_text),
                    cross_references=self._extract_cross_refs(sub_rule_text, rule_number),
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
                    panchayat_category=panchayat_category,
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
                    panchayat_category=panchayat_category,
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
                    panchayat_category=panchayat_category,
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
                        panchayat_category=panchayat_category,
                        topic_tags=topic_tags,
                        parent_clause_id=f"{document_id}-table-{idx}",
                    )
                )
                for col_idx, cell in enumerate(row, start=1):
                    cell_text = normalize_text(cell)
                    if not cell_text:
                        continue
                    header = table.headers[col_idx - 1] if col_idx - 1 < len(table.headers) else f"col_{col_idx}"
                    nodes.append(
                        ClauseNode(
                            clause_id=f"{document_id}-table-{idx}-row-{row_idx}-cell-{col_idx}",
                            clause_type=ClauseType.TABLE_CELL,
                            state=self.state,
                            jurisdiction_type=self.jurisdiction_type,
                            ruleset_id=self.ruleset_id,
                            ruleset_version=self.ruleset_version,
                            chapter_number=chapter_number,
                            chapter_title=chapter_title,
                            rule_number=rule_number,
                            rule_title=rule_title,
                            sub_rule_path=f"table-{idx}.row-{row_idx}.cell-{col_idx}",
                            display_citation=f"{display_citation} Table {idx} Row {row_idx} {header}",
                            source_file=source_file,
                            anchor_id=f"{anchor}-table-{idx}-row-{row_idx}-cell-{col_idx}",
                            raw_text=f"{header}: {cell_text}",
                            normalized_text=normalize_text(f"{header} {cell_text}"),
                            occupancy_groups=occupancy_groups,
                            panchayat_category=panchayat_category,
                            topic_tags=topic_tags,
                            parent_clause_id=row_clause_id,
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
