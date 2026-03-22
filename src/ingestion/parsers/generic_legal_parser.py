from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.cleaners import clean_markdown_noise, strip_frontpage_javascript
from src.ingestion.normalizers import normalize_text
from src.models import ClauseNode, ClauseType, CrossReference, RuleDocument, TableData

from .base import RulesetParser

_NAMED_HEADING_RE = re.compile(
    r"^(?P<label>(?:section|sec\.?|rule|regulation|reg\.?|clause|article|chapter|part|schedule|annexure|annex|appendix)\s+"
    r"(?P<number>[A-Za-z0-9()./\-]+))\s*[:.\-]?\s*(?P<title>.*)$",
    flags=re.IGNORECASE,
)
_NUMERIC_HEADING_RE = re.compile(
    r"^(?P<number>\d+(?:[A-Za-z]|\.\d+|\([a-z0-9]+\))*)(?:[.)-])\s*(?P<title>.+)$",
    flags=re.IGNORECASE,
)
_SUBCLAUSE_RE = re.compile(r"^\s*(?P<token>\([a-z0-9ivx]+\)|\d+(?:\.\d+)+)\s*(?P<text>.*)$", flags=re.IGNORECASE)

_CROSS_REF_PATTERNS = [
    (r"\b(?:Rule|Section|Sec\.?|Clause|Article|Regulation|Reg\.?)\s+[A-Za-z0-9()./\-]+\b", "provision"),
    (r"\b(?:Chapter|Part|Schedule|Appendix|Annexure|Annex)\s+[A-Za-z0-9()./\-]+\b", "structural"),
    (r"\bTable\s+[A-Za-z0-9()./\-]+\b", "table"),
]


class GenericLegalParser(RulesetParser):
    """
    Generic parser for legal/compliance corpora across html/markdown/txt/pdf.
    Produces section-level documents with optional sub-clause nodes.
    """

    def parse_file(self, file_path: Path) -> list[RuleDocument]:
        text, file_tables, parser_notes = self._extract_text_and_tables(file_path)
        lines = [self._clean_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]

        if not lines:
            return []

        sections = self._split_sections(lines)
        chapter_title_default = self._fallback_chapter_title(file_path)
        documents: list[RuleDocument] = []
        for idx, section in enumerate(sections, start=1):
            rule_number = section["number"] or str(idx)
            rule_title = section["title"] or f"Section {rule_number}"
            section_lines = section["lines"]
            full_text = normalize_text("\n".join(section_lines))
            if not full_text:
                continue

            chapter_number = section["chapter_number"] or 0
            chapter_title = section["chapter_title"] or chapter_title_default
            anchor = self._slugify(f"{file_path.stem}-{rule_number}-{idx}")
            document_id = f"{self.ruleset_id}-{anchor}"
            cross_refs = self._extract_cross_refs(full_text, source_clause_id=f"{document_id}-rule")
            tables = self._extract_markdown_style_tables(section_lines, document_id)
            if idx == 1 and file_tables:
                tables = [*tables, *file_tables]

            notes = [*parser_notes]
            clause_nodes = self._build_clause_nodes(
                document_id=document_id,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
                rule_number=rule_number,
                rule_title=rule_title,
                full_text=full_text,
                source_file=str(file_path),
                anchor=anchor,
                notes=notes,
                tables=tables,
                cross_refs=cross_refs,
            )
            documents.append(
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
                    rule_number=rule_number,
                    rule_title=rule_title,
                    full_text=full_text,
                    anchor_id=anchor,
                    occupancy_groups=[],
                    is_generic=True,
                    panchayat_category=None,
                    conditions={},
                    numeric_values={},
                    provisos=[],
                    notes=notes,
                    tables=tables,
                    cross_references=cross_refs,
                    amendments=[],
                    clause_nodes=clause_nodes,
                )
            )
        return documents

    def _extract_text_and_tables(self, file_path: Path) -> tuple[str, list[TableData], list[str]]:
        ext = file_path.suffix.lower()
        if ext in {".md", ".markdown"}:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            return clean_markdown_noise(raw), [], []
        if ext in {".txt"}:
            return file_path.read_text(encoding="utf-8", errors="ignore"), [], []
        if ext in {".html", ".htm"}:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned = strip_frontpage_javascript(raw)
            soup = BeautifulSoup(cleaned, "lxml")
            tables = self._extract_html_tables(soup, file_path)
            return soup.get_text("\n", strip=True), tables, []
        if ext == ".pdf":
            return self._extract_pdf_text(file_path)
        raise ValueError(f"Unsupported file extension '{ext}' for generic parser: {file_path}")

    def _extract_pdf_text(self, file_path: Path) -> tuple[str, list[TableData], list[str]]:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as exc:
            raise ValueError("PDF parsing requires optional dependency 'pypdf'.") from exc

        reader = PdfReader(str(file_path))
        page_text: list[str] = []
        empty_pages = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            normalized = normalize_text(text)
            if not normalized:
                empty_pages += 1
            page_text.append(normalized)

        joined = "\n\n".join([text for text in page_text if text.strip()])
        if not joined:
            raise ValueError(f"No extractable text from PDF '{file_path}'. It may be scanned-only.")
        alpha_ratio = self._alpha_ratio(joined)
        notes: list[str] = []
        if alpha_ratio < 0.45 or empty_pages > 0:
            notes.append(
                f"Low-confidence text extraction detected (alpha_ratio={alpha_ratio:.2f}, empty_pages={empty_pages})."
            )
        return joined, [], notes

    def _split_sections(self, lines: list[str]) -> list[dict[str, object]]:
        sections: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        for line in lines:
            heading = self._parse_heading(line)
            if heading:
                if current and current.get("lines"):
                    sections.append(current)
                current = {
                    "number": heading["number"],
                    "title": heading["title"],
                    "chapter_number": heading["chapter_number"],
                    "chapter_title": heading["chapter_title"],
                    "lines": [line],
                }
                continue
            if current is None:
                current = {
                    "number": "0",
                    "title": "",
                    "chapter_number": 0,
                    "chapter_title": "",
                    "lines": [],
                }
            current["lines"].append(line)
        if current and current.get("lines"):
            sections.append(current)
        return sections

    def _parse_heading(self, line: str) -> dict[str, object] | None:
        named = _NAMED_HEADING_RE.match(line)
        if named:
            label = named.group("label")
            number = named.group("number").strip().rstrip(".-:;")
            title = normalize_text(named.group("title"))
            chapter_number = self._chapter_number_from_named_heading(label)
            chapter_title = title if label.lower().startswith(("chapter", "part")) else ""
            return {
                "number": number,
                "title": title or label,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
            }
        numbered = _NUMERIC_HEADING_RE.match(line)
        if numbered and len(numbered.group("title").split()) >= 2:
            return {
                "number": numbered.group("number").strip().rstrip(".-:;"),
                "title": normalize_text(numbered.group("title")),
                "chapter_number": 0,
                "chapter_title": "",
            }
        return None

    def _chapter_number_from_named_heading(self, heading: str) -> int:
        if not heading.lower().startswith("chapter"):
            return 0
        match = re.search(r"([A-Za-z0-9()./\-]+)$", heading)
        if not match:
            return 0
        token = match.group(1)
        if token.isdigit():
            return int(token)
        roman = token.upper()
        if re.fullmatch(r"[IVXLCM]+", roman):
            return self._roman_to_int(roman)
        return 0

    def _roman_to_int(self, roman: str) -> int:
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "M": 1000}
        total = 0
        previous = 0
        for char in reversed(roman):
            value = values.get(char, 0)
            if value < previous:
                total -= value
            else:
                total += value
                previous = value
        return total

    def _build_clause_nodes(
        self,
        *,
        document_id: str,
        chapter_number: int,
        chapter_title: str,
        rule_number: str,
        rule_title: str,
        full_text: str,
        source_file: str,
        anchor: str,
        notes: list[str],
        tables: list[TableData],
        cross_refs: list[CrossReference],
    ) -> list[ClauseNode]:
        nodes: list[ClauseNode] = []
        rule_clause_id = f"{document_id}-rule"
        display_citation = f"Rule {rule_number}"
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
                occupancy_groups=[],
                topic_tags=[],
                is_generic=True,
                conditions={},
                numeric_values={},
                cross_references=cross_refs,
            )
        )

        for token, text in self._extract_subclauses(full_text):
            sub_id = self._slugify(token)
            nodes.append(
                ClauseNode(
                    clause_id=f"{document_id}-sub-{sub_id}",
                    clause_type=ClauseType.SUB_RULE,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    sub_rule_path=token,
                    display_citation=f"{display_citation} {token}",
                    source_file=source_file,
                    anchor_id=f"{anchor}-sub-{sub_id}",
                    raw_text=text,
                    normalized_text=normalize_text(text),
                    occupancy_groups=[],
                    topic_tags=[],
                    is_generic=True,
                    parent_clause_id=rule_clause_id,
                )
            )

        for index, table in enumerate(tables, start=1):
            nodes.append(
                ClauseNode(
                    clause_id=f"{document_id}-table-{index}",
                    clause_type=ClauseType.TABLE,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    sub_rule_path=f"table-{index}",
                    display_citation=f"{display_citation} Table {index}",
                    source_file=source_file,
                    anchor_id=f"{anchor}-table-{index}",
                    raw_text=table.raw_text,
                    normalized_text=normalize_text(table.raw_text),
                    occupancy_groups=[],
                    topic_tags=[],
                    is_generic=True,
                    table_data=table,
                    parent_clause_id=rule_clause_id,
                )
            )

        for index, note in enumerate(notes, start=1):
            nodes.append(
                ClauseNode(
                    clause_id=f"{document_id}-note-{index}",
                    clause_type=ClauseType.NOTE,
                    state=self.state,
                    jurisdiction_type=self.jurisdiction_type,
                    ruleset_id=self.ruleset_id,
                    ruleset_version=self.ruleset_version,
                    chapter_number=chapter_number,
                    chapter_title=chapter_title,
                    rule_number=rule_number,
                    rule_title=rule_title,
                    sub_rule_path=f"note-{index}",
                    display_citation=f"{display_citation} Note {index}",
                    source_file=source_file,
                    anchor_id=f"{anchor}-note-{index}",
                    raw_text=note,
                    normalized_text=normalize_text(note),
                    occupancy_groups=[],
                    topic_tags=[],
                    is_generic=True,
                    parent_clause_id=rule_clause_id,
                )
            )
        return nodes

    def _extract_subclauses(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_token: str | None = None
        buffer: list[str] = []
        for line in text.splitlines():
            match = _SUBCLAUSE_RE.match(line)
            if match:
                if current_token and buffer:
                    sections.append((current_token, normalize_text("\n".join(buffer))))
                current_token = match.group("token")
                seed = normalize_text(match.group("text"))
                buffer = [seed] if seed else []
                continue
            if current_token:
                buffer.append(line)
        if current_token and buffer:
            sections.append((current_token, normalize_text("\n".join(buffer))))
        return sections

    def _extract_cross_refs(self, text: str, source_clause_id: str) -> list[CrossReference]:
        refs: list[CrossReference] = []
        seen: set[tuple[str, str]] = set()
        for pattern, target_type in _CROSS_REF_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                target_ref = normalize_text(match.group(0))
                normalized = self._slugify(target_ref)
                key = (target_type, normalized)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(
                    CrossReference(
                        source_clause_id=source_clause_id,
                        target_ref=target_ref,
                        target_type=target_type,
                        normalized_target_id=normalized,
                    )
                )
        return refs

    def _extract_markdown_style_tables(self, lines: list[str], document_id: str) -> list[TableData]:
        tables: list[TableData] = []
        table_lines: list[str] = []
        for line in lines:
            if line.count("|") >= 2:
                table_lines.append(line.strip())
                continue
            if table_lines:
                maybe_table = self._table_from_pipe_lines(table_lines, document_id, len(tables) + 1)
                if maybe_table:
                    tables.append(maybe_table)
                table_lines = []
        if table_lines:
            maybe_table = self._table_from_pipe_lines(table_lines, document_id, len(tables) + 1)
            if maybe_table:
                tables.append(maybe_table)
        return tables

    def _table_from_pipe_lines(self, lines: list[str], document_id: str, index: int) -> TableData | None:
        if len(lines) < 2:
            return None
        rows = [self._split_pipe_row(line) for line in lines]
        rows = [row for row in rows if row]
        if len(rows) < 2:
            return None
        headers = rows[0]
        body = rows[1:]
        if body and all(re.fullmatch(r"-{2,}", cell.replace(":", "").strip()) for cell in body[0]):
            body = body[1:]
        if not body:
            return None
        table_id = f"{document_id}-table-{index}"
        return TableData(table_id=table_id, caption=f"Table {index}", headers=headers, rows=body, raw_text="\n".join(lines))

    def _extract_html_tables(self, soup: BeautifulSoup, file_path: Path) -> list[TableData]:
        tables: list[TableData] = []
        for index, table in enumerate(soup.find_all("table"), start=1):
            rows: list[list[str]] = []
            for tr in table.find_all("tr"):
                cells = [self._clean_line(td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(cells)
            if len(rows) < 2:
                continue
            headers = rows[0]
            body = rows[1:]
            table_id = f"{self.ruleset_id}-{self._slugify(file_path.stem)}-table-{index}"
            tables.append(TableData(table_id=table_id, caption=f"Table {index}", headers=headers, rows=body, raw_text=str(table)))
        return tables

    def _split_pipe_row(self, line: str) -> list[str]:
        stripped = line.strip().strip("|")
        if not stripped:
            return []
        return [self._clean_line(cell) for cell in stripped.split("|")]

    def _clean_line(self, line: str) -> str:
        compact = line.replace("\xa0", " ")
        compact = re.sub(r"[ \t]+", " ", compact)
        return compact.strip()

    def _fallback_chapter_title(self, file_path: Path) -> str:
        token = file_path.stem.replace("_", " ").replace("-", " ").strip()
        return normalize_text(token.title()) or "General"

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "x"

    def _alpha_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        alpha = sum(1 for ch in text if ch.isalpha())
        return alpha / float(len(text))
