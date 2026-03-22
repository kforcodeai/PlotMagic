from __future__ import annotations

import json
import re
from pathlib import Path

from src.models import AmendmentRecord, RuleDocument, TableData


class KMBRMarkdownExporter:
    def render_chapter(self, docs: list[RuleDocument], source_html: Path) -> str:
        if not docs:
            raise ValueError("No rule documents provided for chapter export.")

        ordered_docs = sorted(docs, key=lambda item: self._rule_sort_key(item.rule_number))
        first_doc = ordered_docs[0]
        chapter_number = first_doc.chapter_number
        chapter_title = first_doc.chapter_title

        lines = [
            f"# Chapter {chapter_number}: {chapter_title}",
            "",
            f"**Source HTML:** `{source_html.name}`",
            "",
        ]

        for doc in ordered_docs:
            lines.extend(self._render_rule(doc, source_html.name))

        return "\n".join(lines).rstrip() + "\n"

    def _render_rule(self, doc: RuleDocument, source_name: str) -> list[str]:
        source_ref = f"{source_name}#{doc.anchor_id}"
        citation_payload = json.dumps(
            {
                "chapter": doc.chapter_number,
                "rule": doc.rule_number,
                "citation": f"Chapter {doc.chapter_number}, Rule {doc.rule_number}",
                "anchor": doc.anchor_id,
                "source_html": source_ref,
            },
            ensure_ascii=False,
        )
        lines = [
            f"<!-- citation: {citation_payload} -->",
            f"<a id=\"{doc.anchor_id}\"></a>",
            f"## {doc.rule_number}. {doc.rule_title}",
            "",
            f"**Citation:** Chapter {doc.chapter_number}, Rule {doc.rule_number}",
            f"**Source:** `{source_ref}`",
            "",
        ]

        body = self._strip_rule_header(doc.full_text, doc.rule_number)
        body_lines = self._text_to_paragraphs(body)
        if body_lines:
            lines.extend(body_lines)
            lines.append("")

        if doc.amendments:
            lines.append("**Amendment History:**")
            for amendment in doc.amendments:
                lines.append(f"- {self._format_amendment(amendment)}")
            lines.append("")

        if doc.tables:
            lines.append("**Tables:**")
            lines.append("")
            for idx, table in enumerate(doc.tables, start=1):
                lines.extend(self._render_table(table, idx))
                lines.append("")

        lines.append("---")
        lines.append("")
        return lines

    def _rule_sort_key(self, rule_number: str) -> tuple[int, str]:
        match = re.match(r"(?P<number>\d+)(?P<suffix>[A-Za-z]*)", rule_number)
        if not match:
            return (10_000, rule_number)
        return (int(match.group("number")), match.group("suffix"))

    def _strip_rule_header(self, full_text: str, rule_number: str) -> str:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        if not lines:
            return ""
        if re.match(rf"^{re.escape(rule_number)}\s*\.", lines[0]):
            lines = lines[1:]
        return "\n".join(lines).strip()

    def _text_to_paragraphs(self, text: str) -> list[str]:
        if not text:
            return []
        lines: list[str] = []
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            lines.append(cleaned)
        return lines

    def _format_amendment(self, amendment: AmendmentRecord) -> str:
        tokens = [amendment.action.value.title(), amendment.reference_text]
        if amendment.effective_from:
            tokens.append(f"Effective: {amendment.effective_from.isoformat()}")
        return " | ".join(tokens)

    def _render_table(self, table: TableData, table_index: int) -> list[str]:
        caption = table.caption.strip() if table.caption else table.table_id
        rows = [list(row) for row in table.rows]
        headers = list(table.headers) if table.headers else []

        if not headers:
            max_len = max((len(row) for row in rows), default=1)
            headers = [f"col_{idx}" for idx in range(1, max_len + 1)]

        width = max(len(headers), max((len(row) for row in rows), default=0))
        headers = self._pad_cells(headers, width)
        rows = [self._pad_cells(row, width) for row in rows]

        lines = [
            f"### Table {table_index}: {caption}",
            f"| {' | '.join(self._escape_cell(cell) for cell in headers)} |",
            f"| {' | '.join('---' for _ in headers)} |",
        ]
        for row in rows:
            lines.append(f"| {' | '.join(self._escape_cell(cell) for cell in row)} |")
        return lines

    def _pad_cells(self, cells: list[str], width: int) -> list[str]:
        if len(cells) >= width:
            return cells[:width]
        return cells + [""] * (width - len(cells))

    def _escape_cell(self, text: str) -> str:
        escaped = text.replace("|", "\\|").replace("\n", "<br>")
        return escaped.strip()
