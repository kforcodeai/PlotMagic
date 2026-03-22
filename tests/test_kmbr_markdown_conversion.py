from __future__ import annotations

from pathlib import Path

from src.ingestion.kmbr_markdown_exporter import KMBRMarkdownExporter
from src.ingestion.parsers.kmbr_html_parser import KMBRHTMLParser


def _build_parser() -> KMBRHTMLParser:
    return KMBRHTMLParser(
        state="kerala",
        jurisdiction_type="municipality",
        ruleset_id="KMBR_1999",
        ruleset_version="1999",
    )


def test_kmbr_parser_recovers_rules_from_empty_anchor_headers() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = _build_parser()

    chapter1_docs = parser.parse_file(root / "data" / "kerala" / "kmbr_muncipal_rules" / "chapter1.html")
    chapter1_rules = {doc.rule_number for doc in chapter1_docs}
    assert {"1", "2", "3"}.issubset(chapter1_rules)

    chapter17_docs = parser.parse_file(root / "data" / "kerala" / "kmbr_muncipal_rules" / "chapter17.html")
    assert any(doc.rule_number == "115" for doc in chapter17_docs)


def test_kmbr_markdown_export_preserves_anchor_and_citation_block() -> None:
    root = Path(__file__).resolve().parents[1]
    html_file = root / "data" / "kerala" / "kmbr_muncipal_rules" / "chapter2.html"

    parser = _build_parser()
    docs = parser.parse_file(html_file)
    markdown = KMBRMarkdownExporter().render_chapter(docs, html_file)

    assert "<!-- citation:" in markdown
    assert '<a id="chapter2-1"></a>' in markdown
    assert "**Citation:** Chapter 2, Rule 4" in markdown
    assert "**Source:** `chapter2.html#chapter2-1`" in markdown
