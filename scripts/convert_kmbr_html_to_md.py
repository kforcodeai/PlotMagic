from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.kmbr_markdown_exporter import KMBRMarkdownExporter
from src.ingestion.parsers import KMBRHTMLParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert KMBR chapter HTML files into citation-rich Markdown.")
    parser.add_argument(
        "--input-dir",
        default="data/kerala/kmbr_muncipal_rules",
        help="Directory with chapter*.html files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/kerala/kmbr_muncipal_rules_md",
        help="Directory for generated markdown chapter files.",
    )
    parser.add_argument("--glob", default="chapter*.html", help="Input filename pattern.")
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="municipality")
    parser.add_argument("--ruleset-id", default="KMBR_1999")
    parser.add_argument("--ruleset-version", default="1999")
    parser.add_argument(
        "--manifest",
        default="citation_manifest.jsonl",
        help="Manifest filename to emit in output-dir.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = (ROOT / args.input_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = KMBRHTMLParser(
        state=args.state,
        jurisdiction_type=args.jurisdiction,
        ruleset_id=args.ruleset_id,
        ruleset_version=args.ruleset_version,
    )
    exporter = KMBRMarkdownExporter()

    summary: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for html_file in sorted(input_dir.glob(args.glob)):
        docs = parser.parse_file(html_file)
        if not docs:
            continue

        markdown = exporter.render_chapter(docs, source_html=html_file)
        output_path = output_dir / f"{html_file.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")

        summary.append(
            {
                "source_html": str(html_file.relative_to(ROOT)),
                "output_markdown": str(output_path.relative_to(ROOT)),
                "rules_exported": len(docs),
            }
        )
        for doc in docs:
            manifest_rows.append(
                {
                    "document_id": doc.document_id,
                    "chapter_number": doc.chapter_number,
                    "chapter_title": doc.chapter_title,
                    "rule_number": doc.rule_number,
                    "rule_title": doc.rule_title,
                    "display_citation": f"Chapter {doc.chapter_number}, Rule {doc.rule_number}",
                    "anchor_id": doc.anchor_id,
                    "source_html": f"{html_file.name}#{doc.anchor_id}",
                    "source_markdown": f"{output_path.name}#{doc.anchor_id}",
                }
            )

    manifest_path = output_dir / args.manifest
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "chapters_exported": len(summary),
        "rules_exported": sum(item["rules_exported"] for item in summary),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "files": summary,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
