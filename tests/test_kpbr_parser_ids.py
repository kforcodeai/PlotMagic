from __future__ import annotations

from pathlib import Path

from src.ingestion.parsers.kpbr_markdown_parser import KPBRMarkdownParser


def test_kpbr_sub_rule_ids_are_collision_safe_with_ordinals(tmp_path: Path) -> None:
    source = tmp_path / "kpbr_sample.md"
    source.write_text(
        "\n".join(
            [
                "# CHAPTER I",
                "1.- Sample Rule.",
                "Category II applies for this clause.",
                "(1) First requirement text.",
                "(1) Second requirement text with same marker.",
                "(2) Third requirement text.",
            ]
        ),
        encoding="utf-8",
    )

    parser = KPBRMarkdownParser(
        state="kerala",
        jurisdiction_type="panchayat",
        ruleset_id="KPBR_2011",
        ruleset_version="2011",
        issuing_authority="LSGD",
    )
    docs = parser.parse_file(source)
    assert len(docs) == 1

    sub_rules = [node for node in docs[0].clause_nodes if node.clause_type.value == "sub_rule"]
    assert len(sub_rules) == 3
    assert len({node.clause_id for node in sub_rules}) == 3
    assert all("-o" in node.clause_id for node in sub_rules)
    assert all(node.panchayat_category == "Category-II" for node in docs[0].clause_nodes)
