from __future__ import annotations

from src.ingestion.cleaners import clean_markdown_noise


def test_clean_markdown_noise_removes_page_markers_code_fences_and_separators() -> None:
    raw = "\n".join(
        [
            "KERALA PANCHAYAT BUILDING RULES 2011",
            "KERALA PANCHAYAT BUILDING RULES 2011",
            "KERALA PANCHAYAT BUILDING RULES 2011",
            "Page 1 of 200",
            "```",
            "Code block heading",
            "```",
            "-----",
            "1.- Rule heading.",
            "Applicable text.",
        ]
    )
    cleaned = clean_markdown_noise(raw)

    assert "Page 1 of 200" not in cleaned
    assert "```" not in cleaned
    assert "-----" not in cleaned
    assert "1.- Rule heading." in cleaned
    assert "Applicable text." in cleaned
    assert "KERALA PANCHAYAT BUILDING RULES 2011" not in cleaned
