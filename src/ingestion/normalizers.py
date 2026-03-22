from __future__ import annotations

import re
import unicodedata
from datetime import date


UNIT_MAP = {
    r"\bsq\.?\s*met(res|ers)?\b": "sq.m",
    r"\bsquare\s+met(res|ers)\b": "sq.m",
    r"\bm2\b": "sq.m",
    r"\bper\s+cent\b": "%",
    r"\bpercent\b": "%",
}


OCR_NORMALIZATIONS = {
    "Group AI": "Group A1",
    "Group GI": "Group G1",
    "Group 1(1)": "Group I(1)",
    "Group 1(2)": "Group I(2)",
    "Resdidential": "Residential",
}


def normalize_units(text: str) -> str:
    output = text
    for pattern, replacement in UNIT_MAP.items():
        output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)
    return output


def normalize_ocr_tokens(text: str, replacements: dict[str, str] | None = None) -> str:
    mapping = replacements or OCR_NORMALIZATIONS
    normalized = text
    for bad, good in mapping.items():
        normalized = normalized.replace(bad, good)
    return normalized


def normalize_text(
    text: str,
    *,
    preserve_newlines: bool = True,
    ocr_replacements: dict[str, str] | None = None,
    apply_ocr_replacements: bool = False,
) -> str:
    if not text:
        return ""
    output = unicodedata.normalize("NFKC", text)
    output = output.replace("\r\n", "\n").replace("\r", "\n")
    output = output.replace("\u200b", "").replace("\ufeff", "")
    output = normalize_units(output)
    if apply_ocr_replacements and ocr_replacements:
        output = normalize_ocr_tokens(output, replacements=ocr_replacements)

    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in output.split("\n")]
        compact: list[str] = []
        empty_run = 0
        for line in lines:
            if not line:
                empty_run += 1
                if empty_run <= 1:
                    compact.append("")
                continue
            empty_run = 0
            compact.append(line)
        return "\n".join(compact).strip()

    output = re.sub(r"\s+", " ", output)
    return output.strip()


def parse_effective_date(reference: str) -> date | None:
    """
    Parse simple date formats in amendment references.
    Example: dt.22-2-2001
    """
    patterns = [
        re.compile(r"dt\.?\s*(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", flags=re.IGNORECASE),
        re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", flags=re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(reference)
        if not match:
            continue
        if pattern is patterns[0]:
            day, month, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        else:
            year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return None
