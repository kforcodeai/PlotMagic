from __future__ import annotations

import re

CLEANING_VERSION = "markdown_cleaning.v2"


def collapse_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def clean_markdown_noise(raw_text: str) -> str:
    text = raw_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    lines = _strip_code_fence_markers(lines)
    lines = _drop_repeated_margin_lines(lines)
    lines = _drop_repeated_heading_lines(lines)
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _is_likely_page_marker(stripped):
            continue
        if _is_separator_only_fragment(stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def strip_frontpage_javascript(raw_html: str) -> str:
    text = re.sub(r"<script\b.*?</script>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"MM_showHideLayers\([^)]*\)", "", text)
    text = re.sub(r"MM_reloadPage\([^)]*\)", "", text)
    return text


def _is_likely_page_marker(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"^(?:page|pg\.?)\s*\d+(?:\s*(?:of|/)\s*\d+)?$",
        r"^\d+\s*(?:of|/)\s*\d+$",
        r"^\d{1,4}$",
    ]
    return any(re.fullmatch(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _drop_repeated_margin_lines(lines: list[str], min_repetitions: int = 4) -> list[str]:
    counts: dict[str, int] = {}
    for line in lines:
        key = collapse_spaces(line)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1

    repeated_margin = {
        key
        for key, count in counts.items()
        if count >= min_repetitions and _looks_like_margin_text(key)
    }
    if not repeated_margin:
        return lines
    return [line for line in lines if collapse_spaces(line) not in repeated_margin]


def _drop_repeated_heading_lines(lines: list[str], min_repetitions: int = 3) -> list[str]:
    counts: dict[str, int] = {}
    for line in lines:
        key = collapse_spaces(line)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1

    repeated = {
        key
        for key, count in counts.items()
        if count >= min_repetitions and _looks_like_heading_noise(key)
    }
    if not repeated:
        return lines
    return [line for line in lines if collapse_spaces(line) not in repeated]


def _looks_like_margin_text(text: str) -> bool:
    if len(text) > 120:
        return False
    token_count = len(text.split())
    if token_count > 15:
        return False
    if _is_likely_page_marker(text):
        return True
    return bool(re.fullmatch(r"[A-Z0-9 .,_()/-]+", text))


def _looks_like_heading_noise(text: str) -> bool:
    lowered = text.lower()
    if len(text) > 180:
        return False
    if len(lowered.split()) > 18:
        return False
    if re.search(r"\b(chapter|rule|appendix|building rules|panchayat|kerala)\b", lowered):
        return True
    if text.startswith("#"):
        return True
    if bool(re.fullmatch(r"[A-Za-z0-9 .,:;()/_-]+", text)) and text == text.title():
        return True
    return False


def _is_separator_only_fragment(text: str) -> bool:
    if not text:
        return False
    if "|" in text:
        return False
    return bool(re.fullmatch(r"[-_=~`*.:]{3,}", text))


def _strip_code_fence_markers(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"`{3,}.*", stripped) or re.fullmatch(r"~{3,}.*", stripped):
            continue
        cleaned.append(line)
    return cleaned
