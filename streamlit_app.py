from __future__ import annotations

import html
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.api.service import ComplianceEngine
from src.models.schemas import AnswerResponse, ComplianceBriefPayload, QueryRequest


ROOT = Path(__file__).resolve().parent

EXAMPLE_QUERIES_PANCHAYAT = [
    "What permits are needed to construct a residential building in a panchayat area?",
    "What are the setback rules for Category-II panchayat buildings?",
    "Is a completion certificate required for buildings under 100 sq metres?",
    "What are the penalties for unauthorized construction in a panchayat?",
]

EXAMPLE_QUERIES_MUNICIPALITY = [
    "What is the maximum FAR for residential buildings in a municipality?",
    "What permits are needed for commercial construction in a municipality?",
    "What are the parking requirements for a hostel in a municipality area?",
    "What are the open space requirements for multi-storey buildings?",
]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        cleaned = value.strip().strip('"').strip("'")
        os.environ[key] = cleaned


@st.cache_resource(show_spinner=False)
def get_engine(llm_provider_id: str) -> ComplianceEngine:
    _load_env_file(ROOT / ".env")
    os.environ["PLOTMAGIC_LLM_PROVIDER"] = llm_provider_id
    return ComplianceEngine(root=ROOT)


def inject_chat_styles() -> None:
    st.markdown(
        """
<style>
    /* ── Foundation ── */
    :root {
        --pm-surface: #ffffff;
        --pm-surface-raised: #f8fafc;
        --pm-border: #e2e8f0;
        --pm-border-light: #f1f5f9;
        --pm-muted: #64748b;
        --pm-accent: #0f766e;
        --pm-accent-light: #ccfbf1;
        --pm-accent-muted: #99f6e4;
        --pm-user-bg: #eff6ff;
        --pm-danger: #dc2626;
        --pm-danger-bg: #fef2f2;
        --pm-warning-bg: #fffbeb;
        --pm-success: #16a34a;
        --pm-success-bg: #f0fdf4;
        --pm-radius: 12px;
        --pm-radius-lg: 18px;
        --pm-chat-width: 820px;
        --pm-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
        --pm-shadow-md: 0 8px 22px rgba(15, 23, 42, 0.09);
    }

    /* ── Main canvas ── */
    [data-testid="stMain"],
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .main .block-container {
        max-width: 940px;
        padding-top: 1rem;
        padding-bottom: 8.4rem;
    }
    .main .block-container a {
        color: var(--pm-accent) !important;
    }
    .main [data-testid="stCaptionContainer"] {
        color: var(--pm-muted) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0c1222 0%, #111827 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stToggle label {
        color: #94a3b8 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] .stButton button {
        border: 1px solid #334155 !important;
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #334155 !important;
        border-color: #475569 !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.2rem 0.1rem;
        margin-bottom: 0.75rem;
        max-width: var(--pm-chat-width);
        width: min(var(--pm-chat-width), 100%);
        margin-left: auto;
        margin-right: auto;
        transition: background 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        background: rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--pm-user-bg) !important;
        border: 1px solid var(--pm-border) !important;
        border-radius: var(--pm-radius-lg) !important;
        padding: 0.5rem 0.92rem;
        box-shadow: var(--pm-shadow-sm) !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #0f172a !important;
        line-height: 1.6;
    }

    /* ── Chat input ── */
    [data-testid="stChatInputContainer"] {
        background: linear-gradient(to top, rgba(241, 245, 249, 0.98) 64%, rgba(241, 245, 249, 0)) !important;
        border-top: 0 !important;
    }
    [data-testid="stChatInputContainer"] > div {
        max-width: var(--pm-chat-width) !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding: 0.42rem 0 0.62rem 0;
        display: flex;
        justify-content: center;
    }
    [data-testid="stChatInputContainer"] > div > div {
        width: 100% !important;
        max-width: var(--pm-chat-width) !important;
        margin: 0 auto !important;
    }
    [data-testid="stChatInput"] {
        width: 100% !important;
        max-width: var(--pm-chat-width) !important;
        margin: 0 auto !important;
        border: 1px solid var(--pm-border) !important;
        border-radius: 28px !important;
        background: #ffffff !important;
        backdrop-filter: blur(8px);
        box-shadow:
            0 10px 24px rgba(15, 23, 42, 0.09),
            0 1px 2px rgba(15, 23, 42, 0.08) !important;
        min-height: 58px;
    }
    [data-testid="stChatInput"] > div {
        padding: 0.2rem 0.4rem;
    }
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        font-size: 0.98rem !important;
        line-height: 1.45 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    [data-testid="stChatInput"] button {
        border-radius: 999px !important;
        width: 32px !important;
        height: 32px !important;
        background: #f1f5f9 !important;
        border: 1px solid #d1d9e8 !important;
        color: #334155 !important;
    }

    /* ── Verdict badges ── */
    .pm-verdict {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .pm-verdict-compliant {
        background: var(--pm-success-bg);
        color: var(--pm-success);
        border: 1px solid #bbf7d0;
    }
    .pm-verdict-noncompliant {
        background: var(--pm-danger-bg);
        color: var(--pm-danger);
        border: 1px solid #fecaca;
    }
    .pm-verdict-insufficient {
        background: var(--pm-warning-bg);
        color: #d97706;
        border: 1px solid #fde68a;
    }
    .pm-verdict-depends {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }

    /* ── Provider chip ── */
    .pm-provider-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        border: 1px solid #bfd9d5;
        color: var(--pm-accent);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        background: var(--pm-accent-light);
    }

    /* ── Section cards ── */
    .pm-section-card {
        background: var(--pm-surface);
        border: 1px solid var(--pm-border);
        border-radius: var(--pm-radius);
        padding: 1rem 1.15rem;
        margin-bottom: 0.75rem;
        box-shadow: var(--pm-shadow-sm);
    }
    .pm-section-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--pm-muted);
        font-weight: 700;
    }

    /* ── Risk flag pills ── */
    .pm-risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: var(--pm-danger-bg);
        color: var(--pm-danger);
        border: 1px solid #fecaca;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    /* ── Grounding meter ── */
    .pm-meter-track {
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin: 0.4rem 0;
    }
    .pm-meter-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    /* ── Latency bar ── */
    .pm-latency-bar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.2rem 0;
        font-size: 0.8rem;
    }
    .pm-latency-bar-fill {
        height: 6px;
        border-radius: 999px;
        background: var(--pm-accent);
        transition: width 0.4s ease;
    }

    /* ── Citation card ── */
    .pm-citation-card {
        background: var(--pm-surface-raised);
        border-left: 3px solid var(--pm-accent);
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    /* ── Empty state ── */
    .pm-empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
    }
    .pm-empty-state h2 {
        color: #0f172a;
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .pm-empty-state p {
        color: var(--pm-muted);
        font-size: 0.95rem;
        max-width: 480px;
        margin: 0 auto 1.5rem auto;
    }

    /* ── Example query buttons ── */
    .pm-example-btn {
        display: block;
        width: 100%;
        padding: 0.65rem 1rem;
        margin-bottom: 0.45rem;
        background: var(--pm-surface);
        border: 1px solid var(--pm-border);
        border-radius: 10px;
        color: #334155;
        font-size: 0.88rem;
        text-align: left;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .pm-example-btn:hover {
        background: var(--pm-accent-light);
        border-color: var(--pm-accent-muted);
        color: var(--pm-accent);
    }

    .main .stButton button {
        background: #ffffff !important;
        color: #334155 !important;
        border: 1px solid var(--pm-border) !important;
        border-radius: 14px !important;
        box-shadow: var(--pm-shadow-sm) !important;
    }
    .main .stButton button:hover {
        background: #f8fafc !important;
        border-color: #cbd5e1 !important;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px 8px 0 0 !important;
    }

    /* ── Expander polish ── */
    [data-testid="stExpander"] {
        border: 1px solid var(--pm-border) !important;
        border-radius: var(--pm-radius) !important;
        box-shadow: var(--pm-shadow-sm);
        margin-bottom: 0.5rem;
        background: var(--pm-surface) !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #334155 !important;
    }
    @media (max-width: 900px) {
        :root {
            --pm-chat-width: calc(100vw - 1.25rem);
        }
        .main .block-container {
            padding-bottom: 7rem;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_event_line(event: dict[str, Any]) -> str:
    status = str(event.get("status", ""))
    icon = {
        "running": "\u23f3",
        "ok": "\u2705",
        "skipped": "\u23ed\ufe0f",
        "abstained": "\U0001f6d1",
        "needs_clarification": "\u2753",
    }.get(status, "\u2022")
    step = str(event.get("step", "event"))
    details = event.get("details", {})
    details_text = json.dumps(details, ensure_ascii=False) if details else "{}"
    return f"{icon} `{step}` \u2014 `{status}`  \n`{details_text}`"


def citation_id_for_payload(citation: Any) -> str:
    return f"{citation.ruleset_id}:{citation.rule_number}:{citation.anchor_id}"


def build_citation_lookup(response: Any) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for citation in response.citations:
        lookup[citation_id_for_payload(citation)] = citation
    return lookup


def render_inline_citation_links(citation_ids: list[str], citation_lookup: dict[str, Any]) -> str:
    links: list[str] = []
    for citation_id in citation_ids:
        citation = citation_lookup.get(citation_id)
        if citation is None:
            links.append(f"`{citation_id}`")
            continue
        label = citation.display_citation or citation_id
        if citation.source_url:
            links.append(f"[{label}]({citation.source_url})")
        else:
            links.append(f"`{label}`")
    return " ".join(links)


def render_inline_citation_links_html(citation_ids: list[str], citation_lookup: dict[str, Any]) -> str:
    links: list[str] = []
    for citation_id in citation_ids:
        citation = citation_lookup.get(citation_id)
        if citation is None:
            links.append(f"<code>{html.escape(citation_id)}</code>")
            continue
        label = html.escape(citation.display_citation or citation_id)
        if citation.source_url:
            safe_url = html.escape(citation.source_url, quote=True)
            links.append(f"<a href='{safe_url}' target='_blank'>{label}</a>")
        else:
            links.append(f"<code>{label}</code>")
    return " ".join(links)


def choose_dynamic_top_k(query: str, jurisdiction: str, llm_provider_id: str) -> tuple[int, str]:
    text = query.strip().lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    word_count = len(tokens)
    top_k = 8
    reasons: list[str] = ["base=8"]

    if jurisdiction.strip().lower() == "municipality":
        top_k += 1
        reasons.append("+1 municipality scope")

    if word_count >= 22:
        top_k += 4
        reasons.append("+4 long query")
    elif word_count >= 16:
        top_k += 3
        reasons.append("+3 medium-long query")
    elif word_count >= 10:
        top_k += 2
        reasons.append("+2 medium query")
    elif word_count >= 6:
        top_k += 1
        reasons.append("+1 short query")

    if len(re.findall(r"\b(and|or|except|unless|provided)\b", text)) >= 2:
        top_k += 2
        reasons.append("+2 multi-condition intent")

    if re.search(r"\b\d+(?:\.\d+)?\b", text):
        top_k += 2
        reasons.append("+2 numeric constraints")

    high_detail_terms = {
        "setback",
        "distance",
        "clearance",
        "penalty",
        "compounding",
        "documents",
        "certificate",
        "exemption",
        "occupancy",
        "fee",
        "regularisation",
        "regularization",
    }
    overlap = len(high_detail_terms.intersection(set(tokens)))
    if overlap >= 3:
        top_k += 3
        reasons.append("+3 dense compliance topics")
    elif overlap >= 1:
        top_k += 1
        reasons.append("+1 compliance topic keyword")

    if llm_provider_id == "no_llm":
        top_k += 1
        reasons.append("+1 retrieval-only mode")

    bounded = max(6, min(24, top_k))
    if bounded != top_k:
        reasons.append(f"clamped->{bounded}")
    return bounded, "; ".join(reasons)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_transform_fragment(text: str) -> str:
    clean = _normalize_whitespace(str(text or ""))
    clean = re.sub(r"^[\-\*\u2022]\s+", "", clean)
    clean = re.sub(r"^\d+\s*[.)]\s*", "", clean)
    clean = re.sub(r"\([^)]*\)", "", clean)
    clean = re.sub(r"\bAppendix\s*[A-Za-z0-9\-]+\b", "required form", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bRule\s*\d+[A-Za-z\-]*\b", "the rule", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bSection\s*\d+[A-Za-z\-]*\b", "the law", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:KPBR|KMBR)\s*[_\-\dA-Za-z.]*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^\b(?:as cited|as stated|as per evidence)\b[:,]?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(in|under|as per|according to)\s*,", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+,", ",", clean)
    clean = re.sub(r",\s*,+", ", ", clean)
    clean = re.sub(r"\.{2,}", ".", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" ,;:.-")
    return clean


def _truncate_words(text: str, max_words: int, *, ellipsis: bool = True) -> str:
    clean = _clean_transform_fragment(text)
    if max_words <= 0:
        return ""
    words = clean.split()
    if len(words) <= max_words:
        return clean
    suffix = "..." if ellipsis else ""
    return " ".join(words[:max_words]).rstrip(" ,;:.") + suffix


def _split_sentences(text: str) -> list[str]:
    clean = _normalize_whitespace(str(text or ""))
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+|;\s+", clean)
    out: list[str] = []
    seen: set[str] = set()
    for raw in parts:
        item = _clean_transform_fragment(raw.strip(" -\t"))
        if len(item) < 4:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    if out:
        return out
    return [clean]


def _transform_items(text: str, *, max_items: int, words_per_item: int, ellipsis: bool = True) -> list[str]:
    items: list[str] = []
    for sentence in _split_sentences(text):
        clipped = _truncate_words(sentence, words_per_item, ellipsis=ellipsis)
        if clipped:
            items.append(clipped)
        if len(items) >= max_items:
            break
    return items


def _finalize_sentence(text: str) -> str:
    clean = _normalize_whitespace(str(text or ""))
    clean = re.sub(r"\b(?:under|in|for|to|of|the|a|an)\s*\.$", ".", clean, flags=re.IGNORECASE)
    clean = clean.strip()
    if not clean:
        return ""
    if clean[-1] not in ".!?":
        clean = f"{clean}."
    return clean[0].upper() + clean[1:]


def _coerce_transform_summary(style: str, text: str, *, fallback: str = "") -> str:
    normalized_style = _normalize_whitespace(style).lower()
    base = _normalize_whitespace(text or "") or _normalize_whitespace(fallback or "")
    if normalized_style == "show_retrieved":
        return "Showing previously retrieved evidence from earlier turns."
    if not base:
        return ""
    if normalized_style == "summary":
        sentences = _transform_items(base, max_items=2, words_per_item=28, ellipsis=True)
        if not sentences:
            sentences = [_truncate_words(base, 40, ellipsis=True)]
        merged = " ".join(_finalize_sentence(item.rstrip(".")) for item in sentences[:2])
        return _normalize_whitespace(merged)
    if normalized_style == "eli5":
        simple = " ".join(_split_sentences(base)[:3])
        replacements = [
            (r"\bshall\b", "must"),
            (r"\bpursuant to\b", "under"),
            (r"\bthereof\b", "of it"),
            (r"\btherein\b", "in it"),
            (r"\bherein\b", "here"),
            (r"\bapplicant\b", "you"),
            (r"\bSecretary\b", "local authority"),
            (r"\bdevelopment certificate\b", "approval paper"),
            (r"\boccupancy certificate\b", "use certificate"),
            (r"\bcompletion certificate\b", "finish-work certificate"),
        ]
        for pattern, replacement in replacements:
            simple = re.sub(pattern, replacement, simple, flags=re.IGNORECASE)
        items = _transform_items(simple, max_items=2, words_per_item=26, ellipsis=True)
        if not items:
            items = [_truncate_words(simple, 30, ellipsis=True)]
        merged = " ".join(_finalize_sentence(item.rstrip(".")) for item in items[:2])
        return _normalize_whitespace(merged)
    if normalized_style == "bullets":
        bullet_source = re.sub(r"\s+\d+\s*[.)]\s+", ". ", base)
        items = _transform_items(bullet_source, max_items=4, words_per_item=20, ellipsis=True)
        if not items:
            items = [_truncate_words(base, 20, ellipsis=True)]
        return "\n".join(f"- {_finalize_sentence(item.rstrip('.'))}" for item in items[:4])
    if normalized_style == "checklist":
        checklist_source = re.sub(r"\s+\d+\s*[.)]\s+", ". ", base)
        items = _transform_items(checklist_source, max_items=4, words_per_item=20, ellipsis=True)
        if not items:
            items = [_truncate_words(base, 20, ellipsis=True)]
        return "\n".join(f"{idx}. {_finalize_sentence(item.rstrip('.'))}" for idx, item in enumerate(items[:4], start=1))
    return base


def _sanitize_followup_suggestion(text: str, *, max_chars: int = 84) -> str:
    clean = _normalize_whitespace(str(text or ""))
    if not clean:
        return ""
    clean = re.sub(r"\((?:[^)]*(?:KPBR|KMBR|Rule|Section)[^)]*)\)", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip(" -")
    if len(clean) > max_chars:
        clipped = clean[: max_chars - 3]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        clean = clipped.rstrip(" ,;:.") + "..."
    if clean and not clean.endswith("?"):
        clean = clean.rstrip(".") + "?"
    return clean


def _assistant_summary_from_message(message: dict[str, Any]) -> str:
    response = message.get("response")
    if response is None:
        return ""

    final_answer = getattr(response, "final_answer", None)
    if final_answer is not None:
        return _normalize_whitespace(str(getattr(final_answer, "short_summary", "")))

    if isinstance(response, dict):
        final_answer_dict = response.get("final_answer") or {}
        if isinstance(final_answer_dict, dict):
            return _normalize_whitespace(str(final_answer_dict.get("short_summary", "")))
    return ""


def _coerce_assistant_response(response: Any) -> Any | None:
    if response is None:
        return None
    if isinstance(response, AnswerResponse):
        return response if response.final_answer is not None else None
    if getattr(response, "final_answer", None) is not None and callable(getattr(response, "model_copy", None)):
        return response

    candidate_raw: dict[str, Any] | None = None
    if isinstance(response, dict):
        candidate_raw = response
    elif callable(getattr(response, "model_dump", None)):
        try:
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                candidate_raw = dumped
        except Exception:
            candidate_raw = None

    if isinstance(candidate_raw, dict):
        try:
            candidate = AnswerResponse.model_validate(candidate_raw)
            if candidate.final_answer is not None:
                return candidate
        except Exception:
            return None
    return None


def _recent_conversation_turns(chat_history: list[dict[str, Any]], *, limit: int) -> list[tuple[str, str]]:
    if limit <= 0:
        return []
    turns: list[tuple[str, str]] = []
    pending_user = ""
    for message in chat_history:
        role = str(message.get("role", ""))
        if role == "user":
            pending_user = _normalize_whitespace(str(message.get("content", "")))
            continue
        if role != "assistant" or not pending_user:
            continue
        turns.append((pending_user, _assistant_summary_from_message(message)))
        pending_user = ""
    return turns[-limit:]


def _query_looks_followup(query: str) -> bool:
    lowered = _normalize_whitespace(query.lower())
    if lowered.startswith(("and ", "also ", "what about", "how about", "compare ", "combine ")):
        return True
    markers = [
        r"\b(previous|earlier|above|that answer|last answer|same as above)\b",
        r"\b(combine|merge|compare|difference between|summari[sz]e both)\b",
        r"\b(use that|from that|based on that)\b",
    ]
    return any(re.search(pattern, lowered) for pattern in markers)


def _explicit_transform_style(query: str) -> str | None:
    lowered = _normalize_whitespace(query.lower())
    if not lowered:
        return None
    if "show me all that you have retrieved" in lowered or "show retrieved evidence" in lowered or "show all retrieved" in lowered:
        return "show_retrieved"
    reference_markers = ("this", "that", "above", "previous", "last answer", "the answer", "it")
    references_prior = any(marker in lowered for marker in reference_markers)
    if "eli5" in lowered or "simple terms" in lowered or "plain language" in lowered:
        return "eli5"
    if "checklist" in lowered or "check list" in lowered:
        if references_prior or lowered.startswith(("make", "convert", "turn")):
            return "checklist"
    if "bullet" in lowered or "bulleted" in lowered:
        if references_prior or lowered.startswith(("make", "convert", "turn")):
            return "bullets"
    if "summarize" in lowered or "summarise" in lowered or "summary" in lowered:
        if references_prior or lowered.startswith(("summarize", "summarise")):
            return "summary"
    return None


def build_contextual_query(
    *,
    raw_query: str,
    chat_history: list[dict[str, Any]],
    enabled: bool,
    max_turns: int,
    always_include: bool = False,
) -> tuple[str, dict[str, Any]]:
    clean_query = _normalize_whitespace(raw_query)
    meta = {
        "enabled": bool(enabled),
        "applied": False,
        "always_include": bool(always_include),
        "available_turns": 0,
        "turns_used": 0,
        "followup_detected": False,
    }
    if not enabled:
        return clean_query, meta

    turns = _recent_conversation_turns(chat_history, limit=max_turns)
    meta["available_turns"] = len(turns)
    if not turns:
        return clean_query, meta

    followup = _query_looks_followup(clean_query)
    meta["followup_detected"] = followup
    if not always_include and not followup:
        return clean_query, meta

    blocks: list[str] = []
    for idx, (user_text, assistant_summary) in enumerate(turns, start=1):
        blocks.append(f"Turn {idx} user: {user_text}")
        if assistant_summary:
            blocks.append(f"Turn {idx} assistant: {assistant_summary}")
        else:
            blocks.append(f"Turn {idx} assistant: [summary unavailable]")
    context = "\n".join(blocks)
    max_context_chars = 3200
    if len(context) > max_context_chars:
        context = context[-max_context_chars:]
        newline_idx = context.find("\n")
        if newline_idx >= 0:
            context = context[newline_idx + 1 :]

    contextual_query = (
        "Conversation context from prior turns:\n"
        f"{context}\n\n"
        "Current user question:\n"
        f"{clean_query}\n\n"
        "Instruction: Use prior context only when needed to resolve references "
        "or combine earlier answers."
    )
    meta["applied"] = True
    meta["turns_used"] = len(turns)
    meta["query_chars"] = len(contextual_query)
    return contextual_query, meta


def build_followup_suggestions(response: Any, user_query: str) -> list[str]:
    suggestions: list[str] = []
    final_answer = getattr(response, "final_answer", None)
    verdict = str(getattr(response, "verdict", "") or "").strip().lower()

    if final_answer is not None:
        clarifications = list(getattr(final_answer, "clarifications_needed", []) or [])
        for item in clarifications[:2]:
            text = _normalize_whitespace(str(item))
            if not text:
                continue
            if not text.endswith("?"):
                text = f"{text}?"
            suggestions.append(text)

    # Generate contextual defaults based on verdict and answer content.
    if _query_looks_followup(user_query):
        defaults = [
            "Can you combine this with the previous answer into one plan?",
            "What changed from the previous answer?",
            "What should I do next, in order?",
        ]
    elif verdict == "non_compliant":
        defaults = [
            "What are the penalties if I proceed without compliance?",
            "What steps do I need to take to become compliant?",
            "Are there any exemptions or exceptions that could apply?",
        ]
    elif verdict == "compliant":
        defaults = [
            "What documents do I need to prepare for the permit application?",
            "Can you turn this into a step-by-step checklist?",
            "What are the conditions I must follow after approval?",
        ]
    elif verdict == "insufficient_evidence":
        defaults = [
            "Can you try with a more specific question?",
            "What information should I provide for a better answer?",
            "Can you check the rules for a related topic instead?",
        ]
    else:
        # "depends" / informational queries
        has_conditions = final_answer and bool(getattr(final_answer, "conditions_and_exceptions", []))
        has_risks = final_answer and bool(getattr(final_answer, "risk_flags", []))
        defaults = []
        if has_conditions:
            defaults.append("Which of these conditions apply to my specific case?")
        if has_risks:
            defaults.append("What are the penalties for non-compliance with these rules?")
        defaults.extend([
            "Can you turn this into a step-by-step checklist?",
            "What documents should I prepare first?",
        ])

    for item in defaults:
        if len(suggestions) >= 3:
            break
        suggestions.append(item)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in suggestions:
        normalized = _sanitize_followup_suggestion(item)
        key = normalized.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized.strip())
    return deduped[:3]


def _latest_assistant_response(chat_history: list[dict[str, Any]]) -> Any | None:
    for message in reversed(chat_history):
        if str(message.get("role", "")) != "assistant":
            continue
        if candidate := _coerce_assistant_response(message.get("response")):
            return candidate
    return None


def _select_chat_agent_provider(base_llm_provider: Any) -> Any:
    if base_llm_provider is None:
        return None
    provider_id = str(getattr(base_llm_provider, "provider_id", ""))
    if provider_id != "openai_responses_llm":
        return base_llm_provider
    desired_model = os.getenv("PLOTMAGIC_CHAT_AGENT_MODEL", "gpt-5.2").strip()
    if not desired_model:
        return base_llm_provider
    current_model = str(getattr(base_llm_provider, "model", "") or "").strip()
    if current_model == desired_model:
        return base_llm_provider
    settings = getattr(base_llm_provider, "settings", None)
    if settings is None:
        return base_llm_provider
    try:
        from src.providers.adapters.openai_responses_llm import OpenAIResponsesLLMProvider
        from src.providers.config import ProviderSettings

        upgraded_settings = ProviderSettings(
            provider_id=settings.provider_id,
            model=desired_model,
            api_key_env=settings.api_key_env,
            api_key=settings.api_key,
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
            retry_backoff_s=settings.retry_backoff_s,
            dim=settings.dim,
            top_n=settings.top_n,
            enabled=settings.enabled,
            extras=dict(settings.extras),
        )
        return OpenAIResponsesLLMProvider(upgraded_settings)
    except Exception:
        return base_llm_provider


def _openai_router_function_calls(response: Any, *, function_name: str) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return calls
    for item in output:
        if str(getattr(item, "type", "")).strip() != "function_call":
            continue
        if str(getattr(item, "name", "")).strip() != function_name:
            continue
        call_id = str(getattr(item, "call_id", "") or "").strip()
        arguments_raw = str(getattr(item, "arguments", "") or "").strip()
        rewritten_query = ""
        if arguments_raw:
            try:
                parsed = json.loads(arguments_raw)
                if isinstance(parsed, dict):
                    rewritten_query = _normalize_whitespace(
                        str(parsed.get("query", "") or parsed.get("rewritten_query", ""))
                    )
            except Exception:
                rewritten_query = ""
        calls.append(
            {
                "call_id": call_id,
                "rewritten_query": rewritten_query,
            }
        )
    return calls


def _extract_openai_router_payload(response: Any, llm_provider: Any) -> dict[str, Any] | None:
    extractor = getattr(llm_provider, "_extract_text", None)
    extracted: Any = None
    if callable(extractor):
        try:
            extracted = extractor(response)
        except Exception:
            extracted = None
    if extracted is None:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            extracted = output_text
    if isinstance(extracted, dict):
        return extracted
    if isinstance(extracted, str):
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            raw = extracted
            brace_start = raw.find("{")
            brace_end = raw.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                candidate = raw[brace_start : brace_end + 1]
                try:
                    parsed_candidate = json.loads(candidate)
                    if isinstance(parsed_candidate, dict):
                        return parsed_candidate
                except Exception:
                    pass
            action_match = re.search(r'"action"\s*:\s*"([^"]+)"', raw)
            style_match = re.search(r'"response_style"\s*:\s*"([^"]+)"', raw)
            confidence_match = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
            if action_match or style_match:
                action = _normalize_whitespace(action_match.group(1) if action_match else "use_retrieval")
                style = _normalize_whitespace(style_match.group(1) if style_match else "normal")
                try:
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.6
                except Exception:
                    confidence = 0.6
                return {
                    "action": action,
                    "response_style": style,
                    "rewritten_query": "",
                    "context_short_summary": "",
                    "reason": "router_partial_json_recovered",
                    "confidence": confidence,
                }
            return None
    return None


def _build_router_decision_via_openai_tool(
    *,
    query: str,
    turns: list[dict[str, str]],
    latest_snapshot: dict[str, Any],
    llm_provider: Any,
    decision_schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool, str]:
    provider_id = str(getattr(llm_provider, "provider_id", "")).strip()
    client = getattr(llm_provider, "_client", None)
    model = str(getattr(llm_provider, "model", "")).strip()
    if provider_id != "openai_responses_llm" or client is None or not model:
        return None, False, query
    responses_api = getattr(client, "responses", None)
    create_fn = getattr(responses_api, "create", None)
    if not callable(create_fn):
        return None, False, query

    normalize_schema = getattr(llm_provider, "_normalize_for_openai_strict_schema", None)
    normalized_schema = decision_schema
    if callable(normalize_schema):
        try:
            normalized_schema = normalize_schema(decision_schema)
        except Exception:
            normalized_schema = decision_schema

    timeout_s = float(getattr(llm_provider, "timeout_s", 20.0) or 20.0)
    router_instructions = (
        "You are the chat agent router for a legal compliance assistant.\n"
        "Use tool `retrieval_query` only when the user needs fresh legal lookup: new facts, changed constraints, "
        "new location/jurisdiction/category, new rule topic, comparisons, or uncertainty.\n"
        "Do not call the tool when the user only asks to transform the previous answer format/style "
        "(summary/checklist/bullets/ELI5/show_retrieved).\n"
        "Transformation-only examples that MUST be `respond_from_context` and MUST NOT call the tool:\n"
        "- 'eli5 the above answer'\n"
        "- 'summarize this in bullet points'\n"
        "- 'make it a checklist'\n"
        "- 'show me all that you have retrieved'\n"
        "Fresh-lookup examples that MUST call `retrieval_query`:\n"
        "- 'now compare this with municipality requirements'\n"
        "- 'what are the penalties for unauthorized construction?'\n"
        "If you call `retrieval_query`, pass the best standalone rewritten query with explicit context.\n"
        "Return only strict JSON with keys: "
        "action, response_style, rewritten_query, context_short_summary, reason, confidence.\n"
        "When `respond_from_context`, include the user-ready answer in `context_short_summary`.\n"
        "When `use_retrieval`, set `context_short_summary` to an empty string.\n"
        "Keep `context_short_summary` concise: summary/eli5 <= 90 words, checklist/bullets <= 6 short lines, "
        "show_retrieved <= 20 words, plain text only (no markdown headings).\n"
        "Allowed `action`: use_retrieval, respond_from_context.\n"
        "Allowed `response_style`: normal, summary, checklist, bullets, eli5, show_retrieved."
    )
    user_payload = {
        "query": query,
        "conversation_turns": turns,
        "latest_answer_snapshot": latest_snapshot,
    }
    tools = [
        {
            "type": "function",
            "name": "retrieval_query",
            "description": "Request fresh retrieval by providing a standalone legal query for the retrieval pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Standalone retrieval query with relevant context.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    rewritten_from_tool = query
    request_args: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": router_instructions,
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tool_calls": 1,
        "temperature": 0.0,
        "max_output_tokens": 420,
        "timeout": timeout_s,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "conversation_router_decision",
                "schema": normalized_schema,
                "strict": True,
            }
        },
    }
    response = create_fn(**request_args)
    function_calls = _openai_router_function_calls(response, function_name="retrieval_query")
    if function_calls:
        first = function_calls[0]
        rewritten = _normalize_whitespace(first.get("rewritten_query", "")) or query
        rewritten_from_tool = rewritten
        return (
            {
                "action": "use_retrieval",
                "response_style": "normal",
                "rewritten_query": rewritten_from_tool,
                "reason": "retrieval_tool_called",
                "confidence": 0.95,
                "context_short_summary": "",
            },
            True,
            rewritten_from_tool,
        )

    payload = _extract_openai_router_payload(response, llm_provider)
    if isinstance(payload, dict):
        return payload, False, rewritten_from_tool
    return None, False, query


def build_conversation_transform_response(
    *,
    query: str,
    chat_history: list[dict[str, Any]],
    llm_provider: Any,
) -> tuple[AnswerResponse | None, str | None, dict[str, Any]]:
    previous = _latest_assistant_response(chat_history)
    default_decision = {
        "action": "use_retrieval",
        "response_style": "normal",
        "rewritten_query": query,
        "reason": "no prior assistant answer available",
        "confidence": 0.0,
    }
    if previous is None or previous.final_answer is None:
        return None, None, default_decision

    if llm_provider is None:
        return None, None, {**default_decision, "reason": "router_provider_unavailable"}

    turns: list[dict[str, str]] = []
    for user_text, assistant_text in _recent_conversation_turns(chat_history, limit=6):
        turns.append({"user": user_text, "assistant": assistant_text})

    latest = previous.final_answer
    latest_snapshot = {
        "verdict": latest.verdict,
        "short_summary": latest.short_summary,
        "required_actions": [item.text for item in latest.required_actions[:6]],
        "conditions_and_exceptions": [item.text for item in latest.conditions_and_exceptions[:6]],
        "risk_flags": latest.risk_flags[:6],
    }

    decision_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["use_retrieval", "respond_from_context"]},
            "response_style": {
                "type": "string",
                "enum": ["normal", "summary", "checklist", "bullets", "eli5", "show_retrieved"],
            },
            "rewritten_query": {"type": "string"},
            "context_short_summary": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["action", "response_style", "rewritten_query", "context_short_summary", "reason", "confidence"],
        "additionalProperties": False,
    }
    router_instructions = (
        "You are the conversation controller for a legal compliance assistant.\n"
        "Available tool: retrieval_query(query).\n"
        "Decide whether to call retrieval_query or answer from existing conversation context.\n"
        "Use `respond_from_context` only when the user is asking for transformation of prior content "
        "(summary, checklist, bullets, ELI5, or show previously retrieved evidence) and no new legal lookup is needed.\n"
        "Transformation-only examples that MUST be `respond_from_context`: "
        "'eli5 the above answer', 'summarize this', 'make it checklist', 'show me all that you have retrieved'.\n"
        "Use `use_retrieval` for new facts, changed constraints, new jurisdiction/location/category, "
        "new legal topic, or any uncertainty that prior context is sufficient.\n"
        "When `use_retrieval`, write `rewritten_query` as the best standalone retrieval query with explicit context.\n"
        "When `respond_from_context`, keep `rewritten_query` equal to the user query.\n"
        "When `respond_from_context`, provide the full user-ready answer in `context_short_summary`.\n"
        "When `use_retrieval`, keep `context_short_summary` empty.\n"
        "Keep `context_short_summary` concise: summary/eli5 <= 90 words, checklist/bullets <= 6 short lines, "
        "show_retrieved <= 20 words, plain text only (no markdown headings).\n"
        "Set `response_style` to one of: normal, summary, checklist, bullets, eli5, show_retrieved.\n"
        "Provide a short `reason` and calibrated `confidence` in [0, 1].\n"
        "Return strict JSON only."
    )
    raw_decision: dict[str, Any] | None = None
    tool_invoked = False
    tool_rewritten_query = query
    try:
        raw_decision, tool_invoked, tool_rewritten_query = _build_router_decision_via_openai_tool(
            query=query,
            turns=turns,
            latest_snapshot=latest_snapshot,
            llm_provider=llm_provider,
            decision_schema=decision_schema,
        )
    except Exception:
        raw_decision = None
        tool_invoked = False
        tool_rewritten_query = query

    if raw_decision is None:
        try:
            raw_decision = llm_provider.generate_structured(
                task="conversation_turn_router",
                payload={
                    "instructions": router_instructions,
                    "query": query,
                    "conversation_turns": turns,
                    "latest_answer_snapshot": latest_snapshot,
                },
                json_schema=decision_schema,
                temperature=0.0,
                max_output_tokens=320,
            )
        except Exception:
            return None, None, {**default_decision, "reason": "router_llm_unavailable"}

    action = str((raw_decision or {}).get("action", "use_retrieval")).strip().lower()
    if action not in {"use_retrieval", "respond_from_context"}:
        action = "use_retrieval"
    style = str((raw_decision or {}).get("response_style", "normal")).strip().lower()
    if style not in {"normal", "summary", "checklist", "bullets", "eli5", "show_retrieved"}:
        style = "normal"
    rewritten_query = _normalize_whitespace(str((raw_decision or {}).get("rewritten_query", query) or query))
    context_short_summary = _normalize_whitespace(str((raw_decision or {}).get("context_short_summary", "") or ""))
    if action == "use_retrieval" and tool_invoked and not rewritten_query:
        rewritten_query = _normalize_whitespace(tool_rewritten_query) or query
    reason = _normalize_whitespace(str((raw_decision or {}).get("reason", ""))) or "router_decision"
    confidence_raw = (raw_decision or {}).get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except Exception:
        confidence = 0.0
    decision = {
        "action": action,
        "response_style": style,
        "rewritten_query": rewritten_query or query,
        "context_short_summary": context_short_summary,
        "reason": reason,
        "confidence": confidence,
        "tool_invoked": bool(tool_invoked and action == "use_retrieval"),
    }
    if transform_override := _explicit_transform_style(query):
        if decision["action"] == "use_retrieval":
            decision["action"] = "respond_from_context"
            decision["response_style"] = transform_override
            decision["tool_invoked"] = False
            decision["reason"] = f"{decision['reason']} | transform_followup_override"
    transform_styles = {"summary", "checklist", "bullets", "eli5", "show_retrieved"}
    if decision["action"] == "use_retrieval" and decision["response_style"] in transform_styles and not bool(decision["tool_invoked"]):
        decision["action"] = "respond_from_context"
        decision["reason"] = f"{decision['reason']} | normalized_to_context_style"
    action = str(decision["action"])
    style = str(decision["response_style"])
    if action != "respond_from_context":
        return None, None, decision

    rewrite_schema = {
        "type": "object",
        "properties": {
            "short_summary": {"type": "string"},
        },
        "required": ["short_summary"],
        "additionalProperties": False,
    }
    rewrite_instructions = (
        "You are rewriting the assistant's prior answer from provided conversation context only.\n"
        "Do not introduce new legal facts, new citations, or new conclusions.\n"
        "Apply the requested response_style faithfully:\n"
        "- summary: concise plain summary, max 120 words.\n"
        "- checklist: 4-7 short numbered steps.\n"
        "- bullets: 4-7 short bullet points.\n"
        "- eli5: very simple language, max 120 words.\n"
        "- show_retrieved: one sentence intro only, max 25 words.\n"
        "Be concise and avoid repeating near-identical lines.\n"
        "Return strict JSON only."
    )
    rewrite_max_output_tokens = 260
    if style in {"summary", "eli5", "show_retrieved"}:
        rewrite_max_output_tokens = 180
    style_source_parts: list[str] = []
    if transformed_source := _normalize_whitespace(latest.short_summary):
        style_source_parts.append(transformed_source)
    style_source_parts.extend(
        _normalize_whitespace(str(item.text))
        for item in (latest.required_actions or [])[:4]
        if _normalize_whitespace(str(item.text))
    )
    style_source_parts.extend(
        _normalize_whitespace(str(item.text))
        for item in (latest.conditions_and_exceptions or [])[:2]
        if _normalize_whitespace(str(item.text))
    )
    style_source_text = " ".join(style_source_parts).strip() or latest.short_summary
    if context_short_summary:
        start = time.perf_counter()
        transformed_summary = _coerce_transform_summary(style, context_short_summary, fallback=latest.short_summary)
        if not transformed_summary:
            transformed_summary = latest.short_summary
        transformed = previous.model_copy(deep=True)
        transformed.final_answer = ComplianceBriefPayload(
            verdict=latest.verdict,
            short_summary=transformed_summary,
            applicable_rules=latest.applicable_rules,
            conditions_and_exceptions=latest.conditions_and_exceptions,
            required_actions=latest.required_actions,
            risk_flags=latest.risk_flags,
            clarifications_needed=[],
        )
        transformed.verdict = latest.verdict
        transformed.agent_trace = []
        transformed.latency_ms = {
            "conversation_transform_ms": (time.perf_counter() - start) * 1000,
            "llm_used": 1.0,
            "conversation_transform_strategy": 1.0,
        }
        return transformed, style, decision
    try:
        rewrite = llm_provider.generate_structured(
            task="conversation_context_rewrite",
            payload={
                "instructions": rewrite_instructions,
                "query": query,
                "response_style": style,
                "conversation_turns": turns,
                "latest_answer_snapshot": latest_snapshot,
            },
            json_schema=rewrite_schema,
            temperature=0.0,
            max_output_tokens=rewrite_max_output_tokens,
        )
    except Exception:
        start = time.perf_counter()
        fallback_summary = _coerce_transform_summary(style, style_source_text, fallback=latest.short_summary)
        fallback = previous.model_copy(deep=True)
        fallback.final_answer = ComplianceBriefPayload(
            verdict=latest.verdict,
            short_summary=fallback_summary or latest.short_summary,
            applicable_rules=latest.applicable_rules,
            conditions_and_exceptions=latest.conditions_and_exceptions,
            required_actions=latest.required_actions,
            risk_flags=latest.risk_flags,
            clarifications_needed=[],
        )
        fallback.verdict = latest.verdict
        fallback.agent_trace = []
        fallback.latency_ms = {
            "conversation_transform_ms": (time.perf_counter() - start) * 1000,
            "llm_used": 1.0,
        }
        return fallback, style, {**decision, "action": "respond_from_context", "reason": "context_rewrite_llm_unavailable_fallback"}

    transformed_summary_raw = _normalize_whitespace(str((rewrite or {}).get("short_summary", "")))
    transform_input = transformed_summary_raw if transformed_summary_raw else style_source_text
    transformed_summary = _coerce_transform_summary(style, transform_input, fallback=latest.short_summary)
    if not transformed_summary:
        transformed_summary = latest.short_summary

    start = time.perf_counter()
    transformed = previous.model_copy(deep=True)
    transformed.final_answer = ComplianceBriefPayload(
        verdict=latest.verdict,
        short_summary=transformed_summary,
        applicable_rules=latest.applicable_rules,
        conditions_and_exceptions=latest.conditions_and_exceptions,
        required_actions=latest.required_actions,
        risk_flags=latest.risk_flags,
        clarifications_needed=[],
    )
    transformed.verdict = latest.verdict
    transformed.agent_trace = []
    transformed.latency_ms = {
        "conversation_transform_ms": (time.perf_counter() - start) * 1000,
        "llm_used": 1.0,
    }
    return transformed, style, decision


def run_query_with_events(
    engine: ComplianceEngine,
    request: QueryRequest,
    show_live_events: bool = False,
) -> tuple[Any, list[dict[str, Any]], str | None]:
    event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    events: list[dict[str, Any]] = []
    result_holder: dict[str, Any] = {"response": None, "error": None}

    start = time.perf_counter()

    def sink(event: dict[str, Any]) -> None:
        event_queue.put(
            {
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                **event,
            }
        )

    def worker() -> None:
        try:
            result_holder["response"] = engine.query(request=request, event_sink=sink)
        except Exception as exc:  # pragma: no cover - runtime surface
            result_holder["error"] = str(exc)
        finally:
            event_queue.put({"elapsed_ms": round((time.perf_counter() - start) * 1000, 2), "step": "worker.done", "status": "ok", "details": {}})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    status_box = st.status("Analyzing compliance rules...", expanded=True) if show_live_events else None
    latest_step = st.empty() if show_live_events else None
    timeline = st.empty() if show_live_events else None
    while thread.is_alive() or not event_queue.empty():
        while not event_queue.empty():
            event = event_queue.get()
            events.append(event)
        if show_live_events and events and latest_step is not None:
            last_event = events[-1]
            step_name = str(last_event.get("step", "event")).replace(".", " > ").replace("_", " ").title()
            latest_step.caption(f"\u23f3 {step_name}")
            if timeline is not None:
                rendered = [f"**{event['elapsed_ms']}ms**  \n{render_event_line(event)}" for event in events[-12:]]
                timeline.markdown("\n\n".join(rendered))
        time.sleep(0.05)

    thread.join()
    total_ms = events[-1]["elapsed_ms"] if events else 0
    if show_live_events and latest_step is not None and status_box is not None:
        latest_step.empty()
        if total_ms < 1000:
            label = f"Completed in {total_ms:.0f} ms"
        else:
            label = f"Completed in {total_ms / 1000:.1f}s"
        status_box.update(label=label, state="complete", expanded=False)
    return result_holder["response"], events, result_holder["error"]


# ── Verdict rendering ──────────────────────────────────────────────

def _verdict_html(verdict: str) -> str:
    v = verdict.strip().lower()
    label = verdict.upper().replace("_", " ")
    if v in ("compliant", "allowed", "permitted"):
        css = "pm-verdict-compliant"
        icon = "\u2705"
    elif v == "depends":
        css = "pm-verdict-depends"
        icon = "\u2696\ufe0f"
        label = "INFORMATIONAL"
    elif v == "insufficient_evidence":
        css = "pm-verdict-insufficient"
        icon = "\u26a0\ufe0f"
        label = "INSUFFICIENT EVIDENCE"
    else:
        css = "pm-verdict-noncompliant"
        icon = "\u274c"
    return f"<span class='pm-verdict {css}'>{icon} {label}</span>"


def _build_claim_list_html(items: list[Any], show_inline_citations: bool, citation_lookup: dict[str, Any]) -> str:
    if not items:
        return "<p style='color:#94a3b8;font-size:0.85rem;margin:0'>None identified.</p>"
    lines: list[str] = []
    for item in items:
        text_html = html.escape(item.text)
        if show_inline_citations and item.citation_ids:
            links_html = render_inline_citation_links_html(item.citation_ids, citation_lookup)
            if links_html:
                text_html = f"{text_html} {links_html}"
        ref_html = ""
        if item.citation_ids and not show_inline_citations:
            refs = ", ".join(f"<code>{html.escape(c)}</code>" for c in item.citation_ids)
            ref_html = f"<br><span style='color:#94a3b8;font-size:0.78rem;padding-left:1rem'>Refs: {refs}</span>"
        lines.append(f"<li style='margin-bottom:0.35rem'>{text_html}{ref_html}</li>")
    return "<ul style='margin:0;padding-left:1.2rem'>" + "".join(lines) + "</ul>"


def render_final_answer(response: Any, llm_provider_id: str) -> None:
    if response.final_answer is None:
        st.warning("No structured final answer payload returned.")
        return

    verdict = response.final_answer.verdict
    st.markdown(_verdict_html(verdict), unsafe_allow_html=True)
    st.write("")

    summary = response.final_answer.short_summary.strip()
    if summary:
        st.markdown(summary)
    else:
        st.caption("_No summary returned._")

    citation_lookup = build_citation_lookup(response)
    show_inline_citations = llm_provider_id == "openai_responses_llm"

    def render_claim_section(icon: str, title: str, items: list[Any]) -> None:
        body = _build_claim_list_html(items, show_inline_citations, citation_lookup)
        st.markdown(
            f"<div class='pm-section-card'><h4>{icon} {title}</h4>{body}</div>",
            unsafe_allow_html=True,
        )

    render_claim_section("\U0001f4dc", "Applicable Rules", response.final_answer.applicable_rules)
    render_claim_section("\u2696\ufe0f", "Conditions & Exceptions", response.final_answer.conditions_and_exceptions)
    render_claim_section("\u2705", "Required Actions", response.final_answer.required_actions)

    # Risk flags as pills -- single HTML block
    if response.final_answer.risk_flags:
        pills_html = "".join(
            f"<span class='pm-risk-pill'>{risk}</span>" for risk in response.final_answer.risk_flags
        )
        st.markdown(
            f"<div class='pm-section-card'><h4>\u26a0\ufe0f Risk Flags</h4>{pills_html}</div>",
            unsafe_allow_html=True,
        )

    # Clarifications -- single HTML block
    if response.final_answer.clarifications_needed:
        items_html = "".join(f"<li>{q}</li>" for q in response.final_answer.clarifications_needed)
        st.markdown(
            f"<div class='pm-section-card'><h4>\u2753 Clarifications Needed</h4>"
            f"<ul style='margin:0;padding-left:1.2rem'>{items_html}</ul></div>",
            unsafe_allow_html=True,
        )


def _final_answer_citation_ids(final_answer: Any) -> list[str]:
    ids: list[str] = []
    if final_answer is None:
        return ids
    sections = [
        getattr(final_answer, "applicable_rules", []) or [],
        getattr(final_answer, "conditions_and_exceptions", []) or [],
        getattr(final_answer, "required_actions", []) or [],
    ]
    for section in sections:
        for item in section:
            citation_ids = getattr(item, "citation_ids", []) or []
            for citation_id in citation_ids:
                citation_text = str(citation_id).strip()
                if citation_text:
                    ids.append(citation_text)
    deduped: list[str] = []
    seen: set[str] = set()
    for citation_id in ids:
        if citation_id in seen:
            continue
        seen.add(citation_id)
        deduped.append(citation_id)
    return deduped


def render_llm_chat_answer(
    response: Any,
    llm_provider_id: str,
    *,
    show_analysis_details: bool = False,
    show_refine_prompts: bool = False,
) -> None:
    if response.final_answer is None:
        st.warning("No structured final answer payload returned.")
        return

    summary = str(response.final_answer.short_summary or "").strip()
    if summary:
        st.markdown(summary)
    else:
        st.caption("_No summary returned._")

    citation_lookup = build_citation_lookup(response)
    citation_ids = _final_answer_citation_ids(response.final_answer)
    if citation_ids:
        with st.expander("Sources", expanded=False):
            for citation_id in citation_ids[:8]:
                citation = citation_lookup.get(citation_id)
                if citation is None:
                    st.markdown(f"- `{citation_id}`")
                    continue
                label = citation.display_citation or citation_id
                if citation.source_url:
                    st.markdown(f"- [{label}]({citation.source_url})")
                else:
                    st.markdown(f"- `{label}`")

        with st.expander("Cited passages", expanded=False):
            for citation_id in citation_ids[:6]:
                citation = citation_lookup.get(citation_id)
                if citation is None:
                    continue
                quote = (citation.quote_excerpt or "").strip()
                snippet = quote[:420] + ("..." if len(quote) > 420 else "")
                link_html = ""
                if citation.source_url:
                    link_html = f" &middot; <a href='{citation.source_url}' style='color:var(--pm-accent)'>Open source</a>"
                st.markdown(
                    f"""<div class='pm-citation-card'>
                        <strong>{citation.display_citation}</strong>{link_html}
                        <br><span style='color:#475569;font-size:0.84rem'>{html.escape(snippet)}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    clarifications = list(getattr(response.final_answer, "clarifications_needed", []) or [])
    if clarifications and show_refine_prompts:
        with st.expander("Clarify for higher precision", expanded=False):
            st.markdown("\n".join(f"- {item}" for item in clarifications[:4]))

    if show_analysis_details:
        with st.expander("Analysis details", expanded=False):
            render_final_answer(response, llm_provider_id)


def _parse_clarification_item(item: Any) -> tuple[str, list[str]]:
    if hasattr(item, "question"):
        question = str(getattr(item, "question", "") or "").strip()
        options = [str(opt).strip() for opt in (getattr(item, "options", []) or []) if str(opt).strip()]
        return question, options
    if isinstance(item, dict):
        question = str(item.get("question", "") or "").strip()
        options = [str(opt).strip() for opt in (item.get("options", []) or []) if str(opt).strip()]
        return question, options
    return "", []


def render_clarification_turn(response: Any, *, message_key: str = "") -> None:
    clarifications = list(getattr(response, "clarifications", []) or [])
    if not clarifications:
        st.warning("No answer payload was returned for this turn.")
        return

    # Find the original user question so clarification answers include context.
    original_question = ""
    for msg in reversed(st.session_state.get("chat_history", [])):
        if str(msg.get("role", "")) == "user":
            original_question = _normalize_whitespace(str(msg.get("content", "")))
            break

    st.markdown("I need one clarification before I can answer accurately:")
    for idx, item in enumerate(clarifications[:3]):
        question, options = _parse_clarification_item(item)
        if question:
            st.markdown(f"**{question}**")
        else:
            st.markdown("**Please clarify your request.**")

        if options:
            cols = st.columns(min(len(options), 3))
            for opt_idx, option in enumerate(options[:6]):
                key = f"clarify_{message_key}_{idx}_{opt_idx}"
                col = cols[opt_idx % len(cols)]
                if col.button(option, key=key, use_container_width=True):
                    # Include original question context so the clarification
                    # answer doesn't lose the user's intent.
                    if original_question:
                        st.session_state.pending_query = f"{original_question} ({option})"
                    else:
                        st.session_state.pending_query = option
                    st.rerun()

    st.caption("You can click an option above or type your clarification in chat.")


# ── Grounding confidence ───────────────────────────────────────────

def render_grounding_summary(response: Any) -> None:
    if not response.grounding:
        st.caption("No grounding data available.")
        return

    g = response.grounding
    ratio = getattr(g, "support_ratio", None)
    supported = getattr(g, "supported_claim_count", 0)
    unsupported = getattr(g, "unsupported_claim_count", 0)
    conflicting = getattr(g, "conflicting_claim_count", 0)
    total = supported + unsupported + conflicting

    if ratio is not None:
        pct = round(ratio * 100)
        if pct >= 80:
            color = "#16a34a"
        elif pct >= 50:
            color = "#d97706"
        else:
            color = "#dc2626"
        st.markdown(
            f"""
            <div style="margin-bottom:0.75rem">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#64748b;margin-bottom:0.25rem">
                    <span>Grounding Confidence</span><span style="font-weight:700;color:{color}">{pct}%</span>
                </div>
                <div class="pm-meter-track">
                    <div class="pm-meter-fill" style="width:{pct}%;background:{color}"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if total:
        col1, col2, col3 = st.columns(3)
        col1.metric("Supported", supported)
        col2.metric("Unsupported", unsupported)
        col3.metric("Conflicting", conflicting)


# ── Latency visualization ─────────────────────────────────────────

def render_latency(response: Any) -> None:
    latency = response.latency_ms or {}
    if not latency:
        st.caption("No latency data.")
        return

    total = latency.get("total_ms", max(latency.values())) if latency else 1
    if total == 0:
        total = 1

    for key, ms in sorted(latency.items(), key=lambda kv: -kv[1]):
        pct = min(round(ms / total * 100), 100)
        label = key.replace("_ms", "").replace("_", " ").title()
        st.markdown(
            f"""<div class="pm-latency-bar">
                <span style="min-width:130px;color:#475569">{label}</span>
                <div style="flex:1;background:#e2e8f0;border-radius:999px;height:6px">
                    <div class="pm-latency-bar-fill" style="width:{pct}%"></div>
                </div>
                <span style="min-width:70px;text-align:right;font-weight:600;color:#334155">{ms:.0f}ms</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Retrieved chunks ──────────────────────────────────────────────

def _primary_retrieval_score(scores: dict[str, Any]) -> tuple[str, float]:
    ordered_keys = [
        "rrf_score",
        "retrieval_score",
        "query_relevance",
        "rerank_score",
        "vector_score",
        "lexical_score",
    ]
    for key in ordered_keys:
        value = scores.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    numeric_items = [(str(k), float(v)) for k, v in scores.items() if isinstance(v, (int, float))]
    if numeric_items:
        return max(numeric_items, key=lambda item: item[1])
    return "score", 0.0


def _evidence_sort_key(item: Any) -> tuple[float, str]:
    scores = getattr(item, "scores", {}) or {}
    _score_key, score = _primary_retrieval_score(scores)
    return score, str(getattr(item, "chunk_id", ""))


def _chunk_preview(text: str, limit: int = 1100) -> tuple[str, bool]:
    clean_text = (text or "").strip()
    if len(clean_text) <= limit:
        return clean_text, False
    clipped = clean_text[:limit].rsplit(" ", 1)[0].strip()
    return f"{clipped}...", True


def render_retrieved_chunks(
    response: Any,
    emphasize_ranking: bool = False,
    *,
    max_items: int | None = None,
) -> None:
    if not response.evidence_matrix:
        st.caption("No retrieved chunks available.")
        return

    sorted_items = sorted(response.evidence_matrix, key=_evidence_sort_key, reverse=True)
    items_to_render = sorted_items if not max_items else sorted_items[:max_items]
    if sorted_items:
        top_score_key, top_score_value = _primary_retrieval_score(getattr(sorted_items[0], "scores", {}) or {})
        st.caption(
            f"Retrieved chunks: {len(sorted_items)} total, showing {len(items_to_render)}. "
            f"Sorted by `{top_score_key}` (highest first). Top score: {top_score_value:.3f}"
        )

    if emphasize_ranking and sorted_items:
        st.markdown("**Top Retrieved Items (By Score)**")
        for rank, item in enumerate(sorted_items[:8], start=1):
            scores = getattr(item, "scores", {}) or {}
            score_key, score_value = _primary_retrieval_score(scores)
            st.markdown(
                f"{rank}. `{item.chunk_id}` \u2014 `{score_key}={score_value:.3f}` \u2014 claim `{item.claim_id}`"
            )

    citations_by_claim: dict[str, list[Any]] = {}
    for citation in response.citations:
        citations_by_claim.setdefault(citation.claim_id, []).append(citation)

    for rank, item in enumerate(items_to_render, start=1):
        scores = getattr(item, "scores", {}) or {}
        score_key, score_value = _primary_retrieval_score(scores)
        with st.expander(
            f"\U0001f4c4 #{rank} {item.claim_id} \u2014 {item.chunk_id} \u00b7 {score_key}={score_value:.3f}",
            expanded=False,
        ):
            if scores:
                top_scores = sorted(
                    [(str(k), float(v)) for k, v in scores.items() if isinstance(v, (int, float))],
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:4]
                if top_scores:
                    score_text = " | ".join(f"{k}={v:.3f}" for k, v in top_scores)
                    st.caption(f"Scores: {score_text}")
            preview, truncated = _chunk_preview(str(item.text or ""))
            st.markdown(preview or "_No text_")
            if truncated:
                with st.expander("Show full chunk text", expanded=False):
                    st.markdown(item.text or "_No text_")
            claim_citations = citations_by_claim.get(item.claim_id, [])
            if claim_citations:
                for citation in claim_citations:
                    link = citation.source_url or ""
                    label = citation.display_citation
                    if link:
                        st.markdown(
                            f"<div class='pm-citation-card'><a href='{link}'>{label}</a></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='pm-citation-card'>{label}</div>",
                            unsafe_allow_html=True,
                        )


# ── Citation explorer ─────────────────────────────────────────────

def render_citation_explorer(response: Any) -> None:
    if not response.citations:
        st.caption("No citations returned.")
        return

    for citation in response.citations:
        excerpt = citation.quote_excerpt[:220] if citation.quote_excerpt else ""
        link_html = ""
        if citation.source_url:
            link_html = f" &middot; <a href='{citation.source_url}' style='color:var(--pm-accent)'>Open source</a>"
        st.markdown(
            f"""<div class='pm-citation-card'>
                <strong>{citation.display_citation}</strong>
                <span style='color:var(--pm-muted);font-size:0.78rem'> (claim: {citation.claim_id}){link_html}</span>
                <br><span style='color:#475569;font-size:0.84rem'>{excerpt}</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Thinking / trace panel ────────────────────────────────────────

def _tool_steps_from_events(events: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(event.get("step", ""))
            for event in events
            if str(event.get("step", "")).startswith("tool.")
        }
    )


def render_thinking_panel(response: Any, events: list[dict[str, Any]]) -> None:
    tool_steps = _tool_steps_from_events(events)
    if tool_steps:
        st.markdown("**Tools Used**")
        cols = st.columns(min(len(tool_steps), 4))
        for i, step in enumerate(tool_steps):
            cols[i % len(cols)].code(step.replace("tool.", ""), language=None)

    if response.agent_trace:
        st.markdown("**Agent Trace**")
        for step in response.agent_trace:
            with st.expander(f"{step.step} \u2014 {step.status}", expanded=False):
                st.json(step.details)

    with st.expander("Pipeline Events", expanded=False):
        if not events:
            st.caption("No events recorded.")
        else:
            rendered = [f"**{event['elapsed_ms']}ms**  \n{render_event_line(event)}" for event in events]
            st.markdown("\n\n".join(rendered))


def render_diagnostics_tabs(
    response: Any,
    events: list[dict[str, Any]],
    *,
    retrieval_primary: bool = False,
) -> None:
    st.write("")
    if retrieval_primary:
        st.caption(
            "Debug diagnostics for retrieval mode. Ranked evidence is shown above; these tabs expose internals."
        )
        tabs = st.tabs(["\U0001f4d1 Citations", "\U0001f4ca Grounding", "\u23f1\ufe0f Latency", "\U0001f9e0 Trace", "\U0001f4e6 Raw"])
        with tabs[0]:
            render_citation_explorer(response)
        with tabs[1]:
            render_grounding_summary(response)
        with tabs[2]:
            render_latency(response)
        with tabs[3]:
            render_thinking_panel(response, events)
        with tabs[4]:
            st.json(response.model_dump())
            with st.expander("Events JSON", expanded=False):
                st.json(events)
    else:
        st.caption(
            "Debug diagnostics: inspect citations, evidence ranking, grounding, latency, trace, and raw payload."
        )
        tabs = st.tabs(["\U0001f4d1 Citations", "\U0001f50d Evidence", "\U0001f4ca Grounding", "\u23f1\ufe0f Latency", "\U0001f9e0 Trace", "\U0001f4e6 Raw"])
        with tabs[0]:
            render_citation_explorer(response)
        with tabs[1]:
            llm_used_flag = float((response.latency_ms or {}).get("llm_used", 0.0))
            render_retrieved_chunks(response, emphasize_ranking=llm_used_flag < 0.5)
        with tabs[2]:
            render_grounding_summary(response)
        with tabs[3]:
            render_latency(response)
        with tabs[4]:
            render_thinking_panel(response, events)
        with tabs[5]:
            st.json(response.model_dump())
            with st.expander("Events JSON", expanded=False):
                st.json(events)


# ── Assistant payload (full response card) ─────────────────────────

def render_assistant_payload(
    payload: dict[str, Any],
    *,
    show_diagnostics: bool = False,
    show_suggestions: bool = False,
    message_key: str = "",
) -> None:
    requested_provider = str(payload.get("provider", "no_llm"))
    provider = str(payload.get("provider_actual", requested_provider))
    response = payload.get("response")
    events = payload.get("events", [])
    error = payload.get("error")
    conversation_transform = bool(payload.get("conversation_transform", False))
    conversation_transform_kind = str(payload.get("conversation_transform_kind", "")).strip()
    conversation_router = payload.get("conversation_router") or {}

    if provider == "no_llm":
        st.markdown("<span class='pm-provider-chip'>Retrieval Only</span>", unsafe_allow_html=True)
    if requested_provider != provider:
        st.warning(
            f"Requested mode `{requested_provider}` is unavailable; using `{provider}`. "
            "Check provider/API-key configuration."
        )

    if error:
        st.error(f"Query failed: {error}")
        return
    if response is None:
        st.error("No response produced.")
        return
    if getattr(response, "final_answer", None) is None:
        render_clarification_turn(response, message_key=message_key)
        return

    if conversation_transform:
        kind_label = conversation_transform_kind.replace("_", " ") if conversation_transform_kind else "formatting"
        st.caption(f"Conversation follow-up ({kind_label}): reused previous answer context; no new retrieval run.")

    top_k_selected = payload.get("top_k_selected")
    top_k_reason = str(payload.get("top_k_reason", "")).strip()
    llm_used_flag = float((response.latency_ms or {}).get("llm_used", 0.0))
    llm_runtime_label = "LLM synthesis: ON" if llm_used_flag >= 0.5 else "LLM synthesis: OFF (deterministic)"

    conversation_meta = payload.get("conversation_meta") or {}
    tool_steps = _tool_steps_from_events(events)
    user_query_payload = _normalize_whitespace(str(payload.get("user_query", "") or ""))

    def render_runtime_details() -> None:
        with st.expander("Runtime details", expanded=False):
            if conversation_transform:
                st.markdown("- Mode: `conversation_transform` (no retrieval/grounding run)")
            if isinstance(conversation_router, dict) and conversation_router:
                action = str(conversation_router.get("action", "")).strip() or "use_retrieval"
                style = str(conversation_router.get("response_style", "")).strip() or "normal"
                reason = str(conversation_router.get("reason", "")).strip() or "router_decision"
                confidence = conversation_router.get("confidence")
                if isinstance(confidence, (int, float)):
                    st.markdown(f"- Router decision: `{action}` (`style={style}`, `confidence={float(confidence):.2f}`)")
                else:
                    st.markdown(f"- Router decision: `{action}` (`style={style}`)")
                if bool(conversation_router.get("tool_invoked")):
                    st.markdown("- Router tool use: `retrieval_query`")
                st.markdown(f"- Router reason: {reason}")
                rewritten_query = _normalize_whitespace(str(conversation_router.get("rewritten_query", "")))
                if action == "use_retrieval" and rewritten_query and rewritten_query != user_query_payload:
                    st.markdown(f"- Router retrieval query: `{rewritten_query}`")
            if isinstance(top_k_selected, int):
                if top_k_reason:
                    st.markdown(f"- Retrieval depth: `top-k={top_k_selected}` (`{top_k_reason}`)")
                else:
                    st.markdown(f"- Retrieval depth: `top-k={top_k_selected}`")
            st.markdown(f"- Runtime: {llm_runtime_label} (`requested={requested_provider}`, `actual={provider}`)")

            if isinstance(conversation_meta, dict) and conversation_meta.get("enabled"):
                if conversation_meta.get("applied"):
                    turns_used = int(conversation_meta.get("turns_used", 0))
                    st.markdown(f"- Conversation memory: ON (`{turns_used}` prior turns applied)")
                else:
                    available_turns = int(conversation_meta.get("available_turns", 0))
                    followup_detected = bool(conversation_meta.get("followup_detected", False))
                    if available_turns == 0:
                        st.markdown("- Conversation memory: ON (no prior assistant turns)")
                    elif not followup_detected and not bool(conversation_meta.get("always_include", False)):
                        st.markdown("- Conversation memory: ON (not applied; no follow-up reference)")
                    else:
                        st.markdown("- Conversation memory: ON (not applied)")

            if tool_steps:
                compact = [step.replace("tool.", "") for step in tool_steps[:8]]
                suffix = ""
                if len(tool_steps) > 8:
                    suffix = f", +{len(tool_steps) - 8} more"
                st.markdown(f"- Agentic tools: `{', '.join(compact)}{suffix}`")

    retrieval_primary = provider == "no_llm"
    if retrieval_primary:
        if isinstance(top_k_selected, int):
            if top_k_reason:
                st.caption(f"Retrieval depth: top-k={top_k_selected} ({top_k_reason})")
            else:
                st.caption(f"Retrieval depth: top-k={top_k_selected}")
        st.caption(f"{llm_runtime_label}  |  Requested: `{requested_provider}`  |  Runtime: `{provider}`")
        st.markdown("**Retrieved Evidence (Sorted by Relevance)**")
        st.caption(
            "Retrieval-only mode shows all retrieved chunks ranked by relevance. "
            "No LLM synthesis is used in this view."
        )
        render_retrieved_chunks(response, emphasize_ranking=True)
        if show_diagnostics:
            with st.expander("Deterministic synthesis (debug)", expanded=False):
                render_final_answer(response, provider)
    else:
        st.markdown("<span class='pm-provider-chip'>LLM + Citations</span>", unsafe_allow_html=True)
        render_llm_chat_answer(
            response,
            provider,
            show_analysis_details=show_diagnostics,
            show_refine_prompts=show_diagnostics,
        )
        if conversation_transform and conversation_transform_kind == "show_retrieved":
            st.markdown("**Retrieved Evidence (From Previous Turn)**")
            render_retrieved_chunks(response, emphasize_ranking=True)
        if llm_used_flag < 0.5 and not conversation_transform:
            st.warning("The AI synthesis step was skipped for this answer (using rule-based summary instead). This may happen due to a temporary API issue.")

        if show_suggestions:
            user_query = str(payload.get("user_query", "") or "")
            suggestions = build_followup_suggestions(response, user_query)
            if suggestions:
                st.caption("Suggested follow-ups")
                cols = st.columns(len(suggestions))
                for idx, suggestion in enumerate(suggestions):
                    key = f"followup_{message_key}_{idx}"
                    if cols[idx].button(suggestion, key=key, use_container_width=True):
                        st.session_state.pending_query = suggestion
                        st.rerun()
            st.caption("Continue in the chat input at the bottom.")

    if show_diagnostics:
        render_runtime_details()
        render_diagnostics_tabs(response, events, retrieval_primary=retrieval_primary)


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="PlotMagic Compliance", page_icon="\U0001f3db\ufe0f", layout="wide")
    inject_chat_styles()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### \U0001f3db\ufe0f PlotMagic")
        st.caption("Building Compliance Assistant")
        st.write("")

        st.markdown("##### Query Settings")
        llm_provider = st.selectbox(
            "MODE",
            options=["no_llm", "openai_responses_llm"],
            index=1,
            format_func=lambda x: "Retrieval Only" if x == "no_llm" else "LLM + Citations",
        )
        state = st.selectbox("STATE", options=["kerala"], index=0)
        jurisdiction = st.selectbox("JURISDICTION", options=["panchayat", "municipality"], index=0)
        location = st.text_input(
            "LOCATION (optional)",
            value="",
            placeholder="e.g. Thrissur, Anthikkad",
            help="Enter a city or local body name to improve scope resolution.",
        )
        category = st.selectbox(
            "PANCHAYAT CATEGORY",
            options=["Category-II", "Category-I"],
            index=0,
            disabled=jurisdiction != "panchayat",
        )

        st.write("")
        st.markdown("##### Experience")
        conversation_memory = st.toggle(
            "Conversation memory (LLM mode)",
            value=True,
            disabled=llm_provider != "openai_responses_llm",
        )
        with st.expander("Advanced controls", expanded=False):
            auto_top_k = st.toggle("Dynamic retrieval depth (auto top-k)", value=True)
            top_k_label = "Minimum retrieval depth (TOP-K floor)" if auto_top_k else "Manual retrieval depth (TOP-K)"
            manual_top_k = st.slider(
                top_k_label,
                min_value=3,
                max_value=30,
                value=12,
                step=1,
                help=(
                    "When dynamic top-k is ON, this acts as a floor (effective top-k = max(dynamic, floor)). "
                    "When OFF, this is the fixed retrieval depth."
                ),
            )
            debug_trace = st.toggle("Debug trace", value=False)
            show_live_events = st.toggle("Live pipeline events", value=False)
            if auto_top_k:
                st.caption("Top-k is selected per query using dynamic complexity heuristics, with your floor applied.")

            conversation_turns = st.slider(
                "Conversation memory turns",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
                disabled=(not conversation_memory) or llm_provider != "openai_responses_llm",
            )
            always_include_memory = st.toggle(
                "Always include memory context",
                value=True,
                disabled=(not conversation_memory) or llm_provider != "openai_responses_llm",
                help=(
                    "When OFF, memory context is only injected for follow-up style questions. "
                    "When ON, context is injected on every LLM-mode query."
                ),
            )
            show_diagnostics_panels = st.toggle(
                "Show diagnostics panels",
                value=False,
                help="Show citations/evidence/grounding/latency/trace tabs under each assistant message.",
            )

        st.write("")
        st.markdown("---")
        clear_chat = st.button("\U0001f5d1\ufe0f  Clear conversation", use_container_width=True)

    # ── Session state ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.pending_query = None
        st.rerun()

    pending_query = st.session_state.pending_query
    st.session_state.pending_query = None

    # ── Empty state with examples ──
    if not st.session_state.chat_history and not pending_query:
        st.write("")
        st.markdown(
            """<div class='pm-empty-state'>
                <h2>\U0001f3db\ufe0f PlotMagic Compliance</h2>
                <p>Ask questions about building rules, permits, setbacks, and regulatory compliance for Kerala local bodies.</p>
            </div>""",
            unsafe_allow_html=True,
        )

        examples = EXAMPLE_QUERIES_MUNICIPALITY if jurisdiction == "municipality" else EXAMPLE_QUERIES_PANCHAYAT
        cols = st.columns(2)
        for i, example in enumerate(examples):
            if cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()
        return

    # ── Chat history ──
    last_assistant_idx = -1
    for idx, message in enumerate(st.session_state.chat_history):
        if str(message.get("role", "")) == "assistant":
            last_assistant_idx = idx

    for idx, message in enumerate(st.session_state.chat_history):
        role = str(message.get("role", "assistant"))
        with st.chat_message(role):
            if role == "user":
                st.markdown(str(message.get("content", "")))
            else:
                render_assistant_payload(
                    message,
                    show_diagnostics=show_diagnostics_panels,
                    show_suggestions=(idx == last_assistant_idx),
                    message_key=f"hist_{idx}",
                )

    # ── Chat input ──
    typed_query = st.chat_input("Ask a compliance question...")
    query = pending_query if pending_query else typed_query
    if not query:
        return
    query_text = query.strip()
    if len(query_text) < 3:
        st.error("Query is too short.")
        return

    contextual_query, conversation_meta = build_contextual_query(
        raw_query=query_text,
        chat_history=st.session_state.chat_history,
        enabled=(llm_provider == "openai_responses_llm" and conversation_memory),
        max_turns=int(conversation_turns),
        always_include=bool(always_include_memory),
    )
    query_for_retrieval = query_text
    transform_response: AnswerResponse | None = None
    transform_kind: str | None = None
    conversation_router: dict[str, Any] = {}

    with st.spinner("Initializing engine..."):
        engine = get_engine(llm_provider)
    llm_provider_obj = getattr(engine, "llm_provider", None)
    provider_actual = str(getattr(llm_provider_obj, "provider_id", llm_provider))

    if llm_provider == "openai_responses_llm" and provider_actual != "openai_responses_llm":
        if auto_top_k:
            dynamic_top_k, dynamic_reason = choose_dynamic_top_k(query_text, jurisdiction, llm_provider)
            floor_top_k = int(manual_top_k)
            selected_top_k = max(dynamic_top_k, floor_top_k)
            top_k_reason = f"{dynamic_reason}; floor->{floor_top_k}" if selected_top_k > dynamic_top_k else dynamic_reason
        else:
            selected_top_k, top_k_reason = int(manual_top_k), "manual override"

        st.session_state.chat_history.append({"role": "user", "content": query_text})
        with st.chat_message("user"):
            st.markdown(query_text)
        with st.chat_message("assistant"):
            payload = {
                "role": "assistant",
                "provider": llm_provider,
                "provider_actual": provider_actual,
                "response": None,
                "events": [],
                "top_k_selected": selected_top_k,
                "top_k_reason": top_k_reason,
                "conversation_meta": conversation_meta,
                "conversation_router": conversation_router,
                "user_query": query_text,
                "error": (
                    "LLM mode is not available — the OpenAI API key is missing or invalid. "
                    "To fix: add your OPENAI_API_KEY to the `.env` file and restart the app. "
                    "You can still use **Retrieval Only** mode from the sidebar."
                ),
            }
            render_assistant_payload(
                payload,
                show_diagnostics=show_diagnostics_panels,
                show_suggestions=True,
                message_key=f"live_{len(st.session_state.chat_history)}",
            )
        st.session_state.chat_history.append(payload)
        return

    if llm_provider == "openai_responses_llm":
        router_provider = _select_chat_agent_provider(llm_provider_obj)
        transform_response, transform_kind, conversation_router = build_conversation_transform_response(
            query=query_text,
            chat_history=st.session_state.chat_history,
            llm_provider=router_provider,
        )
        if transform_response is None and conversation_router.get("action") == "use_retrieval":
            rewritten = _normalize_whitespace(str(conversation_router.get("rewritten_query", "")))
            if rewritten:
                query_for_retrieval = rewritten
                if query_for_retrieval != query_text:
                    contextual_query, conversation_meta = build_contextual_query(
                        raw_query=query_for_retrieval,
                        chat_history=st.session_state.chat_history,
                        enabled=(llm_provider == "openai_responses_llm" and conversation_memory),
                        max_turns=int(conversation_turns),
                        always_include=bool(always_include_memory),
                    )
                    conversation_meta["router_rewrite_applied"] = True
                    conversation_meta["router_rewritten_query"] = query_for_retrieval

    st.session_state.chat_history.append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)

    with st.chat_message("assistant"):
        if transform_response is not None:
            payload = {
                "role": "assistant",
                "provider": llm_provider,
                "provider_actual": provider_actual,
                "response": transform_response,
                "events": [],
                "top_k_selected": None,
                "top_k_reason": "conversation transform",
                "conversation_meta": conversation_meta,
                "conversation_router": conversation_router,
                "user_query": query_text,
                "conversation_transform": True,
                "conversation_transform_kind": transform_kind,
                "error": None,
            }
            render_assistant_payload(
                payload,
                show_diagnostics=show_diagnostics_panels,
                show_suggestions=True,
                message_key=f"live_{len(st.session_state.chat_history)}",
            )
            st.session_state.chat_history.append(payload)
            return

        if auto_top_k:
            dynamic_top_k, dynamic_reason = choose_dynamic_top_k(query_for_retrieval, jurisdiction, llm_provider)
            floor_top_k = int(manual_top_k)
            selected_top_k = max(dynamic_top_k, floor_top_k)
            if selected_top_k > dynamic_top_k:
                top_k_reason = f"{dynamic_reason}; floor->{floor_top_k}"
            else:
                top_k_reason = dynamic_reason
        else:
            selected_top_k, top_k_reason = int(manual_top_k), "manual override"

        request = QueryRequest(
            query=contextual_query,
            state=state,
            location=location.strip() if location.strip() else None,
            jurisdiction_type=jurisdiction,
            panchayat_category=category if jurisdiction == "panchayat" else None,
            top_k=selected_top_k,
            debug_trace=debug_trace,
        )

        with st.spinner("Searching rules and building your compliance answer..."):
            response, events, error = run_query_with_events(
                engine,
                request,
                show_live_events=show_live_events,
            )

        payload = {
            "role": "assistant",
            "provider": llm_provider,
            "provider_actual": provider_actual,
            "response": response,
            "events": events,
            "top_k_selected": selected_top_k,
            "top_k_reason": top_k_reason,
            "conversation_meta": conversation_meta,
            "conversation_router": conversation_router,
            "user_query": query_text,
            "error": error,
        }
        render_assistant_payload(
            payload,
            show_diagnostics=show_diagnostics_panels,
            show_suggestions=True,
            message_key=f"live_{len(st.session_state.chat_history)}",
        )

    st.session_state.chat_history.append(payload)


if __name__ == "__main__":
    main()
