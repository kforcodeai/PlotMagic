from __future__ import annotations

from streamlit_app import build_contextual_query


def _assistant_message(summary: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "response": {
            "final_answer": {
                "short_summary": summary,
            }
        },
    }


def test_build_contextual_query_disabled_returns_raw_query() -> None:
    raw = "What permits are needed?"
    query, meta = build_contextual_query(
        raw_query=raw,
        chat_history=[],
        enabled=False,
        max_turns=4,
        always_include=False,
    )
    assert query == raw
    assert meta["applied"] is False


def test_build_contextual_query_applies_for_followup_question() -> None:
    chat_history = [
        {"role": "user", "content": "What are permit documents?"},
        _assistant_message("Permit needs Appendix A form and ownership proof."),
        {"role": "user", "content": "Any fee concessions?"},
        _assistant_message("Category I residential up to 150 sq.m gets 50% fee concession."),
    ]
    query, meta = build_contextual_query(
        raw_query="Can you combine that with the previous answer?",
        chat_history=chat_history,
        enabled=True,
        max_turns=3,
        always_include=False,
    )
    assert meta["applied"] is True
    assert "Conversation context from prior turns:" in query
    assert "Current user question:" in query
    assert "What are permit documents?" in query
    assert "Any fee concessions?" in query


def test_build_contextual_query_skips_non_followup_when_auto_mode() -> None:
    chat_history = [
        {"role": "user", "content": "What are permit documents?"},
        _assistant_message("Permit needs Appendix A form and ownership proof."),
    ]
    raw = "What is the height limit for this area?"
    query, meta = build_contextual_query(
        raw_query=raw,
        chat_history=chat_history,
        enabled=True,
        max_turns=4,
        always_include=False,
    )
    assert meta["applied"] is False
    assert meta["followup_detected"] is False
    assert query == raw

