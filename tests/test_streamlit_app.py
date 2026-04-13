from __future__ import annotations

import json
from textwrap import dedent

import pytest
from streamlit.testing.v1 import AppTest

from src.models.schemas import AnswerResponse, ComplianceBriefPayload
from streamlit_app import (
    EXAMPLE_QUERIES_PANCHAYAT,
    EXAMPLE_QUERIES_MUNICIPALITY,
    _coerce_transform_summary,
    _explicit_transform_style,
    _sanitize_followup_suggestion,
    build_conversation_transform_response,
)


TEST_APP = dedent(
    """
    import streamlit as st
    import streamlit_app
    from src.models.schemas import AnswerResponse, ComplianceBriefPayload

    class FakeLLMProvider:
        provider_id = "openai_responses_llm"
        model = "gpt-5.2"
        settings = None

        def generate_structured(self, *, task, payload, json_schema, temperature=0.0, max_output_tokens=1200):
            query = str(payload.get("query", "") or "").lower()
            if task == "conversation_turn_router":
                if (
                    "show me all that you have retrieved" in query
                    or "show retrieved evidence" in query
                    or "show all retrieved" in query
                ):
                    return {
                        "action": "respond_from_context",
                        "response_style": "show_retrieved",
                        "rewritten_query": str(payload.get("query", "")),
                        "reason": "test_followup_show_retrieved",
                        "confidence": 0.96,
                    }
                if ("summarize" in query or "summarise" in query) and "checklist" in query:
                    return {
                        "action": "respond_from_context",
                        "response_style": "checklist",
                        "rewritten_query": str(payload.get("query", "")),
                        "reason": "test_followup_checklist",
                        "confidence": 0.97,
                    }
                if "checklist" in query:
                    return {
                        "action": "respond_from_context",
                        "response_style": "checklist",
                        "rewritten_query": str(payload.get("query", "")),
                        "reason": "test_followup_checklist_general",
                        "confidence": 0.95,
                    }
                if "bullet" in query:
                    return {
                        "action": "respond_from_context",
                        "response_style": "bullets",
                        "rewritten_query": str(payload.get("query", "")),
                        "reason": "test_followup_bullets",
                        "confidence": 0.94,
                    }
                if "eli5" in query or "simple terms" in query or "plain language" in query:
                    return {
                        "action": "respond_from_context",
                        "response_style": "eli5",
                        "rewritten_query": str(payload.get("query", "")),
                        "reason": "test_followup_eli5",
                        "confidence": 0.94,
                    }
                if "summarize" in query or "summarise" in query or "summary" in query:
                    return {
                        "action": "respond_from_context",
                        "response_style": "summary",
                        "rewritten_query": str(payload.get("query", "")),
                        "reason": "test_followup_summary",
                        "confidence": 0.94,
                    }
                return {
                    "action": "use_retrieval",
                    "response_style": "normal",
                    "rewritten_query": str(payload.get("query", "")),
                    "reason": "test_new_query",
                    "confidence": 0.93,
                }
            if task == "conversation_context_rewrite":
                style = str(payload.get("response_style", "normal"))
                return {"short_summary": f"Transformed summary ({style})"}
            return {"short_summary": "Transformed summary"}

    class FakeEngine:
        def __init__(self, provider):
            self.llm_provider = FakeLLMProvider() if provider == "openai_responses_llm" else None

        def query(self, request, event_sink=None):
            brief = ComplianceBriefPayload(
                verdict="depends",
                short_summary=f"Summary for {request.jurisdiction_type}",
                applicable_rules=[],
                conditions_and_exceptions=[],
                required_actions=[],
                risk_flags=[],
                clarifications_needed=[],
            )
            return AnswerResponse(
                jurisdiction=f"{request.state}::{request.jurisdiction_type}",
                occupancy_groups=[],
                citations=[],
                verdict="depends",
                final_answer=brief,
            )

    def fake_get_engine(provider):
        st.session_state.setdefault("provider_log", []).append(provider)
        return FakeEngine(provider)

    def fake_run_query_with_events(engine, request, show_live_events=False):
        st.session_state.setdefault("request_log", []).append(
            {
                "query": request.query,
                "state": request.state,
                "jurisdiction_type": request.jurisdiction_type,
                "panchayat_category": request.panchayat_category,
                "top_k": request.top_k,
                "debug_trace": request.debug_trace,
            }
        )
        events = [
            {"elapsed_ms": 1.0, "step": "tool.scope_resolver", "status": "ok", "details": {}},
            {"elapsed_ms": 2.0, "step": "tool.query_planner", "status": "ok", "details": {}},
            {"elapsed_ms": 3.0, "step": "tool.occupancy_resolver", "status": "ok", "details": {}},
            {"elapsed_ms": 4.0, "step": "tool.scope_filter", "status": "ok", "details": {}},
            {"elapsed_ms": 5.0, "step": "tool.evidence_judge", "status": "ok", "details": {}},
            {"elapsed_ms": 6.0, "step": "tool.query_complete", "status": "ok", "details": {}},
        ]
        return engine.query(request=request), events, None

    streamlit_app.get_engine = fake_get_engine
    streamlit_app.run_query_with_events = fake_run_query_with_events
    streamlit_app.main()
    """
)

TEST_APP_CLARIFICATION = dedent(
    """
    import streamlit as st
    import streamlit_app
    from src.models.schemas import AnswerResponse

    class FakeEngine:
        def query(self, request, event_sink=None):
            return AnswerResponse(
                jurisdiction=f"{request.state}::{request.jurisdiction_type}",
                occupancy_groups=[],
                clarifications=[
                    {
                        "code": "OCCUPANCY_REQUIRED",
                        "question": "What is the occupancy type?",
                        "options": ["residential", "commercial"],
                    }
                ],
            )

    def fake_get_engine(provider):
        return FakeEngine()

    def fake_run_query_with_events(engine, request, show_live_events=False):
        return engine.query(request=request), [], None

    streamlit_app.get_engine = fake_get_engine
    streamlit_app.run_query_with_events = fake_run_query_with_events
    streamlit_app.main()
    """
)


def _new_app() -> AppTest:
    return AppTest.from_string(TEST_APP)


def _new_clarification_app() -> AppTest:
    return AppTest.from_string(TEST_APP_CLARIFICATION)


def _selectbox(at: AppTest, label: str):
    for item in at.selectbox:
        if item.label == label:
            return item
    raise AssertionError(f"Selectbox not found: {label}")


def _button(at: AppTest, label: str):
    for item in at.button:
        if item.label == label:
            return item
    raise AssertionError(f"Button not found: {label}")


def _toggle(at: AppTest, label: str):
    for item in at.toggle:
        if item.label == label:
            return item
    raise AssertionError(f"Toggle not found: {label}")


def _slider(at: AppTest, label: str):
    for item in at.slider:
        if item.label == label:
            return item
    raise AssertionError(f"Slider not found: {label}")


def _latest_assistant_payload(at: AppTest) -> dict:
    for item in reversed(list(at.session_state["chat_history"])):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return item
    raise AssertionError("Assistant payload not found in chat history")


@pytest.mark.parametrize("question", EXAMPLE_QUERIES_PANCHAYAT)
def test_front_screen_example_questions_submit_and_answer(question: str) -> None:
    at = _new_app()
    at.run()

    _button(at, question).click().run()

    assert len(at.exception) == 0
    assert at.session_state["request_log"][-1]["query"] == question
    assert [message.name for message in at.chat_message] == ["user", "assistant"]


@pytest.mark.parametrize("mode", ["no_llm", "openai_responses_llm"])
@pytest.mark.parametrize("question", EXAMPLE_QUERIES_MUNICIPALITY)
def test_municipality_example_questions_submit_and_answer_across_modes(mode: str, question: str) -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value(mode).run()
    _selectbox(at, "JURISDICTION").set_value("municipality").run()
    _button(at, question).click().run()

    assert len(at.exception) == 0
    request = at.session_state["request_log"][-1]
    assert request["query"] == question
    assert request["jurisdiction_type"] == "municipality"
    assert request["panchayat_category"] is None


@pytest.mark.parametrize("mode", ["no_llm", "openai_responses_llm"])
@pytest.mark.parametrize("jurisdiction", ["panchayat", "municipality"])
def test_mode_jurisdiction_permutations(mode: str, jurisdiction: str) -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value(mode).run()
    _selectbox(at, "JURISDICTION").set_value(jurisdiction).run()
    examples = EXAMPLE_QUERIES_MUNICIPALITY if jurisdiction == "municipality" else EXAMPLE_QUERIES_PANCHAYAT
    _button(at, examples[0]).click().run()

    assert len(at.exception) == 0
    request = at.session_state["request_log"][-1]
    assert request["jurisdiction_type"] == jurisdiction
    assert request["panchayat_category"] == ("Category-II" if jurisdiction == "panchayat" else None)
    assert at.session_state["provider_log"][-1] == mode


@pytest.mark.parametrize("mode", ["no_llm", "openai_responses_llm"])
@pytest.mark.parametrize("jurisdiction", ["panchayat", "municipality"])
@pytest.mark.parametrize("auto_top_k", [True, False])
def test_mode_jurisdiction_topk_strategy_matrix(mode: str, jurisdiction: str, auto_top_k: bool) -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value(mode).run()
    _selectbox(at, "JURISDICTION").set_value(jurisdiction).run()
    _toggle(at, "Dynamic retrieval depth (auto top-k)").set_value(auto_top_k).run()

    if auto_top_k:
        _slider(at, "Minimum retrieval depth (TOP-K floor)").set_value(9).run()
    else:
        _slider(at, "Manual retrieval depth (TOP-K)").set_value(7).run()

    examples = EXAMPLE_QUERIES_MUNICIPALITY if jurisdiction == "municipality" else EXAMPLE_QUERIES_PANCHAYAT
    _button(at, examples[-1]).click().run()

    assert len(at.exception) == 0
    request = at.session_state["request_log"][-1]
    assert request["jurisdiction_type"] == jurisdiction
    assert request["panchayat_category"] == ("Category-II" if jurisdiction == "panchayat" else None)
    if auto_top_k:
        assert request["top_k"] >= 9
    else:
        assert request["top_k"] == 7


def test_trace_panel_handles_more_than_four_tool_steps() -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value("no_llm").run()
    _toggle(at, "Show diagnostics panels").set_value(True).run()
    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()

    assert len(at.exception) == 0
    assert len(at.code) >= 6


def test_llm_mode_also_shows_diagnostics_tabs() -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value("openai_responses_llm").run()
    _toggle(at, "Show diagnostics panels").set_value(True).run()
    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()

    assert len(at.exception) == 0
    assert len(at.code) >= 6
    assert any(
        "Debug diagnostics:" in str(caption.value or "")
        for caption in at.caption
    )


def test_retrieval_only_mode_shows_ranked_retrieved_items_as_primary_output() -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value("no_llm").run()
    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()

    assert len(at.exception) == 0
    assert any(
        "Retrieval-only mode shows all retrieved chunks ranked by relevance." in str(caption.value or "")
        and "No LLM synthesis is used in this view." in str(caption.value or "")
        for caption in at.caption
    )


def test_clarification_payload_renders_question_in_chat() -> None:
    at = _new_clarification_app()
    at.run()

    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()

    assert len(at.exception) == 0
    assert any("I need one clarification before I can answer accurately:" in str(md.value or "") for md in at.markdown)
    assert any("What is the occupancy type?" in str(md.value or "") for md in at.markdown)


def test_llm_mode_avoids_inline_followup_textbox() -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value("openai_responses_llm").run()
    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()

    assert len(at.exception) == 0
    button_labels = [item.label for item in at.button]
    assert "Send follow-up" not in button_labels
    assert not any(label == "Continue this chat" for label in button_labels)


def test_chat_input_remains_visible_after_example_click() -> None:
    at = _new_app()
    at.run()

    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()

    assert len(at.exception) == 0
    assert len(at.chat_input) == 1


def test_llm_followup_summary_uses_conversation_transform_without_new_retrieval() -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value("openai_responses_llm").run()
    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()
    initial_requests = len(at.session_state["request_log"])
    assert initial_requests == 1

    at.chat_input[0].set_value("Can you summarize this answer as a checklist?").run()

    assert len(at.exception) == 0
    assert len(at.session_state["request_log"]) == initial_requests
    assert any(
        "Conversation follow-up (checklist): reused previous answer context; no new retrieval run."
        in str(caption.value or "")
        for caption in at.caption
    )


def test_llm_followup_show_retrieved_uses_previous_turn_without_new_retrieval() -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value("openai_responses_llm").run()
    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()
    initial_requests = len(at.session_state["request_log"])
    assert initial_requests == 1

    at.chat_input[0].set_value("show me all that you have retrieved").run()

    assert len(at.exception) == 0
    assert len(at.session_state["request_log"]) == initial_requests
    assert any(
        "Conversation follow-up (show retrieved): reused previous answer context; no new retrieval run."
        in str(caption.value or "")
        for caption in at.caption
    )


@pytest.mark.parametrize(
    ("followups", "expected_request_counts", "expected_actions"),
    [
        (
            ["What are the penalties for unauthorized construction in a panchayat?"],
            [1, 2],
            ["use_retrieval", "use_retrieval"],
        ),
        (
            [
                "Can you summarize this answer as a checklist?",
                "show me all that you have retrieved",
                "What are the setback rules for Category-II panchayat buildings?",
            ],
            [1, 1, 1, 2],
            ["use_retrieval", "respond_from_context", "respond_from_context", "use_retrieval"],
        ),
        (
            [
                "eli5 this in simple terms",
                "What documents are needed for permit application in panchayat?",
            ],
            [1, 1, 2],
            ["use_retrieval", "respond_from_context", "use_retrieval"],
        ),
        (
            [
                "Can you format this in bullet points?",
                "summarize this again in one short paragraph",
                "Now compare this with municipality requirements",
            ],
            [1, 1, 1, 2],
            ["use_retrieval", "respond_from_context", "respond_from_context", "use_retrieval"],
        ),
    ],
)
def test_llm_router_permutation_sequences_respect_retrieval_decisions(
    followups: list[str],
    expected_request_counts: list[int],
    expected_actions: list[str],
) -> None:
    at = _new_app()
    at.run()

    _selectbox(at, "MODE").set_value("openai_responses_llm").run()
    _button(at, EXAMPLE_QUERIES_PANCHAYAT[0]).click().run()
    assert len(at.exception) == 0

    actions_seen: list[str] = []
    counts_seen: list[int] = [len(at.session_state["request_log"])]
    first_payload = _latest_assistant_payload(at)
    first_router = first_payload.get("conversation_router") or {}
    actions_seen.append(str(first_router.get("action", "")))
    assert str(first_router.get("reason", "")).strip()
    confidence0 = first_router.get("confidence")
    assert isinstance(confidence0, (int, float))

    for query in followups:
        at.chat_input[0].set_value(query).run()
        assert len(at.exception) == 0

        counts_seen.append(len(at.session_state["request_log"]))
        payload = _latest_assistant_payload(at)
        router = payload.get("conversation_router") or {}
        actions_seen.append(str(router.get("action", "")))
        assert str(router.get("reason", "")).strip()
        confidence = router.get("confidence")
        assert isinstance(confidence, (int, float))

    assert counts_seen == expected_request_counts
    assert actions_seen == expected_actions


def test_first_turn_non_retrieval_style_query_defaults_to_use_retrieval() -> None:
    class DummyProvider:
        def generate_structured(self, **kwargs):
            return {"action": "respond_from_context"}

    transformed, kind, decision = build_conversation_transform_response(
        query="summarize this answer as a checklist",
        chat_history=[],
        llm_provider=DummyProvider(),
    )

    assert transformed is None
    assert kind is None
    assert decision["action"] == "use_retrieval"
    assert decision["response_style"] == "normal"
    assert decision["reason"] == "no prior assistant answer available"


def test_router_uses_prior_answer_from_dict_payload_without_forcing_retrieval() -> None:
    class DummyProvider:
        def generate_structured(self, **kwargs):
            task = kwargs.get("task")
            if task == "conversation_turn_router":
                return {
                    "action": "respond_from_context",
                    "response_style": "eli5",
                    "rewritten_query": "eli5 the above answer",
                    "reason": "transform_only",
                    "confidence": 0.93,
                }
            return {"short_summary": "Simple explanation."}

    prior = ComplianceBriefPayload(
        verdict="depends",
        short_summary="Prior summary",
        applicable_rules=[],
        conditions_and_exceptions=[],
        required_actions=[],
        risk_flags=[],
        clarifications_needed=[],
    )
    prior_response = AnswerResponse(
        jurisdiction="kerala::panchayat",
        occupancy_groups=[],
        citations=[],
        verdict="depends",
        final_answer=prior,
    )
    chat_history = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "response": prior_response.model_dump()},
    ]

    transformed, kind, decision = build_conversation_transform_response(
        query="eli5 the above answer",
        chat_history=chat_history,
        llm_provider=DummyProvider(),
    )

    assert transformed is not None
    assert kind == "eli5"
    assert decision["action"] == "respond_from_context"


def test_router_normalizes_transform_style_conflict_to_context() -> None:
    class DummyProvider:
        def generate_structured(self, **kwargs):
            task = kwargs.get("task")
            if task == "conversation_turn_router":
                return {
                    "action": "use_retrieval",
                    "response_style": "eli5",
                    "rewritten_query": "eli5 the above answer",
                    "reason": "conflict_for_test",
                    "confidence": 0.67,
                }
            return {"short_summary": "Simple explanation."}

    prior = ComplianceBriefPayload(
        verdict="depends",
        short_summary="Prior summary",
        applicable_rules=[],
        conditions_and_exceptions=[],
        required_actions=[],
        risk_flags=[],
        clarifications_needed=[],
    )
    prior_response = AnswerResponse(
        jurisdiction="kerala::panchayat",
        occupancy_groups=[],
        citations=[],
        verdict="depends",
        final_answer=prior,
    )
    chat_history = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "response": prior_response},
    ]

    transformed, kind, decision = build_conversation_transform_response(
        query="eli5 the above answer",
        chat_history=chat_history,
        llm_provider=DummyProvider(),
    )

    assert transformed is not None
    assert kind == "eli5"
    assert decision["action"] == "respond_from_context"
    assert (
        "normalized_to_context_style" in str(decision["reason"])
        or "transform_followup_override" in str(decision["reason"])
    )


def test_openai_router_uses_retrieval_query_function_tool_for_fresh_lookup() -> None:
    class _FunctionCall:
        type = "function_call"
        name = "retrieval_query"

        def __init__(self, call_id: str, rewritten_query: str) -> None:
            self.call_id = call_id
            self.arguments = json.dumps({"query": rewritten_query})

    class _Response:
        def __init__(self, *, response_id: str, output: list[object] | None = None, output_text: str = "") -> None:
            self.id = response_id
            self.output = output or []
            self.output_text = output_text

    class _ResponsesAPI:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self._count = 0

        def create(self, **kwargs):
            self.calls.append(kwargs)
            self._count += 1
            if self._count == 1:
                return _Response(
                    response_id="resp_1",
                    output=[_FunctionCall("call_1", "penalties unauthorized construction panchayat kerala")],
                )
            return _Response(
                response_id="resp_2",
                output_text=json.dumps(
                    {
                        "action": "use_retrieval",
                        "response_style": "normal",
                        "rewritten_query": "penalties unauthorized construction panchayat kerala",
                        "reason": "requires fresh legal lookup",
                        "confidence": 0.98,
                    }
                ),
            )

    class _Client:
        def __init__(self) -> None:
            self.responses = _ResponsesAPI()

    class _ToolProvider:
        provider_id = "openai_responses_llm"
        model = "gpt-5.2"
        timeout_s = 12.0

        def __init__(self) -> None:
            self._client = _Client()

        def _normalize_for_openai_strict_schema(self, schema: dict) -> dict:
            return schema

        def _extract_text(self, response: object) -> str:
            return str(getattr(response, "output_text", "") or "")

        def generate_structured(self, **kwargs):
            raise AssertionError("legacy router path should not be called in tool-router test")

    prior = ComplianceBriefPayload(
        verdict="depends",
        short_summary="Prior answer summary",
        applicable_rules=[],
        conditions_and_exceptions=[],
        required_actions=[],
        risk_flags=[],
        clarifications_needed=[],
    )
    prior_response = AnswerResponse(
        jurisdiction="kerala::panchayat",
        occupancy_groups=[],
        citations=[],
        verdict="depends",
        final_answer=prior,
    )
    chat_history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "response": prior_response},
    ]
    provider = _ToolProvider()

    transformed, kind, decision = build_conversation_transform_response(
        query="What are the penalties for unauthorized construction?",
        chat_history=chat_history,
        llm_provider=provider,
    )

    assert transformed is None
    assert kind is None
    assert decision["action"] == "use_retrieval"
    assert decision["tool_invoked"] is True
    assert "penalties unauthorized construction panchayat kerala" in decision["rewritten_query"]
    assert decision["context_short_summary"] == ""
    first_call = provider._client.responses.calls[0]
    assert first_call.get("tools")
    assert len(provider._client.responses.calls) == 1


def test_tool_call_misfire_on_format_followup_is_overridden_to_context() -> None:
    class _FunctionCall:
        type = "function_call"
        name = "retrieval_query"

        def __init__(self) -> None:
            self.call_id = "call_1"
            self.arguments = json.dumps({"query": "irrelevant retrieval query"})

    class _Response:
        def __init__(self) -> None:
            self.id = "resp_1"
            self.output = [_FunctionCall()]
            self.output_text = ""

    class _ResponsesAPI:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _Response()

    class _Client:
        def __init__(self) -> None:
            self.responses = _ResponsesAPI()

    class _ToolProvider:
        provider_id = "openai_responses_llm"
        model = "gpt-5.2"
        timeout_s = 12.0

        def __init__(self) -> None:
            self._client = _Client()

        def _normalize_for_openai_strict_schema(self, schema: dict) -> dict:
            return schema

        def _extract_text(self, response: object) -> str:
            return str(getattr(response, "output_text", "") or "")

        def generate_structured(self, **kwargs):
            if kwargs.get("task") == "conversation_context_rewrite":
                return {"short_summary": "1. Step one\n2. Step two"}
            raise AssertionError("legacy conversation router should not be called")

    prior = ComplianceBriefPayload(
        verdict="depends",
        short_summary="Prior answer summary",
        applicable_rules=[],
        conditions_and_exceptions=[],
        required_actions=[],
        risk_flags=[],
        clarifications_needed=[],
    )
    prior_response = AnswerResponse(
        jurisdiction="kerala::panchayat",
        occupancy_groups=[],
        citations=[],
        verdict="depends",
        final_answer=prior,
    )
    chat_history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "response": prior_response},
    ]
    provider = _ToolProvider()

    transformed, kind, decision = build_conversation_transform_response(
        query="make it as checklist",
        chat_history=chat_history,
        llm_provider=provider,
    )

    assert transformed is not None
    assert kind == "checklist"
    assert decision["action"] == "respond_from_context"
    assert decision["tool_invoked"] is False
    assert "transform_followup_override" in str(decision["reason"])


def test_transform_style_coercion_keeps_followup_outputs_concise() -> None:
    long_text = (
        "A completion certificate is required when work is carried out under a permit and submitted to the Secretary. "
        "The Secretary issues a development certificate and occupancy certificate if rules are met. "
        "For telecommunication towers, structural safety and stability certificates are also required before a use certificate."
    )

    eli5 = _coerce_transform_summary("eli5", long_text)
    summary = _coerce_transform_summary("summary", long_text)
    checklist = _coerce_transform_summary("checklist", long_text)
    bullets = _coerce_transform_summary("bullets", long_text)

    assert len(eli5.split()) <= 55
    assert len(summary.split()) <= 65
    assert checklist.startswith("1.")
    assert len([line for line in checklist.splitlines() if line.strip()]) <= 4
    assert bullets.startswith("- ")
    assert len([line for line in bullets.splitlines() if line.strip()]) <= 4


def test_explicit_transform_style_detection() -> None:
    assert _explicit_transform_style("eli5 the above answer") == "eli5"
    assert _explicit_transform_style("make it as checklist") == "checklist"
    assert _explicit_transform_style("summarize this in bullet points") == "bullets"
    assert _explicit_transform_style("show me all that you have retrieved") == "show_retrieved"
    assert _explicit_transform_style("What are the penalties for unauthorized construction?") is None


def test_followup_suggestion_sanitizer_trims_noise_and_length() -> None:
    raw = (
        "Is the building work covered by permit-not-necessary category under item 1327 "
        "(KPBR 2011 Rule 25 and Section 235P details) and other long statutory references?"
    )
    clean = _sanitize_followup_suggestion(raw)

    assert clean.endswith("?")
    assert len(clean) <= 108
    assert "KPBR" not in clean
