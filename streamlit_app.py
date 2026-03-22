from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.api.service import ComplianceEngine
from src.models.schemas import QueryRequest


ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def get_engine(llm_provider_id: str) -> ComplianceEngine:
    os.environ["PLOTMAGIC_LLM_PROVIDER"] = llm_provider_id
    return ComplianceEngine(root=ROOT)


def inject_chat_styles() -> None:
    st.markdown(
        """
<style>
    :root {
        --pm-surface: #ffffff;
        --pm-border: #dbe3ee;
        --pm-muted: #4b5563;
        --pm-accent: #0f766e;
        --pm-user-bg: #eff6ff;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 100% 0%, rgba(226, 239, 255, 0.75) 0%, rgba(248, 250, 252, 0.95) 40%, #f8fafc 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] .stButton button {
        border: 1px solid #334155;
        background: #1e293b;
        color: #e5e7eb;
    }
    [data-testid="stChatMessage"] {
        background: var(--pm-surface);
        border: 1px solid var(--pm-border);
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
        padding: 0.35rem 0.8rem;
        margin-bottom: 0.85rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--pm-user-bg);
        border-color: #bfdbfe;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #111827;
    }
    [data-testid="stChatInput"] {
        border-top: 1px solid var(--pm-border);
        background: rgba(248, 250, 252, 0.94);
    }
    .pm-provider-chip {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        border: 1px solid #bfd9d5;
        color: #0f766e;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_event_line(event: dict[str, Any]) -> str:
    status = str(event.get("status", ""))
    icon = {
        "running": "⏳",
        "ok": "✅",
        "skipped": "⏭️",
        "abstained": "🛑",
        "needs_clarification": "❓",
    }.get(status, "•")
    step = str(event.get("step", "event"))
    details = event.get("details", {})
    details_text = json.dumps(details, ensure_ascii=False) if details else "{}"
    return f"{icon} `{step}` - `{status}`  \n`{details_text}`"


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

    status_box = st.status("Running query...", expanded=show_live_events)
    latest_step = st.empty()
    timeline = st.empty() if show_live_events else None
    while thread.is_alive() or not event_queue.empty():
        while not event_queue.empty():
            event = event_queue.get()
            events.append(event)
        if events:
            last_event = events[-1]
            latest_step.caption(
                f"Current step: `{last_event.get('step', 'event')}` ({last_event.get('status', 'running')})"
            )
            if timeline is not None:
                rendered = [f"**{event['elapsed_ms']}ms**  \n{render_event_line(event)}" for event in events[-12:]]
                timeline.markdown("\n\n".join(rendered))
        time.sleep(0.05)

    thread.join()
    total_ms = events[-1]["elapsed_ms"] if events else 0
    latest_step.empty()
    status_box.update(label=f"Query completed in {total_ms} ms", state="complete", expanded=False)
    return result_holder["response"], events, result_holder["error"]


def render_final_answer(response: Any, llm_provider_id: str) -> None:
    if response.final_answer is None:
        st.warning("No structured final answer payload returned.")
        return

    verdict = response.final_answer.verdict.upper()
    if response.verdict == "insufficient_evidence":
        st.error(f"Verdict: {verdict}")
    else:
        st.success(f"Verdict: {verdict}")

    summary = response.final_answer.short_summary.strip()
    st.markdown(summary if summary else "_No summary returned._")

    citation_lookup = build_citation_lookup(response)
    show_inline_citations = llm_provider_id == "openai_responses_llm"

    def render_claim_section(title: str, items: list[Any]) -> None:
        st.markdown(f"**{title}**")
        if not items:
            st.caption("No items.")
            return
        for item in items:
            if show_inline_citations:
                links = render_inline_citation_links(item.citation_ids, citation_lookup)
                suffix = f" {links}" if links else ""
                st.markdown(f"- {item.text}{suffix}")
            else:
                st.markdown(f"- `{item.claim_id}`: {item.text}")
            if item.citation_ids and not show_inline_citations:
                st.caption("Citations: " + ", ".join(item.citation_ids))

    render_claim_section("Applicable Rules", response.final_answer.applicable_rules)
    render_claim_section("Conditions and Exceptions", response.final_answer.conditions_and_exceptions)
    render_claim_section("Required Actions", response.final_answer.required_actions)

    st.markdown("### Risk Flags")
    for risk in response.final_answer.risk_flags:
        st.markdown(f"- {risk}")

    st.markdown("### Clarifications Needed")
    if response.final_answer.clarifications_needed:
        for question in response.final_answer.clarifications_needed:
            st.markdown(f"- {question}")
    else:
        st.caption("No clarification required.")


def render_retrieved_chunks(response: Any) -> None:
    if not response.evidence_matrix:
        st.info("No retrieved chunks available.")
        return

    citations_by_claim: dict[str, list[Any]] = {}
    for citation in response.citations:
        citations_by_claim.setdefault(citation.claim_id, []).append(citation)

    for item in response.evidence_matrix:
        with st.expander(f"{item.claim_id} ({item.chunk_id})", expanded=False):
            st.markdown(item.text or "_No text_")
            claim_citations = citations_by_claim.get(item.claim_id, [])
            if claim_citations:
                st.markdown("**Citations**")
                for citation in claim_citations:
                    link = citation.source_url or ""
                    if link:
                        st.markdown(f"- [{citation.display_citation}]({link})")
                    else:
                        st.markdown(f"- `{citation.display_citation}`")
            else:
                st.caption("No citations attached to this chunk.")


def render_citation_explorer(response: Any) -> None:
    if not response.citations:
        st.info("No citations returned.")
        return

    for citation in response.citations:
        summary = f"- **{citation.display_citation}** (`claim={citation.claim_id}`)"
        st.markdown(summary)
        if citation.source_url:
            st.markdown(f"  [Open Source]({citation.source_url})")
        st.markdown(f"  `{citation.quote_excerpt[:220]}`")


def render_thinking_panel(response: Any, events: list[dict[str, Any]]) -> None:
    st.caption("High-level reasoning trace and tool-stage telemetry (not raw chain-of-thought).")

    tool_steps = sorted({str(event.get("step", "")) for event in events if str(event.get("step", "")).startswith("tool.")})
    if tool_steps:
        st.markdown("**Tools used**")
        for step in tool_steps:
            st.markdown(f"- `{step}`")

    st.markdown("**Agent Trace**")
    if response.agent_trace:
        for step in response.agent_trace:
            st.markdown(f"- `{step.step}` / `{step.status}` / `{json.dumps(step.details, ensure_ascii=False)}`")
    else:
        st.caption("No explicit agent trace (enable `debug_trace`).")

    st.markdown("**Grounding**")
    if response.grounding:
        st.json(response.grounding.model_dump())
    else:
        st.caption("No grounding payload.")

    st.markdown("**Latency Map**")
    st.json(response.latency_ms)

    with st.expander("Pipeline Events", expanded=False):
        if not events:
            st.caption("No events recorded.")
        else:
            rendered = [f"**{event['elapsed_ms']}ms**  \n{render_event_line(event)}" for event in events]
            st.markdown("\n\n".join(rendered))


def render_assistant_payload(payload: dict[str, Any]) -> None:
    provider = str(payload.get("provider", "no_llm"))
    response = payload.get("response")
    events = payload.get("events", [])
    error = payload.get("error")

    chip = provider.replace("_", " ").upper()
    st.markdown(f"<span class='pm-provider-chip'>{chip}</span>", unsafe_allow_html=True)

    if error:
        st.error(f"Query failed: {error}")
        return
    if response is None:
        st.error("No response produced.")
        return

    render_final_answer(response, provider)

    if provider == "openai_responses_llm":
        return

    st.info("LLM disabled (`no_llm`). Showing retrieval-grounded artifacts for verification.")
    with st.expander("Retrieved Chunks", expanded=True):
        render_retrieved_chunks(response)
    with st.expander("Citations", expanded=True):
        render_citation_explorer(response)
    with st.expander("Thinking & Tools", expanded=False):
        render_thinking_panel(response, events)
    with st.expander("Raw JSON", expanded=False):
        st.json(response.model_dump())
        st.markdown("### Live Events")
        st.json(events)


def main() -> None:
    st.set_page_config(page_title="PlotMagic Compliance Chat", page_icon="📚", layout="wide")
    inject_chat_styles()
    st.title("PlotMagic Compliance Chat")
    st.caption("Chat-style compliance assistant with provider-aware grounding controls.")

    with st.sidebar:
        st.header("Assistant Controls")
        llm_provider = st.selectbox("LLM Provider", options=["no_llm", "openai_responses_llm"], index=0)
        state = st.selectbox("State", options=["kerala"], index=0)
        jurisdiction = st.selectbox("Jurisdiction", options=["panchayat", "municipality"], index=0)
        category = st.selectbox("Panchayat Category", options=["Category-II", "Category-I"], index=0)
        top_k = st.slider("Top-K Retrieval", min_value=3, max_value=30, value=12, step=1)
        debug_trace = st.toggle("Enable debug trace", value=True)
        show_live_events = st.toggle("Show live events while running", value=False)

        st.markdown("---")
        clear_chat = st.button("Clear Chat", use_container_width=True)
        st.caption("Tip: first run may be slower due to lazy ingestion.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if clear_chat:
        st.session_state.chat_history = []
        st.rerun()

    if not st.session_state.chat_history:
        st.info(
            "Ask a compliance question to begin. Use `no_llm` for full retrieval artifacts, or "
            "`openai_responses_llm` for refined final answers with inline citations."
        )

    for message in st.session_state.chat_history:
        role = str(message.get("role", "assistant"))
        with st.chat_message(role):
            if role == "user":
                st.markdown(str(message.get("content", "")))
            else:
                render_assistant_payload(message)

    query = st.chat_input("Ask a compliance question...")
    if not query:
        return
    if len(query.strip()) < 3:
        st.error("Query is too short.")
        return

    st.session_state.chat_history.append({"role": "user", "content": query.strip()})
    with st.chat_message("user"):
        st.markdown(query.strip())

    with st.chat_message("assistant"):
        with st.spinner("Initializing engine..."):
            engine = get_engine(llm_provider)

        request = QueryRequest(
            query=query.strip(),
            state=state,
            jurisdiction_type=jurisdiction,
            panchayat_category=category if jurisdiction == "panchayat" else None,
            top_k=top_k,
            debug_trace=debug_trace,
        )

        response, events, error = run_query_with_events(
            engine,
            request,
            show_live_events=show_live_events,
        )

        payload = {
            "role": "assistant",
            "provider": llm_provider,
            "response": response,
            "events": events,
            "error": error,
        }
        render_assistant_payload(payload)

    st.session_state.chat_history.append(payload)


if __name__ == "__main__":
    main()
