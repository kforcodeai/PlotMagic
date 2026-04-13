#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.models.schemas import AnswerResponse, QueryRequest
from streamlit_app import (
    _load_env_file,
    _select_chat_agent_provider,
    build_contextual_query,
    build_conversation_transform_response,
    choose_dynamic_top_k,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: list[str]
    expected_retrieval: list[bool]


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split())


def _assistant_summary(response: AnswerResponse) -> str:
    final = getattr(response, "final_answer", None)
    if final is None:
        return ""
    return _normalize(str(getattr(final, "short_summary", "")))


SCENARIOS: list[Scenario] = [
    Scenario(
        name="retrieval_after_retrieval",
        turns=[
            "Is a completion certificate required for buildings under 100 sq metres?",
            "What are the penalties for unauthorized construction in a panchayat?",
        ],
        expected_retrieval=[True, True],
    ),
    Scenario(
        name="retrieval_non_non_retrieval",
        turns=[
            "Is a completion certificate required for buildings under 100 sq metres?",
            "eli5",
            "make it as checklist",
            "What are the setback rules for Category-II panchayat buildings?",
        ],
        expected_retrieval=[True, False, False, True],
    ),
    Scenario(
        name="non_then_retrieval",
        turns=[
            "What permits are needed to construct a residential building in a panchayat area?",
            "summarize this in bullet points",
            "What are the penalties for unauthorized construction in a panchayat?",
        ],
        expected_retrieval=[True, False, True],
    ),
    Scenario(
        name="non_non_then_retrieval",
        turns=[
            "What are the penalties for unauthorized construction in a panchayat?",
            "show me all that you have retrieved",
            "summarize in one short paragraph",
            "Now compare this with municipality requirements",
        ],
        expected_retrieval=[True, False, False, True],
    ),
    Scenario(
        name="retrieval_non_retrieval_non_retrieval_retrieval",
        turns=[
            "What are the setback rules for Category-II panchayat buildings?",
            "Can you explain that in simple terms?",
            "What is the permit fee concession for Category I residential buildings up to 150 sq.m?",
            "make this as checklist",
            "What permits are needed for commercial construction in a municipality?",
        ],
        expected_retrieval=[True, False, True, False, True],
    ),
]


def run_scenario(
    *,
    engine: ComplianceEngine,
    scenario: Scenario,
    state: str,
    jurisdiction: str,
    category: str,
    location: str | None,
    max_turns: int,
    always_include_memory: bool,
    manual_top_k: int,
    auto_top_k: bool,
) -> dict[str, Any]:
    chat_history: list[dict[str, Any]] = []
    llm_provider_obj = getattr(engine, "llm_provider", None)
    router_provider = _select_chat_agent_provider(llm_provider_obj)
    turn_results: list[dict[str, Any]] = []
    mismatches: list[str] = []

    for i, query in enumerate(scenario.turns):
        print(f"[{scenario.name}] turn {i + 1}/{len(scenario.turns)}: {query}", flush=True)
        contextual_query, conversation_meta = build_contextual_query(
            raw_query=query,
            chat_history=chat_history,
            enabled=True,
            max_turns=max_turns,
            always_include=always_include_memory,
        )
        transform_response, transform_kind, decision = build_conversation_transform_response(
            query=query,
            chat_history=chat_history,
            llm_provider=router_provider,
        )

        retrieval_executed = False
        events: list[dict[str, Any]] = []
        top_k_selected: int | None = None
        retrieval_query = query
        response: AnswerResponse

        if transform_response is not None:
            response = transform_response
            retrieval_query = "[none]"
        else:
            rewritten = _normalize(str(decision.get("rewritten_query", "")))
            if str(decision.get("action", "")) == "use_retrieval" and rewritten:
                retrieval_query = rewritten
                if rewritten != query:
                    contextual_query, conversation_meta = build_contextual_query(
                        raw_query=rewritten,
                        chat_history=chat_history,
                        enabled=True,
                        max_turns=max_turns,
                        always_include=always_include_memory,
                    )
                    conversation_meta["router_rewrite_applied"] = True
            else:
                retrieval_query = query

            if auto_top_k:
                dyn_k, _ = choose_dynamic_top_k(retrieval_query, jurisdiction, "openai_responses_llm")
                top_k_selected = max(int(manual_top_k), int(dyn_k))
            else:
                top_k_selected = int(manual_top_k)

            request = QueryRequest(
                query=contextual_query,
                state=state,
                location=location,
                jurisdiction_type=jurisdiction,
                panchayat_category=category if jurisdiction == "panchayat" else None,
                top_k=top_k_selected,
                debug_trace=True,
            )

            def sink(event: dict[str, Any]) -> None:
                events.append(event)

            response = engine.query(request, event_sink=sink)
            retrieval_executed = True

        chat_history.append({"role": "user", "content": query})
        chat_history.append(
            {
                "role": "assistant",
                "response": response,
                "conversation_router": decision,
                "conversation_transform_kind": transform_kind,
            }
        )

        expected_retrieval = scenario.expected_retrieval[i]
        if retrieval_executed != expected_retrieval:
            mismatches.append(
                f"turn {i + 1}: expected retrieval={expected_retrieval}, got retrieval={retrieval_executed} "
                f"(action={decision.get('action')}, style={decision.get('response_style')})"
            )

        turn_results.append(
            {
                "turn_index": i + 1,
                "query": query,
                "expected_retrieval": expected_retrieval,
                "retrieval_executed": retrieval_executed,
                "router_action": decision.get("action"),
                "router_style": decision.get("response_style"),
                "router_reason": decision.get("reason"),
                "router_confidence": decision.get("confidence"),
                "transform_kind": transform_kind,
                "retrieval_query": retrieval_query,
                "top_k_selected": top_k_selected,
                "conversation_meta": conversation_meta,
                "event_steps": [str(item.get("step", "")) for item in events[:12]],
                "answer_summary": _assistant_summary(response),
            }
        )

    return {
        "scenario": scenario.name,
        "turns": turn_results,
        "expected_retrieval_pattern": scenario.expected_retrieval,
        "actual_retrieval_pattern": [bool(item["retrieval_executed"]) for item in turn_results],
        "matched_expected_pattern": not mismatches,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM router retrieval-vs-context behavior across turn permutations.")
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="panchayat", choices=["panchayat", "municipality"])
    parser.add_argument("--category", default="Category-II")
    parser.add_argument("--location", default=None)
    parser.add_argument("--conversation-turns", type=int, default=4)
    parser.add_argument("--manual-top-k", type=int, default=12)
    parser.add_argument("--auto-top-k", action="store_true", default=True)
    parser.add_argument("--no-auto-top-k", action="store_false", dest="auto_top_k")
    parser.add_argument("--always-include-memory", action="store_true", default=True)
    parser.add_argument("--no-always-include-memory", action="store_false", dest="always_include_memory")
    parser.add_argument("--output-json", default="router_permutation_report.json")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run only selected scenario name(s). Can be passed multiple times.",
    )
    parser.add_argument("--max-scenarios", type=int, default=0, help="Run only first N scenarios after filtering. 0 = all.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any permutation mismatches expected retrieval pattern.")
    args = parser.parse_args()

    _load_env_file(ROOT / ".env")
    os.environ["PLOTMAGIC_LLM_PROVIDER"] = "openai_responses_llm"
    os.environ.setdefault("PLOTMAGIC_CHAT_AGENT_MODEL", "gpt-5.2")

    engine = ComplianceEngine(ROOT)
    provider_actual = str(getattr(getattr(engine, "llm_provider", None), "provider_id", "unknown"))
    provider_model = str(getattr(getattr(engine, "llm_provider", None), "model", "unknown"))
    router_model = os.getenv("PLOTMAGIC_CHAT_AGENT_MODEL", "")

    selected = list(SCENARIOS)
    if args.scenario:
        wanted = {str(item).strip() for item in args.scenario if str(item).strip()}
        selected = [item for item in selected if item.name in wanted]
    if int(args.max_scenarios) > 0:
        selected = selected[: int(args.max_scenarios)]
    if not selected:
        raise SystemExit("No scenarios selected.")

    overall: dict[str, Any] = {
        "provider_requested": "openai_responses_llm",
        "provider_actual": provider_actual,
        "provider_model": provider_model,
        "router_model": router_model,
        "scenarios": [],
    }

    print(
        f"Running {len(selected)} scenario(s) with provider={provider_actual}, model={provider_model}, router_model={router_model}",
        flush=True,
    )
    for scenario in selected:
        print(f"=== Scenario: {scenario.name} ===", flush=True)
        result = run_scenario(
            engine=engine,
            scenario=scenario,
            state=args.state,
            jurisdiction=args.jurisdiction,
            category=args.category,
            location=args.location,
            max_turns=int(args.conversation_turns),
            always_include_memory=bool(args.always_include_memory),
            manual_top_k=int(args.manual_top_k),
            auto_top_k=bool(args.auto_top_k),
        )
        overall["scenarios"].append(result)
        print(
            f"Scenario result: matched={result['matched_expected_pattern']} "
            f"actual={result['actual_retrieval_pattern']} expected={result['expected_retrieval_pattern']}",
            flush=True,
        )

    matched = [bool(item.get("matched_expected_pattern")) for item in overall["scenarios"]]
    overall["all_patterns_matched"] = all(matched) if matched else False
    overall["total_scenarios"] = len(matched)
    overall["matched_scenarios"] = sum(1 for item in matched if item)

    out_path = ROOT / args.output_json
    out_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(
        "Summary:",
        json.dumps(
            {
                "provider_actual": overall["provider_actual"],
                "provider_model": overall["provider_model"],
                "router_model": overall["router_model"],
                "matched_scenarios": overall["matched_scenarios"],
                "total_scenarios": overall["total_scenarios"],
                "all_patterns_matched": overall["all_patterns_matched"],
            },
            ensure_ascii=False,
        ),
    )

    if args.strict and not overall["all_patterns_matched"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
