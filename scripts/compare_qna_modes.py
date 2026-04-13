#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.models.schemas import QueryRequest


DEFAULT_QUESTIONS = [
    "What permits are needed to construct a residential building in a panchayat area?",
    "What are the setback rules for Category-II panchayat buildings?",
    "Is a completion certificate required for buildings under 100 sq metres?",
    "What are the penalties for unauthorized construction?",
    "What documents are required with the building permit application?",
]


@dataclass(slots=True)
class ModeRun:
    question: str
    requested_mode: str
    runtime_provider: str
    provider_diagnostics: list[str]
    llm_used: float
    fallback_used: float
    response: dict[str, Any] | None
    events: list[dict[str, Any]]
    error: str | None
    traceback_text: str | None


def _load_env_file(path: Path, override_existing: bool = False) -> None:
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
        if not key:
            continue
        if key in os.environ and not override_existing:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def _parse_questions(question_values: list[str], questions_file: str | None) -> list[str]:
    out: list[str] = [item.strip() for item in question_values if item and item.strip()]
    if questions_file:
        path = Path(questions_file)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            loaded = json.loads(text)
            if isinstance(loaded, list):
                out.extend(str(item).strip() for item in loaded if str(item).strip())
            elif isinstance(loaded, dict) and isinstance(loaded.get("questions"), list):
                out.extend(str(item).strip() for item in loaded["questions"] if str(item).strip())
            else:
                raise ValueError("JSON questions file must be a list[str] or {'questions': list[str]}")
        elif suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, str):
                    value = row.strip()
                elif isinstance(row, dict):
                    value = str(row.get("question") or row.get("query") or "").strip()
                else:
                    value = ""
                if value:
                    out.append(value)
        else:
            out.extend(line.strip() for line in text.splitlines() if line.strip())
    if not out:
        out = list(DEFAULT_QUESTIONS)
    seen: set[str] = set()
    deduped: list[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _build_engine(mode: str) -> ComplianceEngine:
    os.environ["PLOTMAGIC_LLM_PROVIDER"] = mode
    return ComplianceEngine(root=ROOT)


def _run_with_events(engine: ComplianceEngine, request: QueryRequest) -> tuple[Any, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    start = time.perf_counter()

    def sink(event: dict[str, Any]) -> None:
        events.append(
            {
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                **event,
            }
        )

    response = engine.query(request=request, event_sink=sink)
    return response, events


def _citation_id(citation: dict[str, Any]) -> str:
    return f"{citation.get('ruleset_id')}:{citation.get('rule_number')}:{citation.get('anchor_id')}"


def _citation_lookup(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for citation in response.get("citations", []) or []:
        if isinstance(citation, dict):
            lookup[_citation_id(citation)] = citation
    return lookup


def _render_claim_lines(
    items: list[dict[str, Any]],
    inline_links: bool,
    citation_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    if not items:
        return ["- None identified."]
    lines: list[str] = []
    for item in items:
        text = str(item.get("text", ""))
        citation_ids = [str(cid) for cid in (item.get("citation_ids", []) or [])]
        if inline_links and citation_ids:
            links: list[str] = []
            for citation_id in citation_ids:
                citation = citation_lookup.get(citation_id)
                if citation is None:
                    links.append(f"`{citation_id}`")
                    continue
                label = str(citation.get("display_citation") or citation_id)
                url = str(citation.get("source_url") or "")
                if url:
                    links.append(f"[{label}]({url})")
                else:
                    links.append(f"`{label}`")
            if links:
                text = f"{text} {' '.join(links)}"
        elif citation_ids:
            text = f"{text}\n  Refs: {', '.join(citation_ids)}"
        lines.append(f"- {text}")
    return lines


def _json_block(data: Any) -> str:
    return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"


def _text_block(text: str) -> str:
    return "```text\n" + text + "\n```"


def _mode_metrics_table(left: ModeRun, right: ModeRun) -> str:
    left_resp = left.response or {}
    right_resp = right.response or {}
    left_fa = left_resp.get("final_answer") or {}
    right_fa = right_resp.get("final_answer") or {}

    left_citations = len(left_resp.get("citations") or [])
    right_citations = len(right_resp.get("citations") or [])
    left_counts = (
        len(left_fa.get("applicable_rules") or []),
        len(left_fa.get("conditions_and_exceptions") or []),
        len(left_fa.get("required_actions") or []),
    )
    right_counts = (
        len(right_fa.get("applicable_rules") or []),
        len(right_fa.get("conditions_and_exceptions") or []),
        len(right_fa.get("required_actions") or []),
    )

    lines = [
        "| Metric | LLM + Citations | Retrieval Only |",
        "|---|---|---|",
        f"| Requested mode | `{left.requested_mode}` | `{right.requested_mode}` |",
        f"| Runtime provider | `{left.runtime_provider}` | `{right.runtime_provider}` |",
        f"| LLM used | `{left.llm_used}` | `{right.llm_used}` |",
        f"| Fallback used | `{left.fallback_used}` | `{right.fallback_used}` |",
        f"| Verdict | `{left_fa.get('verdict')}` | `{right_fa.get('verdict')}` |",
        f"| Citations returned | `{left_citations}` | `{right_citations}` |",
        f"| Section counts (rules/conditions/actions) | `{left_counts[0]} / {left_counts[1]} / {left_counts[2]}` | `{right_counts[0]} / {right_counts[1]} / {right_counts[2]}` |",
        f"| Risk flags / clarifications | `{len(left_fa.get('risk_flags') or [])} / {len(left_fa.get('clarifications_needed') or [])}` | `{len(right_fa.get('risk_flags') or [])} / {len(right_fa.get('clarifications_needed') or [])}` |",
    ]
    return "\n".join(lines)


def _render_mode_section(run: ModeRun) -> str:
    lines: list[str] = []
    lines.append(f"#### Mode: `{run.requested_mode}`")
    lines.append(f"- Runtime provider: `{run.runtime_provider}`")
    lines.append(f"- LLM used: `{run.llm_used}`")
    lines.append(f"- Fallback used: `{run.fallback_used}`")
    if run.provider_diagnostics:
        lines.append("- Provider diagnostics:")
        lines.extend(f"  - {item}" for item in run.provider_diagnostics)
    if run.error:
        lines.append("- Error:")
        lines.append(_text_block(run.error))
        if run.traceback_text:
            lines.append("- Traceback:")
            lines.append(_text_block(run.traceback_text))
        return "\n".join(lines)

    response = run.response or {}
    final_answer = response.get("final_answer") or {}
    citation_lookup = _citation_lookup(response)
    inline_links = run.runtime_provider == "openai_responses_llm"

    lines.append("")
    lines.append("##### Final Answer (Rendered)")
    lines.append(f"- Verdict: `{final_answer.get('verdict')}`")
    lines.append("- Short summary:")
    lines.append(_text_block(str(final_answer.get("short_summary") or "")))

    lines.append("- Applicable rules:")
    lines.extend(_render_claim_lines(final_answer.get("applicable_rules") or [], inline_links, citation_lookup))

    lines.append("- Conditions and exceptions:")
    lines.extend(_render_claim_lines(final_answer.get("conditions_and_exceptions") or [], inline_links, citation_lookup))

    lines.append("- Required actions:")
    lines.extend(_render_claim_lines(final_answer.get("required_actions") or [], inline_links, citation_lookup))

    lines.append("- Risk flags:")
    risk_flags = final_answer.get("risk_flags") or []
    if risk_flags:
        lines.extend(f"  - {item}" for item in risk_flags)
    else:
        lines.append("  - None")

    lines.append("- Clarifications needed:")
    clarifications = final_answer.get("clarifications_needed") or []
    if clarifications:
        lines.extend(f"  - {item}" for item in clarifications)
    else:
        lines.append("  - None")

    lines.append("")
    lines.append("##### Diagnostics: Citations")
    lines.append(_json_block(response.get("citations") or []))

    lines.append("")
    lines.append("##### Diagnostics: Evidence")
    lines.append(_json_block(response.get("evidence_matrix") or []))

    lines.append("")
    lines.append("##### Diagnostics: Grounding")
    lines.append(_json_block(response.get("grounding")))

    lines.append("")
    lines.append("##### Diagnostics: Latency")
    lines.append(_json_block(response.get("latency_ms") or {}))

    lines.append("")
    lines.append("##### Diagnostics: Trace")
    tool_steps = sorted(
        {
            str(event.get("step", ""))
            for event in run.events
            if str(event.get("step", "")).startswith("tool.")
        }
    )
    lines.append("- Tools used:")
    if tool_steps:
        lines.extend(f"  - `{step}`" for step in tool_steps)
    else:
        lines.append("  - None")
    lines.append("- Agent trace:")
    lines.append(_json_block(response.get("agent_trace") or []))
    lines.append("- Pipeline events:")
    lines.append(_json_block(run.events))

    lines.append("")
    lines.append("##### Raw Response JSON")
    lines.append(_json_block(response))
    return "\n".join(lines)


def _write_markdown(
    output_md: Path,
    questions: list[str],
    runs_by_question: dict[str, dict[str, ModeRun]],
    modes: list[str],
    state: str,
    jurisdiction: str,
    panchayat_category: str | None,
    top_k: int,
    debug_trace: bool,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append("# QnA Compare: LLM + Citations vs Retrieval Only")
    lines.append("")
    lines.append("Generated by `scripts/compare_qna_modes.py`.")
    lines.append("")
    lines.append("Run context:")
    lines.append(f"- Generated at: {ts}")
    lines.append(f"- State: `{state}`")
    lines.append(f"- Jurisdiction: `{jurisdiction}`")
    lines.append(f"- Panchayat category: `{panchayat_category}`")
    lines.append(f"- `top_k={top_k}`, `debug_trace={str(debug_trace).lower()}`")
    lines.append(f"- Modes: {', '.join(f'`{m}`' for m in modes)}")
    lines.append("")

    for idx, question in enumerate(questions, start=1):
        lines.append(f"## Q{idx}. {question}")
        lines.append("")
        mode_runs = runs_by_question.get(question, {})
        llm_run = mode_runs.get("openai_responses_llm")
        no_llm_run = mode_runs.get("no_llm")
        if llm_run and no_llm_run:
            lines.append(_mode_metrics_table(llm_run, no_llm_run))
            lines.append("")
        for mode in modes:
            mode_run = mode_runs.get(mode)
            if mode_run is None:
                lines.append(f"#### Mode: `{mode}`")
                lines.append("- Error:")
                lines.append(_text_block("Mode run missing from results."))
            else:
                lines.append(_render_mode_section(mode_run))
            lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PlotMagic QnA outputs side-by-side across LLM and retrieval-only modes."
    )
    parser.add_argument("--question", action="append", default=[], help="Question text. Repeatable.")
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Questions input file (.txt/.json/.jsonl). If omitted and no --question is provided, defaults are used.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["openai_responses_llm", "no_llm"],
        help="Modes to compare.",
    )
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="panchayat")
    parser.add_argument("--panchayat-category", default="Category-II")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--debug-trace", action="store_true", default=True)
    parser.add_argument("--no-debug-trace", dest="debug_trace", action="store_false")
    parser.add_argument("--output-md", default="qna_compare.md")
    parser.add_argument("--output-json", default="qna_compare.json")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--override-env", action="store_true", default=False)
    args = parser.parse_args()

    _load_env_file(Path(args.env_file), override_existing=args.override_env)
    questions = _parse_questions(args.question, args.questions_file)
    modes = [str(mode).strip() for mode in args.modes if str(mode).strip()]
    if not modes:
        raise ValueError("At least one mode is required.")

    engines: dict[str, ComplianceEngine] = {}
    for mode in modes:
        engines[mode] = _build_engine(mode)

    runs_by_question: dict[str, dict[str, ModeRun]] = {question: {} for question in questions}
    all_runs: list[ModeRun] = []
    for question in questions:
        for mode in modes:
            engine = engines[mode]
            runtime_provider = str(getattr(getattr(engine, "llm_provider", None), "provider_id", mode))
            provider_diagnostics = list(getattr(engine, "provider_diagnostics", []))
            request = QueryRequest(
                query=question,
                state=args.state,
                jurisdiction_type=args.jurisdiction,
                panchayat_category=(args.panchayat_category if args.jurisdiction == "panchayat" else None),
                top_k=args.top_k,
                debug_trace=args.debug_trace,
            )

            try:
                response, events = _run_with_events(engine, request)
                response_dict = response.model_dump()
                latency = response_dict.get("latency_ms") or {}
                llm_used = float(latency.get("llm_used", 0.0))
                fallback_used = float(latency.get("llm_fallback_used", 0.0))
                run = ModeRun(
                    question=question,
                    requested_mode=mode,
                    runtime_provider=runtime_provider,
                    provider_diagnostics=provider_diagnostics,
                    llm_used=llm_used,
                    fallback_used=fallback_used,
                    response=response_dict,
                    events=events,
                    error=None,
                    traceback_text=None,
                )
            except Exception as exc:  # pragma: no cover - runtime/debug utility
                run = ModeRun(
                    question=question,
                    requested_mode=mode,
                    runtime_provider=runtime_provider,
                    provider_diagnostics=provider_diagnostics,
                    llm_used=0.0,
                    fallback_used=0.0,
                    response=None,
                    events=[],
                    error=str(exc),
                    traceback_text=traceback.format_exc(),
                )

            runs_by_question[question][mode] = run
            all_runs.append(run)

    output_md = Path(args.output_md)
    _write_markdown(
        output_md=output_md,
        questions=questions,
        runs_by_question=runs_by_question,
        modes=modes,
        state=args.state,
        jurisdiction=args.jurisdiction,
        panchayat_category=(args.panchayat_category if args.jurisdiction == "panchayat" else None),
        top_k=args.top_k,
        debug_trace=args.debug_trace,
    )

    output_json = Path(args.output_json)
    output_json_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state": args.state,
        "jurisdiction": args.jurisdiction,
        "panchayat_category": args.panchayat_category if args.jurisdiction == "panchayat" else None,
        "top_k": args.top_k,
        "debug_trace": args.debug_trace,
        "modes": modes,
        "questions": questions,
        "runs": [
            {
                "question": run.question,
                "requested_mode": run.requested_mode,
                "runtime_provider": run.runtime_provider,
                "provider_diagnostics": run.provider_diagnostics,
                "llm_used": run.llm_used,
                "fallback_used": run.fallback_used,
                "error": run.error,
                "traceback": run.traceback_text,
                "events": run.events,
                "response": run.response,
            }
            for run in all_runs
        ],
    }
    output_json.write_text(json.dumps(output_json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote markdown comparison: {output_md}")
    print(f"Wrote json comparison: {output_json}")
    print(f"Questions: {len(questions)}  Modes: {len(modes)}  Runs: {len(all_runs)}")


if __name__ == "__main__":
    main()
