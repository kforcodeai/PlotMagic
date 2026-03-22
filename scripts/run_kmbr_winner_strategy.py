from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.models.schemas import QueryRequest


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON array or JSONL stream.")
    return [item for item in payload if isinstance(item, dict)]


def _final_answer_text(payload: dict[str, Any]) -> str:
    final_answer = payload.get("final_answer") or {}
    if not isinstance(final_answer, dict):
        return ""
    lines: list[str] = []
    summary = str(final_answer.get("short_summary", "")).strip()
    if summary:
        lines.append(summary)
    for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
        section = final_answer.get(section_name, [])
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def _sorted_chunks(payload: dict[str, Any], max_chunks: int) -> list[dict[str, Any]]:
    evidence = payload.get("evidence_matrix", [])
    if not isinstance(evidence, list):
        return []
    chunks: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        score = float((item.get("scores") or {}).get("rrf_score", 0.0) or 0.0)
        chunks.append(
            {
                "claim_id": str(item.get("claim_id", "")),
                "topic": str(item.get("topic", "")),
                "document_id": str(item.get("document_id", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "rrf_score": score,
                "text": text,
            }
        )
    chunks.sort(key=lambda row: row["rrf_score"], reverse=True)
    return chunks[:max_chunks]


def run(args: argparse.Namespace) -> Path:
    dataset_path = (ROOT / args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")
    rows = _load_dataset(dataset_path)
    if not rows:
        raise SystemExit(f"No rows found in dataset: {dataset_path}")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"kmbr_winner_answers_with_chunks_{run_id}.json"
    jsonl_path = output_dir / f"kmbr_winner_answers_with_chunks_{run_id}.jsonl"

    engine = ComplianceEngine(ROOT)
    engine.ingest(state=args.state, jurisdiction_type=args.jurisdiction)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row.get("id", "")).strip()
        query = str(row.get("query", "")).strip()
        if not query_id or not query:
            continue
        request = QueryRequest(
            query=query,
            state=args.state,
            jurisdiction_type=args.jurisdiction,
            top_k=args.top_k,
            retrieval_mode="lexical_only_bm25",
            debug_trace=False,
        )
        response = engine.query(request)
        payload = response.model_dump()

        final_answer = payload.get("final_answer") if isinstance(payload.get("final_answer"), dict) else {}
        out_rows.append(
            {
                "id": query_id,
                "query": query,
                "verdict": payload.get("verdict"),
                "final_answer_text": _final_answer_text(payload),
                "final_answer": final_answer,
                "retrieved_chunks": _sorted_chunks(payload, max_chunks=args.max_chunks),
                "citations": payload.get("citations", []),
            }
        )

    artifact = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "state": args.state,
        "jurisdiction": args.jurisdiction,
        "retrieval_mode": "lexical_only_bm25",
        "synthesis_mode": "deterministic_draft_only",
        "top_k": args.top_k,
        "query_count": len(out_rows),
        "rows": out_rows,
    }
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in out_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "query_count": len(out_rows),
                "json": str(json_path),
                "jsonl": str(jsonl_path),
            },
            indent=2,
        )
    )
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run winning KMBR strategy and export answers with retrieved chunks.")
    parser.add_argument("--dataset", default="evaluation/kmbr/kmbr_multihop_retrieval_dataset.jsonl")
    parser.add_argument("--output-dir", default="evaluation/kmbr")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="municipality")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-chunks", type=int, default=10)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
