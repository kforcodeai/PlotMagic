from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine
from src.models.schemas import QueryRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Query PlotMagic from CLI.")
    parser.add_argument("--query", help="Query text. If omitted, interactive mode starts.")
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--location", default=None)
    parser.add_argument("--jurisdiction", default=None, choices=["municipality", "panchayat"])
    parser.add_argument("--category", default=None, help="Panchayat category: Category-I or Category-II")
    parser.add_argument("--occupancy", default=None, help="Explicit occupancy code, e.g., A1, F")
    parser.add_argument("--top-k", default=12, type=int)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/query")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Query directly in-process instead of hitting localhost API.",
    )
    args = parser.parse_args()

    if args.query:
        run_once(args, args.query)
        return

    interactive_loop(args)


def interactive_loop(args: argparse.Namespace) -> None:
    print("PlotMagic CLI interactive mode. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except EOFError:
            print()
            return
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return
        run_once(args, query)


def run_once(args: argparse.Namespace, query: str) -> None:
    if args.direct:
        response = query_direct(args, query)
    else:
        response = query_api(args, query)

    print("\n=== PlotMagic Response ===")
    print(json.dumps(response, indent=2, ensure_ascii=False))


def query_direct(args: argparse.Namespace, query: str) -> dict[str, object]:
    engine = ComplianceEngine(ROOT)
    engine.ingest(state=args.state, jurisdiction_type="municipality")
    engine.ingest(state=args.state, jurisdiction_type="panchayat")
    request = QueryRequest(
        query=query,
        state=args.state,
        location=args.location,
        jurisdiction_type=args.jurisdiction,
        panchayat_category=args.category,
        explicit_occupancy=args.occupancy,
        top_k=args.top_k,
    )
    response = engine.query(request)
    return response.model_dump()


def query_api(args: argparse.Namespace, query: str) -> dict[str, object]:
    payload = {
        "query": query,
        "state": args.state,
        "location": args.location,
        "jurisdiction_type": args.jurisdiction,
        "panchayat_category": args.category,
        "explicit_occupancy": args.occupancy,
        "top_k": args.top_k,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.URLError as exc:
        message = (
            "Failed to reach API. Start server first with:\n"
            "  .venv/bin/python scripts/run_local.py --warm-ingest\n"
            f"Error: {exc}"
        )
        raise SystemExit(message) from exc


if __name__ == "__main__":
    main()

