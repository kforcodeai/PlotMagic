from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.multihop_eval import run_multihop_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KPBR multi-hop (20) retrieval + synthesis pipeline.")
    parser.add_argument(
        "--dataset",
        default="evaluation/kpbr/kpbr_multihop_retrieval_dataset_20_enriched.jsonl",
    )
    parser.add_argument("--tier", choices=["core", "exhaustive"], default="core")
    parser.add_argument("--state", default="kerala")
    parser.add_argument("--jurisdiction", default="panchayat")
    parser.add_argument("--category", default="Category-II")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--strict-providers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vector-backend", choices=["qdrant_local"], default="qdrant_local")
    parser.add_argument("--vector-db-path", default=".cache/qdrant_kpbr")
    parser.add_argument("--embedding-provider", choices=["openai_embedding"], default="openai_embedding")
    parser.add_argument("--output-dir", default="evaluation/kpbr")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    dataset_path = (ROOT / args.dataset).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    vector_db_path = (ROOT / args.vector_db_path).resolve()
    vector_db_path.mkdir(parents=True, exist_ok=True)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = run_multihop_evaluation(
        root=ROOT,
        dataset_path=dataset_path,
        output_dir=output_dir,
        run_id=run_id,
        tier=args.tier,
        state=args.state,
        jurisdiction=args.jurisdiction,
        category=args.category,
        top_k=args.top_k,
        strict_providers=args.strict_providers,
        vector_backend=args.vector_backend,
        vector_db_path=vector_db_path,
        embedding_provider=args.embedding_provider,
    )

    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "output_dir": result["output_dir"],
                "winner_report": str(Path(result["output_dir"]) / "winner_report.json"),
                "best_retrieval_strategy": result["winner_report"]["best_retrieval_strategy"],
                "best_synthesis_strategy": result["winner_report"]["best_synthesis_strategy"],
                "best_end_to_end_strategy_or_no_winner": result["winner_report"]["best_end_to_end_strategy_or_no_winner"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
