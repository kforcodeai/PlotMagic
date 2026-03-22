from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PlotMagic API locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    parser.add_argument("--state", default="kerala", help="State used for warm-ingest and preflight evaluation.")
    parser.add_argument(
        "--warm-ingest",
        action="store_true",
        help="Ingest Kerala municipality and panchayat before starting server.",
    )
    parser.add_argument(
        "--preflight-eval",
        action="store_true",
        help="Run evaluation gate before startup (recommended for canary/promotion).",
    )
    parser.add_argument(
        "--gold-file",
        default="tests/evaluation/gold_queries_kerala.json",
        help="Gold dataset for preflight eval.",
    )
    parser.add_argument(
        "--eval-output",
        default="evaluation/preflight_eval.json",
        help="Output artifact for preflight eval.",
    )
    parser.add_argument(
        "--baseline-artifact",
        default=None,
        help="Optional baseline artifact for regression checks during preflight eval.",
    )
    parser.add_argument(
        "--shadow-legacy",
        action="store_true",
        help="Run legacy-provider shadow comparison during preflight eval.",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Abort startup when preflight evaluation gate fails.",
    )
    parser.add_argument(
        "--canary-percent",
        default=100,
        type=int,
        help="Traffic slice percentage to annotate for canary deployments.",
    )
    args = parser.parse_args()

    if args.canary_percent < 1 or args.canary_percent > 100:
        raise SystemExit("--canary-percent must be between 1 and 100")
    os.environ["PLOTMAGIC_CANARY_PERCENT"] = str(args.canary_percent)

    if args.preflight_eval:
        eval_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--state",
            args.state,
            "--gold-file",
            args.gold_file,
            "--output",
            args.eval_output,
        ]
        if args.baseline_artifact:
            eval_cmd.extend(["--baseline-artifact", args.baseline_artifact])
        if args.shadow_legacy:
            eval_cmd.append("--shadow-legacy")
        if args.fail_on_gate:
            eval_cmd.append("--fail-on-gate")
        subprocess.run(eval_cmd, check=True, cwd=ROOT)

    if args.warm_ingest:
        engine = ComplianceEngine(ROOT)
        print(f"Warming indexes for state={args.state}: municipality...")
        engine.ingest(state=args.state, jurisdiction_type="municipality")
        print("Warming indexes: panchayat...")
        engine.ingest(state=args.state, jurisdiction_type="panchayat")
        print("Warm ingest complete.")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
