from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PlotMagic ingestion pipeline.")
    parser.add_argument("--state", required=True, help="State code, e.g. kerala")
    parser.add_argument("--jurisdiction", default=None, help="Optional jurisdiction: municipality/panchayat")
    args = parser.parse_args()

    engine = ComplianceEngine(root=ROOT)
    result = engine.ingest(state=args.state, jurisdiction_type=args.jurisdiction)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()

