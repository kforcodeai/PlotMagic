from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statepack import kerala_statepack


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate StatePack configuration.")
    parser.add_argument("--state", default="kerala")
    args = parser.parse_args()

    if args.state != "kerala":
        raise ValueError("Only kerala statepack is currently implemented.")

    pack = kerala_statepack(ROOT)
    warnings = pack.validate()
    print(
        json.dumps(
            {
                "state": pack.state_code,
                "valid": not warnings,
                "warnings": warnings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

