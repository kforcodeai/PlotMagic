from __future__ import annotations

from pathlib import Path

from src.statepack.base import StatePack


def kerala_statepack(root: Path) -> StatePack:
    return StatePack(
        state_code="kerala",
        parser_registry={
            "municipality": "KMBRHTMLParser",
            "panchayat": "KPBRMarkdownParser",
        },
        scope_resolver_config=root / "config" / "local_bodies" / "kerala_local_bodies.yaml",
        occupancy_mapping_config=root / "config" / "states.yaml",
        precedence_policy={
            "global": [
                "effective_version",
                "specific_over_generic",
                "most_restrictive_for_mixed_use",
                "proviso_over_parent_clause",
            ]
        },
        normalizer_rules={
            "Group AI": "Group A1",
            "Group GI": "Group G1",
            "Group 1(1)": "Group I(1)",
            "Group 1(2)": "Group I(2)",
        },
        tests_path=root / "tests" / "statepacks" / "kerala",
    )

