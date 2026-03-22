from __future__ import annotations

from pathlib import Path

import yaml

from src.statepack.base import StatePack


class StatePackFactory:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, state_code: str) -> StatePack:
        normalized = state_code.strip().lower()
        manifest = self.root / "data" / "statepacks" / f"{normalized}_statepack.yaml"
        if not manifest.exists():
            raise FileNotFoundError(f"StatePack manifest not found: {manifest}")
        cfg = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        raw_profiles = cfg.get("policy_profiles", {})
        policy_profiles: dict[str, Path] = {}
        if isinstance(raw_profiles, dict):
            for profile_name, profile_path in raw_profiles.items():
                policy_profiles[str(profile_name)] = self.root / str(profile_path)
        return StatePack(
            state_code=cfg["state_code"],
            parser_registry=cfg["parser_registry"],
            scope_resolver_config=self.root / cfg["scope_resolver_config"],
            occupancy_mapping_config=self.root / cfg["occupancy_mapping_config"],
            precedence_policy={"global": cfg.get("precedence_policy", [])},
            policy_profiles=policy_profiles,
            normalizer_rules=cfg.get("normalizer_rules", {}),
            tests_path=self.root / cfg["tests_path"] if cfg.get("tests_path") else None,
        )
