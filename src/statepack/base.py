from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class StatePack:
    state_code: str
    parser_registry: dict[str, str]
    scope_resolver_config: Path
    occupancy_mapping_config: Path
    precedence_policy: dict[str, list[str]]
    policy_profiles: dict[str, Path] = field(default_factory=dict)
    normalizer_rules: dict[str, str] = field(default_factory=dict)
    tests_path: Path | None = None

    def validate(self) -> list[str]:
        warnings: list[str] = []
        if not self.parser_registry:
            warnings.append("StatePack parser registry is empty.")
        if not self.scope_resolver_config.exists():
            warnings.append(f"Missing scope resolver config: {self.scope_resolver_config}")
        if not self.occupancy_mapping_config.exists():
            warnings.append(f"Missing occupancy mapping config: {self.occupancy_mapping_config}")
        for profile_name, profile_path in self.policy_profiles.items():
            if not profile_path.exists():
                warnings.append(f"Missing policy profile '{profile_name}': {profile_path}")
        if not self.precedence_policy:
            warnings.append("Missing precedence policy.")
        return warnings
