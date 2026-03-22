from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class ScopeResolution:
    resolved: bool
    state: str | None
    jurisdiction_type: str | None
    ruleset_id: str | None
    local_body: str | None = None
    local_body_type: str | None = None
    panchayat_category: str | None = None
    reasons: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)


class ScopeResolver:
    def __init__(self, states_config_path: Path, local_bodies_path: Path) -> None:
        self.states_cfg = yaml.safe_load(states_config_path.read_text(encoding="utf-8"))
        self.local_bodies_cfg = yaml.safe_load(local_bodies_path.read_text(encoding="utf-8"))

    def resolve(
        self,
        location: str | None,
        state_hint: str | None = None,
        jurisdiction_hint: str | None = None,
        panchayat_category_hint: str | None = None,
    ) -> ScopeResolution:
        state = (state_hint or self.local_bodies_cfg.get("state") or "").lower() or None
        if not state:
            return ScopeResolution(
                resolved=False,
                state=None,
                jurisdiction_type=None,
                ruleset_id=None,
                clarification_questions=["Which state is this property located in?"],
            )
        if state not in self.states_cfg["states"]:
            return ScopeResolution(
                resolved=False,
                state=state,
                jurisdiction_type=None,
                ruleset_id=None,
                clarification_questions=[f"State '{state}' is not yet configured in this deployment."],
            )

        if location:
            location_key = location.strip().lower()
            match = self._match_local_body(location_key)
            if match:
                jurisdiction = match["jurisdiction_type"]
                if jurisdiction_hint and jurisdiction_hint.lower().strip() != jurisdiction:
                    return ScopeResolution(
                        resolved=False,
                        state=state,
                        jurisdiction_type=None,
                        ruleset_id=None,
                        clarification_questions=[
                            f"Location '{location}' maps to '{jurisdiction}', but query requested '{jurisdiction_hint}'. Please confirm jurisdiction."
                        ],
                    )
                ruleset_id = self.states_cfg["states"][state]["jurisdictions"][jurisdiction]["ruleset_id"]
                category = panchayat_category_hint or match.get("panchayat_category")
                if jurisdiction == "panchayat" and not category:
                    return ScopeResolution(
                        resolved=False,
                        state=state,
                        jurisdiction_type=jurisdiction,
                        ruleset_id=ruleset_id,
                        local_body=match["canonical_name"],
                        local_body_type=match.get("local_body_type"),
                        clarification_questions=["Is this Category-I or Category-II village panchayat?"],
                    )

                return ScopeResolution(
                    resolved=True,
                    state=state,
                    jurisdiction_type=jurisdiction,
                    ruleset_id=ruleset_id,
                    local_body=match["canonical_name"],
                    local_body_type=match.get("local_body_type"),
                    panchayat_category=category,
                    reasons=[f"Matched location '{location}' to local body '{match['canonical_name']}'."],
                )

        if jurisdiction_hint:
            return self._resolve_from_jurisdiction_hint(state, jurisdiction_hint, panchayat_category_hint)

        if not location:
            return ScopeResolution(
                resolved=False,
                state=state,
                jurisdiction_type=None,
                ruleset_id=None,
                clarification_questions=[
                    "Provide city/town/village name to deterministically resolve municipality vs panchayat.",
                ],
            )
        location_key = location.strip().lower()
        match = self._match_local_body(location_key)
        if not match:
            return ScopeResolution(
                resolved=False,
                state=state,
                jurisdiction_type=None,
                ruleset_id=None,
                clarification_questions=[
                    f"Could not map '{location}' to a known local body. Is it municipality/corporation or panchayat?",
                ],
            )

        jurisdiction = match["jurisdiction_type"]
        ruleset_id = self.states_cfg["states"][state]["jurisdictions"][jurisdiction]["ruleset_id"]
        category = match.get("panchayat_category")
        if jurisdiction == "panchayat" and not category and not panchayat_category_hint:
            return ScopeResolution(
                resolved=False,
                state=state,
                jurisdiction_type=jurisdiction,
                ruleset_id=ruleset_id,
                local_body=match["canonical_name"],
                local_body_type=match.get("local_body_type"),
                clarification_questions=["Is this Category-I or Category-II village panchayat?"],
            )

        return ScopeResolution(
            resolved=True,
            state=state,
            jurisdiction_type=jurisdiction,
            ruleset_id=ruleset_id,
            local_body=match["canonical_name"],
            local_body_type=match.get("local_body_type"),
            panchayat_category=panchayat_category_hint or category,
            reasons=[f"Matched location '{location}' to local body '{match['canonical_name']}'."],
        )

    def _resolve_from_jurisdiction_hint(
        self,
        state: str,
        jurisdiction_hint: str,
        panchayat_category_hint: str | None,
    ) -> ScopeResolution:
        jurisdiction = jurisdiction_hint.lower().strip()
        if jurisdiction not in self.states_cfg["states"][state]["jurisdictions"]:
            return ScopeResolution(
                resolved=False,
                state=state,
                jurisdiction_type=None,
                ruleset_id=None,
                clarification_questions=[f"Unsupported jurisdiction '{jurisdiction_hint}' for state '{state}'."],
            )
        ruleset_id = self.states_cfg["states"][state]["jurisdictions"][jurisdiction]["ruleset_id"]
        if jurisdiction == "panchayat" and not panchayat_category_hint:
            return ScopeResolution(
                resolved=False,
                state=state,
                jurisdiction_type=jurisdiction,
                ruleset_id=ruleset_id,
                clarification_questions=["Provide panchayat category (Category-I or Category-II)."],
            )
        return ScopeResolution(
            resolved=True,
            state=state,
            jurisdiction_type=jurisdiction,
            ruleset_id=ruleset_id,
            panchayat_category=panchayat_category_hint,
            reasons=["Used explicit jurisdiction hint."],
        )

    def _match_local_body(self, location_key: str) -> dict[str, str] | None:
        for local_body in self.local_bodies_cfg.get("local_bodies", []):
            aliases = [alias.lower() for alias in local_body.get("aliases", [])]
            if location_key in aliases:
                return local_body
            for alias in aliases:
                if alias and alias in location_key:
                    return local_body
        return None

