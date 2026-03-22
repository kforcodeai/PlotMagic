from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml


@dataclass(slots=True)
class OccupancyCandidate:
    code: str
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OccupancyResolution:
    resolved: bool
    candidates: list[OccupancyCandidate] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)


class OccupancyResolver:
    def __init__(self, states_config_path: Path) -> None:
        config = yaml.safe_load(states_config_path.read_text(encoding="utf-8"))
        self.states = config["states"]

    def resolve(
        self,
        state: str,
        building_description: str | None,
        explicit_occupancy: str | None = None,
    ) -> OccupancyResolution:
        occupancy_cfg = self.states[state]["occupancy_groups"]
        if explicit_occupancy:
            normalized = explicit_occupancy.strip().upper()
            if normalized in occupancy_cfg:
                return OccupancyResolution(
                    resolved=True,
                    candidates=[OccupancyCandidate(code=normalized, score=1.0, reasons=["Explicit occupancy provided."])],
                    selected=[normalized],
                )

        if not building_description:
            return OccupancyResolution(
                resolved=False,
                clarification_questions=["Describe building intent (e.g., residential flat, hotel, school, hospital)."],
            )

        text = building_description.lower()
        explicit_groups = self._extract_explicit_groups(building_description)
        if explicit_groups:
            return OccupancyResolution(
                resolved=True,
                candidates=[OccupancyCandidate(code=code, score=1.0, reasons=["Explicit group code mentioned in query."]) for code in explicit_groups],
                selected=explicit_groups,
            )

        candidates: list[OccupancyCandidate] = []
        for code, cfg in occupancy_cfg.items():
            score = 0.0
            reasons: list[str] = []
            for keyword in cfg.get("keywords", []):
                if keyword.lower() in text:
                    score += 1.0
                    reasons.append(f"Matched keyword '{keyword}'.")
            # Light threshold logic for area-based differentiators between A1/A2.
            if code == "A2" and any(token in text for token in ["hotel", "hostel", "lodging"]):
                score += 0.5
                reasons.append("Hospitality intent pushes towards A2.")
            if code == "A1" and any(token in text for token in ["flat", "apartment", "dwelling", "residential"]):
                score += 0.5
                reasons.append("Residential intent pushes towards A1.")
            if score > 0:
                candidates.append(OccupancyCandidate(code=code, score=score, reasons=reasons))

        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        if not candidates:
            return OccupancyResolution(
                resolved=False,
                clarification_questions=["Could not classify occupancy. Choose one: A1, A2, B, C, D, E, F, G1, G2, H, I(1), I(2)."],
            )

        top_score = candidates[0].score
        selected = [candidate.code for candidate in candidates if candidate.score >= max(1.0, top_score - 0.25)]
        if len(selected) > 1:
            if self._is_mixed_use_request(text):
                return OccupancyResolution(
                    resolved=True,
                    candidates=candidates[:5],
                    selected=selected,
                )
            return OccupancyResolution(
                resolved=False,
                candidates=candidates[:3],
                selected=selected,
                clarification_questions=[f"Ambiguous occupancy: {', '.join(selected)}. Please choose one primary occupancy."],
            )

        if len(selected) == 1 and self._is_mixed_use_request(text):
            secondary = [candidate.code for candidate in candidates[1:] if candidate.score >= 1.0]
            if secondary:
                return OccupancyResolution(
                    resolved=True,
                    candidates=candidates[:5],
                    selected=[selected[0], *secondary],
                )

        return OccupancyResolution(resolved=True, candidates=candidates[:3], selected=selected)

    def _extract_explicit_groups(self, text: str) -> list[str]:
        normalized = text.upper().replace(" ", "")
        matches = re.findall(r"GROUP([A-Z]\(?\d?\)?)", normalized)
        if not matches:
            matches = re.findall(r"\b(A1|A2|B|C|D|E|F|G1|G2|H|I\(1\)|I\(2\))\b", text.upper())
        canonical: list[str] = []
        for match in matches:
            token = match
            if token == "I1":
                token = "I(1)"
            if token == "I2":
                token = "I(2)"
            if token in {"A1", "A2", "B", "C", "D", "E", "F", "G1", "G2", "H", "I(1)", "I(2)"}:
                canonical.append(token)
        return sorted(set(canonical))

    def _is_mixed_use_request(self, text: str) -> bool:
        hints = [
            "mixed use",
            "mixed-use",
            "more than one use",
            "more than one occupancy",
            "shops and apartments",
        ]
        return any(hint in text for hint in hints)

