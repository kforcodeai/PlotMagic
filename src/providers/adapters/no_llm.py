from __future__ import annotations

from typing import Any

from src.providers.base import LLMProvider, ProviderHealth
from src.providers.config import ProviderSettings


class NoLLMProvider(LLMProvider):
    provider_id = "no_llm"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings

    def generate_structured(
        self,
        *,
        task: str,
        payload: dict[str, Any],
        json_schema: dict[str, Any],
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> dict[str, Any]:
        if task != "compliance_brief":
            return payload.get("deterministic_output", {})
        draft = payload.get("draft", {})
        return {
            "verdict": str(draft.get("verdict", "depends")),
            "short_summary": str(draft.get("short_summary", "")),
            "applicable_rules": self._items(draft.get("applicable_rules", [])),
            "conditions_and_exceptions": self._items(draft.get("conditions_and_exceptions", [])),
            "required_actions": self._items(draft.get("required_actions", [])),
            "risk_flags": [str(item) for item in draft.get("risk_flags", [])],
            "clarifications_needed": [str(item) for item in draft.get("clarifications_needed", [])],
        }

    def _items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "claim_id": str(item.get("claim_id", "")),
                    "text": str(item.get("text", "")),
                    "citation_ids": [str(citation_id) for citation_id in item.get("citation_ids", []) if citation_id],
                }
            )
        return out

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            available=True,
            capabilities=["generate_structured"],
            details={"mode": "deterministic"},
        )
