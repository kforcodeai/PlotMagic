from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from src.models import RuleDocument


class RulesetParser(ABC):
    def __init__(
        self,
        state: str,
        jurisdiction_type: str,
        ruleset_id: str,
        ruleset_version: str,
        *,
        issuing_authority: str | None = None,
        default_effective_from: date | None = None,
    ) -> None:
        self.state = state
        self.jurisdiction_type = jurisdiction_type
        self.ruleset_id = ruleset_id
        self.ruleset_version = ruleset_version
        self.issuing_authority = issuing_authority
        self.default_effective_from = default_effective_from

    @abstractmethod
    def parse_file(self, file_path: Path) -> list[RuleDocument]:
        raise NotImplementedError

    def parse_paths(self, file_paths: list[Path]) -> list[RuleDocument]:
        documents: list[RuleDocument] = []
        for path in sorted(file_paths):
            documents.extend(self.parse_file(path))
        return documents
