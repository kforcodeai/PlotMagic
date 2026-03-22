from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import re
from typing import Any

import yaml

from src.ingestion.parsers import GenericLegalParser, KMBRHTMLParser, KPBRMarkdownParser, RulesetParser
from src.models import RuleDocument

_SUPPORTED_SOURCE_EXTENSIONS = {".html", ".htm", ".md", ".markdown", ".txt", ".pdf"}


@dataclass(slots=True)
class PipelineStats:
    parsed_rules: int = 0
    parsed_clauses: int = 0
    parsed_tables: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    parse_quality_score: float | None = None
    parse_quality: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self._load_yaml(config_path)
        self.root = config_path.resolve().parents[1]

    def ingest_state(self, state: str, jurisdiction_type: str | None = None) -> tuple[list[RuleDocument], PipelineStats]:
        state_key = state.strip().lower()
        states_cfg = self.config.get("states", {})
        if state_key not in states_cfg:
            raise ValueError(f"State '{state}' is not configured.")
        state_cfg = states_cfg[state_key]
        jurisdictions = state_cfg["jurisdictions"]
        selected = [jurisdiction_type.strip().lower()] if jurisdiction_type else list(jurisdictions.keys())
        docs_by_id: dict[str, RuleDocument] = {}
        stats = PipelineStats()

        for jurisdiction in selected:
            if jurisdiction not in jurisdictions:
                stats.warnings.append(f"Unknown jurisdiction '{jurisdiction}' for state '{state_key}'.")
                continue
            j_cfg = jurisdictions[jurisdiction]
            parser = self._build_parser(
                state=state_key,
                jurisdiction_type=jurisdiction,
                ruleset_id=j_cfg["ruleset_id"],
                ruleset_version=str(j_cfg.get("ruleset_version", "")),
                parser_class=str(j_cfg.get("parser_class", "GenericLegalParser")),
                issuing_authority=j_cfg.get("issuing_authority"),
                default_effective_from=self._parse_optional_date(j_cfg.get("effective_from")),
            )
            source_path = self._resolve_path(Path(str(j_cfg["source_path"])))
            paths = self._resolve_source_paths(
                source_path,
                source_glob=j_cfg.get("source_glob"),
                source_format=j_cfg.get("source_format"),
            )
            if not paths:
                stats.warnings.append(f"No source files found for {state_key}:{jurisdiction} at '{source_path}'.")
                continue
            for path in paths:
                try:
                    docs = parser.parse_file(path)
                except Exception as exc:
                    stats.failed_files += 1
                    stats.warnings.append(f"Failed to parse '{path}': {exc}")
                    continue
                stats.parsed_files += 1
                for doc in docs:
                    docs_by_id[doc.document_id] = doc
                stats.parsed_rules += len(docs)
                stats.parsed_clauses += sum(len(doc.clause_nodes) for doc in docs)
                stats.parsed_tables += sum(len(doc.tables) for doc in docs)

        all_docs = list(docs_by_id.values())
        self._run_qc(all_docs, stats)
        return all_docs, stats

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def _build_parser(
        self,
        state: str,
        jurisdiction_type: str,
        ruleset_id: str,
        ruleset_version: str,
        parser_class: str,
        issuing_authority: str | None = None,
        default_effective_from: date | None = None,
    ) -> RulesetParser:
        parser_map = {
            "KMBRHTMLParser": KMBRHTMLParser,
            "KPBRMarkdownParser": KPBRMarkdownParser,
            "GenericLegalParser": GenericLegalParser,
        }
        if parser_class not in parser_map:
            raise ValueError(f"Unsupported parser class '{parser_class}'")
        return parser_map[parser_class](
            state=state,
            jurisdiction_type=jurisdiction_type,
            ruleset_id=ruleset_id,
            ruleset_version=ruleset_version,
            issuing_authority=issuing_authority,
            default_effective_from=default_effective_from,
        )

    def _resolve_source_paths(
        self,
        source: Path,
        *,
        source_glob: str | None = None,
        source_format: str | None = None,
    ) -> list[Path]:
        if source.is_file():
            return [source]
        if source.is_dir():
            if source_glob:
                paths = sorted(source.glob(source_glob))
            else:
                paths = sorted(source.rglob("*"))
            filtered = [path for path in paths if path.is_file() and path.suffix.lower() in _SUPPORTED_SOURCE_EXTENSIONS]
            if source_format:
                format_hint = source_format.strip().lower()
                if format_hint in {"html", "htm"}:
                    filtered = [path for path in filtered if path.suffix.lower() in {".html", ".htm"}]
                elif format_hint in {"markdown", "md"}:
                    filtered = [path for path in filtered if path.suffix.lower() in {".md", ".markdown"}]
                elif format_hint in {"text", "txt"}:
                    filtered = [path for path in filtered if path.suffix.lower() == ".txt"]
                elif format_hint == "pdf":
                    filtered = [path for path in filtered if path.suffix.lower() == ".pdf"]
            return filtered
        return []

    def _run_qc(self, docs: list[RuleDocument], stats: PipelineStats) -> None:
        seen_ids: set[str] = set()
        severe_anomalies: list[str] = []
        table_header_quality_total = 0
        sub_rule_count = 0
        clause_id_counts: dict[str, int] = {}
        noisy_clause_count = 0
        total_clause_count = 0
        for doc in docs:
            if not doc.rule_number:
                stats.warnings.append(f"Missing rule number in {doc.source_file}")
            if doc.document_id in seen_ids:
                severe_anomalies.append(f"Duplicate document_id detected: {doc.document_id}")
            seen_ids.add(doc.document_id)
            if not doc.anchor_id:
                stats.warnings.append(f"Missing anchor for {doc.document_id}")
            if not doc.full_text.strip():
                severe_anomalies.append(f"Document has empty text: {doc.document_id}")
            if not doc.clause_nodes:
                stats.warnings.append(f"No clause nodes emitted for {doc.document_id}")
            for note in doc.notes:
                if "low-confidence text extraction" in note.lower():
                    stats.warnings.append(f"{doc.document_id}: {note}")
            for table in doc.tables:
                if not table.headers:
                    stats.warnings.append(f"Table with no headers in {doc.document_id}")
                if table.headers and table.headers != ["raw_row"]:
                    table_header_quality_total += 1
            sub_rule_count += sum(1 for clause in doc.clause_nodes if clause.clause_type.value == "sub_rule")
            for clause in doc.clause_nodes:
                total_clause_count += 1
                clause_id_counts[clause.clause_id] = clause_id_counts.get(clause.clause_id, 0) + 1
                if self._is_noisy_clause_text(clause.normalized_text):
                    noisy_clause_count += 1

        if not docs:
            severe_anomalies.append("No documents parsed. Validate source paths and parser patterns.")

        total_files = max(1, stats.parsed_files + stats.failed_files)
        failed_file_ratio = stats.failed_files / float(total_files)
        avg_clause_per_rule = stats.parsed_clauses / float(max(1, stats.parsed_rules))
        table_header_coverage = (
            table_header_quality_total / float(max(1, stats.parsed_tables)) if stats.parsed_tables > 0 else 1.0
        )
        sub_rule_ratio = sub_rule_count / float(max(1, stats.parsed_clauses))
        clause_collision_count = sum(1 for count in clause_id_counts.values() if count > 1)
        clause_noise_ratio = noisy_clause_count / float(max(1, total_clause_count))
        stats.parse_quality = {
            "failed_file_ratio": failed_file_ratio,
            "avg_clause_per_rule": avg_clause_per_rule,
            "table_header_coverage": table_header_coverage,
            "sub_rule_ratio": sub_rule_ratio,
            "clause_id_collision_count": float(clause_collision_count),
            "clause_noise_ratio": clause_noise_ratio,
        }

        thresholds = self._qc_thresholds()
        requires_sub_rules = any(str(doc.ruleset_id).upper().startswith("KPBR") for doc in docs)
        min_sub_rule_ratio_required = thresholds["min_sub_rule_ratio"] if requires_sub_rules else 0.0
        if failed_file_ratio > thresholds["max_failed_file_ratio"]:
            severe_anomalies.append(
                f"Failed file ratio {failed_file_ratio:.3f} exceeds threshold {thresholds['max_failed_file_ratio']:.3f}."
            )
        if avg_clause_per_rule < thresholds["min_clause_per_rule"]:
            severe_anomalies.append(
                f"Average clauses/rule {avg_clause_per_rule:.3f} below threshold {thresholds['min_clause_per_rule']:.3f}."
            )
        if table_header_coverage < thresholds["min_table_header_coverage"]:
            severe_anomalies.append(
                f"Table header coverage {table_header_coverage:.3f} below threshold {thresholds['min_table_header_coverage']:.3f}."
            )
        if min_sub_rule_ratio_required > 0 and sub_rule_ratio < min_sub_rule_ratio_required:
            severe_anomalies.append(
                f"Sub-rule ratio {sub_rule_ratio:.3f} below threshold {min_sub_rule_ratio_required:.3f}."
            )
        if clause_collision_count > 0:
            severe_anomalies.append(f"Clause ID collision count {clause_collision_count} exceeds threshold 0.")
        if clause_noise_ratio > thresholds["max_clause_noise_ratio"]:
            severe_anomalies.append(
                f"Clause noise ratio {clause_noise_ratio:.3f} exceeds threshold {thresholds['max_clause_noise_ratio']:.3f}."
            )

        sub_rule_component = 1.0
        if min_sub_rule_ratio_required > 0:
            sub_rule_component = min(1.0, sub_rule_ratio / max(1e-6, min_sub_rule_ratio_required))

        stats.parse_quality_score = max(
            0.0,
            min(
                1.0,
                (
                    (1.0 - failed_file_ratio)
                    + min(1.0, avg_clause_per_rule / max(1e-6, thresholds["min_clause_per_rule"]))
                    + min(1.0, table_header_coverage / max(1e-6, thresholds["min_table_header_coverage"]))
                    + sub_rule_component
                    + max(0.0, 1.0 - (clause_noise_ratio / max(1e-6, thresholds["max_clause_noise_ratio"])))
                )
                / 5.0,
            ),
        )

        if severe_anomalies:
            stats.warnings.extend(severe_anomalies)
            raise ValueError("Ingestion QC failed: " + "; ".join(severe_anomalies))

    def _qc_thresholds(self) -> dict[str, float]:
        raw = self.config.get("ingestion_qc", {})
        return {
            "max_failed_file_ratio": float(raw.get("max_failed_file_ratio", 0.10)),
            "min_clause_per_rule": float(raw.get("min_clause_per_rule", 1.0)),
            "min_table_header_coverage": float(raw.get("min_table_header_coverage", 0.20)),
            "min_sub_rule_ratio": float(raw.get("min_sub_rule_ratio", 0.01)),
            "max_clause_noise_ratio": float(raw.get("max_clause_noise_ratio", 0.20)),
        }

    def _is_noisy_clause_text(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        if len(stripped) < 18:
            return True
        if re.fullmatch(r"[-_=~`*.:| ]{3,}", stripped):
            return True
        alpha_count = sum(1 for ch in stripped if ch.isalpha())
        symbol_count = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())
        token_count = len(re.findall(r"[A-Za-z0-9]+", stripped))
        if token_count <= 2 and alpha_count < 8:
            return True
        return symbol_count / float(max(1, len(stripped))) > 0.40

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    def _parse_optional_date(self, raw: object) -> date | None:
        if raw is None:
            return None
        if isinstance(raw, date):
            return raw
        value = str(raw).strip()
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
