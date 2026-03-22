from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class ClauseType(str, Enum):
    CHAPTER = "chapter"
    RULE = "rule"
    SUB_RULE = "sub_rule"
    PROVISO = "proviso"
    NOTE = "note"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    APPENDIX = "appendix"
    SCHEDULE = "schedule"
    DEFINITION = "definition"


class AmendmentAction(str, Enum):
    INSERTED = "inserted"
    SUBSTITUTED = "substituted"
    OMITTED = "omitted"
    ADDED = "added"
    OTHER = "other"


@dataclass(slots=True)
class AmendmentRecord:
    action: AmendmentAction
    reference_text: str
    sro_number: str | None = None
    gazette_reference: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    old_text: str | None = None
    new_text: str | None = None


@dataclass(slots=True)
class TableData:
    table_id: str
    caption: str
    headers: list[str]
    rows: list[list[str]]
    raw_text: str = ""


@dataclass(slots=True)
class CrossReference:
    source_clause_id: str
    target_ref: str
    target_type: str
    normalized_target_id: str | None = None


@dataclass(slots=True)
class ClauseNode:
    clause_id: str
    clause_type: ClauseType
    state: str
    jurisdiction_type: str
    ruleset_id: str
    ruleset_version: str
    chapter_number: int | None
    chapter_title: str
    rule_number: str | None
    rule_title: str
    sub_rule_path: str
    display_citation: str
    source_file: str
    anchor_id: str
    raw_text: str
    normalized_text: str
    effective_from: date | None = None
    effective_to: date | None = None
    occupancy_groups: list[str] = field(default_factory=list)
    panchayat_category: str | None = None
    topic_tags: list[str] = field(default_factory=list)
    is_generic: bool = False
    conditions: dict[str, Any] = field(default_factory=dict)
    numeric_values: dict[str, float] = field(default_factory=dict)
    amendments: list[AmendmentRecord] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
    table_data: TableData | None = None
    parent_clause_id: str | None = None


@dataclass(slots=True)
class RuleDocument:
    document_id: str
    state: str
    jurisdiction_type: str
    ruleset_id: str
    ruleset_version: str
    issuing_authority: str
    effective_from: date | None
    effective_to: date | None
    source_file: str
    chapter_number: int
    chapter_title: str
    rule_number: str
    rule_title: str
    full_text: str
    anchor_id: str
    occupancy_groups: list[str] = field(default_factory=list)
    is_generic: bool = False
    panchayat_category: str | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    numeric_values: dict[str, float] = field(default_factory=dict)
    provisos: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
    amendments: list[AmendmentRecord] = field(default_factory=list)
    clause_nodes: list[ClauseNode] = field(default_factory=list)


@dataclass(slots=True)
class Citation:
    claim_id: str
    ruleset_id: str
    ruleset_name: str
    chapter_number: int | None
    rule_number: str | None
    sub_rule_path: str
    table_ref: str | None
    anchor_id: str
    source_file: str
    display_citation: str
    quote_excerpt: str
    document_id: str | None = None


@dataclass(slots=True)
class QueryFact:
    state: str | None = None
    location_text: str | None = None
    jurisdiction_type: str | None = None
    panchayat_category: str | None = None
    occupancies: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    height_m: float | None = None
    floor_area_sqm: float | None = None
    plot_area_sqm: float | None = None
    floors: int | None = None
    query_date: date | None = None
    mentioned_rules: list[str] = field(default_factory=list)
    query_intent: str | None = None
