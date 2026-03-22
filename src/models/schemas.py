from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=3)
    location: str | None = None
    state: str | None = None
    jurisdiction_type: str | None = None
    panchayat_category: str | None = None
    query_date: date | None = None
    explicit_occupancy: str | None = None
    top_k: int = Field(default=12, ge=1, le=50)
    retrieval_mode: Literal[
        "vector_only",
        "hybrid_no_reranker",
        "lexical_only_bm25",
        "hybrid_reranker",
        "hybrid_graph_reranker",
        "hybrid_graph_no_reranker",
        "agentic_dynamic",
    ] = "hybrid_no_reranker"
    debug_trace: bool = False


class ApplicabilityRequest(BaseModel):
    location: str
    state: str | None = None
    jurisdiction_type: str | None = None
    panchayat_category: str | None = None
    building_description: str | None = None


class ClarificationQuestion(BaseModel):
    code: str
    question: str
    options: list[str] = Field(default_factory=list)


class CitationPayload(BaseModel):
    claim_id: str
    ruleset_id: str
    chapter_number: int | None
    rule_number: str | None
    sub_rule_path: str = ""
    table_ref: str | None = None
    anchor_id: str
    source_file: str
    display_citation: str
    quote_excerpt: str
    document_id: str | None = None
    source_url: str | None = None


class EvidenceItem(BaseModel):
    claim_id: str
    text: str
    chunk_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    citations: list[CitationPayload] = Field(default_factory=list)


class ComplianceBriefItem(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)


class ComplianceBriefPayload(BaseModel):
    verdict: Literal["compliant", "non_compliant", "depends", "insufficient_evidence"]
    short_summary: str = Field(min_length=1)
    applicable_rules: list[ComplianceBriefItem] = Field(default_factory=list)
    conditions_and_exceptions: list[ComplianceBriefItem] = Field(default_factory=list)
    required_actions: list[ComplianceBriefItem] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    clarifications_needed: list[str] = Field(default_factory=list)


class GroundingReportPayload(BaseModel):
    supported_claim_count: int
    unsupported_claim_count: int
    conflicting_claim_count: int
    missing_topics: list[str] = Field(default_factory=list)
    sufficient: bool = False
    partial: bool = False
    support_ratio: float = 0.0
    insufficiency_reasons: list[str] = Field(default_factory=list)
    abstained: bool = False


class AgentTraceStep(BaseModel):
    step: str
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    jurisdiction: str
    occupancy_groups: list[str]
    assumptions: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    answer_sections: list[dict[str, Any]] = Field(default_factory=list)
    evidence_matrix: list[EvidenceItem] = Field(default_factory=list)
    citations: list[CitationPayload] = Field(default_factory=list)
    verdict: Literal["compliant", "non_compliant", "depends", "insufficient_evidence"] | None = None
    final_answer: ComplianceBriefPayload | None = None
    grounding: GroundingReportPayload | None = None
    claim_citations: dict[str, list[str]] = Field(default_factory=dict)
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    clarifications: list[ClarificationQuestion] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    state: str
    jurisdiction_type: str | None = None
    source_glob: str | None = None


class IngestResult(BaseModel):
    state: str
    ruleset_ids: list[str]
    parsed_rules: int
    parsed_clauses: int
    parsed_tables: int
    parsed_files: int = 0
    failed_files: int = 0
    parse_quality_score: float | None = None
    parse_quality: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
