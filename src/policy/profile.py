from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RetrievalPolicy:
    vector_rrf_weight: float = 1.0
    lexical_rrf_weight: float = 0.85
    structured_rrf_weight: float = 0.35
    cross_ref_score_decay: float = 0.85
    rrf_k: int = 60
    default_evidence_docs_per_topic: int = 4
    topic_min_docs: int = 2
    max_excerpt_sentences: int = 8
    max_excerpt_chars: int = 1600
    min_evidence_score: float = 0.08
    min_query_relevance: float = 0.06
    query_relevance_weight: float = 0.2
    topic_match_weight: float = 0.05
    max_topic_likes: int = 6
    per_topic_structured_limit_floor: int = 8
    candidate_pool_factor: float = 4.0
    max_claims_per_doc: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RetrievalPolicy":
        payload = data or {}
        return cls(
            vector_rrf_weight=float(payload.get("vector_rrf_weight", 1.0)),
            lexical_rrf_weight=float(payload.get("lexical_rrf_weight", 0.85)),
            structured_rrf_weight=float(payload.get("structured_rrf_weight", 0.35)),
            cross_ref_score_decay=float(payload.get("cross_ref_score_decay", 0.85)),
            rrf_k=int(payload.get("rrf_k", 60)),
            default_evidence_docs_per_topic=int(payload.get("default_evidence_docs_per_topic", 4)),
            topic_min_docs=int(payload.get("topic_min_docs", 2)),
            max_excerpt_sentences=int(payload.get("max_excerpt_sentences", 8)),
            max_excerpt_chars=int(payload.get("max_excerpt_chars", 1600)),
            min_evidence_score=float(payload.get("min_evidence_score", 0.08)),
            min_query_relevance=float(payload.get("min_query_relevance", 0.06)),
            query_relevance_weight=float(payload.get("query_relevance_weight", 0.2)),
            topic_match_weight=float(payload.get("topic_match_weight", 0.05)),
            max_topic_likes=int(payload.get("max_topic_likes", 6)),
            per_topic_structured_limit_floor=int(payload.get("per_topic_structured_limit_floor", 8)),
            candidate_pool_factor=float(payload.get("candidate_pool_factor", 4.0)),
            max_claims_per_doc=int(payload.get("max_claims_per_doc", 2)),
        )


@dataclass(slots=True)
class ApplicabilityPolicy:
    procedural_occupancy_agnostic: bool = False
    generic_fallback_enabled: bool = True
    allow_generic_without_occupancy: bool = False
    strict_topic_matching: bool = True
    broad_procedural_topics: set[str] = field(
        default_factory=lambda: {"permit", "timeline", "fees", "registration", "appeal", "penalty", "definition"}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ApplicabilityPolicy":
        payload = data or {}
        broad = payload.get("broad_procedural_topics")
        if isinstance(broad, list):
            broad_topics = {str(item).strip().lower() for item in broad if str(item).strip()}
        else:
            broad_topics = {"permit", "timeline", "fees", "registration", "appeal", "penalty", "definition"}
        return cls(
            procedural_occupancy_agnostic=bool(payload.get("procedural_occupancy_agnostic", False)),
            generic_fallback_enabled=bool(payload.get("generic_fallback_enabled", True)),
            allow_generic_without_occupancy=bool(payload.get("allow_generic_without_occupancy", False)),
            strict_topic_matching=bool(payload.get("strict_topic_matching", True)),
            broad_procedural_topics=broad_topics,
        )


@dataclass(slots=True)
class GenerationPolicy:
    max_output_tokens: int = 1200
    max_claims_per_section: int = 2
    summary_char_limit: int = 280
    concise_claim_char_limit: int = 280
    excavation_claim_char_limit: int = 480
    enforce_non_empty_summary: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationPolicy":
        payload = data or {}
        return cls(
            max_output_tokens=int(payload.get("max_output_tokens", 1200)),
            max_claims_per_section=int(payload.get("max_claims_per_section", 2)),
            summary_char_limit=int(payload.get("summary_char_limit", 280)),
            concise_claim_char_limit=int(payload.get("concise_claim_char_limit", 280)),
            excavation_claim_char_limit=int(payload.get("excavation_claim_char_limit", 480)),
            enforce_non_empty_summary=bool(payload.get("enforce_non_empty_summary", True)),
        )


@dataclass(slots=True)
class AbstentionPolicy:
    min_support_ratio: float = 0.55
    partial_support_ratio: float = 0.12
    min_evidence_score: float = 0.08
    partial_min_evidence_score: float = 0.04
    conservative_mandatory_components: bool = True
    contradiction_blocking: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AbstentionPolicy":
        payload = data or {}
        return cls(
            min_support_ratio=float(payload.get("min_support_ratio", 0.55)),
            partial_support_ratio=float(payload.get("partial_support_ratio", 0.12)),
            min_evidence_score=float(payload.get("min_evidence_score", 0.08)),
            partial_min_evidence_score=float(payload.get("partial_min_evidence_score", 0.04)),
            conservative_mandatory_components=bool(payload.get("conservative_mandatory_components", True)),
            contradiction_blocking=bool(payload.get("contradiction_blocking", True)),
        )


@dataclass(slots=True)
class LegalRagPolicyPack:
    retrieval: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    applicability: ApplicabilityPolicy = field(default_factory=ApplicabilityPolicy)
    generation: GenerationPolicy = field(default_factory=GenerationPolicy)
    abstention: AbstentionPolicy = field(default_factory=AbstentionPolicy)


class PolicyLoader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, profile_paths: dict[str, Path]) -> LegalRagPolicyPack:
        retrieval_cfg = self._load_yaml(profile_paths.get("retrieval"))
        applicability_cfg = self._load_yaml(profile_paths.get("applicability"))
        generation_cfg = self._load_yaml(profile_paths.get("generation"))
        abstention_cfg = self._load_yaml(profile_paths.get("abstention"))

        return LegalRagPolicyPack(
            retrieval=RetrievalPolicy.from_dict(retrieval_cfg),
            applicability=ApplicabilityPolicy.from_dict(applicability_cfg),
            generation=GenerationPolicy.from_dict(generation_cfg),
            abstention=AbstentionPolicy.from_dict(abstention_cfg),
        )

    def _load_yaml(self, path: Path | None) -> dict[str, Any]:
        if not path:
            return {}
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"Policy file must contain a mapping: {path}")
        return raw
