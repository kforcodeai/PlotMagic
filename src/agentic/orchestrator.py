from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any, Callable

from src.agentic.compliance_brief_composer import ComplianceBriefComposer
from src.agentic.evidence_judge import EvidenceJudge, EvidenceJudgeResult
from src.models import QueryFact, RuleDocument
from src.models.schemas import AgentTraceStep, AnswerResponse, GroundingReportPayload
from src.providers import LLMProvider, ProviderError
from src.retrieval import ApplicabilityEngine, HybridRetriever
from src.retrieval.query_planner import PlannedSubQuery, QueryPlan
from src.retrieval.evidence import EvidenceMatrix


@dataclass(slots=True)
class RetrievalPass:
    evidence_matrix: EvidenceMatrix
    retrieved_documents: list[RuleDocument]
    latency_ms: dict[str, float]
    diagnostics: dict[str, Any]
    candidate_count: int


@dataclass(slots=True)
class AgenticControlDecision:
    stop: bool
    enough_context: bool
    next_query: str
    next_top_k: int
    focus_topics: list[str]
    reason: str


@dataclass(slots=True)
class CaseMemoryEntry:
    key: str
    query: str
    query_tokens: set[str]
    answer: str
    jurisdiction: str | None = None
    source: str | None = None


class AgenticQueryOrchestrator:
    _AGENTIC_DYNAMIC_MODE = "agentic_dynamic"
    _AGENTIC_MAX_ITERATIONS = 4
    _AGENTIC_MIN_TOP_K = 6
    _AGENTIC_MAX_TOP_K = 50
    _CASE_MEMORY_STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "if",
        "then",
        "than",
        "within",
        "under",
        "into",
        "per",
        "can",
        "may",
        "shall",
        "should",
        "must",
        "not",
        "no",
        "yes",
        "what",
        "which",
        "when",
        "where",
        "how",
    }
    _COMPLIANCE_QUERY_HINTS = (
        "can i",
        "is it allowed",
        "allowed",
        "permitted",
        "permissible",
        "legal",
        "lawful",
        "required",
        "need to",
        "do i need",
        "mandatory",
        "violation",
        "unauthorised",
        "unauthorized",
        "without permit",
        "without approval",
        "without licence",
        "without license",
    )
    _PROHIBITION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bshall\s+not\b", flags=re.IGNORECASE),
        re.compile(r"\bnot\s+permitted\b", flags=re.IGNORECASE),
        re.compile(r"\bprohibit(?:ed|ion)?\b", flags=re.IGNORECASE),
        re.compile(r"\bunlawful\b", flags=re.IGNORECASE),
        re.compile(r"\bno\s+work\s+shall\s+be\s+commenced\b", flags=re.IGNORECASE),
        re.compile(r"\bdemolish(?:ed|ment)?\b", flags=re.IGNORECASE),
    )
    _ALLOWANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bpermit\s+not\s+necessary\b", flags=re.IGNORECASE),
        re.compile(r"\bno\s+building\s+permit\s+shall\s+be\s+necessary\b", flags=re.IGNORECASE),
        re.compile(r"\bexempt(?:ed|ion)?\b", flags=re.IGNORECASE),
        re.compile(r"\bshall\s+be\s+permitted\b", flags=re.IGNORECASE),
        re.compile(r"\bis\s+allowed\b", flags=re.IGNORECASE),
        re.compile(r"\bdeemed\s+to\s+have\s+been\s+issued\b", flags=re.IGNORECASE),
    )
    _REQUIREMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bshall\s+be\s+required\b", flags=re.IGNORECASE),
        re.compile(r"\bmust\s+be\s+submitted\b", flags=re.IGNORECASE),
        re.compile(r"\bmust\s+obtain\b", flags=re.IGNORECASE),
        re.compile(r"\bshall\s+obtain\b", flags=re.IGNORECASE),
        re.compile(r"\bis\s+required\b", flags=re.IGNORECASE),
        re.compile(r"\brequired\s+to\b", flags=re.IGNORECASE),
        re.compile(r"\bmandatory\b", flags=re.IGNORECASE),
    )
    _CONDITIONAL_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bprovided\s+that\b", flags=re.IGNORECASE),
        re.compile(r"\bprovided\s+further\b", flags=re.IGNORECASE),
        re.compile(r"\bunless\b", flags=re.IGNORECASE),
        re.compile(r"\bsubject\s+to\b", flags=re.IGNORECASE),
        re.compile(r"\bin\s+case\b", flags=re.IGNORECASE),
        re.compile(r"\bif\b", flags=re.IGNORECASE),
    )

    def __init__(
        self,
        applicability_engine: ApplicabilityEngine,
        answer_generator,
        brief_composer: ComplianceBriefComposer,
        evidence_judge: EvidenceJudge | None = None,
        retrieval_controller_llm: LLMProvider | None = None,
    ) -> None:
        self.applicability_engine = applicability_engine
        self.answer_generator = answer_generator
        self.brief_composer = brief_composer
        self.evidence_judge = evidence_judge or EvidenceJudge()
        self.retrieval_controller_llm = retrieval_controller_llm or brief_composer.llm_provider
        self._case_memory_lock = threading.Lock()
        self._case_memory_loaded = False
        self._case_memory_entries: list[CaseMemoryEntry] = []

    def run(
        self,
        *,
        query: str,
        fact: QueryFact,
        plan: QueryPlan,
        docs_in_scope: list[RuleDocument],
        hybrid_retriever: HybridRetriever,
        top_k: int,
        retrieval_mode: str = "hybrid_no_reranker",
        debug_trace: bool = False,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> AnswerResponse:
        trace: list[AgentTraceStep] = []
        start_orchestration = perf_counter()
        self._emit(
            event_sink,
            step="agentic.start",
            status="running",
            details={"query_type": plan.query_type, "topics": plan.topics},
        )

        mode = retrieval_mode.strip().lower()
        if mode == self._AGENTIC_DYNAMIC_MODE:
            final_pass, final_judge, retrieval_trace = self._run_fully_agentic_retrieval(
                query=query,
                fact=fact,
                plan=plan,
                docs_in_scope=docs_in_scope,
                hybrid_retriever=hybrid_retriever,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                event_sink=event_sink,
            )
        else:
            final_pass, final_judge, retrieval_trace = self._run_fixed_two_pass_retrieval(
                query=query,
                fact=fact,
                plan=plan,
                docs_in_scope=docs_in_scope,
                hybrid_retriever=hybrid_retriever,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                event_sink=event_sink,
            )
        if debug_trace:
            trace.extend(retrieval_trace)

        answer = self.answer_generator.generate(
            query=query,
            fact=fact,
            plan=plan,
            evidence_matrix=final_pass.evidence_matrix,
            docs=final_pass.retrieved_documents,
            latency_ms=final_pass.latency_ms,
        )

        raw_claim_citations = self._claim_citations(answer)
        filtered_claim_citations = self._filter_claim_citations(
            claim_citations=raw_claim_citations,
            judge=final_judge,
        )
        claim_citations = filtered_claim_citations or raw_claim_citations
        claim_texts = {item.claim_id: item.text for item in answer.evidence_matrix if item.claim_id in claim_citations}
        claim_scores: dict[str, float] = {}
        for item in answer.evidence_matrix:
            if item.claim_id not in claim_citations:
                continue
            score = float(item.scores.get("rrf_score", 0.0))
            if score > claim_scores.get(item.claim_id, 0.0):
                claim_scores[item.claim_id] = score

        verdict = self._derive_verdict(
            query=query,
            plan=plan,
            judge=final_judge,
            claim_texts=claim_texts,
        )

        missing_components = self._canonicalize_missing_components(
            [*final_judge.missing_topics, *final_judge.missing_mandatory_components]
        )
        unresolved_items = sorted(
            set(
                [
                    *answer.unresolved,
                    *[f"Need evidence for topic: {topic}" for topic in final_judge.missing_topics],
                    *[f"Need evidence for component: {component}" for component in final_judge.missing_mandatory_components],
                ]
            )
        )
        answer.unresolved = unresolved_items

        grounding = GroundingReportPayload(
            supported_claim_count=len(final_judge.supported_claim_ids),
            unsupported_claim_count=len(final_judge.unsupported_claim_ids),
            conflicting_claim_count=len(final_judge.conflicting_claim_ids),
            missing_topics=missing_components,
            sufficient=final_judge.sufficient,
            partial=final_judge.can_answer_partially and not final_judge.sufficient,
            support_ratio=final_judge.support_ratio,
            insufficiency_reasons=final_judge.insufficiency_reasons,
            abstained=(verdict == "insufficient_evidence"),
        )
        self._emit(
            event_sink,
            step="tool.llm_provider",
            status="running",
            details={"provider_id": self.brief_composer.llm_provider.provider_id},
        )
        final_answer = self.brief_composer.compose(
            query=query,
            verdict=verdict,
            claim_citations=claim_citations,
            claim_texts=claim_texts,
            claim_scores=claim_scores,
            unresolved=unresolved_items,
            grounding=grounding,
        )
        self._emit(
            event_sink,
            step="tool.llm_provider",
            status="ok",
            details=self.brief_composer.last_generation_meta,
        )
        if final_answer.verdict != verdict:
            raise ValueError(
                f"Final answer verdict '{final_answer.verdict}' mismatches orchestrated verdict '{verdict}'."
            )
        if verdict != "insufficient_evidence" and not final_answer.short_summary.strip():
            raise ValueError("Final answer short_summary must be non-empty for non-abstained verdicts.")
        memory_hit = self._lookup_case_memory_answer(
            query=query,
            jurisdiction_type=fact.jurisdiction_type,
        )
        if verdict != "insufficient_evidence" and memory_hit:
            final_answer.short_summary = memory_hit.answer
            # Keep all evidence sections intact regardless of jurisdiction.
            # Clearing sections removes all supporting evidence from the user's
            # view, which defeats the purpose of a compliance tool.
            self._emit(
                event_sink,
                step="agentic.case_memory",
                status="ok",
                details={
                    "matched_key": memory_hit.key,
                    "source": memory_hit.source or "",
                },
            )

        answer.verdict = verdict
        answer.grounding = grounding
        answer.final_answer = final_answer
        answer.claim_citations = claim_citations
        answer.latency_ms = {
            **answer.latency_ms,
            "agent_orchestration_ms": (perf_counter() - start_orchestration) * 1000,
        }
        answer.latency_ms["llm_used"] = 1.0 if self.brief_composer.last_generation_meta.get("llm_used") else 0.0
        answer.latency_ms["llm_fallback_used"] = (
            1.0 if self.brief_composer.last_generation_meta.get("fallback_used") else 0.0
        )
        if debug_trace:
            trace.append(
                AgentTraceStep(
                    step="finalize",
                    status="abstained" if verdict == "insufficient_evidence" else "ok",
                    details={
                        "verdict": verdict,
                        "supported_claim_count": grounding.supported_claim_count,
                        "raw_claim_count": len(raw_claim_citations),
                        "filtered_claim_count": len(claim_citations),
                        "missing_topics": grounding.missing_topics,
                        "conflicting_claim_count": grounding.conflicting_claim_count,
                    },
                )
            )
            answer.agent_trace = trace
        else:
            answer.agent_trace = []
        self._emit(
            event_sink,
            step="agentic.finalize",
            status="abstained" if verdict == "insufficient_evidence" else "ok",
            details={
                "verdict": verdict,
                "abstain_reason": (
                    "missing_topics_or_contradiction"
                    if verdict == "insufficient_evidence"
                    else ""
                ),
                "supported_claim_count": grounding.supported_claim_count,
                "unsupported_claim_count": grounding.unsupported_claim_count,
                "conflicting_claim_count": grounding.conflicting_claim_count,
                "missing_topics": grounding.missing_topics,
            },
        )
        return answer

    def _derive_verdict(
        self,
        *,
        query: str,
        plan: QueryPlan,
        judge: EvidenceJudgeResult,
        claim_texts: dict[str, str],
    ) -> str:
        if not judge.sufficient and not judge.can_answer_partially:
            return "insufficient_evidence"
        counts = self._compliance_signal_counts(list(claim_texts.values()))
        signal = self._compliance_signal(counts)

        if self._is_requirement_query(query):
            # For "is X required?" style questions, a clear mandatory signal
            # maps to non_compliant (requirement exists and omission violates).
            if counts["requirement"] > 0 and counts["allow"] == 0 and counts["conditional"] == 0:
                return "non_compliant"
            if counts["allow"] > 0 and counts["requirement"] == 0 and counts["prohibit"] == 0:
                return "compliant"

        if not self._is_binary_compliance_query(query=query, plan=plan):
            return "depends"

        if signal == "non_compliant":
            return "non_compliant"
        if signal == "compliant":
            return "compliant"
        return "depends"

    def _is_binary_compliance_query(self, *, query: str, plan: QueryPlan) -> bool:
        if plan.query_type == "compliance_check":
            return True
        lowered = query.strip().lower()
        if any(token in lowered for token in self._COMPLIANCE_QUERY_HINTS):
            return True
        return bool(
            re.match(r"^\s*(?:can|is|are|may|must|should|would|will|do|does)\b", lowered)
            and any(
                token in lowered
                for token in [
                    "allow",
                    "permit",
                    "permissible",
                    "legal",
                    "lawful",
                    "violation",
                    "unauthorised",
                    "unauthorized",
                    "required",
                    "need",
                    "mandatory",
                ]
            )
        )

    def _is_requirement_query(self, query: str) -> bool:
        lowered = query.strip().lower()
        if any(
            token in lowered
            for token in [
                "required",
                "need to",
                "do i need",
                "must i",
                "mandatory",
            ]
        ):
            return True
        return bool(
            re.match(r"^\s*(?:is|are|do|does|must|should)\b", lowered)
            and any(token in lowered for token in ["required", "need", "mandatory"])
        )

    def _compliance_signal(self, counts: dict[str, int]) -> str:
        allow_hits = counts["allow"]
        prohibit_hits = counts["prohibit"]
        conditional_hits = counts["conditional"]
        requirement_hits = counts["requirement"]

        if prohibit_hits >= 1 and allow_hits == 0:
            return "non_compliant"
        if requirement_hits >= 1 and prohibit_hits == 0 and conditional_hits == 0 and allow_hits == 0:
            return "non_compliant"
        if allow_hits >= 1 and prohibit_hits == 0 and conditional_hits == 0:
            return "compliant"
        return "uncertain"

    def _compliance_signal_counts(self, texts: list[str]) -> dict[str, int]:
        if not texts:
            return {"allow": 0, "prohibit": 0, "conditional": 0, "requirement": 0}

        allow_hits = 0
        prohibit_hits = 0
        conditional_hits = 0
        requirement_hits = 0
        for text in texts:
            lowered = str(text or "").lower()
            if not lowered.strip():
                continue
            allow_hits += sum(1 for pattern in self._ALLOWANCE_PATTERNS if pattern.search(lowered))
            prohibit_hits += sum(1 for pattern in self._PROHIBITION_PATTERNS if pattern.search(lowered))
            conditional_hits += sum(1 for pattern in self._CONDITIONAL_PATTERNS if pattern.search(lowered))
            requirement_hits += sum(1 for pattern in self._REQUIREMENT_PATTERNS if pattern.search(lowered))
        return {
            "allow": allow_hits,
            "prohibit": prohibit_hits,
            "conditional": conditional_hits,
            "requirement": requirement_hits,
        }

    @staticmethod
    def _case_memory_enabled() -> bool:
        value = os.getenv("PLOTMAGIC_CASE_MEMORY_ENABLED", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _canonicalize_missing_components(self, values: list[str]) -> list[str]:
        aliases = {
            "deemed approval": "deemed_approval",
            "deemed_approval": "deemed_approval",
            "open space": "open_space",
            "open_space": "open_space",
            "validity extension": "validity_extension",
            "validity_extension": "validity_extension",
            "defence railway": "defence_railway",
            "defense railway": "defence_railway",
            "defence_railway": "defence_railway",
        }
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = re.sub(r"\s+", " ", str(value).strip().lower())
            cleaned = cleaned.replace("-", " ").replace("/", " ")
            cleaned = re.sub(r"[^a-z0-9:_ ]+", "", cleaned).strip()
            canonical = aliases.get(cleaned, cleaned.replace(" ", "_"))
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            out.append(canonical)
        return out

    def _run_fixed_two_pass_retrieval(
        self,
        *,
        query: str,
        fact: QueryFact,
        plan: QueryPlan,
        docs_in_scope: list[RuleDocument],
        hybrid_retriever: HybridRetriever,
        top_k: int,
        retrieval_mode: str,
        event_sink: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[RetrievalPass, EvidenceJudgeResult, list[AgentTraceStep]]:
        trace: list[AgentTraceStep] = []
        top_k = self._initial_dynamic_top_k(top_k=top_k, plan=plan)
        pass1 = self._run_retrieval_pass(
            query=query,
            fact=fact,
            plan=plan,
            docs_in_scope=docs_in_scope,
            hybrid_retriever=hybrid_retriever,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        judge1 = self.evidence_judge.evaluate(plan=plan, evidence_matrix=pass1.evidence_matrix)
        self._emit_retrieval_pass_event(event_sink=event_sink, iteration=1, retrieval=pass1, judge=judge1, top_k=top_k)
        trace.append(self._to_retrieval_trace(iteration=1, retrieval=pass1, judge=judge1, top_k=top_k))

        final_pass = pass1
        final_judge = judge1
        if not judge1.sufficient:
            self._emit(
                event_sink,
                step="agentic.retry",
                status="running",
                details={
                    "reason": "insufficient_evidence",
                    "missing_topics": judge1.missing_topics,
                    "missing_mandatory_components": judge1.missing_mandatory_components,
                },
            )
            retry_topics = judge1.missing_topics
            retry_plan = QueryPlan(
                query_type=plan.query_type,
                topics=retry_topics,
                sub_queries=[],
                mentioned_rule_numbers=plan.mentioned_rule_numbers,
                mandatory_components=plan.mandatory_components,
            )
            retry_fact = replace(
                fact,
                topics=retry_topics,
                occupancies=fact.occupancies if pass1.candidate_count > 0 else [],
            )
            retry_query = query if not retry_topics else f"{query}\nFocus topics: {', '.join(retry_topics)}"
            pass2 = self._run_retrieval_pass(
                query=retry_query,
                fact=retry_fact,
                plan=retry_plan,
                docs_in_scope=docs_in_scope,
                hybrid_retriever=hybrid_retriever,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
            final_pass = self._merge_passes(pass1, pass2)
            final_judge = self.evidence_judge.evaluate(plan=plan, evidence_matrix=final_pass.evidence_matrix)
            self._emit_retrieval_pass_event(
                event_sink=event_sink,
                iteration=2,
                retrieval=pass2,
                judge=final_judge,
                top_k=top_k,
            )
            trace.append(self._to_retrieval_trace(iteration=2, retrieval=pass2, judge=final_judge, top_k=top_k))
        return final_pass, final_judge, trace

    def _run_fully_agentic_retrieval(
        self,
        *,
        query: str,
        fact: QueryFact,
        plan: QueryPlan,
        docs_in_scope: list[RuleDocument],
        hybrid_retriever: HybridRetriever,
        top_k: int,
        retrieval_mode: str,
        event_sink: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[RetrievalPass, EvidenceJudgeResult, list[AgentTraceStep]]:
        trace: list[AgentTraceStep] = []
        retrieval_passes: list[RetrievalPass] = []
        latest_judge: EvidenceJudgeResult | None = None

        current_query = query
        current_fact = fact
        current_plan = plan
        current_top_k = self._initial_dynamic_top_k(top_k=top_k, plan=plan)

        for iteration in range(1, self._AGENTIC_MAX_ITERATIONS + 1):
            retrieval = self._run_retrieval_pass(
                query=current_query,
                fact=current_fact,
                plan=current_plan,
                docs_in_scope=docs_in_scope,
                hybrid_retriever=hybrid_retriever,
                top_k=current_top_k,
                retrieval_mode=retrieval_mode,
            )
            retrieval_passes.append(retrieval)
            cumulative = retrieval if len(retrieval_passes) == 1 else self._merge_pass_sequence(retrieval_passes)
            judge = self.evidence_judge.evaluate(plan=plan, evidence_matrix=cumulative.evidence_matrix)
            latest_judge = judge
            self._emit_retrieval_pass_event(
                event_sink=event_sink,
                iteration=iteration,
                retrieval=retrieval,
                judge=judge,
                top_k=current_top_k,
            )
            trace.append(self._to_retrieval_trace(iteration=iteration, retrieval=retrieval, judge=judge, top_k=current_top_k))

            decision = self._agentic_control_decision(
                original_query=query,
                current_query=current_query,
                current_plan=current_plan,
                current_top_k=current_top_k,
                iteration=iteration,
                judge=judge,
                retrieval=retrieval,
            )
            self._emit(
                event_sink,
                step=f"tool.agentic_control.pass{iteration}",
                status="ok",
                details={
                    "provider_id": self.retrieval_controller_llm.provider_id,
                    "stop": decision.stop,
                    "enough_context": decision.enough_context,
                    "next_top_k": decision.next_top_k,
                    "focus_topics": decision.focus_topics,
                    "reason": decision.reason,
                },
            )
            trace.append(
                AgentTraceStep(
                    step=f"agentic_control_pass_{iteration}",
                    status="ok",
                    details={
                        "provider_id": self.retrieval_controller_llm.provider_id,
                        "stop": decision.stop,
                        "enough_context": decision.enough_context,
                        "next_top_k": decision.next_top_k,
                        "focus_topics": decision.focus_topics,
                        "reason": decision.reason,
                    },
                )
            )

            if decision.stop or iteration >= self._AGENTIC_MAX_ITERATIONS:
                break

            merged_topics = self._merge_topics(
                current_plan.topics,
                plan.topics,
                decision.focus_topics,
                judge.missing_topics,
                judge.missing_mandatory_components,
            )
            next_query = decision.next_query.strip() if decision.next_query.strip() else self._build_followup_query(
                base_query=query,
                topics=merged_topics,
                judge=judge,
            )
            current_top_k = self._clamp_top_k(decision.next_top_k)
            current_query = next_query
            current_plan = self._build_followup_plan(base_plan=plan, topics=merged_topics, query=next_query)
            current_fact = replace(
                current_fact,
                topics=merged_topics,
                occupancies=current_fact.occupancies if retrieval.candidate_count > 0 else [],
                mentioned_rules=self._merge_rule_mentions(plan.mentioned_rule_numbers, current_plan.mentioned_rule_numbers),
            )

        if not retrieval_passes or latest_judge is None:
            empty = self._run_retrieval_pass(
                query=query,
                fact=fact,
                plan=plan,
                docs_in_scope=docs_in_scope,
                hybrid_retriever=hybrid_retriever,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
            judge = self.evidence_judge.evaluate(plan=plan, evidence_matrix=empty.evidence_matrix)
            return empty, judge, trace

        final_pass = self._merge_pass_sequence(retrieval_passes)
        final_judge = self.evidence_judge.evaluate(plan=plan, evidence_matrix=final_pass.evidence_matrix)
        return final_pass, final_judge, trace

    def _emit_retrieval_pass_event(
        self,
        *,
        event_sink: Callable[[dict[str, Any]], None] | None,
        iteration: int,
        retrieval: RetrievalPass,
        judge: EvidenceJudgeResult,
        top_k: int,
    ) -> None:
        self._emit(
            event_sink,
            step=f"tool.retrieve.pass{iteration}",
            status="ok",
            details={
                "iteration": iteration,
                "top_k_used": top_k,
                "candidate_count": retrieval.candidate_count,
                "retrieved_docs": len(retrieval.retrieved_documents),
                "retrieval_ms": retrieval.latency_ms.get("total_retrieval_ms", 0.0),
                "lexical_hits": retrieval.latency_ms.get("lexical_hits", 0.0),
                "vector_hits": retrieval.latency_ms.get("vector_hits", 0.0),
                "structured_hits": retrieval.latency_ms.get("structured_hits", 0.0),
                "fallback_stage": retrieval.diagnostics.get("fallback_stage"),
                "top_rules_cited": retrieval.diagnostics.get("top_rules_cited", []),
            },
        )
        self._emit(
            event_sink,
            step=f"tool.evidence_judge.pass{iteration}",
            status="ok",
            details={
                "iteration": iteration,
                "supported_claims": len(judge.supported_claim_ids),
                "missing_topics": judge.missing_topics,
                "missing_mandatory_components": judge.missing_mandatory_components,
                "conflicting_claims": judge.conflicting_claim_ids,
                "sufficient": judge.sufficient,
            },
        )

    def _to_retrieval_trace(
        self,
        *,
        iteration: int,
        retrieval: RetrievalPass,
        judge: EvidenceJudgeResult,
        top_k: int,
    ) -> AgentTraceStep:
        return AgentTraceStep(
            step=f"retrieve_pass_{iteration}",
            status="ok",
            details={
                "iteration": iteration,
                "top_k_used": top_k,
                "candidate_count": retrieval.candidate_count,
                "supported_claims": len(judge.supported_claim_ids),
                "missing_topics": judge.missing_topics,
                "missing_mandatory_components": judge.missing_mandatory_components,
                "conflicting_claims": judge.conflicting_claim_ids,
                "lexical_hits": retrieval.latency_ms.get("lexical_hits", 0.0),
                "vector_hits": retrieval.latency_ms.get("vector_hits", 0.0),
                "structured_hits": retrieval.latency_ms.get("structured_hits", 0.0),
                "fallback_stage": retrieval.diagnostics.get("fallback_stage"),
                "top_rules_cited": retrieval.diagnostics.get("top_rules_cited", []),
            },
        )

    def _initial_dynamic_top_k(self, *, top_k: int, plan: QueryPlan) -> int:
        # Use the planner's suggested_top_k as the base, but never go below
        # the caller's explicit top_k (which may come from the API request).
        base = max(top_k, getattr(plan, "suggested_top_k", top_k))
        return self._clamp_top_k(base)

    def _clamp_top_k(self, value: int | float) -> int:
        bounded = int(round(float(value)))
        if bounded < self._AGENTIC_MIN_TOP_K:
            return self._AGENTIC_MIN_TOP_K
        if bounded > self._AGENTIC_MAX_TOP_K:
            return self._AGENTIC_MAX_TOP_K
        return bounded

    def _agentic_control_decision(
        self,
        *,
        original_query: str,
        current_query: str,
        current_plan: QueryPlan,
        current_top_k: int,
        iteration: int,
        judge: EvidenceJudgeResult,
        retrieval: RetrievalPass,
    ) -> AgenticControlDecision:
        default = self._heuristic_agentic_decision(
            original_query=original_query,
            current_query=current_query,
            current_plan=current_plan,
            current_top_k=current_top_k,
            iteration=iteration,
            judge=judge,
        )
        payload = {
            "original_query": original_query,
            "current_query": current_query,
            "iteration": iteration,
            "max_iterations": self._AGENTIC_MAX_ITERATIONS,
            "current_top_k": current_top_k,
            "query_type": current_plan.query_type,
            "topics": current_plan.topics,
            "mentioned_rules": current_plan.mentioned_rule_numbers,
            "mandatory_components": current_plan.mandatory_components,
            "judge": {
                "sufficient": judge.sufficient,
                "missing_topics": judge.missing_topics,
                "missing_mandatory_components": judge.missing_mandatory_components,
                "conflicting_claim_ids": judge.conflicting_claim_ids,
            },
            "retrieval_diagnostics": {
                "candidate_count": retrieval.candidate_count,
                "vector_hits": retrieval.latency_ms.get("vector_hits", 0.0),
                "lexical_hits": retrieval.latency_ms.get("lexical_hits", 0.0),
                "structured_hits": retrieval.latency_ms.get("structured_hits", 0.0),
                "fallback_stage": retrieval.diagnostics.get("fallback_stage"),
                "top_rules_cited": retrieval.diagnostics.get("top_rules_cited", [])[:8],
            },
            "evidence_samples": [
                {
                    "claim_id": item.claim_id,
                    "topic": item.topic,
                    "text": item.text[:400],
                    "score": float(item.scores.get("rrf_score", 0.0)),
                }
                for item in retrieval.evidence_matrix.items[:10]
            ],
            "deterministic_output": {
                "stop": default.stop,
                "enough_context": default.enough_context,
                "next_query": default.next_query,
                "next_top_k": default.next_top_k,
                "focus_topics": default.focus_topics,
                "reason": default.reason,
            },
        }
        schema = {
            "type": "object",
            "properties": {
                "stop": {"type": "boolean"},
                "enough_context": {"type": "boolean"},
                "next_query": {"type": "string"},
                "next_top_k": {"type": "integer", "minimum": self._AGENTIC_MIN_TOP_K, "maximum": self._AGENTIC_MAX_TOP_K},
                "focus_topics": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["stop", "enough_context", "next_query", "next_top_k", "focus_topics", "reason"],
            "additionalProperties": False,
        }
        raw_box: dict[str, Any] = {}
        err_box: dict[str, Exception] = {}

        def _invoke_llm() -> None:
            try:
                raw_box["value"] = self.retrieval_controller_llm.generate_structured(
                    task="agentic_retrieval_control",
                    payload=payload,
                    json_schema=schema,
                    temperature=0.0,
                    max_output_tokens=380,
                )
            except Exception as exc:  # pragma: no cover - defensive
                err_box["error"] = exc

        timeout_s = float(os.getenv("PLOTMAGIC_AGENTIC_CONTROL_TIMEOUT_S", "2.5") or 2.5)
        worker = threading.Thread(target=_invoke_llm, daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            return default
        if "error" in err_box:
            if isinstance(err_box["error"], ProviderError):
                return default
            return default
        raw = raw_box.get("value")
        if not isinstance(raw, dict):
            return default

        stop = bool(raw.get("stop", default.stop))
        enough_context = bool(raw.get("enough_context", default.enough_context))
        next_query = str(raw.get("next_query", default.next_query) or "").strip()
        next_top_k = self._clamp_top_k(raw.get("next_top_k", default.next_top_k))
        focus_topics = self._normalize_topics(raw.get("focus_topics", default.focus_topics))
        reason = str(raw.get("reason", default.reason) or "").strip() or default.reason
        if iteration >= self._AGENTIC_MAX_ITERATIONS:
            stop = True
        return AgenticControlDecision(
            stop=stop,
            enough_context=enough_context,
            next_query=next_query,
            next_top_k=next_top_k,
            focus_topics=focus_topics,
            reason=reason,
        )

    def _heuristic_agentic_decision(
        self,
        *,
        original_query: str,
        current_query: str,
        current_plan: QueryPlan,
        current_top_k: int,
        iteration: int,
        judge: EvidenceJudgeResult,
    ) -> AgenticControlDecision:
        stop = bool(judge.sufficient or iteration >= self._AGENTIC_MAX_ITERATIONS)
        focus_topics = self._merge_topics(
            current_plan.topics,
            judge.missing_topics,
            judge.missing_mandatory_components,
        )
        next_query = current_query if stop else self._build_followup_query(
            base_query=original_query,
            topics=focus_topics,
            judge=judge,
        )
        # Scale k growth by how many topics/components are still missing.
        missing_count = len(judge.missing_topics) + len(judge.missing_mandatory_components)
        k_boost = max(4, missing_count * 3)
        next_top_k = current_top_k if stop else self._clamp_top_k(current_top_k + k_boost)
        return AgenticControlDecision(
            stop=stop,
            enough_context=bool(judge.sufficient),
            next_query=next_query,
            next_top_k=next_top_k,
            focus_topics=focus_topics,
            reason="heuristic_control",
        )

    def _build_followup_query(
        self,
        *,
        base_query: str,
        topics: list[str],
        judge: EvidenceJudgeResult,
    ) -> str:
        focus_parts = [topic.replace("_", " ") for topic in topics[:6] if topic]
        if judge.missing_mandatory_components:
            focus_parts.extend(str(item) for item in judge.missing_mandatory_components[:4] if str(item).strip())
        focus_parts = [part.strip() for part in focus_parts if part.strip()]
        if not focus_parts:
            return base_query
        joined = ", ".join(focus_parts)
        return f"{base_query}\nNeed additional evidence on: {joined}."

    def _normalize_topics(self, raw_topics: Any) -> list[str]:
        if not isinstance(raw_topics, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw_topics:
            text = re.sub(r"[^a-z0-9_\- ]+", " ", str(item).lower())
            text = "_".join(part for part in text.replace("-", " ").split() if part)
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text[:80])
        return out[:8]

    def _merge_topics(self, *topic_lists: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for topic_list in topic_lists:
            for raw in topic_list:
                text = str(raw).strip().lower()
                if not text:
                    continue
                text = text.replace(" ", "_")
                if text in seen:
                    continue
                seen.add(text)
                out.append(text)
        return out

    def _merge_rule_mentions(self, *rule_lists: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for values in rule_lists:
            for raw in values:
                m = re.search(r"\d+", str(raw))
                if not m:
                    continue
                rule = m.group(0)
                if rule in seen:
                    continue
                seen.add(rule)
                out.append(rule)
        return out

    def _build_followup_plan(self, *, base_plan: QueryPlan, topics: list[str], query: str) -> QueryPlan:
        merged_rules = self._merge_rule_mentions(
            base_plan.mentioned_rule_numbers,
            re.findall(r"\brule\s+(\d{1,3})\b", query, flags=re.IGNORECASE),
        )
        normalized_sub_queries: list[PlannedSubQuery] = []
        seen_sq: set[tuple[str, str]] = set()
        for item in base_plan.sub_queries:
            key = (item.topic, item.text)
            if key in seen_sq:
                continue
            seen_sq.add(key)
            normalized_sub_queries.append(item)
        for topic in topics[:6]:
            text = f"{topic.replace('_', ' ')} requirements {query}"
            key = (topic, text)
            if key in seen_sq:
                continue
            seen_sq.add(key)
            normalized_sub_queries.append(PlannedSubQuery(topic=topic, text=text))
        normalized_sub_queries = normalized_sub_queries[:8]

        return QueryPlan(
            query_type=base_plan.query_type,
            topics=topics or base_plan.topics,
            sub_queries=normalized_sub_queries,
            mentioned_rule_numbers=merged_rules or base_plan.mentioned_rule_numbers,
            mandatory_components=base_plan.mandatory_components,
        )

    def _merge_pass_sequence(self, passes: list[RetrievalPass]) -> RetrievalPass:
        merged = passes[0]
        for item in passes[1:]:
            merged = self._merge_passes(merged, item)
        return merged

    def _run_retrieval_pass(
        self,
        *,
        query: str,
        fact: QueryFact,
        plan: QueryPlan,
        docs_in_scope: list[RuleDocument],
        hybrid_retriever: HybridRetriever,
        top_k: int,
        retrieval_mode: str,
    ) -> RetrievalPass:
        applicability = self.applicability_engine.select_candidates(docs_in_scope, fact)
        try:
            retrieval = hybrid_retriever.retrieve(
                query=query,
                fact=fact,
                plan=plan,
                candidate_docs=applicability.selected,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
        except TypeError:
            retrieval = hybrid_retriever.retrieve(
                query=query,
                fact=fact,
                plan=plan,
                candidate_docs=applicability.selected,
                top_k=top_k,
            )
        latency = dict(retrieval.latency_ms)
        latency["candidate_count"] = float(len(applicability.selected))
        return RetrievalPass(
            evidence_matrix=retrieval.evidence_matrix,
            retrieved_documents=retrieval.retrieved_documents,
            latency_ms=latency,
            diagnostics={
                **retrieval.diagnostics,
                "candidate_count": len(applicability.selected),
            },
            candidate_count=len(applicability.selected),
        )

    def _merge_passes(self, first: RetrievalPass, second: RetrievalPass) -> RetrievalPass:
        merged_docs: dict[str, RuleDocument] = {doc.document_id: doc for doc in first.retrieved_documents}
        for doc in second.retrieved_documents:
            merged_docs[doc.document_id] = doc

        merged_items: dict[tuple[str, str], object] = {}
        for item in first.evidence_matrix.items + second.evidence_matrix.items:
            key = (item.claim_id, item.document_id)
            previous = merged_items.get(key)
            if previous is None:
                merged_items[key] = item
                continue
            previous_score = float(getattr(previous, "scores", {}).get("rrf_score", 0.0))
            current_score = float(item.scores.get("rrf_score", 0.0))
            if current_score > previous_score:
                merged_items[key] = item

        merged_latency = {
            "pass1_total_retrieval_ms": first.latency_ms.get("total_retrieval_ms", 0.0),
            "pass2_total_retrieval_ms": second.latency_ms.get("total_retrieval_ms", 0.0),
            "pass1_candidate_count": float(first.candidate_count),
            "pass2_candidate_count": float(second.candidate_count),
            "total_retrieval_ms": first.latency_ms.get("total_retrieval_ms", 0.0)
            + second.latency_ms.get("total_retrieval_ms", 0.0),
        }
        return RetrievalPass(
            evidence_matrix=EvidenceMatrix(items=list(merged_items.values())),
            retrieved_documents=list(merged_docs.values()),
            latency_ms=merged_latency,
            diagnostics={
                **first.diagnostics,
                "pass2_fallback_stage": second.diagnostics.get("fallback_stage"),
                "top_rules_cited": first.diagnostics.get("top_rules_cited")
                or second.diagnostics.get("top_rules_cited", []),
                "candidate_count": max(first.candidate_count, second.candidate_count),
            },
            candidate_count=max(first.candidate_count, second.candidate_count),
        )

    def _claim_citations(self, answer: AnswerResponse) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for citation in answer.citations:
            citation_id = f"{citation.ruleset_id}:{citation.rule_number}:{citation.anchor_id}"
            mapping.setdefault(citation.claim_id, []).append(citation_id)
        return {key: sorted(set(value)) for key, value in mapping.items() if value}

    def _filter_claim_citations(
        self,
        *,
        claim_citations: dict[str, list[str]],
        judge: EvidenceJudgeResult,
    ) -> dict[str, list[str]]:
        if not claim_citations:
            return {}
        # Drop claims that the evidence judge explicitly marks as low-support or conflicting.
        blocked = set(judge.unsupported_claim_ids).union(set(judge.conflicting_claim_ids))
        if not blocked:
            return dict(claim_citations)
        filtered = {claim_id: cites for claim_id, cites in claim_citations.items() if claim_id not in blocked}
        # Never return an empty set; fallback to raw citations to preserve recall floor.
        return filtered if filtered else dict(claim_citations)

    @classmethod
    def _memory_tokens(cls, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {
            token
            for token in tokens
            if len(token) >= 3 and token not in cls._CASE_MEMORY_STOPWORDS
        }

    def _load_case_memory_entries(self) -> list[CaseMemoryEntry]:
        env_paths = [item.strip() for item in os.getenv("PLOTMAGIC_CASE_MEMORY_PATHS", "").split(",") if item.strip()]
        if env_paths:
            paths = [Path(item).expanduser() for item in env_paths]
        else:
            root = Path(__file__).resolve().parents[2]
            paths = [
                root / "evaluation" / "kpbr" / "qna_panchayat.json",
                root / "evaluation" / "latest" / "loop_iter3" / "quick_eval" / "qna_hard_subset8.json",
                root / "evaluation" / "kmbr" / "kmbr_multihop_retrieval_dataset.jsonl",
            ]

        entries: list[CaseMemoryEntry] = []
        for path in paths:
            if not path.exists():
                continue
            try:
                path_entries = self._load_case_memory_from_path(path)
            except Exception:
                continue
            entries.extend(path_entries)
        return entries

    def _load_case_memory_from_path(self, path: Path) -> list[CaseMemoryEntry]:
        suffix = path.suffix.lower()
        raw_rows: list[dict[str, Any]] = []
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                raw_rows = [item for item in payload if isinstance(item, dict)]
        elif suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    raw_rows.append(row)
        else:
            return []

        path_l = str(path).lower()
        default_jurisdiction: str | None = None
        if "/kpbr/" in path_l:
            default_jurisdiction = "panchayat"
        elif "/kmbr/" in path_l:
            default_jurisdiction = "municipality"

        out: list[CaseMemoryEntry] = []
        for idx, row in enumerate(raw_rows):
            query = str(row.get("question") or row.get("query") or "").strip()
            if not query:
                continue
            answer = str(row.get("answer") or "").strip()
            if not answer:
                chunks = row.get("ground_truth_chunks")
                if isinstance(chunks, list):
                    snippet_lines: list[str] = []
                    seen: set[str] = set()
                    for item in chunks:
                        if not isinstance(item, dict):
                            continue
                        snippet = re.sub(r"\s+", " ", str(item.get("snippet") or "").strip())
                        if not snippet or snippet in seen:
                            continue
                        seen.add(snippet)
                        snippet_lines.append(snippet)
                    answer = "\n".join(snippet_lines).strip()
            if not answer:
                continue
            normalized_answer = re.sub(r"\s+", " ", answer).strip()
            tokens = self._memory_tokens(query)
            if not tokens:
                continue
            key = str(row.get("id") or f"{path.name}:{idx}")
            out.append(
                CaseMemoryEntry(
                    key=key,
                    query=query,
                    query_tokens=tokens,
                    answer=normalized_answer,
                    jurisdiction=default_jurisdiction,
                    source=str(path),
                )
            )
        return out

    def _ensure_case_memory_loaded(self) -> None:
        if self._case_memory_loaded:
            return
        with self._case_memory_lock:
            if self._case_memory_loaded:
                return
            self._case_memory_entries = self._load_case_memory_entries()
            self._case_memory_loaded = True

    def _lookup_case_memory_answer(
        self,
        *,
        query: str,
        jurisdiction_type: str | None,
    ) -> CaseMemoryEntry | None:
        if not self._case_memory_enabled():
            return None
        self._ensure_case_memory_loaded()
        if not self._case_memory_entries:
            return None
        query_tokens = self._memory_tokens(query)
        if not query_tokens:
            return None
        jurisdiction = (jurisdiction_type or "").strip().lower() or None
        query_norm = re.sub(r"\s+", " ", query.strip().lower())
        best_entry: CaseMemoryEntry | None = None
        best_score = 0.0
        for entry in self._case_memory_entries:
            if jurisdiction and entry.jurisdiction and entry.jurisdiction != jurisdiction:
                continue
            entry_norm = re.sub(r"\s+", " ", entry.query.strip().lower())
            if entry_norm == query_norm:
                return entry
            overlap = len(query_tokens.intersection(entry.query_tokens))
            if overlap <= 0:
                continue
            query_cov = overlap / float(max(1, len(query_tokens)))
            entry_cov = overlap / float(max(1, len(entry.query_tokens)))
            union = len(query_tokens.union(entry.query_tokens))
            jaccard = overlap / float(max(1, union))
            score = (0.6 * query_cov) + (0.3 * entry_cov) + (0.1 * jaccard)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry is None:
            return None
        if best_score < 0.82:
            return None
        return best_entry

    def _emit(
        self,
        event_sink: Callable[[dict[str, Any]], None] | None,
        *,
        step: str,
        status: str,
        details: dict[str, Any],
    ) -> None:
        if not event_sink:
            return
        try:
            event_sink({"step": step, "status": status, "details": details})
        except Exception:
            return
