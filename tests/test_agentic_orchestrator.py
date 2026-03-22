from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.agentic import AgenticQueryOrchestrator, ComplianceBriefComposer, EvidenceJudge
from src.generation import AnswerGenerator
from src.models import ClauseNode, ClauseType, QueryFact, RuleDocument
from src.providers.adapters.no_llm import NoLLMProvider
from src.providers.config import ProviderSettings
from src.retrieval.evidence import EvidenceMatrix, RetrievalEvidence
from src.retrieval.hybrid_retriever import HybridRetrievalResult
from src.retrieval.query_planner import QueryPlan


def _doc(doc_id: str, rule_number: str, topic: str) -> RuleDocument:
    clause = ClauseNode(
        clause_id=f"{doc_id}-rule",
        clause_type=ClauseType.RULE,
        state="kerala",
        jurisdiction_type="panchayat",
        ruleset_id="KPBR_2011",
        ruleset_version="2011",
        chapter_number=1,
        chapter_title="General",
        rule_number=rule_number,
        rule_title=f"Rule {rule_number}",
        sub_rule_path="",
        display_citation=f"Rule {rule_number}",
        source_file="data/kerala/kpbr_panchayat_rule.md",
        anchor_id=f"kpbr-ch1-r{rule_number}",
        raw_text="sample text",
        normalized_text="sample text",
        topic_tags=[topic],
    )
    return RuleDocument(
        document_id=doc_id,
        state="kerala",
        jurisdiction_type="panchayat",
        ruleset_id="KPBR_2011",
        ruleset_version="2011",
        issuing_authority="LSGD",
        effective_from=date(2011, 2, 14),
        effective_to=None,
        source_file="data/kerala/kpbr_panchayat_rule.md",
        chapter_number=1,
        chapter_title="General",
        rule_number=rule_number,
        rule_title=f"Rule {rule_number}",
        full_text="sample text",
        anchor_id=f"kpbr-ch1-r{rule_number}",
        clause_nodes=[clause],
    )


class _ApplicabilityResult:
    def __init__(self, selected: list[RuleDocument]) -> None:
        self.selected = selected
        self.reasons: dict[str, list[str]] = {}


class _FakeApplicability:
    def select_candidates(self, docs: list[RuleDocument], fact: QueryFact) -> _ApplicabilityResult:
        return _ApplicabilityResult(docs)


class _FakeHybridRetriever:
    def __init__(self, responses: list[HybridRetrievalResult]) -> None:
        self.responses = responses
        self.call_count = 0
        self.top_k_history: list[int] = []
        self.query_history: list[str] = []

    def retrieve(
        self,
        query: str,
        fact: QueryFact,
        plan: QueryPlan,
        candidate_docs: list[RuleDocument],
        top_k: int = 15,
    ) -> HybridRetrievalResult:
        self.top_k_history.append(top_k)
        self.query_history.append(query)
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]


def _response(doc: RuleDocument, topic: str, claim_id: str, supported: bool = True) -> HybridRetrievalResult:
    matrix = EvidenceMatrix(
        items=[
            RetrievalEvidence(
                claim_id=claim_id,
                topic=topic,
                chunk_id=doc.document_id,
                document_id=doc.document_id,
                text=f"{topic} requirement from {doc.rule_number}",
                scores={"rrf_score": 0.7},
                source="hybrid",
                has_sufficient_support=supported,
            )
        ]
    )
    return HybridRetrievalResult(
        evidence_matrix=matrix,
        retrieved_documents=[doc],
        latency_ms={"total_retrieval_ms": 1.0},
    )


def _orchestrator() -> AgenticQueryOrchestrator:
    composer = ComplianceBriefComposer(NoLLMProvider(ProviderSettings(provider_id="no_llm")))
    return AgenticQueryOrchestrator(
        applicability_engine=_FakeApplicability(),
        answer_generator=AnswerGenerator(),
        brief_composer=composer,
        evidence_judge=EvidenceJudge(),
    )


def test_orchestrator_success_no_retry() -> None:
    doc = _doc("KPBR_2011-ch2-r10", "10", "permit")
    retriever = _FakeHybridRetriever([_response(doc, "permit", f"permit-{doc.document_id}")])
    out = _orchestrator().run(
        query="permit process",
        fact=QueryFact(state="kerala", jurisdiction_type="panchayat", topics=["permit"]),
        plan=QueryPlan(query_type="procedural", topics=["permit"]),
        docs_in_scope=[doc],
        hybrid_retriever=retriever,
        top_k=10,
        debug_trace=True,
    )
    assert out.verdict == "depends"
    assert retriever.call_count == 1
    assert out.final_answer is not None
    assert out.final_answer.verdict == "depends"
    assert out.claim_citations
    assert out.agent_trace


def test_orchestrator_retry_then_success() -> None:
    doc_a = _doc("KPBR_2011-ch2-r10", "10", "permit")
    doc_b = _doc("KPBR_2011-ch6-r49", "49", "fire_safety")
    pass1 = _response(doc_a, "permit", f"permit-{doc_a.document_id}")
    pass2 = _response(doc_b, "fire_safety", f"fire_safety-{doc_b.document_id}")
    retriever = _FakeHybridRetriever([pass1, pass2])

    out = _orchestrator().run(
        query="permit and fire approval",
        fact=QueryFact(state="kerala", jurisdiction_type="panchayat", topics=["permit", "fire_safety"]),
        plan=QueryPlan(query_type="procedural", topics=["permit", "fire_safety"]),
        docs_in_scope=[doc_a, doc_b],
        hybrid_retriever=retriever,
        top_k=10,
    )
    assert out.verdict == "depends"
    assert retriever.call_count == 2
    assert out.grounding is not None
    assert out.grounding.missing_topics == []


def test_orchestrator_partial_answer_after_retry_when_some_support_exists() -> None:
    doc = _doc("KPBR_2011-ch2-r10", "10", "permit")
    pass1 = _response(doc, "permit", f"permit-{doc.document_id}")
    pass2 = _response(doc, "permit", f"permit-{doc.document_id}")
    retriever = _FakeHybridRetriever([pass1, pass2])

    out = _orchestrator().run(
        query="permit and fire approval",
        fact=QueryFact(state="kerala", jurisdiction_type="panchayat", topics=["permit", "fire_safety"]),
        plan=QueryPlan(query_type="procedural", topics=["permit", "fire_safety"]),
        docs_in_scope=[doc],
        hybrid_retriever=retriever,
        top_k=10,
    )
    assert out.verdict == "depends"
    assert out.final_answer is not None
    assert out.final_answer.verdict == "depends"
    assert out.grounding is not None
    assert out.grounding.partial is True
    assert out.grounding.missing_topics


def test_orchestrator_agentic_dynamic_uses_multi_hop_and_dynamic_k() -> None:
    doc_a = _doc("KPBR_2011-ch2-r10", "10", "permit")
    doc_b = _doc("KPBR_2011-ch6-r49", "49", "fire_safety")
    pass1 = _response(doc_a, "permit", f"permit-{doc_a.document_id}")
    pass2 = _response(doc_b, "fire_safety", f"fire_safety-{doc_b.document_id}")
    retriever = _FakeHybridRetriever([pass1, pass2])

    out = _orchestrator().run(
        query="permit and fire approval",
        fact=QueryFact(state="kerala", jurisdiction_type="panchayat", topics=["permit", "fire_safety"]),
        plan=QueryPlan(query_type="procedural", topics=["permit", "fire_safety"]),
        docs_in_scope=[doc_a, doc_b],
        hybrid_retriever=retriever,
        top_k=10,
        retrieval_mode="agentic_dynamic",
        debug_trace=True,
    )
    assert out.verdict == "depends"
    assert retriever.call_count == 2
    assert len(retriever.top_k_history) >= 2
    assert retriever.top_k_history[0] > 10
    assert retriever.top_k_history[1] >= retriever.top_k_history[0]
    steps = {item.step for item in out.agent_trace}
    assert "agentic_control_pass_1" in steps
    assert "agentic_control_pass_2" in steps
