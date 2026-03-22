from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable

from src.agentic import AgenticQueryOrchestrator, ComplianceBriefComposer, EvidenceJudge
from src.generation import AnswerGenerator
from src.indexing import InMemoryVectorStore, LexicalIndex, PersistentLocalVectorStore, PgVectorStore, StructuredStore
from src.indexing import QdrantLocalVectorStore
from src.indexing.vector_store import VectorStore
from src.ingestion import IngestionPipeline
from src.models import ClauseType, QueryFact, RuleDocument
from src.models.schemas import AnswerResponse, ApplicabilityRequest, IngestResult, QueryRequest
from src.policy import LegalRagPolicyPack, PolicyLoader
from src.providers import ProviderFactory, build_default_registry, load_providers_config
from src.retrieval import (
    ApplicabilityEngine,
    FactExtractor,
    HybridRetriever,
    OccupancyResolver,
    QueryPlanner,
    ScopeResolver,
)
from src.statepack import StatePackFactory


@dataclass(slots=True)
class RuntimeState:
    docs: list[RuleDocument]
    hybrid_retriever: HybridRetriever | None


class ComplianceEngine:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pipeline = IngestionPipeline(root / "config" / "states.yaml")
        self.states_config = self.pipeline.config.get("states", {})
        self.statepack_factory = StatePackFactory(root)
        self.policy_loader = PolicyLoader(root)
        self.policy_cache: dict[str, LegalRagPolicyPack] = {}
        self.active_policy_state: str | None = None
        self.scope_resolvers: dict[str, ScopeResolver] = {}
        self.occupancy_resolvers: dict[str, OccupancyResolver] = {}
        self.applicability_engine = ApplicabilityEngine()
        self.fact_extractor = FactExtractor()
        self.query_planner = QueryPlanner()
        self.answer_generator = AnswerGenerator()

        (root / ".cache").mkdir(exist_ok=True)
        self.providers_config = load_providers_config(root / "config" / "providers.yaml")
        self.provider_registry = build_default_registry()
        self.provider_factory = ProviderFactory(self.provider_registry, self.providers_config)
        self.embedding_provider = self.provider_factory.create_embedding_provider()
        self.reranker_provider = self.provider_factory.create_reranker_provider()
        self.llm_provider = self.provider_factory.create_llm_provider()
        self.provider_health = self.provider_factory.health_snapshot(
            embedding_provider=self.embedding_provider,
            reranker_provider=self.reranker_provider,
            llm_provider=self.llm_provider,
        )
        self.provider_diagnostics = list(self.provider_factory.diagnostics)

        self.structured_store = StructuredStore(root / ".cache" / "structured_rules.db")
        self.lexical_index = LexicalIndex()
        self.vector_store = self._build_vector_store()
        self.agentic_orchestrator = AgenticQueryOrchestrator(
            applicability_engine=self.applicability_engine,
            answer_generator=self.answer_generator,
            brief_composer=ComplianceBriefComposer(self.llm_provider),
        )
        self.state = RuntimeState(docs=[], hybrid_retriever=None)

    def ingest(self, state: str, jurisdiction_type: str | None = None) -> IngestResult:
        state_code = state.strip().lower()
        docs, stats = self.pipeline.ingest_state(state=state_code, jurisdiction_type=jurisdiction_type)
        if jurisdiction_type:
            # Replace docs for selected jurisdiction.
            retained = [
                item
                for item in self.state.docs
                if not (item.state == state_code and item.jurisdiction_type == jurisdiction_type)
            ]
            self.state.docs = retained + docs
        else:
            self.state.docs = [item for item in self.state.docs if item.state != state_code] + docs

        self.structured_store.upsert_documents(docs)
        self.lexical_index.build(self.state.docs)
        vector_clauses = self._dedupe_clauses_for_vector_index(self.state.docs)
        self.vector_store.upsert_clauses(vector_clauses)
        self.state.hybrid_retriever = self._build_hybrid_retriever()
        self._activate_policy_runtime(state_code)
        return IngestResult(
            state=state_code,
            ruleset_ids=sorted(set(item.ruleset_id for item in docs)),
            parsed_rules=stats.parsed_rules,
            parsed_clauses=stats.parsed_clauses,
            parsed_tables=stats.parsed_tables,
            parsed_files=stats.parsed_files,
            failed_files=stats.failed_files,
            parse_quality_score=stats.parse_quality_score,
            parse_quality=stats.parse_quality,
            warnings=stats.warnings,
        )

    def resolve_applicability(self, request: ApplicabilityRequest) -> dict[str, object]:
        inferred = self._infer_scope_hints(request.building_description or "")
        requested_state = (request.state or inferred["state_hint"] or self._default_state()).lower()
        scope = self._scope_resolver(requested_state).resolve(
            location=request.location,
            state_hint=request.state or inferred["state_hint"],
            jurisdiction_hint=request.jurisdiction_type or inferred["jurisdiction_hint"],
            panchayat_category_hint=request.panchayat_category or inferred["panchayat_category_hint"],
        )
        if not scope.resolved:
            return {
                "resolved": False,
                "clarifications": scope.clarification_questions,
                "reasons": scope.reasons,
            }
        occupancy = self._occupancy_resolver(scope.state or requested_state).resolve(
            state=scope.state or requested_state,
            building_description=request.building_description,
            explicit_occupancy=None,
        )
        return {
            "resolved": occupancy.resolved,
            "scope": scope,
            "occupancy": occupancy,
        }

    def query(
        self,
        request: QueryRequest,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> AnswerResponse:
        inferred = self._infer_scope_hints(request.query)
        requested_state = (request.state or inferred["state_hint"] or self._default_state()).lower()
        if not self.state.docs:
            self.ingest(state=requested_state)
        self._activate_policy_runtime(requested_state)

        start_total = perf_counter()
        self._emit(
            event_sink,
            step="tool.query_start",
            status="running",
            details={
                "top_k": request.top_k,
                "debug_trace": request.debug_trace,
                "retrieval_mode": request.retrieval_mode,
            },
        )
        self._emit(event_sink, step="tool.scope_resolver", status="running", details={"state_hint": requested_state})
        scope = self._scope_resolver(requested_state).resolve(
            location=request.location,
            state_hint=request.state or inferred["state_hint"],
            jurisdiction_hint=request.jurisdiction_type or inferred["jurisdiction_hint"],
            panchayat_category_hint=request.panchayat_category or inferred["panchayat_category_hint"],
        )
        if not scope.resolved:
            self._emit(
                event_sink,
                step="tool.scope_resolver",
                status="needs_clarification",
                details={"questions": scope.clarification_questions},
            )
            return AnswerResponse(
                jurisdiction="unknown",
                occupancy_groups=[],
                clarifications=[
                    {
                        "code": "SCOPE_REQUIRED",
                        "question": question,
                        "options": [],
                    }
                    for question in scope.clarification_questions
                ],
            )
        self._emit(
            event_sink,
            step="tool.scope_resolver",
            status="ok",
            details={
                "state": scope.state,
                "jurisdiction_type": scope.jurisdiction_type,
                "ruleset_id": scope.ruleset_id,
            },
        )

        plan = self.query_planner.plan(request.query)
        self._emit(
            event_sink,
            step="tool.query_planner",
            status="ok",
            details={"query_type": plan.query_type, "topics": plan.topics, "mentioned_rules": plan.mentioned_rule_numbers},
        )

        occ_start = perf_counter()
        self._emit(event_sink, step="tool.occupancy_resolver", status="running", details={})
        occupancy = self._occupancy_resolver(scope.state or requested_state).resolve(
            state=scope.state or requested_state,
            building_description=request.query,
            explicit_occupancy=request.explicit_occupancy,
        )
        occupancy_ms = (perf_counter() - occ_start) * 1000
        if self._can_skip_occupancy(plan=plan, query=request.query):
            occupancy = type(occupancy)(resolved=True, candidates=occupancy.candidates, selected=[])
            self._emit(
                event_sink,
                step="tool.occupancy_resolver",
                status="skipped",
                details={"reason": "procedural_or_generic"},
            )
        elif not occupancy.resolved:
            self._emit(
                event_sink,
                step="tool.occupancy_resolver",
                status="needs_clarification",
                details={"questions": occupancy.clarification_questions},
            )
            return AnswerResponse(
                jurisdiction=f"{scope.state}::{scope.jurisdiction_type}",
                occupancy_groups=occupancy.selected,
                clarifications=[
                    {
                        "code": "OCCUPANCY_REQUIRED",
                        "question": question,
                        "options": [candidate.code for candidate in occupancy.candidates],
                    }
                    for question in occupancy.clarification_questions
                ],
                latency_ms={"occupancy_resolution_ms": occupancy_ms},
            )
        else:
            self._emit(
                event_sink,
                step="tool.occupancy_resolver",
                status="ok",
                details={"selected": occupancy.selected},
            )

        fact = QueryFact(
            state=scope.state,
            location_text=request.location,
            jurisdiction_type=scope.jurisdiction_type,
            panchayat_category=scope.panchayat_category,
            occupancies=occupancy.selected,
            topics=plan.topics,
            query_date=request.query_date,
            mentioned_rules=plan.mentioned_rule_numbers,
            query_intent=plan.query_type,
        )
        fact = self.fact_extractor.extract(request.query, seed=fact)
        self._emit(
            event_sink,
            step="tool.fact_extractor",
            status="ok",
            details={
                "height_m": fact.height_m,
                "floor_area_sqm": fact.floor_area_sqm,
                "plot_area_sqm": fact.plot_area_sqm,
                "floors": fact.floors,
            },
        )

        docs_in_scope = [
            doc
            for doc in self.state.docs
            if doc.state == scope.state and doc.jurisdiction_type == scope.jurisdiction_type and doc.ruleset_id == scope.ruleset_id
        ]
        self._emit(
            event_sink,
            step="tool.scope_filter",
            status="ok",
            details={"docs_in_scope": len(docs_in_scope)},
        )

        if not self.state.hybrid_retriever:
            self.state.hybrid_retriever = self._build_hybrid_retriever()

        answer = self.agentic_orchestrator.run(
            query=request.query,
            fact=fact,
            plan=plan,
            docs_in_scope=docs_in_scope,
            hybrid_retriever=self.state.hybrid_retriever,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            debug_trace=request.debug_trace,
            event_sink=event_sink,
        )
        answer.latency_ms = {
            **answer.latency_ms,
            "occupancy_resolution_ms": occupancy_ms,
            "total_ms": (perf_counter() - start_total) * 1000,
        }
        self._emit(
            event_sink,
            step="tool.query_complete",
            status="ok",
            details={"verdict": answer.verdict, "total_ms": answer.latency_ms.get("total_ms", 0.0)},
        )
        return answer

    def _scope_resolver(self, state: str) -> ScopeResolver:
        state = state.strip().lower()
        if state not in self.scope_resolvers:
            pack = self.statepack_factory.load(state)
            self.scope_resolvers[state] = ScopeResolver(
                self.root / "config" / "states.yaml",
                pack.scope_resolver_config,
            )
        return self.scope_resolvers[state]

    def _occupancy_resolver(self, state: str) -> OccupancyResolver:
        state = state.strip().lower()
        if state not in self.occupancy_resolvers:
            pack = self.statepack_factory.load(state)
            self.occupancy_resolvers[state] = OccupancyResolver(pack.occupancy_mapping_config)
        return self.occupancy_resolvers[state]

    def _infer_scope_hints(self, text: str) -> dict[str, str | None]:
        lowered = text.lower()
        jurisdiction_hint: str | None = None
        if any(token in lowered for token in ["panchayat", "grama panchayat", "village panchayat"]):
            jurisdiction_hint = "panchayat"
        if any(token in lowered for token in ["municipality", "municipal", "corporation", "city"]):
            jurisdiction_hint = "municipality" if jurisdiction_hint is None else jurisdiction_hint

        for state_code, state_cfg in self.states_config.items():
            for j_type, j_cfg in state_cfg.get("jurisdictions", {}).items():
                ruleset_id = str(j_cfg.get("ruleset_id", "")).lower()
                if ruleset_id and ruleset_id.lower() in lowered:
                    jurisdiction_hint = j_type if jurisdiction_hint is None else jurisdiction_hint

        category_hint: str | None = None
        if re.search(r"category[\s-]*ii\b", lowered):
            category_hint = "Category-II"
        elif re.search(r"category[\s-]*i\b", lowered):
            category_hint = "Category-I"

        state_hint: str | None = None
        for state_code in self.states_config.keys():
            if state_code.lower() in lowered:
                state_hint = state_code
                break
        return {
            "state_hint": state_hint,
            "jurisdiction_hint": jurisdiction_hint,
            "panchayat_category_hint": category_hint,
        }

    def _default_state(self) -> str:
        if not self.states_config:
            raise ValueError("No states configured. Cannot determine default state.")
        return sorted(self.states_config.keys())[0]

    def _allow_generic_without_occupancy(self, query: str) -> bool:
        lowered = query.lower()
        project_specific_markers = [
            "can i build",
            "can we build",
            "proposed building",
            "proposed house",
            "for my building",
            "for my house",
            "for my plot",
            "on my land",
            "i am planning",
            "i plan to",
            "i want to construct",
            "how much can i build",
        ]
        if any(marker in lowered for marker in project_specific_markers):
            return False
        if (" my " in f" {lowered} " or " our " in f" {lowered} ") and any(
            token in lowered for token in ["plot", "building", "house", "land"]
        ):
            return False
        return True

    def _can_skip_occupancy(self, plan, query: str) -> bool:
        state = self.active_policy_state or self._default_state()
        policy = self._policy_pack(state).applicability
        if not policy.allow_generic_without_occupancy:
            return False
        if plan.query_type == "procedural" and not policy.procedural_occupancy_agnostic:
            return False
        return self._allow_generic_without_occupancy(query)

    def _build_hybrid_retriever(self) -> HybridRetriever:
        state = self.active_policy_state or self._default_state()
        retrieval_policy = self._policy_pack(state).retrieval
        min_evidence_env = os.getenv("PLOTMAGIC_RETRIEVAL_MIN_EVIDENCE_SCORE", "").strip()
        pool_factor_env = os.getenv("PLOTMAGIC_RETRIEVAL_POOL_FACTOR", "").strip()
        if min_evidence_env:
            retrieval_policy = replace(
                retrieval_policy,
                min_evidence_score=float(min_evidence_env),
            )
        if pool_factor_env:
            retrieval_policy = replace(
                retrieval_policy,
                candidate_pool_factor=float(pool_factor_env),
            )
        return HybridRetriever(
            vector_store=self.vector_store,
            lexical_index=self.lexical_index,
            structured_store=self.structured_store,
            all_docs=self.state.docs,
            reranker_provider=self.reranker_provider,
            rerank_top_n=self.providers_config.feature_flags.rerank_top_n,
            policy=retrieval_policy,
        )

    def _policy_pack(self, state: str) -> LegalRagPolicyPack:
        state_l = state.strip().lower()
        if state_l not in self.policy_cache:
            pack = self.statepack_factory.load(state_l)
            self.policy_cache[state_l] = self.policy_loader.load(pack.policy_profiles)
        return self.policy_cache[state_l]

    def _activate_policy_runtime(self, state: str) -> None:
        state_l = state.strip().lower()
        if self.active_policy_state == state_l:
            return
        policy_pack = self._policy_pack(state_l)
        self.applicability_engine = ApplicabilityEngine(policy=policy_pack.applicability)
        self.query_planner = QueryPlanner()
        self.agentic_orchestrator = AgenticQueryOrchestrator(
            applicability_engine=self.applicability_engine,
            answer_generator=self.answer_generator,
            brief_composer=ComplianceBriefComposer(self.llm_provider, policy=policy_pack.generation),
            evidence_judge=EvidenceJudge(policy=policy_pack.abstention),
        )
        self.active_policy_state = state_l
        if self.state.docs:
            self.state.hybrid_retriever = self._build_hybrid_retriever()

    @staticmethod
    def _dedupe_clauses_for_vector_index(docs: list[RuleDocument]) -> list[ClauseNode]:
        """Filter and deduplicate clauses for vector indexing.

        Keeps core clause types (rule, proviso, note, table, appendix, etc.)
        and includes sub_rules only when they add distinct content beyond their
        parent rule text. Excludes table_row and table_cell (redundant with table).
        """
        _ALWAYS_INDEX = {
            ClauseType.RULE, ClauseType.PROVISO, ClauseType.NOTE,
            ClauseType.TABLE, ClauseType.APPENDIX, ClauseType.SCHEDULE,
            ClauseType.DEFINITION, ClauseType.CHAPTER,
        }
        _SKIP = {ClauseType.TABLE_ROW, ClauseType.TABLE_CELL}

        result: list[ClauseNode] = []
        seen_keys: set[str] = set()

        for doc in docs:
            for clause in doc.clause_nodes:
                if clause.clause_type in _SKIP:
                    continue
                if clause.clause_type in _ALWAYS_INDEX:
                    result.append(clause)
                    text_key = re.sub(r"[^a-z0-9]+", " ", (clause.normalized_text or "").lower()).strip()[:200]
                    if text_key:
                        seen_keys.add(text_key)
                    continue
                # SUB_RULE: include only if its text is distinct
                if clause.clause_type == ClauseType.SUB_RULE:
                    text_key = re.sub(r"[^a-z0-9]+", " ", (clause.normalized_text or "").lower()).strip()[:200]
                    if not text_key or text_key in seen_keys:
                        continue
                    seen_keys.add(text_key)
                    result.append(clause)
        return result

    def _build_vector_store(self) -> VectorStore:
        backend = os.getenv("PLOTMAGIC_VECTOR_BACKEND", "sqlite").strip().lower()
        if backend == "in_memory":
            return InMemoryVectorStore(self.embedding_provider)
        if backend == "qdrant_local":
            configured_path = os.getenv("PLOTMAGIC_VECTOR_DB_PATH", ".cache/qdrant_kpbr")
            qdrant_path = Path(configured_path).expanduser()
            if not qdrant_path.is_absolute():
                qdrant_path = (self.root / qdrant_path).resolve()
            collection_name = os.getenv("PLOTMAGIC_QDRANT_COLLECTION", "plotmagic_clauses").strip() or "plotmagic_clauses"
            recreate_collection = os.getenv("PLOTMAGIC_QDRANT_RECREATE_COLLECTION", "false").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            return QdrantLocalVectorStore(
                db_path=qdrant_path,
                embedding_provider=self.embedding_provider,
                collection_name=collection_name,
                recreate_collection=recreate_collection,
            )
        if backend == "pgvector":
            dsn = os.getenv("PLOTMAGIC_PGVECTOR_DSN", "").strip()
            if not dsn:
                raise ValueError(
                    "PLOTMAGIC_VECTOR_BACKEND=pgvector requires PLOTMAGIC_PGVECTOR_DSN to be configured."
                )
            table_name = os.getenv("PLOTMAGIC_PGVECTOR_TABLE", "clause_vectors").strip() or "clause_vectors"
            strict = os.getenv("PLOTMAGIC_VECTOR_BACKEND_STRICT", "true").strip().lower() not in {
                "0",
                "false",
                "no",
            }
            try:
                return PgVectorStore(
                    dsn=dsn,
                    embedding_provider=self.embedding_provider,
                    table_name=table_name,
                )
            except Exception as exc:
                if strict:
                    raise
                self.provider_diagnostics.append(
                    f"PgVector backend unavailable ({exc}); falling back to sqlite vector store."
                )

        configured_path = os.getenv("PLOTMAGIC_VECTOR_DB_PATH", ".cache/vector_index.db")
        db_path = Path(configured_path).expanduser()
        if not db_path.is_absolute():
            db_path = (self.root / db_path).resolve()
        return PersistentLocalVectorStore(db_path=db_path, embedding_provider=self.embedding_provider)

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
