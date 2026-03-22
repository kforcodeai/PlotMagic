# High-Recall Legal RAG Plan (KPBR + KMBR)

## Scope

Datasets in scope:

- `data/kerala/kpbr_panchayat_rule.md`
- `evaluation/kpbr/kpbr_multihop_retrieval_dataset_20_enriched.jsonl`
- `data/kerala/kmbr_muncipal_rules_md/*`
- `evaluation/kmbr/kmbr_multihop_retrieval_dataset.jsonl`

Primary objective: maximize recall without degrading downstream synthesis quality.

Locked decisions:

- Optimize jointly for both KPBR and KMBR (no one-sided wins).
- Run reranker experiments in isolation with explicit `retrieval_mode=hybrid_reranker` and no script-level `no_reranker` forcing.

## Baseline (Artifacts Dated 2026-03-08/09)

- KPBR retrieval benchmark (`evaluation/kpbr/kpbr_multihop_retrieval_benchmark.json`, `top_k=5`):
- `chunk_recall_at_k=0.892`
- `doc_recall_at_k=0.859`
- KPBR end-to-end baseline (`evaluation/kpbr/qna_panchayat_answer_benchmark_final.json`):
- `token_set_f1_mean=0.362`
- `token_set_recall_mean=0.555`
- `numeric_token_recall_mean=0.781`
- KMBR retrieval benchmark (`evaluation/kmbr/kmbr_multihop_retrieval_benchmark.json`, `top_k=20`):
- `chunk_recall_at_k=0.881`
- `doc_recall_at_k=0.786`
- `candidate_nonzero_rate=1.0`
- KMBR decomposed v2 best lexical (`evaluation/kmbr/kmbr_decomposed_comparison_after_v2_openai_qdrant.json`):
- `retrieval_ceiling_recall_mean=0.728`
- `answer_f1_mean=0.292`
- `answer_recall_mean=0.356`
- `answer_numeric_recall_mean=0.266`

## Non-Negotiable Guardrails

## 1) Runtime Health Gate

- No provider fallback to `hash_embedding`, `no_reranker`, or `no_llm` for production-grade comparisons.
- `provider_diagnostics` in runtime snapshots must be empty.
- If this fails, invalidate the run.

## 2) Index Consistency Gate

- Fixed vector backend per experiment batch (`qdrant_local` or `sqlite`, never mixed in same A/B).
- Fixed vector DB path per batch (`PLOTMAGIC_VECTOR_DB_PATH`).
- Stable manifest/source hash and clause counts unless ingestion was intentionally changed.

## 3) Promotion Gates

- `citation_groundedness_mean == 1.0`
- `contract_violation_rate == 0.0`
- `token_set_f1_mean >= 0.30` (precision floor)

## 4) KPBR/KMBR Parity Gate

- A phase is accepted only if both KPBR and KMBR clear gates.
- Improvement in one dataset cannot justify material regression in the other.

## Where Recall Is Lost (Actual Pipeline)

## Retrieval/Evidence Funnel

- Reranker often inactive in benchmark paths:
- default request mode is `hybrid_no_reranker`
- `benchmark_qna_answers.py` currently forces `PLOTMAGIC_RERANK_PROVIDER=no_reranker`
- Hard and soft evidence losses:
- `_topic_family_guard` hard exclusion
- `min_query_relevance` threshold
- `default_evidence_docs_per_topic` cap
- `min_evidence_score` threshold
- `max_claims_per_doc=2` hardcoded in evidence matrix

## Synthesis Funnel

- `max_claims_per_section`, summary and claim char limits can truncate coverage.
- Composer selection uses strict floor:
- `min_priority = max(0.03, max_priority * 0.72)`
- Near-duplicate key is aggressive at 200 chars:
- `text_key = ...[:200]`

## Operating Model for Recall/Precision Tradeoff

Use dual-lane answer composition:

- `recall_pool`: broad evidence for exhaustive applicable-rule listing with citations.
- `precision_core`: tighter high-confidence subset for concise summary/verdict phrasing.

This preserves coverage without turning summary output into noise.

## Implementation Plan

## Phase 0: Baseline Hardening and Reranker Isolation (Mandatory First)

Goals:

- Remove confounded runs.
- Prove reranker value before phase ordering decisions.

Changes:

- Add explicit reranker provider control in benchmark scripts; stop hard-forcing `no_reranker` for reranker experiments.
- Enable reranker experiments at this stage (do not defer to later phases) so widening decisions are made with a realistic precision guard.
- Run paired A/B on both KPBR and KMBR with identical settings:
- `hybrid_no_reranker`
- `hybrid_reranker`
- optional `hybrid_graph_reranker`
- Keep backend/path/manifest fixed across the pair.
- For KPBR retrieval benchmarking in recall mode, use `--top-k 15` (keep legacy `k=5` only as a historical comparator, not primary decision metric).

Exit criteria:

- Valid runtime snapshots with empty `provider_diagnostics`.
- Reranker-isolated comparison results for both datasets.

## Phase 1: Retrieval Funnel Widening (Config-First Ramp)

Files:

- `config/policies/retrieval.default.yaml`
- `config/policies/applicability.default.yaml`
- `config/policies/abstention.default.yaml` (alignment only)
- `src/retrieval/applicability_engine.py` (tier merge in Step B)

Ramp strategy:

- Step A (moderate):
- `min_query_relevance: 0.03 -> 0.025`
- `min_evidence_score: 0.05 -> 0.04`
- `default_evidence_docs_per_topic: 5 -> 7`
- `candidate_pool_factor: 4.0 -> 6.0`
- Step B (full, only if gates still pass and recall headroom remains):
- `min_query_relevance: 0.025 -> 0.02`
- `min_evidence_score: 0.04 -> 0.03`
- `default_evidence_docs_per_topic: 7 -> 8`
- `topic_min_docs: 2 -> 3`
- `max_excerpt_sentences: 8 -> 14`
- `max_excerpt_chars: 1600 -> 2400`
- `strict_topic_matching: true -> false` (applicability YAML)
- Bundle applicability tier merge code change here (same step as `strict_topic_matching=false`):
- update `_select_tier()` to return merged `strict + relaxed_topic + topic-matching generic_fallback` with dedupe.
- Align abstention thresholds with retrieval thresholds:
- `abstention.min_evidence_score: 0.05 -> 0.03`
- `abstention.partial_min_evidence_score: 0.04 -> 0.02`
- Run this phase with reranker lane active in parallel (`hybrid_reranker`) and control lane (`hybrid_no_reranker`) under identical settings.

Exit criteria:

- Retrieval recall up on both datasets.
- Precision floor preserved (`F1 >= 0.30`, groundedness and contract gates pass).

## Phase 2: Reranking Scale-Up and Budget Tuning

Files:

- `config/providers.yaml`
- benchmark/runtime config wiring

Changes:

- Keep reranker-enabled modes active (`hybrid_reranker` / `hybrid_graph_reranker`) and tune budget/latency tradeoff.
- Set reranker provider to `cohere_reranker`.
- Increase rerank budget in provider feature flags (`rerank_top_n: 80 -> 100`), then tune.

Exit criteria:

- MRR gains on low-MRR failures.
- Recall non-regression versus Phase 1.
- Latency within agreed p95 bounds.

## Phase 3: Evidence Matrix and Applicability Code Fixes

Files:

- `src/retrieval/hybrid_retriever.py`
- `src/policy/profile.py`
- `config/policies/retrieval.default.yaml`

Changes:

- Parameterize `max_claims_per_doc` in policy, not hardcoded:
- add `max_claims_per_doc` to `RetrievalPolicy` and `from_dict`.
- set `max_claims_per_doc` in `config/policies/retrieval.default.yaml` (initial target `4`).
- consume policy value in `_build_evidence_matrix`.
- Replace topic-family hard exclusion with score penalty.
- Add diagnostics for per-tier candidate counts and post-filter evidence counts.

Exit criteria:

- Higher retrieval ceiling and answer recall.
- No gate failures.

## Phase 4: Synthesis Funnel Widening (Precision-Safe)

Files:

- `config/policies/generation.default.yaml`
- `src/agentic/compliance_brief_composer.py`

Config targets (ramp):

- `max_claims_per_section: 7 -> 10`
- `summary_char_limit: 480 -> 720`
- `concise_claim_char_limit: 520 -> 680`
- `excavation_claim_char_limit: 560 -> 800`
- optional `max_output_tokens: 1200 -> 2400` after latency/cost review

Code targets:

- Relax priority floor:
- from `max(0.03, max_priority * 0.72)`
- toward `max(0.02, max_priority * 0.45)` (ramp, not one-shot)
- Reduce false dedup collisions:
- dedup key from 200 to 350 chars
- Add numeric-preservation behavior in concise summarization (do not truncate away key numbers/timelines).
- Introduce dual-lane composer behavior (`recall_pool` + `precision_core`).

Exit criteria:

- Answer recall and numeric recall rise without breaching precision gates.

## Phase 5: Retrieval Signal Improvements

Files:

- `config/policies/retrieval.default.yaml`
- `src/retrieval/hybrid_retriever.py`
- `src/retrieval/query_planner.py`

Changes:

- Rebalance RRF weights toward legal lexical strength:
- `lexical_rrf_weight: 0.75 -> 1.0`
- `vector_rrf_weight: 1.0 -> 0.7`
- `structured_rrf_weight: 0.75 -> 0.35`
- Deepen cross-reference expansion:
- depth `1 -> 2`
- `cross_ref_score_decay: 0.85 -> 0.90`
- Add sibling-clause evidence expansion in evidence-selection phase.
- Use clause-level claim IDs for sibling-added evidence to avoid collisions:
- format: `f"{topic}-{doc.document_id}-{clause.clause_id}"`.
- Enhance lexical query expansion with legal synonym variants and component-aware expansions (bounded variant count).

Exit criteria:

- Retrieval ceiling and hop coverage improve on multi-hop misses.
- Gate compliance maintained.

## Phase 6: Adaptive top_k (Advanced)

Files:

- `src/models/schemas.py`
- `src/retrieval/query_planner.py`
- `src/api/service.py` and retrieval call sites

Changes:

- Make `top_k` optional or add `use_adaptive_top_k` flag.
- Add query complexity scoring and adaptive `top_k` bands.
- Use adaptive value only when explicit `top_k` is not provided.

Exit criteria:

- Complex queries get recall headroom.
- Simple queries keep latency controlled.

## Evaluation Protocol (Per Phase)

1. Validate runtime/index guardrails first.
2. Retrieval-only:
- KPBR multihop20 enriched via `scripts/benchmark_retrieval.py` with `--top-k 15` as primary recall run
- KMBR multihop46 via `scripts/benchmark_retrieval.py`
3. End-to-end:
- KPBR via `scripts/benchmark_qna_answers.py`
- KMBR via `scripts/benchmark_kmbr_decomposed.py`
4. Record:
- recall/f1/numeric metrics
- groundedness/contract metrics
- latency p50/p95
- provider and manifest snapshots
- per-query failure tracking:
- `query_id`, `chunk_recall_at_k`, `doc_recall_at_k`, `mrr`
- `candidate_doc_count`, `fallback_stage`, `vector/lexical/structured_hits`
- top blocking queries and recurring failure buckets per phase
5. Fail-fast:
- If any hard gate fails, mark run invalid.
- If `F1 < 0.30`, stop and rollback phase.

Cache/reset rule:

- Clear active vector path from `PLOTMAGIC_VECTOR_DB_PATH` plus `.cache/structured_rules.db`.
- If backend/path changed between runs, also clear stale alternate paths before re-run.

## Final Decision Rule

- Promote only the highest-recall strategy that passes all hard gates for both KPBR and KMBR.
- If two strategies are close on recall, pick lower-latency/lower-complexity option.
- Never trade away citation groundedness or contract safety for recall gains.

## Practical Repo Notes

- Runtime policy is loaded from statepack YAML (`data/statepacks/kerala_statepack.yaml` -> `config/policies/*.yaml`), not from dataclass defaults.
- `rerank_top_n` is controlled by provider feature flags (`config/providers.yaml`), not retrieval policy.
- `hybrid_no_reranker` disables reranking by design.
