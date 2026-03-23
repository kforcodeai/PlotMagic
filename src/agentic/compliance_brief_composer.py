from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from src.models.schemas import ComplianceBriefItem, ComplianceBriefPayload, GroundingReportPayload
from src.policy import GenerationPolicy
from src.providers import LLMProvider, ProviderError


class ComplianceBriefComposer:
    _SLOT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "distance": [
            re.compile(r"\b\d+(?:\.\d+)?\s*(?:m|metre|meter|meters|metres|cm|sq\.?\s*m)\b", flags=re.IGNORECASE),
            re.compile(r"\b(?:distance|setback|clearance|within)\b", flags=re.IGNORECASE),
        ],
        "authority": [
            re.compile(r"\b(?:secretary|council|authority|officer|committee|planner|defen[cs]e|railway)\b", flags=re.IGNORECASE)
        ],
        "timeline": [
            re.compile(r"\b\d+(?:\.\d+)?\s*(?:day|days|month|months|year|years)\b", flags=re.IGNORECASE),
            re.compile(r"\b(?:within|timeline|deadline|time)\b", flags=re.IGNORECASE),
        ],
        "exception": [
            re.compile(r"\b(?:provided that|provided further|except|unless|shall not apply|permit not necessary|waiver)\b", flags=re.IGNORECASE)
        ],
        "fees": [re.compile(r"\b(?:fee|fees|charge|charges|rate|percent|%)\b", flags=re.IGNORECASE)],
        "validity_extension": [
            re.compile(r"\b(?:validity|valid|renewal|renewed|extension|extended)\b", flags=re.IGNORECASE)
        ],
        "procedure": [
            re.compile(r"\b(?:apply|application|submit|submission|issue|issued|certificate|approval)\b", flags=re.IGNORECASE)
        ],
    }

    def __init__(self, llm_provider: LLMProvider, policy: GenerationPolicy | None = None) -> None:
        self.llm_provider = llm_provider
        self.policy = policy or GenerationPolicy()
        self.last_generation_meta: dict[str, Any] = {
            "llm_attempted": False,
            "llm_used": False,
            "fallback_used": True,
            "error": None,
        }

    def compose(
        self,
        *,
        query: str,
        verdict: str,
        claim_citations: dict[str, list[str]],
        claim_texts: dict[str, str],
        claim_scores: dict[str, float] | None = None,
        unresolved: list[str],
        grounding: GroundingReportPayload,
    ) -> ComplianceBriefPayload:
        # Deterministic draft serves as recall-floor fallback.
        draft = self._deterministic_draft(
            query=query,
            verdict=verdict,
            claim_citations=claim_citations,
            claim_texts=claim_texts,
            claim_scores=claim_scores or {},
            unresolved=unresolved,
            grounding=grounding,
        )
        if verdict == "insufficient_evidence":
            self.last_generation_meta = {
                "llm_attempted": False,
                "llm_used": False,
                "fallback_used": True,
                "error": None,
            }
            return draft

        # Use deterministic draft as primary when LLM-bypass is active.
        # Keep import local to avoid module import side-effects in tests.
        import os
        if os.environ.get("PLOTMAGIC_DETERMINISTIC_ONLY", "").lower() in ("1", "true"):
            self.last_generation_meta = {
                "llm_attempted": False,
                "llm_used": False,
                "fallback_used": True,
                "error": "deterministic_only_mode",
            }
            return draft

        query_terms = self._query_terms(query)
        claim_priority: dict[str, float] = {
            claim_id: self._claim_priority(
                claim_id=claim_id,
                claim_texts=claim_texts,
                claim_scores=claim_scores or {},
                query_terms=query_terms,
            )
            for claim_id in claim_citations.keys()
        }
        ordered_claim_ids = sorted(
            claim_priority.keys(),
            key=lambda claim_id: claim_priority.get(claim_id, 0.0),
            reverse=True,
        )
        evidence_budget = min(20, max(10, 8 + (len(query_terms) // 3)))

        # Build full-evidence payload for LLM extraction (no pre-filtering).
        evidence_items = []
        for claim_id in ordered_claim_ids:
            if len(evidence_items) >= evidence_budget:
                break
            text = claim_texts.get(claim_id, "")
            citations = claim_citations.get(claim_id, [])
            if not text.strip() or not citations:
                continue
            compressed = self._summarize_claim_text(text=text, query_terms=query_terms) or text
            compressed = self._truncate_at_sentence_boundary(compressed, 800, preserve_numeric=True)
            evidence_items.append({
                "claim_id": claim_id,
                "text": compressed,
                "citations": citations[:6],
            })

        schema = ComplianceBriefPayload.model_json_schema()
        target_chars = min(600, max(380, len(query) * 3))
        max_chars = min(650, max(450, len(query) * 4))
        payload = {
            "task": "legal_rule_extraction",
            "query": query,
            "required_verdict": verdict,
            "evidence": evidence_items,
            "grounding": grounding.model_dump(),
            "instructions": (
                "You are a legal compliance analyst. Produce a structured compliance brief.\n\n"
                "CRITICAL RULES:\n"
                "0. verdict MUST exactly equal required_verdict.\n"
                f"1. short_summary is THE MAIN ANSWER — a dense paragraph of 3-5 sentences "
                f"(target {target_chars} characters, MAXIMUM {max_chars} characters). "
                "Be CONCISE. Include ALL key numbers, authority names, and conditions "
                "but avoid filler words, redundant phrases, or unnecessary context.\n"
                "2. STYLE: Write direct factual statements. "
                "DO NOT include rule/act references (no 'KPBR 2011', 'Rule 36', 'r.36'). "
                "DO NOT use meta-language ('the evidence states', 'based on the evidence', 'as per'). "
                "DO NOT hedge ('may depend', 'varies', 'it appears'). "
                "State rules as facts: 'The Secretary must consult...' not 'Rule 8 states that...'.\n"
                "3. Section items: 1 concise sentence each with citation backing.\n"
                "4. Use provided claim_id and citation_ids exactly; do not fabricate.\n"
                "5. Do not invent facts beyond the evidence."
            ),
        }

        try:
            self.last_generation_meta = {
                "llm_attempted": True,
                "llm_used": True,
                "fallback_used": False,
                "error": None,
            }
            candidate = self.llm_provider.generate_structured(
                task="compliance_brief",
                payload=payload,
                json_schema=schema,
                temperature=0.0,
                max_output_tokens=self.policy.max_output_tokens,
            )
            cleaned = self._enforce_citation_grounding(candidate, claim_citations)
            validated = ComplianceBriefPayload.model_validate(cleaned)
            # Non-regression guard: if LLM output has fewer grounded claims than
            # deterministic draft, merge missing claims from the draft.
            validated = self._non_regression_merge(
                validated=validated,
                fallback=draft,
                query=query,
                query_terms=self._query_terms(query),
                claim_priority=claim_priority,
                max_claims_per_section=self.policy.max_claims_per_section,
            )
            validated = self._normalize_payload_sections(
                payload=validated,
                query=query,
                query_terms=query_terms,
                claim_priority=claim_priority,
                rebuild_summary=False,
            )
            if validated.verdict != verdict:
                raise ValueError(
                    f"LLM verdict '{validated.verdict}' disagrees with grounding verdict '{verdict}'."
                )
            if self.policy.enforce_non_empty_summary and not validated.short_summary.strip():
                raise ValueError("LLM output violated contract: short_summary cannot be empty.")
            # Adaptive summary limit based on query complexity.
            base_limit = self.policy.summary_char_limit
            query_complexity = len(query_terms) + len(re.findall(r"\?|;|,\s*and\s+what", query, re.IGNORECASE))
            adaptive_limit = base_limit + min(200, max(0, (query_complexity - 8) * 25))
            if len(validated.short_summary) > adaptive_limit:
                validated.short_summary = self._truncate_at_sentence_boundary(
                    validated.short_summary,
                    adaptive_limit,
                    preserve_numeric=True,
                )
            return validated
        except (ProviderError, ValidationError, ValueError, TypeError) as exc:
            self.last_generation_meta = {
                "llm_attempted": True,
                "llm_used": False,
                "fallback_used": True,
                "error": str(exc),
            }
            return draft

    def _deterministic_draft(
        self,
        *,
        query: str,
        verdict: str,
        claim_citations: dict[str, list[str]],
        claim_texts: dict[str, str],
        claim_scores: dict[str, float],
        unresolved: list[str],
        grounding: GroundingReportPayload,
    ) -> ComplianceBriefPayload:
        if verdict == "insufficient_evidence":
            return ComplianceBriefPayload(
                verdict="insufficient_evidence",
                short_summary="Insufficient grounded evidence to provide a definitive compliance conclusion.",
                applicable_rules=[],
                conditions_and_exceptions=[],
                required_actions=[],
                risk_flags=["Definitive compliance determination withheld due to insufficient evidence."],
                clarifications_needed=sorted(
                    set(unresolved + [f"Need evidence for topic: {topic}" for topic in grounding.missing_topics])
                ),
            )

        query_terms = self._query_terms(query)
        claim_priority: dict[str, float] = {
            claim_id: self._claim_priority(
                claim_id=claim_id,
                claim_texts=claim_texts,
                claim_scores=claim_scores,
                query_terms=query_terms,
            )
            for claim_id in claim_citations.keys()
        }
        ordered_claims = sorted(claim_priority.keys(), key=lambda claim_id: claim_priority[claim_id], reverse=True)
        numeric_query = self._is_numeric_query(query)
        max_priority = max(claim_priority.values(), default=0.0)
        min_priority = max(0.015, max_priority * 0.35)
        max_claims_per_section = self.policy.max_claims_per_section + (2 if numeric_query else 1)
        statements: list[ComplianceBriefItem] = []
        conditions: list[ComplianceBriefItem] = []
        actions: list[ComplianceBriefItem] = []
        seen_text_keys: set[str] = set()
        selected_claim_ids: set[str] = set()
        for claim_id in ordered_claims:
            if claim_priority.get(claim_id, 0.0) < min_priority:
                continue
            text = claim_texts.get(claim_id, "").strip()
            citations = claim_citations.get(claim_id, [])
            if not text or not citations:
                continue
            concise = self._summarize_claim_text(text=text, query_terms=query_terms)
            if not concise:
                continue
            # Deduplicate near-identical claim texts across topics.
            text_key = (
                re.sub(r"[^a-z0-9]+", " ", concise.lower()).strip()[:380]
                + "|"
                + str(citations[0]).lower()
            )
            if text_key in seen_text_keys:
                continue
            seen_text_keys.add(text_key)
            item = ComplianceBriefItem(claim_id=claim_id, text=concise, citation_ids=citations[:4])
            selected_claim_ids.add(claim_id)
            lowered = concise.lower()
            if any(
                token in lowered
                for token in ["provided that", "subject to", "except", "unless", "in case", " where ", " if "]
            ):
                conditions.append(item)
            elif any(
                token in lowered
                for token in ["shall", "must", "required", "obtain", "submit", "ensure", "within", "before", "after"]
            ):
                actions.append(item)
            else:
                statements.append(item)

        selected_statements = statements[:max_claims_per_section]
        selected_conditions = conditions[:max_claims_per_section]
        selected_actions = actions[:max_claims_per_section]
        if numeric_query:
            selected_statements, selected_conditions, selected_actions = self._ensure_numeric_claim_coverage(
                ordered_claims=ordered_claims,
                selected_statements=selected_statements,
                selected_conditions=selected_conditions,
                selected_actions=selected_actions,
                selected_claim_ids=selected_claim_ids,
                claim_texts=claim_texts,
                claim_citations=claim_citations,
                query_terms=query_terms,
                max_claims_per_section=max_claims_per_section,
            )
        if not (selected_statements or selected_conditions or selected_actions):
            fallback_item = self._fallback_grounded_item(ordered_claims, claim_texts, claim_citations, query_terms)
            if fallback_item:
                selected_statements = [fallback_item]

        slot_seed = self._extract_slot_seed_items(
            query=query,
            ordered_claims=ordered_claims,
            claim_texts=claim_texts,
            claim_citations=claim_citations,
            claim_scores=claim_scores,
            query_terms=query_terms,
        )
        selected_statements = self._merge_section_items(
            seeds=slot_seed["statements"],
            current=selected_statements,
            limit=max_claims_per_section,
        )
        selected_conditions = self._merge_section_items(
            seeds=slot_seed["conditions"],
            current=selected_conditions,
            limit=max_claims_per_section,
        )
        selected_actions = self._merge_section_items(
            seeds=slot_seed["actions"],
            current=selected_actions,
            limit=max_claims_per_section,
        )
        summary = self._build_short_summary(
            statements=selected_statements,
            conditions=selected_conditions,
            actions=selected_actions,
            unresolved=unresolved,
        )
        payload = ComplianceBriefPayload(
            verdict="depends",
            short_summary=summary,
            applicable_rules=selected_statements,
            conditions_and_exceptions=selected_conditions,
            required_actions=selected_actions,
            risk_flags=["Professional review recommended before permit submission."],
            clarifications_needed=sorted(set(unresolved)),
        )
        return self._normalize_payload_sections(
            payload=payload,
            query=query,
            query_terms=query_terms,
            claim_priority=claim_priority,
            rebuild_summary=True,
            apply_trim=False,
        )

    def _enforce_citation_grounding(
        self,
        payload: dict[str, Any],
        allowed_claim_citations: dict[str, list[str]],
    ) -> dict[str, Any]:
        for section_name in ["applicable_rules", "conditions_and_exceptions", "required_actions"]:
            items = payload.get(section_name, [])
            if not isinstance(items, list):
                payload[section_name] = []
                continue
            cleaned: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                claim_id = str(item.get("claim_id", "")).strip()
                if not claim_id or claim_id not in allowed_claim_citations:
                    continue
                citation_ids = item.get("citation_ids", [])
                if not isinstance(citation_ids, list) or not citation_ids:
                    continue
                allowed = set(allowed_claim_citations.get(claim_id, []))
                filtered = [str(citation_id) for citation_id in citation_ids if str(citation_id) in allowed]
                if not filtered:
                    continue
                cleaned.append(
                    {
                        "claim_id": claim_id,
                        "text": str(item.get("text", "")),
                        "citation_ids": filtered,
                    }
                )
            payload[section_name] = cleaned
        return payload

    _STOPWORDS = {
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "into",
        "has",
        "have",
        "had",
        "not",
        "but",
        "also",
        "any",
        "all",
        "such",
        "than",
        "will",
        "may",
        "shall",
        "can",
        "per",
        "which",
        "where",
        "when",
        "who",
        "how",
        "what",
    }

    def _query_terms(self, query: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        return {token for token in tokens if len(token) >= 3 and token not in self._STOPWORDS}

    def _split_sentences(self, text: str) -> list[str]:
        compact = re.sub(r"[_`]+", " ", text)
        compact = re.sub(r"\s+", " ", compact).strip()
        if not compact:
            return []
        marked = re.sub(r"(\(\d+[A-Za-z]?\))", r"\n\1 ", compact)
        parts = re.split(r"(?<=[.;:])\s+|\n+", marked)
        out: list[str] = []
        for part in parts:
            sentence = part.strip(" -")
            sentence = re.sub(r"^\(?\d+[A-Za-z]?\)?\s*", "", sentence).strip()
            if len(sentence) < 24:
                continue
            if len(sentence) > 340:
                long_parts = re.split(
                    r",\s+(?=(?:provided|subject to|unless|if|where|within|before|after|the secretary|the authority|officer|distance|setback|clearance|metre|meter|day|month|year)\b)",
                    sentence,
                    flags=re.IGNORECASE,
                )
                if len(long_parts) > 1:
                    for long_part in long_parts:
                        candidate = long_part.strip(" -")
                        if len(candidate) >= 24:
                            out.append(candidate)
                    continue
            out.append(sentence)
        return out

    def _compress_sentence_clause(self, *, sentence: str, query_terms: set[str]) -> str:
        compact = re.sub(r"\s+", " ", sentence).strip()
        if len(compact) <= 320:
            return compact
        chunks = re.split(
            r",\s+(?=(?:provided|subject to|unless|if|where|within|before|after|the secretary|the authority|officer|distance|setback|clearance|metre|meter|day|month|year)\b)",
            compact,
            flags=re.IGNORECASE,
        )
        if len(chunks) <= 1:
            return compact
        ranked: list[tuple[float, int, str]] = []
        for idx, chunk in enumerate(chunks):
            lowered = chunk.lower()
            terms = set(re.findall(r"[a-z0-9]+", lowered))
            overlap = len(terms.intersection(query_terms))
            has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", lowered))
            has_legal_verb = any(token in lowered for token in ["shall", "must", "required", "within", "permit"])
            score = float(overlap) + (1.2 if has_number else 0.0) + (0.6 if has_legal_verb else 0.0)
            score += max(0.0, 0.4 - (0.05 * float(idx)))
            ranked.append((score, idx, chunk.strip()))
        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen_idxs: list[int] = []
        for score, idx, _chunk in ranked:
            if score <= 0.0:
                continue
            chosen_idxs.append(idx)
            if len(chosen_idxs) >= 2:
                break
        if not chosen_idxs:
            return chunks[0].strip()
        chosen_idxs.sort()
        merged = "; ".join(chunks[idx].strip() for idx in chosen_idxs if chunks[idx].strip())
        return merged.strip() or compact

    def _summarize_claim_text(self, *, text: str, query_terms: set[str]) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        scored: list[tuple[float, str]] = []
        numeric_query = bool(
            query_terms.intersection({"minimum", "maximum", "distance", "clearance", "days", "years", "fees"})
            or any(token.isdigit() for token in query_terms)
        )
        for idx, sentence in enumerate(sentences):
            lowered = sentence.lower()
            sentence_terms = set(re.findall(r"[a-z0-9]+", lowered))
            overlap = len(query_terms.intersection(sentence_terms))
            score = float(overlap)
            has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", lowered))
            has_time = any(token in lowered for token in ["day", "days", "month", "months", "year", "years"])
            if has_number:
                score += 1.0
            if has_time:
                score += 0.6
            if has_number and has_time:
                score += 0.8
            if any(token in lowered for token in ["shall", "must", "within", "permit", "approval", "fee", "valid"]):
                score += 0.5
            if re.search(r"\b(?:sq\.?m|m\b|metre|meter|percent|%|cm\b)", lowered):
                score += 0.5
            if numeric_query and has_number:
                score += 0.4
            if query_terms.intersection(
                {"drawings", "drawing", "plans", "sections", "protective", "protection", "neighbour", "neighbor"}
            ):
                if any(
                    token in lowered
                    for token in [
                        "drawing",
                        "plan",
                        "section",
                        "protective",
                        "protection",
                        "committee",
                        "neighbour",
                        "neighbor",
                    ]
                ):
                    score += 0.8
            score += max(0.0, 0.8 - (0.08 * float(idx)))
            if any(token in lowered for token in ["before the commencement", "issued before", "prior to commencement"]):
                score -= 0.7
            if overlap == 0 and score < 1.0:
                continue
            scored.append((score, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        excavation_query = bool(query_terms.intersection({"excavation", "cutting", "compensation", "concurrence"}))
        term_count = len(query_terms)
        if term_count >= 9 or excavation_query:
            max_sentences = 7
        elif numeric_query:
            max_sentences = 5
        else:
            max_sentences = 4
        for _score, sentence in scored:
            key = re.sub(r"^\(?\d+[A-Za-z]?\)?\s*", "", sentence.lower())
            key = re.sub(r"[^a-z0-9]+", " ", key).strip()
            core_key = re.sub(r"^(provided\s+that|subject\s+to|in\s+case|if|where)\s+", "", key).strip()[:180]
            if key in seen:
                continue
            if core_key in seen:
                continue
            selected.append(self._compress_sentence_clause(sentence=sentence, query_terms=query_terms))
            seen.add(key)
            seen.add(core_key)
            if len(selected) >= max_sentences:
                break

        if numeric_query and selected:
            covered_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", " ".join(selected)))
            for _score, sentence in scored:
                if len(selected) >= max_sentences + 2:
                    break
                numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", sentence))
                if not numbers or numbers.issubset(covered_numbers):
                    continue
                key = re.sub(r"^\(?\d+[A-Za-z]?\)?\s*", "", sentence.lower())
                key = re.sub(r"[^a-z0-9]+", " ", key).strip()
                core_key = re.sub(r"^(provided\s+that|subject\s+to|in\s+case|if|where)\s+", "", key).strip()[:180]
                if key in seen or core_key in seen:
                    continue
                selected.append(self._compress_sentence_clause(sentence=sentence, query_terms=query_terms))
                seen.add(key)
                seen.add(core_key)
                covered_numbers.update(numbers)

        concise = " ".join(selected).strip()
        concise = re.sub(r"\s+", " ", concise)
        concise = self._normalize_numeric_phrases(concise)
        has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", concise))
        if excavation_query:
            char_limit = self.policy.excavation_claim_char_limit
        elif numeric_query:
            char_limit = max(self.policy.concise_claim_char_limit, 420)
        else:
            char_limit = self.policy.concise_claim_char_limit
        if has_numbers:
            char_limit = int(char_limit * 1.22)
        # Truncate at sentence boundary instead of raw char slice.
        if len(concise) > char_limit:
            concise = self._truncate_at_sentence_boundary(concise, char_limit, preserve_numeric=has_numbers)
        return concise

    def _build_short_summary(
        self,
        *,
        statements: list[ComplianceBriefItem],
        conditions: list[ComplianceBriefItem],
        actions: list[ComplianceBriefItem],
        unresolved: list[str],
    ) -> str:
        """Build a comprehensive summary from all items (not just first two).

        The summary is the primary answer text. It should contain all key facts,
        numbers, authorities, and conditions from the section items.
        """
        all_items = list(statements) + list(actions) + list(conditions)
        if not all_items:
            summary = "Grounded evidence retrieved; verify cited provisions before final compliance action."
            if unresolved:
                unresolved_note = "; ".join(sorted(set(unresolved))[:3])
                summary = f"{summary} Unresolved items: {unresolved_note}"
            return summary[: self.policy.summary_char_limit]

        # Collect unique text fragments from all items, preserving order.
        seen_keys: set[str] = set()
        fragments: list[str] = []
        covered_numbers: set[str] = set()
        for item in all_items:
            text = item.text.strip()
            if not text:
                continue
            key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:200]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fragments.append(text)
            covered_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", text))

        summary = " ".join(fragments)

        # Trim to summary_char_limit at sentence boundary.
        char_limit = self.policy.summary_char_limit
        if len(summary) > char_limit:
            summary = self._truncate_at_sentence_boundary(
                summary, char_limit, preserve_numeric=bool(covered_numbers)
            )
        if unresolved:
            unresolved_note = "; ".join(sorted(set(unresolved))[:3])
            summary = f"{summary} Unresolved items: {unresolved_note}"
        return summary[: int(self.policy.summary_char_limit * 1.2)]

    def _extract_slot_seed_items(
        self,
        *,
        query: str,
        ordered_claims: list[str],
        claim_texts: dict[str, str],
        claim_citations: dict[str, list[str]],
        claim_scores: dict[str, float],
        query_terms: set[str],
    ) -> dict[str, list[ComplianceBriefItem]]:
        slots = self._slots_from_query(query)
        if not slots:
            return {"statements": [], "conditions": [], "actions": []}

        ranked_claims = sorted(
            ordered_claims,
            key=lambda claim_id: (-float(claim_scores.get(claim_id, 0.0)), claim_id),
        )
        used_claim_ids: set[str] = set()
        out: dict[str, list[ComplianceBriefItem]] = {"statements": [], "conditions": [], "actions": []}
        for slot in slots:
            best: ComplianceBriefItem | None = None
            best_claim_id = ""
            best_score = -1.0
            for claim_id in ranked_claims:
                if claim_id in used_claim_ids:
                    continue
                text = claim_texts.get(claim_id, "").strip()
                citations = claim_citations.get(claim_id, [])
                if not text or not citations:
                    continue
                if not self._slot_matches(slot=slot, text=text):
                    continue
                concise = self._summarize_claim_text(text=text, query_terms=query_terms)
                if not concise:
                    continue
                score = float(claim_scores.get(claim_id, 0.0))
                if score <= best_score:
                    continue
                best_score = score
                best_claim_id = claim_id
                best = ComplianceBriefItem(
                    claim_id=claim_id,
                    text=concise,
                    citation_ids=citations[:4],
                )
            if not best or not best_claim_id:
                continue
            used_claim_ids.add(best_claim_id)
            section = self._slot_section(slot)
            out[section].append(best)
        return out

    def _slots_from_query(self, query: str) -> list[str]:
        lowered = query.lower()
        slots: list[str] = []
        for slot, patterns in self._SLOT_PATTERNS.items():
            if any(pattern.search(lowered) for pattern in patterns):
                slots.append(slot)
        seen: set[str] = set()
        deduped: list[str] = []
        for slot in slots:
            if slot in seen:
                continue
            seen.add(slot)
            deduped.append(slot)
        return deduped

    def _slot_matches(self, *, slot: str, text: str) -> bool:
        patterns = self._SLOT_PATTERNS.get(slot, [])
        lowered = text.lower()
        return any(pattern.search(lowered) for pattern in patterns)

    def _slot_section(self, slot: str) -> str:
        if slot == "exception":
            return "conditions"
        if slot in {"timeline", "fees", "validity_extension", "procedure", "registration"}:
            return "actions"
        return "statements"

    def _merge_section_items(
        self,
        *,
        seeds: list[ComplianceBriefItem],
        current: list[ComplianceBriefItem],
        limit: int,
    ) -> list[ComplianceBriefItem]:
        merged: list[ComplianceBriefItem] = []
        seen: set[str] = set()
        for item in [*seeds, *current]:
            key = (
                f"{item.claim_id}:"
                f"{','.join(item.citation_ids[:2]).lower()}:"
                f"{re.sub(r'[^a-z0-9]+', ' ', item.text.lower()).strip()[:260]}"
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def _dynamic_claim_caps(
        self,
        *,
        query: str,
        query_terms: set[str],
        max_claims_per_section: int,
    ) -> tuple[int, int]:
        slot_count = len(self._slots_from_query(query))
        term_count = len(query_terms)
        numeric_query = self._is_numeric_query(query)
        complexity_bonus = 3 if term_count >= 12 else 2 if term_count >= 8 else 1 if term_count >= 5 else 0
        coverage_bonus = min(4, max(0, term_count // 4))
        numeric_bonus = 2 if numeric_query else 0
        target_total = max(10, 8 + slot_count + complexity_bonus + numeric_bonus + coverage_bonus)
        max_total = max(6, max_claims_per_section * 2)
        total_cap = min(target_total, max_total)
        per_section_cap = max(2, min(max_claims_per_section, (total_cap + 2) // 3))
        return total_cap, per_section_cap

    def _item_dedup_key(self, item: ComplianceBriefItem) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", item.text.lower()).strip()
        citation = item.citation_ids[0].lower() if item.citation_ids else ""
        return f"{item.claim_id}:{citation}:{normalized[:240]}"

    def _item_rank_score(
        self,
        *,
        item: ComplianceBriefItem,
        query_terms: set[str],
        numeric_query: bool,
        claim_priority: dict[str, float],
        query_slots: list[str] | None = None,
    ) -> float:
        text = item.text.strip().lower()
        if not text:
            return -1.0
        text_terms = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(text_terms.intersection(query_terms))
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", text))
        slot_hits = 0
        for slot in query_slots or []:
            if self._slot_matches(slot=slot, text=text):
                slot_hits += 1
        base = float(claim_priority.get(item.claim_id, 0.0))
        score = (1.6 * base) + (0.18 * float(overlap)) + (0.16 * float(slot_hits))
        if numeric_query and has_number:
            score += 0.22
        if overlap == 0 and slot_hits == 0 and not has_number:
            score -= 0.55
        score -= min(0.3, float(len(text)) / 2600.0)
        return score

    def _adaptive_trim_sections(
        self,
        *,
        query: str,
        query_terms: set[str],
        claim_priority: dict[str, float],
        statements: list[ComplianceBriefItem],
        conditions: list[ComplianceBriefItem],
        actions: list[ComplianceBriefItem],
        max_claims_per_section: int,
    ) -> tuple[list[ComplianceBriefItem], list[ComplianceBriefItem], list[ComplianceBriefItem]]:
        candidates_with_section: list[tuple[str, ComplianceBriefItem]] = []
        for item in statements:
            candidates_with_section.append(("statements", item))
        for item in conditions:
            candidates_with_section.append(("conditions", item))
        for item in actions:
            candidates_with_section.append(("actions", item))
        if not candidates_with_section:
            return statements, conditions, actions

        numeric_query = self._is_numeric_query(query)
        query_slots = self._slots_from_query(query)
        total_cap, per_section_cap = self._dynamic_claim_caps(
            query=query,
            query_terms=query_terms,
            max_claims_per_section=max_claims_per_section,
        )

        deduped: dict[str, tuple[str, ComplianceBriefItem, float]] = {}
        for section, item in candidates_with_section:
            key = self._item_dedup_key(item)
            rank = self._item_rank_score(
                item=item,
                query_terms=query_terms,
                numeric_query=numeric_query,
                claim_priority=claim_priority,
                query_slots=query_slots,
            )
            current = deduped.get(key)
            if current is None or rank > current[2]:
                deduped[key] = (section, item, rank)
        ranked = sorted(deduped.values(), key=lambda entry: entry[2], reverse=True)
        if not ranked:
            return statements, conditions, actions

        counts = {"statements": 0, "conditions": 0, "actions": 0}
        selected: list[tuple[str, ComplianceBriefItem]] = []
        selected_keys: set[str] = set()

        def try_add(section: str, item: ComplianceBriefItem) -> bool:
            key = self._item_dedup_key(item)
            if key in selected_keys:
                return False
            if counts[section] >= per_section_cap:
                return False
            if len(selected) >= total_cap:
                return False
            selected.append((section, item))
            selected_keys.add(key)
            counts[section] += 1
            return True

        # Seed coverage for query slots before global ranking fill.
        for slot in query_slots:
            best_entry: tuple[str, ComplianceBriefItem, float] | None = None
            best_score = -1e9
            preferred_section = self._slot_section(slot)
            for section, item, rank in ranked:
                if not self._slot_matches(slot=slot, text=item.text):
                    continue
                if section == preferred_section:
                    rank += 0.12
                key = self._item_dedup_key(item)
                if key in selected_keys:
                    continue
                if counts[preferred_section] >= per_section_cap and counts[section] >= per_section_cap:
                    continue
                if rank > best_score:
                    best_entry = (section, item, rank)
                    best_score = rank
            if best_entry is None:
                continue
            section, item, _rank = best_entry
            target_section = preferred_section if counts[preferred_section] < per_section_cap else section
            try_add(target_section, item)

        query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", query))
        for number in sorted(query_numbers):
            for section, item, _rank in ranked:
                if number not in item.text:
                    continue
                if try_add(section, item):
                    break

        selected_terms: set[str] = set()
        selected_numbers: set[str] = set()
        for _section, item in selected:
            selected_terms.update(self._query_terms(item.text))
            selected_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", item.text))

        for section, item, rank in ranked:
            if len(selected) >= total_cap:
                break
            text_terms = self._query_terms(item.text)
            numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", item.text))
            adds_new_terms = bool(text_terms.intersection(query_terms - selected_terms))
            adds_new_numbers = bool(numbers - selected_numbers)
            if rank < -0.25 and not adds_new_terms and not adds_new_numbers:
                continue
            if not adds_new_terms and not adds_new_numbers and len(selected) >= max(8, int(total_cap * 0.95)):
                continue
            if try_add(section, item):
                selected_terms.update(text_terms)
                selected_numbers.update(numbers)

        # Keep at least one item in each non-empty section if possible.
        for section_name in ["statements", "conditions", "actions"]:
            if counts[section_name] > 0:
                continue
            for section, item, _rank in ranked:
                if section != section_name:
                    continue
                if try_add(section, item):
                    break

        out_statements = [item for section, item in selected if section == "statements"]
        out_conditions = [item for section, item in selected if section == "conditions"]
        out_actions = [item for section, item in selected if section == "actions"]
        return out_statements, out_conditions, out_actions

    def _normalize_payload_sections(
        self,
        *,
        payload: ComplianceBriefPayload,
        query: str,
        query_terms: set[str],
        claim_priority: dict[str, float],
        rebuild_summary: bool,
        apply_trim: bool = True,
    ) -> ComplianceBriefPayload:
        if payload.verdict == "insufficient_evidence":
            return payload
        statements = list(payload.applicable_rules)
        conditions = list(payload.conditions_and_exceptions)
        actions = list(payload.required_actions)
        if apply_trim:
            statements, conditions, actions = self._adaptive_trim_sections(
                query=query,
                query_terms=query_terms,
                claim_priority=claim_priority,
                statements=statements,
                conditions=conditions,
                actions=actions,
                max_claims_per_section=self.policy.max_claims_per_section,
            )
        else:
            seen_keys: set[str] = set()
            deduped: list[tuple[str, ComplianceBriefItem]] = []
            for section_name, items in [
                ("statements", statements),
                ("conditions", conditions),
                ("actions", actions),
            ]:
                for item in items:
                    key = self._item_dedup_key(item)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    deduped.append((section_name, item))
            statements = [item for section_name, item in deduped if section_name == "statements"]
            conditions = [item for section_name, item in deduped if section_name == "conditions"]
            actions = [item for section_name, item in deduped if section_name == "actions"]
        payload.applicable_rules = statements
        payload.conditions_and_exceptions = conditions
        payload.required_actions = actions
        if rebuild_summary:
            summary = self._build_short_summary(
                statements=statements,
                conditions=conditions,
                actions=actions,
                unresolved=list(payload.clarifications_needed),
            )
            if summary.strip():
                payload.short_summary = summary
        return payload

    @staticmethod
    def _truncate_at_sentence_boundary(text: str, max_chars: int, preserve_numeric: bool = False) -> str:
        if len(text) <= max_chars:
            return text
        hard_limit = int(max_chars * (1.30 if preserve_numeric else 1.15))
        truncated = text[:hard_limit]
        boundary = max(truncated.rfind(". "), truncated.rfind("; "), truncated.rfind(": "))
        if boundary > max_chars * 0.5:
            out = truncated[: boundary + 1].strip()
        else:
            out = truncated.strip()
        if not preserve_numeric:
            return out
        full_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        out_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", out))
        if not full_numbers or full_numbers.issubset(out_numbers):
            return out
        sentences = [segment.strip() for segment in re.split(r"(?<=[.;:])\s+", text) if segment.strip()]
        numeric_limit = int(max_chars * 1.45)
        for sentence in sentences:
            sentence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", sentence))
            if not sentence_numbers or sentence_numbers.issubset(out_numbers):
                continue
            candidate = (out + " " + sentence).strip()
            if len(candidate) > numeric_limit:
                break
            out = candidate
            out_numbers.update(sentence_numbers)
            if full_numbers.issubset(out_numbers):
                break
        return out

    def _normalize_numeric_phrases(self, text: str) -> str:
        if not text:
            return text
        normalized = text
        word_to_digit = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
            "sixty": "60",
            "seventy": "70",
            "eighty": "80",
            "ninety": "90",
            "hundred": "100",
        }
        units = ["day", "days", "month", "months", "year", "years", "metre", "metres", "meter", "meters", "m", "cm", "%"]
        for word, digit in word_to_digit.items():
            normalized = re.sub(rf"\b{word}\b(?=\s*(?:{'|'.join(units)}))", digit, normalized, flags=re.IGNORECASE)
            normalized = re.sub(rf"\b{word}\b(?=\s*%)", digit, normalized, flags=re.IGNORECASE)
        normalized = normalized.replace("500/@", "50%")
        return normalized

    def _is_numeric_query(self, query: str) -> bool:
        lowered = query.lower()
        if re.search(r"\b\d+(?:\.\d+)?\b", lowered):
            return True
        numeric_tokens = {
            "minimum",
            "maximum",
            "distance",
            "clearance",
            "width",
            "height",
            "setback",
            "timeline",
            "days",
            "months",
            "years",
            "fee",
            "fees",
            "values",
            "cap",
            "formula",
        }
        return any(token in lowered for token in numeric_tokens)

    def _ensure_numeric_claim_coverage(
        self,
        *,
        ordered_claims: list[str],
        selected_statements: list[ComplianceBriefItem],
        selected_conditions: list[ComplianceBriefItem],
        selected_actions: list[ComplianceBriefItem],
        selected_claim_ids: set[str],
        claim_texts: dict[str, str],
        claim_citations: dict[str, list[str]],
        query_terms: set[str],
        max_claims_per_section: int,
    ) -> tuple[list[ComplianceBriefItem], list[ComplianceBriefItem], list[ComplianceBriefItem]]:
        selected_numbers = self._collect_numbers_from_items([*selected_statements, *selected_conditions, *selected_actions])
        available_numbers = self._collect_numbers_from_claims(ordered_claims, claim_texts)
        if not available_numbers:
            return selected_statements, selected_conditions, selected_actions

        target_count = min(4, max(2, len(available_numbers)))
        if len(selected_numbers) >= target_count:
            return selected_statements, selected_conditions, selected_actions

        for claim_id in ordered_claims:
            if claim_id in selected_claim_ids:
                continue
            if len(selected_numbers) >= target_count:
                break
            text = claim_texts.get(claim_id, "").strip()
            citations = claim_citations.get(claim_id, [])
            if not text or not citations:
                continue
            claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
            if not claim_numbers:
                continue
            if claim_numbers.issubset(selected_numbers):
                continue
            concise = self._summarize_claim_text(text=text, query_terms=query_terms)
            if not concise:
                continue
            item = ComplianceBriefItem(claim_id=claim_id, text=concise, citation_ids=citations[:4])
            lowered = concise.lower()
            if any(token in lowered for token in ["provided that", "subject to", "except", "unless", " where ", " if "]):
                if len(selected_conditions) < max_claims_per_section:
                    selected_conditions.append(item)
                elif len(selected_actions) < max_claims_per_section:
                    selected_actions.append(item)
                elif len(selected_statements) < max_claims_per_section:
                    selected_statements.append(item)
                else:
                    break
            elif any(token in lowered for token in ["shall", "must", "required", "within", "before", "after"]):
                if len(selected_actions) < max_claims_per_section:
                    selected_actions.append(item)
                elif len(selected_statements) < max_claims_per_section:
                    selected_statements.append(item)
                elif len(selected_conditions) < max_claims_per_section:
                    selected_conditions.append(item)
                else:
                    break
            else:
                if len(selected_statements) < max_claims_per_section:
                    selected_statements.append(item)
                elif len(selected_actions) < max_claims_per_section:
                    selected_actions.append(item)
                elif len(selected_conditions) < max_claims_per_section:
                    selected_conditions.append(item)
                else:
                    break
            selected_claim_ids.add(claim_id)
            selected_numbers.update(claim_numbers)

        return selected_statements, selected_conditions, selected_actions

    def _collect_numbers_from_items(self, items: list[ComplianceBriefItem]) -> set[str]:
        numbers: set[str] = set()
        for item in items:
            numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", item.text))
        return numbers

    def _collect_numbers_from_claims(self, ordered_claims: list[str], claim_texts: dict[str, str]) -> set[str]:
        numbers: set[str] = set()
        for claim_id in ordered_claims:
            text = claim_texts.get(claim_id, "")
            numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        return numbers

    def _fallback_grounded_item(
        self,
        ordered_claims: list[str],
        claim_texts: dict[str, str],
        claim_citations: dict[str, list[str]],
        query_terms: set[str],
    ) -> ComplianceBriefItem | None:
        for claim_id in ordered_claims:
            text = claim_texts.get(claim_id, "").strip()
            citations = claim_citations.get(claim_id, [])
            if not text or not citations:
                continue
            concise = self._summarize_claim_text(text=text, query_terms=query_terms)
            if not concise:
                continue
            return ComplianceBriefItem(claim_id=claim_id, text=concise, citation_ids=citations[:4])
        return None

    def _non_regression_merge(
        self,
        *,
        validated: ComplianceBriefPayload,
        fallback: ComplianceBriefPayload,
        query: str,
        query_terms: set[str],
        claim_priority: dict[str, float],
        max_claims_per_section: int,
    ) -> ComplianceBriefPayload:
        """Merge missing claims/numbers from deterministic draft to prevent recall regression."""
        llm_items = validated.applicable_rules + validated.conditions_and_exceptions + validated.required_actions
        draft_items = fallback.applicable_rules + fallback.conditions_and_exceptions + fallback.required_actions
        if not llm_items and draft_items:
            return fallback
        if not draft_items:
            return validated

        llm_claim_ids = {item.claim_id for item in llm_items if item.claim_id}
        llm_numbers = set()
        for item in llm_items:
            llm_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", item.text))

        total_cap, _ = self._dynamic_claim_caps(
            query=query,
            query_terms=query_terms,
            max_claims_per_section=max_claims_per_section,
        )
        max_missing_to_add = max(2, total_cap - len(llm_items))
        missing_candidates: list[tuple[float, ComplianceBriefItem]] = []
        for item in draft_items:
            if item.claim_id and item.claim_id in llm_claim_ids:
                continue
            item_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", item.text))
            has_new_numbers = bool(item_numbers - llm_numbers)
            has_new_claim = item.claim_id and item.claim_id not in llm_claim_ids
            if has_new_numbers or has_new_claim:
                compressed_text = self._summarize_claim_text(text=item.text, query_terms=query_terms) or item.text
                merged_item = ComplianceBriefItem(
                    claim_id=item.claim_id,
                    text=compressed_text,
                    citation_ids=item.citation_ids,
                )
                rank = self._item_rank_score(
                    item=merged_item,
                    query_terms=query_terms,
                    numeric_query=self._is_numeric_query(query),
                    claim_priority=claim_priority,
                )
                if has_new_numbers:
                    rank += 0.22
                if has_new_claim:
                    rank += 0.08
                missing_candidates.append((rank, merged_item))

        missing_candidates.sort(key=lambda entry: entry[0], reverse=True)
        missing_items: list[ComplianceBriefItem] = []
        for _rank, item in missing_candidates:
            if len(missing_items) >= max_missing_to_add:
                break
            item_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", item.text))
            if item.claim_id and item.claim_id in llm_claim_ids and not (item_numbers - llm_numbers):
                continue
            missing_items.append(item)
            llm_numbers.update(item_numbers)
            if item.claim_id:
                llm_claim_ids.add(item.claim_id)

        if missing_items:
            extra_rules = list(validated.applicable_rules)
            for item in missing_items:
                lowered = item.text.lower()
                if any(t in lowered for t in ["provided that", "subject to", "except", "unless"]):
                    validated.conditions_and_exceptions.append(item)
                elif any(t in lowered for t in ["shall", "must", "required", "within", "before"]):
                    validated.required_actions.append(item)
                else:
                    extra_rules.append(item)
            validated.applicable_rules = extra_rules
        return validated

    def _ensure_grounded_claim_presence(
        self,
        *,
        validated: ComplianceBriefPayload,
        fallback: ComplianceBriefPayload,
    ) -> ComplianceBriefPayload:
        if any(
            [
                validated.applicable_rules,
                validated.conditions_and_exceptions,
                validated.required_actions,
            ]
        ):
            return validated
        if any(
            [
                fallback.applicable_rules,
                fallback.conditions_and_exceptions,
                fallback.required_actions,
            ]
        ):
            return fallback
        return validated

    def _claim_priority(
        self,
        *,
        claim_id: str,
        claim_texts: dict[str, str],
        claim_scores: dict[str, float],
        query_terms: set[str],
    ) -> float:
        text = claim_texts.get(claim_id, "")
        text_terms = self._query_terms(text)
        overlap = len(query_terms.intersection(text_terms))
        base = float(claim_scores.get(claim_id, 0.0)) + (0.02 * float(overlap))

        topic = claim_id.split("-", 1)[0].strip().lower()
        topic_terms = self._query_terms(topic.replace("_", " "))
        topic_overlap = len(query_terms.intersection(topic_terms))
        base += 0.04 * float(topic_overlap)

        lowered = text.lower()
        looks_table_heavy = any(token in lowered for token in ["schedule", "sl.no", "sl no", "table i", "table ii"])
        table_expected = bool(
            query_terms.intersection(
                {"fee", "fees", "cost", "charge", "clearance", "distance", "minimum", "maximum", "value", "values"}
            )
        )
        if looks_table_heavy and not table_expected:
            base -= 0.15

        looks_definition = ("for the purpose of these rules" in lowered) or bool(re.search(r"\bmeans\b", lowered[:120]))
        if looks_definition and not query_terms.intersection({"definition", "definitions", "means"}):
            base -= 0.12

        mentions_telecom = "telecommunication tower" in lowered or "pole structure" in lowered
        if mentions_telecom and not query_terms.intersection({"telecommunication", "tower", "towers", "pole", "poles"}):
            base -= 0.35

        numeric_or_formula_query = bool(
            query_terms.intersection({"compute", "computed", "formula", "maximum", "minimum", "distance", "height", "setback"})
        )
        has_numeric = bool(re.search(r"\b\d+(?:\.\d+)?\b", lowered))
        if numeric_or_formula_query and not has_numeric:
            base -= 0.18
        if numeric_or_formula_query and has_numeric:
            base += 0.1
        if numeric_or_formula_query and ("certain provisions not to apply" in lowered):
            base -= 0.35
        if query_terms.intersection({"security", "zone"}) and "security zone" in lowered:
            base += 0.12

        if "appendix a2" in lowered and query_terms.intersection(
            {"huts", "special", "chapter", "intimation", "exempt", "replaces"}
        ):
            base += 0.25
        if "permit not necessary" in lowered and query_terms.intersection({"permit", "exempt", "replaces", "necessary"}):
            base += 0.2
        if topic == "flood_crz" and query_terms.intersection({"flood", "crz", "coastal"}):
            base += 0.12
        return base
