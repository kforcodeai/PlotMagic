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

        schema = ComplianceBriefPayload.model_json_schema()
        payload = {
            "query": query,
            "grounding": grounding.model_dump(),
            "draft": draft.model_dump(),
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
            validated = self._ensure_grounded_claim_presence(validated=validated, fallback=draft)
            if validated.verdict != verdict:
                raise ValueError(
                    f"LLM verdict '{validated.verdict}' disagrees with grounding verdict '{verdict}'."
                )
            if self.policy.enforce_non_empty_summary and not validated.short_summary.strip():
                raise ValueError("LLM output violated contract: short_summary cannot be empty.")
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
        min_priority = max(0.02, max_priority * 0.50)
        max_claims_per_section = self.policy.max_claims_per_section + (1 if numeric_query else 0)
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
            text_key = re.sub(r"[^a-z0-9]+", " ", concise.lower()).strip()[:200]
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
        return ComplianceBriefPayload(
            verdict="depends",
            short_summary=summary,
            applicable_rules=selected_statements,
            conditions_and_exceptions=selected_conditions,
            required_actions=selected_actions,
            risk_flags=["Professional review recommended before permit submission."],
            clarifications_needed=sorted(set(unresolved)),
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
            out.append(sentence)
        return out

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
            max_sentences = 6
        elif numeric_query:
            max_sentences = 4
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
            selected.append(sentence)
            seen.add(key)
            seen.add(core_key)
            if len(selected) >= max_sentences:
                break

        if numeric_query and selected:
            covered_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", " ".join(selected)))
            for _score, sentence in scored:
                if len(selected) >= max_sentences + 1:
                    break
                numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", sentence))
                if not numbers or numbers.issubset(covered_numbers):
                    continue
                key = re.sub(r"^\(?\d+[A-Za-z]?\)?\s*", "", sentence.lower())
                key = re.sub(r"[^a-z0-9]+", " ", key).strip()
                core_key = re.sub(r"^(provided\s+that|subject\s+to|in\s+case|if|where)\s+", "", key).strip()[:180]
                if key in seen or core_key in seen:
                    continue
                selected.append(sentence)
                seen.add(key)
                seen.add(core_key)
                covered_numbers.update(numbers)

        concise = " ".join(selected).strip()
        concise = re.sub(r"\s+", " ", concise)
        concise = self._normalize_numeric_phrases(concise)
        if excavation_query:
            char_limit = self.policy.excavation_claim_char_limit
        elif numeric_query:
            char_limit = max(self.policy.concise_claim_char_limit, 420)
        else:
            char_limit = self.policy.concise_claim_char_limit
        return concise[:char_limit]

    def _build_short_summary(
        self,
        *,
        statements: list[ComplianceBriefItem],
        conditions: list[ComplianceBriefItem],
        actions: list[ComplianceBriefItem],
        unresolved: list[str],
    ) -> str:
        summary_fragments: list[str] = []
        if statements:
            summary_fragments.append(statements[0].text)
        if actions:
            summary_fragments.append(actions[0].text)
        if not summary_fragments and conditions:
            summary_fragments.append(conditions[0].text)

        if summary_fragments:
            summary = " ".join(summary_fragments)
        else:
            summary = "Grounded evidence retrieved; verify cited provisions before final compliance action."

        if unresolved:
            unresolved_note = "; ".join(sorted(set(unresolved))[:3])
            summary = f"{summary}\nUnresolved items: {unresolved_note}"
        return summary[: self.policy.summary_char_limit]

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
            key = f"{item.claim_id}:{re.sub(r'[^a-z0-9]+', ' ', item.text.lower()).strip()[:180]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

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
