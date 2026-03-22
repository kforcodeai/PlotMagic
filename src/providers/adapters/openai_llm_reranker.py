from __future__ import annotations

from contextlib import contextmanager
import json
import os
import signal
import threading
import time
from typing import Any

from src.providers.base import ProviderHealth, RerankCandidate, RerankResult, RerankerProvider
from src.providers.config import ProviderSettings
from src.providers.errors import ProviderAuthError, ProviderTimeout, ProviderUnavailable


@contextmanager
def _request_deadline(seconds: float):
    timeout = float(seconds or 0.0)
    if timeout <= 0.0:
        yield
        return
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"OpenAI reranker request exceeded deadline ({timeout:.1f}s)")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


class OpenAILLMRerankerProvider(RerankerProvider):
    provider_id = "openai_llm_reranker"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model = settings.model or "gpt-4.1-mini"
        self.timeout_s = settings.timeout_s
        self.max_retries = max(0, settings.max_retries)
        self.retry_backoff_s = max(0.0, settings.retry_backoff_s)
        self.candidate_limit = max(8, int(settings.extras.get("candidate_limit", 30)))

        api_key = settings.api_key
        if not api_key and settings.api_key_env:
            api_key = os.getenv(settings.api_key_env)
        if not api_key:
            raise ProviderAuthError("OpenAI API key is not configured")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("openai package is not installed") from exc
        openai_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout_s,
            "max_retries": 0,
        }
        try:  # pragma: no cover - optional dependency behavior
            import httpx  # type: ignore

            openai_kwargs["http_client"] = httpx.Client(
                timeout=httpx.Timeout(self.timeout_s),
                limits=httpx.Limits(max_connections=40, max_keepalive_connections=0),
                transport=httpx.HTTPTransport(retries=0),
                http2=False,
            )
        except Exception:
            pass
        try:
            self._client = OpenAI(**openai_kwargs)
        except TypeError:
            # Keep compatibility with simplified/mocked OpenAI clients used in tests.
            self._client = OpenAI(api_key=api_key)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not candidates:
            return []
        top_n = top_n or min(len(candidates), self.settings.top_n)
        shortlist = sorted(candidates, key=lambda row: row.base_score, reverse=True)[: max(top_n * 2, self.candidate_limit)]
        payload_candidates = [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text[:1200],
                "base_score": float(candidate.base_score),
            }
            for candidate in shortlist
        ]
        schema = {
            "type": "object",
            "properties": {
                "ranked_candidate_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            },
            "required": ["ranked_candidate_ids"],
            "additionalProperties": False,
        }
        user_payload = json.dumps(
            {
                "query": query,
                "top_n": top_n,
                "candidates": payload_candidates,
            },
            ensure_ascii=False,
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with _request_deadline(self.timeout_s + 2.0):
                    response = self._client.responses.create(
                        model=self.model,
                        input=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a legal retrieval reranker. "
                                    "Rank candidates by direct answer relevance to the query with emphasis on numeric thresholds, "
                                    "timelines, provisos, rule references, and table rows. Return strict JSON only."
                                ),
                            },
                            {
                                "role": "user",
                                "content": user_payload,
                            },
                        ],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "rerank",
                                "schema": schema,
                                "strict": True,
                            }
                        },
                        temperature=0.0,
                        max_output_tokens=600,
                        timeout=self.timeout_s,
                    )
                parsed = self._parse_response(response)
                ranked_ids = [str(item) for item in parsed.get("ranked_candidate_ids", []) if str(item).strip()]
                if not ranked_ids:
                    raise ProviderUnavailable("OpenAI reranker returned empty ranking")

                known_ids = {item.candidate_id for item in shortlist}
                ordered_ids: list[str] = []
                seen: set[str] = set()
                for cid in ranked_ids:
                    if cid not in known_ids or cid in seen:
                        continue
                    ordered_ids.append(cid)
                    seen.add(cid)
                    if len(ordered_ids) >= top_n:
                        break
                if len(ordered_ids) < top_n:
                    for item in shortlist:
                        if item.candidate_id in seen:
                            continue
                        ordered_ids.append(item.candidate_id)
                        seen.add(item.candidate_id)
                        if len(ordered_ids) >= top_n:
                            break

                out: list[RerankResult] = []
                total = max(1, len(ordered_ids))
                base_map = {item.candidate_id: item.base_score for item in shortlist}
                for idx, cid in enumerate(ordered_ids):
                    rank_score = (total - idx) / float(total)
                    score = rank_score + (0.05 * float(base_map.get(cid, 0.0)))
                    out.append(RerankResult(candidate_id=cid, score=score))
                return out
            except Exception as exc:  # pragma: no cover - provider behavior
                last_error = exc
                message = str(exc).lower()
                if isinstance(exc, TimeoutError) or "timeout" in message or "deadline" in message:
                    mapped: Exception = ProviderTimeout(f"OpenAI reranker timeout: {exc}")
                elif "auth" in message or "401" in message or "invalid api key" in message:
                    mapped = ProviderAuthError(f"OpenAI reranker auth failure: {exc}")
                else:
                    mapped = ProviderUnavailable(f"OpenAI reranker failure: {exc}")
                if attempt >= self.max_retries:
                    raise mapped from exc
                time.sleep(self.retry_backoff_s * (attempt + 1))
        raise ProviderUnavailable(f"OpenAI reranker failed after retries: {last_error}")

    def _parse_response(self, response: Any) -> dict[str, Any]:
        if hasattr(response, "output_text") and response.output_text:
            parsed = json.loads(response.output_text)
            if isinstance(parsed, dict):
                return parsed
        if hasattr(response, "output"):
            out = response.output
            if isinstance(out, list):
                for item in out:
                    content = getattr(item, "content", None)
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        structured = getattr(part, "parsed", None)
                        if isinstance(structured, dict):
                            return structured
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            parsed = json.loads(text)
                            if isinstance(parsed, dict):
                                return parsed
        raise ProviderUnavailable("OpenAI reranker returned no parseable output")

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            available=True,
            capabilities=["rerank"],
            details={"model": self.model, "timeout_s": self.timeout_s},
        )
