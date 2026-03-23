from __future__ import annotations

from contextlib import contextmanager
import json
import os
import signal
import threading
import time
from copy import deepcopy
from typing import Any

from src.providers.base import LLMProvider, ProviderHealth
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
        raise TimeoutError(f"OpenAI request exceeded deadline ({timeout:.1f}s)")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


class OpenAIResponsesLLMProvider(LLMProvider):
    provider_id = "openai_responses_llm"

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.model = settings.model or "gpt-4.1-mini"
        self.timeout_s = settings.timeout_s
        self.max_retries = max(0, settings.max_retries)
        self.retry_backoff_s = max(0.0, settings.retry_backoff_s)

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

    def generate_structured(
        self,
        *,
        task: str,
        payload: dict[str, Any],
        json_schema: dict[str, Any],
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> dict[str, Any]:
        user_payload = json.dumps({"task": task, "payload": payload}, ensure_ascii=False)
        normalized_schema = self._normalize_for_openai_strict_schema(json_schema)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with _request_deadline(self.timeout_s + 2.0):
                    # Use caller-provided instructions if present in payload,
                    # otherwise fall back to default system prompt.
                    _default_system = (
                        "You are a compliance drafting assistant. "
                        "Return strict JSON only and do not invent facts beyond payload."
                    )
                    if isinstance(payload, dict) and payload.get("instructions"):
                        system_content = str(payload["instructions"])
                    else:
                        system_content = _default_system
                    response = self._client.responses.create(
                        model=self.model,
                        input=[
                            {
                                "role": "system",
                                "content": system_content,
                            },
                            {
                                "role": "user",
                                "content": user_payload,
                            },
                        ],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "compliance_brief",
                                "schema": normalized_schema,
                                "strict": True,
                            }
                        },
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        timeout=self.timeout_s,
                    )
                content = self._extract_text(response)
                if isinstance(content, dict):
                    return content
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ProviderUnavailable("OpenAI responses payload is not a JSON object")
                return parsed
            except Exception as exc:  # pragma: no cover - network/provider behavior
                last_error = exc
                message = str(exc).lower()
                if isinstance(exc, TimeoutError) or "timeout" in message or "deadline" in message:
                    mapped: Exception = ProviderTimeout(f"OpenAI responses timeout: {exc}")
                elif "auth" in message or "401" in message or "invalid api key" in message:
                    mapped = ProviderAuthError(f"OpenAI responses auth failure: {exc}")
                else:
                    mapped = ProviderUnavailable(f"OpenAI responses failure: {exc}")
                if attempt >= self.max_retries:
                    raise mapped from exc
                time.sleep(self.retry_backoff_s * (attempt + 1))

        raise ProviderUnavailable(f"OpenAI responses failed after retries: {last_error}")

    def _extract_text(self, response: Any) -> str | dict[str, Any]:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if hasattr(response, "output"):
            out = response.output
            if isinstance(out, list):
                for item in out:
                    content = getattr(item, "content", None)
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        parsed = getattr(part, "parsed", None)
                        if isinstance(parsed, dict):
                            return parsed
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            return text
        raise ProviderUnavailable("OpenAI responses returned no parseable output")

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self.provider_id,
            available=True,
            capabilities=["generate_structured"],
            details={"model": self.model, "timeout_s": self.timeout_s},
        )

    def _normalize_for_openai_strict_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(schema)

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                node_type = node.get("type")
                if node_type == "object":
                    properties = node.get("properties", {})
                    if isinstance(properties, dict):
                        # OpenAI strict json_schema requires explicitly disabling unknown keys.
                        node["additionalProperties"] = False
                        # OpenAI strict json_schema also requires all keys to be listed in required.
                        node["required"] = list(properties.keys())
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(normalized)
        return normalized
