from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProviderSettings:
    provider_id: str
    model: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 1
    retry_backoff_s: float = 0.5
    dim: int = 256
    top_n: int = 80
    enabled: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureFlags:
    rerank_enabled: bool = True
    rerank_top_n: int = 80
    llm_enabled: bool = True


@dataclass(slots=True)
class ProvidersConfig:
    embedding_provider_id: str
    reranker_provider_id: str
    embedding_settings: dict[str, ProviderSettings]
    reranker_settings: dict[str, ProviderSettings]
    feature_flags: FeatureFlags
    llm_provider_id: str = "no_llm"
    llm_settings: dict[str, ProviderSettings] = field(default_factory=dict)

    def embedding(self) -> ProviderSettings:
        return self.embedding_settings[self.embedding_provider_id]

    def reranker(self) -> ProviderSettings:
        return self.reranker_settings[self.reranker_provider_id]

    def llm(self) -> ProviderSettings:
        return self.llm_settings[self.llm_provider_id]


_ENV_OVERRIDES = {
    "embedding_provider_id": "PLOTMAGIC_EMBEDDING_PROVIDER",
    "reranker_provider_id": "PLOTMAGIC_RERANK_PROVIDER",
    "llm_provider_id": "PLOTMAGIC_LLM_PROVIDER",
    "embedding_model": "PLOTMAGIC_EMBEDDING_MODEL",
    "reranker_model": "PLOTMAGIC_RERANK_MODEL",
    "llm_model": "PLOTMAGIC_LLM_MODEL",
    "openai_api_key": "OPENAI_API_KEY",
    "cohere_api_key": "COHERE_API_KEY",
}


def load_providers_config(config_path: Path) -> ProvidersConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    defaults = raw.get("defaults", {})
    providers = raw.get("providers", {})
    embedding_raw = providers.get("embedding", {})
    reranker_raw = providers.get("reranker", {})
    llm_raw = providers.get("llm", {})

    embedding_settings = {
        provider_id: _parse_settings(provider_id, settings)
        for provider_id, settings in embedding_raw.items()
    }
    reranker_settings = {
        provider_id: _parse_settings(provider_id, settings)
        for provider_id, settings in reranker_raw.items()
    }
    llm_settings = {
        provider_id: _parse_settings(provider_id, settings)
        for provider_id, settings in llm_raw.items()
    }

    embedding_provider_id = str(defaults.get("embedding_provider", "hash_embedding"))
    reranker_provider_id = str(defaults.get("reranker_provider", "no_reranker"))
    llm_provider_id = str(defaults.get("llm_provider", "no_llm"))

    embedding_provider_id = os.getenv(_ENV_OVERRIDES["embedding_provider_id"], embedding_provider_id)
    reranker_provider_id = os.getenv(_ENV_OVERRIDES["reranker_provider_id"], reranker_provider_id)
    llm_provider_id = os.getenv(_ENV_OVERRIDES["llm_provider_id"], llm_provider_id)

    if embedding_provider_id not in embedding_settings:
        raise ValueError(f"Unknown embedding provider '{embedding_provider_id}'")
    if reranker_provider_id not in reranker_settings:
        raise ValueError(f"Unknown reranker provider '{reranker_provider_id}'")
    if llm_provider_id not in llm_settings:
        raise ValueError(f"Unknown llm provider '{llm_provider_id}'")

    emb_model_override = os.getenv(_ENV_OVERRIDES["embedding_model"])
    if emb_model_override:
        embedding_settings[embedding_provider_id].model = emb_model_override

    rerank_model_override = os.getenv(_ENV_OVERRIDES["reranker_model"])
    if rerank_model_override:
        reranker_settings[reranker_provider_id].model = rerank_model_override

    llm_model_override = os.getenv(_ENV_OVERRIDES["llm_model"])
    if llm_model_override:
        llm_settings[llm_provider_id].model = llm_model_override

    openai_key = os.getenv(_ENV_OVERRIDES["openai_api_key"])
    if openai_key and "openai_embedding" in embedding_settings:
        embedding_settings["openai_embedding"].api_key = openai_key
    if openai_key and "openai_responses_llm" in llm_settings:
        llm_settings["openai_responses_llm"].api_key = openai_key

    cohere_key = os.getenv(_ENV_OVERRIDES["cohere_api_key"])
    if cohere_key and "cohere_reranker" in reranker_settings:
        reranker_settings["cohere_reranker"].api_key = cohere_key

    feature_flags_raw = raw.get("feature_flags", {})
    feature_flags = FeatureFlags(
        rerank_enabled=bool(feature_flags_raw.get("rerank_enabled", True)),
        rerank_top_n=int(feature_flags_raw.get("rerank_top_n", 80)),
        llm_enabled=bool(feature_flags_raw.get("llm_enabled", True)),
    )

    return ProvidersConfig(
        embedding_provider_id=embedding_provider_id,
        reranker_provider_id=reranker_provider_id,
        embedding_settings=embedding_settings,
        reranker_settings=reranker_settings,
        feature_flags=feature_flags,
        llm_provider_id=llm_provider_id,
        llm_settings=llm_settings,
    )


def _parse_settings(provider_id: str, raw: dict[str, Any]) -> ProviderSettings:
    known_keys = {
        "model",
        "api_key_env",
        "timeout_s",
        "max_retries",
        "retry_backoff_s",
        "dim",
        "top_n",
        "enabled",
    }
    extras = {key: value for key, value in raw.items() if key not in known_keys}
    return ProviderSettings(
        provider_id=provider_id,
        model=raw.get("model"),
        api_key_env=raw.get("api_key_env"),
        timeout_s=float(raw.get("timeout_s", 30.0)),
        max_retries=int(raw.get("max_retries", 1)),
        retry_backoff_s=float(raw.get("retry_backoff_s", 0.5)),
        dim=int(raw.get("dim", 256)),
        top_n=int(raw.get("top_n", 80)),
        enabled=bool(raw.get("enabled", True)),
        extras=extras,
    )
