# Provider Architecture

This codebase uses a provider-agnostic adapter model for embeddings, reranking, and LLM synthesis.

## Contracts

- `EmbeddingProvider`: `embed`, `embed_batch`, `health`
- `RerankerProvider`: `rerank`, `health`
- `LLMProvider`: `generate_structured`, `health`
- `ProviderHealth`: startup/runtime capability snapshot

Contracts are defined in `src/providers/base.py`.

## Composition Root

- `config/providers.yaml` defines active providers and provider settings.
- `src/providers/config.py` loads typed provider config with environment overrides.
- `src/providers/registry.py` maps provider IDs to adapter constructors.
- `src/providers/factory.py` creates concrete providers with local fallbacks:
  - embedding fallback: `hash_embedding`
  - reranker fallback: `no_reranker`
  - llm fallback: `no_llm`

`ComplianceEngine` only consumes the contracts. Retrieval code does not reference provider SDKs.

## Environment Overrides

- `PLOTMAGIC_EMBEDDING_PROVIDER`
- `PLOTMAGIC_RERANK_PROVIDER`
- `PLOTMAGIC_LLM_PROVIDER`
- `PLOTMAGIC_EMBEDDING_MODEL`
- `PLOTMAGIC_RERANK_MODEL`
- `PLOTMAGIC_LLM_MODEL`
- `OPENAI_API_KEY`
- `COHERE_API_KEY`

Environment values take precedence over `config/providers.yaml`.

## Extension Rules

To add a provider:

1. Add one adapter class under `src/providers/adapters/`.
2. Register it in `src/providers/default_registry.py`.
3. Add its stanza in `config/providers.yaml`.

No retrieval/service code changes should be required.

## SDK Boundary Policy

Direct provider SDK imports (`openai`, `cohere`, etc.) are only allowed inside adapter modules under `src/providers/adapters/`.
