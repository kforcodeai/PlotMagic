# State Extension Guide

This project uses a `StatePack` contract so onboarding a new state is deterministic and testable.

## Required Artifacts

1. Source corpus in `data/{state_code}/`.
2. Parser implementation under `src/ingestion/parsers/`.
3. State config in `config/states.yaml`.
4. Local body resolver config in `config/local_bodies/{state}_local_bodies.yaml`.
5. Tests in `tests/statepacks/{state_code}/` and `tests/evaluation/`.

## StatePack Contract

Every state pack must define:

- `parser_registry`: jurisdiction to parser mapping.
- `scope_resolver_config`: location and local body mapping.
- `occupancy_mapping_config`: deterministic occupancy ontology and keywords.
- `precedence_policy`: conflict precedence order.
- `normalizer_rules`: OCR/formatting fix dictionary.

## Validation Checklist

- Rule hierarchy extraction: chapter -> rule -> sub-rule -> proviso -> table.
- Citation integrity: every claim maps to rule and anchor.
- Deterministic routing: location resolves jurisdiction without LLM.
- Mixed-use handling: most restrictive rule logic tested.
- Temporal validity: amendment effective ranges respected.

## Promotion Gates

State is promotable only when:

- Metrics in `config/evaluation/metrics.yaml` pass.
- Gold query suite includes scope, occupancy, table and cross-reference queries.
- Unsupported claim rate remains below threshold.

