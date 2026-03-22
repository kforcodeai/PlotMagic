from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def deterministic_collection_name(
    *,
    state: str,
    jurisdiction: str,
    ruleset_id: str,
    embedding_model: str,
    schema_version: str,
) -> str:
    parts = [state, jurisdiction, ruleset_id, embedding_model, schema_version]
    slugged = [_slug(part) for part in parts if str(part).strip()]
    return "-".join(slugged)[:180] or "plotmagic-collection"


def compute_sources_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\n")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def manifest_matches(existing: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if not existing:
        return False
    for key, value in expected.items():
        if existing.get(key) != value:
            return False
    return True


def build_manifest(
    *,
    state: str,
    jurisdiction: str,
    ruleset_id: str,
    source_hash: str,
    parser_version: str,
    cleaning_version: str,
    embedding_provider: str,
    embedding_model: str,
    vector_dim: int,
    clause_count: int,
    collection_name: str,
) -> dict[str, Any]:
    return {
        "state": state,
        "jurisdiction": jurisdiction,
        "ruleset_id": ruleset_id,
        "source_hash": source_hash,
        "parser_version": parser_version,
        "cleaning_version": cleaning_version,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "vector_dim": vector_dim,
        "clause_count": clause_count,
        "collection_name": collection_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = lowered.strip("-")
    return lowered or "x"
