from __future__ import annotations

from pathlib import Path

from src.indexing.index_manifest import (
    compute_sources_hash,
    deterministic_collection_name,
    manifest_matches,
)


def test_deterministic_collection_name_is_stable_and_slugged() -> None:
    name = deterministic_collection_name(
        state="Kerala",
        jurisdiction="Panchayat",
        ruleset_id="KPBR_2011",
        embedding_model="text-embedding-3-large",
        schema_version="v1",
    )
    assert name == "kerala-panchayat-kpbr-2011-text-embedding-3-large-v1"


def test_manifest_matches_requires_exact_expected_values() -> None:
    expected = {"a": 1, "b": "x"}
    assert manifest_matches({"a": 1, "b": "x", "c": 3}, expected) is True
    assert manifest_matches({"a": 1, "b": "y"}, expected) is False
    assert manifest_matches(None, expected) is False


def test_compute_sources_hash_is_order_stable(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("alpha", encoding="utf-8")
    b.write_text("beta", encoding="utf-8")

    hash_one = compute_sources_hash([b, a])
    hash_two = compute_sources_hash([a, b])
    assert hash_one == hash_two
