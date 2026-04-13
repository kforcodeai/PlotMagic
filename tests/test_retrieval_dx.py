from __future__ import annotations

from dataclasses import dataclass, field

from streamlit_app import _evidence_sort_key, _primary_retrieval_score


@dataclass
class _Item:
    claim_id: str
    chunk_id: str
    scores: dict[str, float] = field(default_factory=dict)


def test_primary_retrieval_score_prefers_ordered_keys() -> None:
    key, value = _primary_retrieval_score(
        {
            "retrieval_score": 0.91,
            "rrf_score": 0.62,
            "query_relevance": 0.77,
        }
    )
    assert key == "rrf_score"
    assert value == 0.62


def test_primary_retrieval_score_falls_back_to_max_numeric() -> None:
    key, value = _primary_retrieval_score({"foo": 0.2, "bar": 0.7})
    assert key == "bar"
    assert value == 0.7


def test_evidence_sort_key_supports_descending_rank() -> None:
    items = [
        _Item(claim_id="c1", chunk_id="a", scores={"rrf_score": 0.2}),
        _Item(claim_id="c2", chunk_id="b", scores={"rrf_score": 0.9}),
        _Item(claim_id="c3", chunk_id="c", scores={"rrf_score": 0.5}),
    ]
    ranked = sorted(items, key=_evidence_sort_key, reverse=True)
    assert [item.chunk_id for item in ranked] == ["b", "c", "a"]

