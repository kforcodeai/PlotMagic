from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.dependencies import get_engine

router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("/browse")
def browse_rules(
    state: str | None = Query(None),
    jurisdiction_type: str = Query(...),
    occupancy: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, object]]:
    engine = get_engine()
    state_code = state or engine._default_state()
    rows = engine.structured_store.search_rules(
        state=state_code,
        jurisdiction_type=jurisdiction_type,
        occupancy=[occupancy] if occupancy else None,
        limit=limit,
    )
    result: list[dict[str, object]] = []
    for row in rows:
        result.append(
            {
                "document_id": row["document_id"],
                "ruleset_id": row["ruleset_id"],
                "chapter_number": row["chapter_number"],
                "rule_number": row["rule_number"],
                "rule_title": row["rule_title"],
                "anchor_id": row["anchor_id"],
            }
        )
    return result


@router.get("/{document_id}/source")
def get_source(document_id: str) -> dict[str, object]:
    engine = get_engine()
    for doc in engine.state.docs:
        if doc.document_id == document_id:
            return {
                "document_id": doc.document_id,
                "source_file": doc.source_file,
                "anchor_id": doc.anchor_id,
                "rule_number": doc.rule_number,
                "rule_title": doc.rule_title,
                "full_text": doc.full_text,
            }
    raise HTTPException(status_code=404, detail=f"Rule document '{document_id}' not found.")
