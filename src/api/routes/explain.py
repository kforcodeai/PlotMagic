from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.dependencies import get_engine

router = APIRouter(prefix="/explain", tags=["explain"])


@router.get("")
def explain_rule(
    rule_number: str = Query(...),
    state: str | None = Query(None),
    jurisdiction_type: str = Query(...),
) -> dict[str, object]:
    engine = get_engine()
    state_code = state or engine._default_state()
    rows = engine.structured_store.get_rule_by_number(
        state=state_code,
        jurisdiction_type=jurisdiction_type,
        rule_number=rule_number,
    )
    if not rows:
        return {"found": False, "message": f"Rule {rule_number} not found for {state_code}:{jurisdiction_type}"}
    return {
        "found": True,
        "matches": [
            {
                "document_id": row["document_id"],
                "ruleset_id": row["ruleset_id"],
                "chapter_number": row["chapter_number"],
                "rule_number": row["rule_number"],
                "rule_title": row["rule_title"],
                "anchor_id": row["anchor_id"],
                "source_file": row["source_file"],
                "full_text": row["full_text"],
            }
            for row in rows
        ],
    }
