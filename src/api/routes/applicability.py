from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import get_engine
from src.models.schemas import ApplicabilityRequest

router = APIRouter(prefix="/resolve-applicability", tags=["applicability"])


@router.post("")
def resolve_applicability(request: ApplicabilityRequest) -> dict[str, object]:
    engine = get_engine()
    return engine.resolve_applicability(request)

