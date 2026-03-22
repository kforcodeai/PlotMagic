from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import get_engine
from src.models.schemas import AnswerResponse, QueryRequest

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=AnswerResponse)
def query(request: QueryRequest) -> AnswerResponse:
    engine = get_engine()
    return engine.query(request)

