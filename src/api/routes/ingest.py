from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import get_engine
from src.models.schemas import IngestRequest, IngestResult

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResult)
def ingest(request: IngestRequest) -> IngestResult:
    engine = get_engine()
    return engine.ingest(state=request.state, jurisdiction_type=request.jurisdiction_type)

