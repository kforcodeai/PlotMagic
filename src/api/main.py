from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import applicability_router, explain_router, ingest_router, query_router, rules_router

app = FastAPI(title="PlotMagic Compliance RAG", version="0.1.0")

app.include_router(ingest_router)
app.include_router(applicability_router)
app.include_router(query_router)
app.include_router(explain_router)
app.include_router(rules_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

