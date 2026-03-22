from __future__ import annotations

from pathlib import Path

from src.api.service import ComplianceEngine
from src.models.schemas import QueryRequest


def test_query_emits_tool_events() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = ComplianceEngine(root=root)
    engine.ingest(state="kerala", jurisdiction_type="panchayat")

    events: list[dict[str, object]] = []

    def sink(event: dict[str, object]) -> None:
        events.append(event)

    _response = engine.query(
        QueryRequest(
            query="What is permit validity under KPBR?",
            state="kerala",
            jurisdiction_type="panchayat",
            panchayat_category="Category-II",
            top_k=8,
            debug_trace=True,
        ),
        event_sink=sink,
    )
    steps = [str(item.get("step")) for item in events]
    assert "tool.scope_resolver" in steps
    assert "tool.query_planner" in steps
    assert "tool.query_complete" in steps
