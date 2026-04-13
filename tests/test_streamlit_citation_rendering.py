from __future__ import annotations

from dataclasses import dataclass

from streamlit_app import render_inline_citation_links_html


@dataclass
class _Citation:
    display_citation: str
    source_url: str | None


def test_render_inline_citation_links_html_renders_anchor_tags() -> None:
    lookup = {
        "c1": _Citation(display_citation="KPBR Rule 10", source_url="/rules/KPBR_2011/source#kpbr-ch2-r10"),
        "c2": _Citation(display_citation="KPBR Rule 7", source_url=None),
    }
    html_text = render_inline_citation_links_html(["c1", "c2", "missing"], lookup)
    assert "<a href=" in html_text
    assert "[KPBR Rule 10](" not in html_text
    assert "<code>KPBR Rule 7</code>" in html_text
    assert "<code>missing</code>" in html_text

