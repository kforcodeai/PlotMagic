from __future__ import annotations

from streamlit_app import _verdict_html


def test_depends_verdict_not_rendered_as_non_compliant() -> None:
    html = _verdict_html("depends")
    assert "pm-verdict-depends" in html
    assert "pm-verdict-noncompliant" not in html

