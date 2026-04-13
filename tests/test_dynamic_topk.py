from __future__ import annotations

from streamlit_app import choose_dynamic_top_k


def test_choose_dynamic_top_k_prefers_higher_for_complex_queries() -> None:
    simple_q = "Permit needed?"
    complex_q = (
        "What are the setback distance and penalty rules with exceptions, "
        "and what documents and certificates are required for regularisation?"
    )
    simple_k, _simple_reason = choose_dynamic_top_k(simple_q, "panchayat", "openai_responses_llm")
    complex_k, _complex_reason = choose_dynamic_top_k(complex_q, "panchayat", "openai_responses_llm")
    assert complex_k > simple_k


def test_choose_dynamic_top_k_adjusts_for_no_llm_mode() -> None:
    q = "What are the setback rules?"
    llm_k, _ = choose_dynamic_top_k(q, "panchayat", "openai_responses_llm")
    no_llm_k, _ = choose_dynamic_top_k(q, "panchayat", "no_llm")
    assert no_llm_k >= llm_k


def test_choose_dynamic_top_k_respects_bounds() -> None:
    very_long = " ".join(
        [
            "setback",
            "distance",
            "clearance",
            "penalty",
            "compounding",
            "documents",
            "certificate",
            "exemption",
            "occupancy",
            "fee",
            "regularisation",
            "provided",
            "unless",
            "except",
            "and",
            "or",
            "100",
            "200",
        ]
        * 4
    )
    k, _ = choose_dynamic_top_k(very_long, "municipality", "no_llm")
    assert 6 <= k <= 24

