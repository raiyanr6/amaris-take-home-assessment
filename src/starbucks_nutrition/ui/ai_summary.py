"""Tab: AI summary (Part 2) — the model narrates facts computed by pandas."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from starbucks_nutrition.llm.client import LLMClient, LLMError, LLMNotConfigured
from starbucks_nutrition.llm.context import build_facts
from starbucks_nutrition.llm.prompts import FOCUSES
from starbucks_nutrition.llm.summarize import (
    SummaryResult,
    focus_rankings,
    generate_summary,
    system_prompt,
)
from starbucks_nutrition.ui.common import source_caption
from starbucks_nutrition.ui.data_source import MenuData
from starbucks_nutrition.ui.llm_access import (
    get_client,
    grounding_badge,
    remaining_calls,
    show_error,
    show_setup_message,
)

RESULT_KEY = "summary_result"


@st.cache_data(show_spinner=False, max_entries=32)
def _facts(
    menu: pd.DataFrame,
    _reports: Any,
    include_unreliable: bool,
    sources: tuple[tuple[str, str], ...],
    expand: tuple[str, ...],
) -> dict[str, Any]:
    # Reports derive from the same file bytes as the menu, so the menu hash covers them.
    return build_facts(menu, _reports, include_unreliable, dict(sources), expand)


@st.cache_data(show_spinner=False, max_entries=64)
def _summary(facts_json: str, focus: str, model: str, _client: LLMClient) -> SummaryResult:
    """Cached by (facts = data + include_unreliable, focus, model): re-clicks don't re-bill."""
    return generate_summary(_client, json.loads(facts_json), focus)


def _render_result(result: SummaryResult, facts: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(result.text)
    grounding_badge(result.grounding)
    st.caption(
        f"Model `{result.model}` · {result.prompt_tokens + result.completion_tokens:,} tokens "
        f"({result.prompt_tokens:,} in / {result.completion_tokens:,} out) · "
        f"{result.latency_s:.1f} s · prompt v{result.prompt_version}"
    )
    with st.expander("What the model was given"):
        st.markdown("**System prompt**")
        st.code(system_prompt(result.focus), language="markdown")
        st.markdown("**FACTS** (computed with pandas; the model sees nothing else)")
        st.json(facts, expanded=False)


def render(data: MenuData, include_unreliable: bool) -> None:
    source_caption(data)
    st.markdown(
        "A plain-language summary written by an LLM (Groq) from figures computed with "
        "pandas. The model only narrates; every figure it uses is checked against the data."
    )
    try:
        client = get_client()
    except LLMNotConfigured:
        show_setup_message()
        return

    cols = st.columns([3, 1])
    focus = cols[0].selectbox("Focus", list(FOCUSES), key="summary_focus")
    cols[1].markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
    clicked = cols[1].button("Generate summary", type="primary", key="summary_generate")

    facts = _facts(
        data.menu,
        data.reports,
        include_unreliable,
        tuple(sorted(data.sources.items())),
        focus_rankings(focus),
    )
    facts_json = json.dumps(facts, ensure_ascii=False, allow_nan=False, sort_keys=True)
    request = (facts_json, focus, client.model)

    if clicked:
        try:
            with st.spinner("Writing the summary…"):
                st.session_state[RESULT_KEY] = (request, _summary(*request, client))
        except LLMError as exc:
            show_error(exc)

    stored = st.session_state.get(RESULT_KEY)
    if stored and stored[0] == request:
        _render_result(stored[1], facts)
    elif stored:
        st.caption("Settings or data changed since the last summary: generate again.")
    st.caption(
        f"AI requests left in this session: {remaining_calls()} (one per summary or question)"
    )
