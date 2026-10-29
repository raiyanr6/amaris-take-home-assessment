"""Tab: Ask the menu (bonus) — natural-language questions answered via tool calling."""

from __future__ import annotations

import streamlit as st

from starbucks_nutrition.llm.client import LLMError, LLMNotConfigured
from starbucks_nutrition.llm.qa import (
    MAX_QUESTION_CHARS,
    QAResult,
    answer_question,
    trace_for_display,
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

CHAT_KEY = "chat_history"
PENDING_KEY = "chat_pending"
EXAMPLES: tuple[str, ...] = (
    "Which drinks have the most carbs?",
    "Which food items are under 500 calories and high in protein?",
    "What's the average caffeine content for drinks?",
    "Which drinks have the most sugar?",
    "How do drinks and food compare on calories?",
)


def _queue(question: str) -> None:
    st.session_state[PENDING_KEY] = question


def _clear() -> None:
    st.session_state[CHAT_KEY] = []


def _render_turn(turn: QAResult) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        st.markdown(turn.answer)
        grounding_badge(turn.grounding)
        with st.expander("How I got this"):
            if not turn.trace:
                st.caption("No data lookups were needed for this reply.")
            for step_no, step in enumerate(trace_for_display(turn.trace), start=1):
                st.markdown(f"**{step_no}. `{step['tool']}`** (round {step['round']})")
                st.code(step["arguments"], language="json")
                st.code(step["result"], language="json")
            st.caption(
                f"Model `{turn.model}` · {turn.total_tokens:,} tokens · {turn.latency_s:.1f} s"
            )


def render(data: MenuData, include_unreliable: bool) -> None:
    source_caption(data)
    st.markdown(
        "Ask a question about the loaded menu. The model answers by calling read-only "
        "lookups (shown under *How I got this*); it never sees the raw files or writes code."
    )
    try:
        client = get_client()
    except LLMNotConfigured:
        show_setup_message()
        return

    history: list[QAResult] = st.session_state.setdefault(CHAT_KEY, [])
    st.caption("Try one of these:")
    buttons = st.columns(len(EXAMPLES))
    for i, (col, example) in enumerate(zip(buttons, EXAMPLES, strict=True)):
        col.button(example, key=f"example_{i}", on_click=_queue, args=(example,))

    for turn in history:
        _render_turn(turn)

    typed = st.chat_input("Ask about the menu…", max_chars=MAX_QUESTION_CHARS, key="chat_box")
    question = typed or st.session_state.pop(PENDING_KEY, None)
    if question:
        try:
            with st.spinner("Looking it up…"):
                result = answer_question(
                    client,
                    question,
                    data.menu,
                    history=[(t.question, t.answer) for t in history],
                    include_unreliable=include_unreliable,
                )
        except LLMError as exc:
            with st.chat_message("user"):
                st.markdown(question)
            show_error(exc)
        else:
            history.append(result)
            _render_turn(result)

    cols = st.columns([1, 3])
    cols[0].button("Clear chat", key="chat_clear", on_click=_clear, disabled=not history)
    cols[1].caption(f"AI requests left in this session: {remaining_calls()}")
