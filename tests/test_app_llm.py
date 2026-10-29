"""AppTest smoke tests for the AI tabs: fake client injected, and no key at all."""

from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from fakes import FakeLLMClient, reply, tool_call
from starbucks_nutrition import config
from starbucks_nutrition.llm.client import LLMRateLimited
from starbucks_nutrition.ui.ask import EXAMPLES
from starbucks_nutrition.ui.llm_access import CLIENT_KEY

pytestmark = pytest.mark.ui

APP = str(config.PROJECT_ROOT / "streamlit_app.py")
SUMMARY_TAB, ASK_TAB = 1, 2


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_data.clear()  # summaries are cached across sessions by design
    yield
    st.cache_data.clear()


def _app(client: FakeLLMClient | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    if client is not None:
        at.session_state[CLIENT_KEY] = client
    at.run()
    assert not at.exception, at.exception
    return at


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for kind in ("info", "success", "warning", "error") for e in at.get(kind)]
    return "\n".join(str(p) for p in parts)


def test_no_key_shows_setup_message_and_app_still_works(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")  # overrides the developer's .env
    at = _app()
    for tab in (at.tabs[SUMMARY_TAB], at.tabs[ASK_TAB]):
        assert any("need a Groq API key" in i.value for i in tab.info)
    assert len(at.tabs[0].metric) == 6  # Overview unaffected


def test_summary_with_fake_client() -> None:
    client = FakeLLMClient(
        [reply("### Headline\n- Drinks with data average 141 kcal; only 48% have data.")]
    )
    at = _app(client)
    at.button(key="summary_generate").click().run()
    assert not at.exception
    text = _all_text(at)
    assert "141 kcal" in text and "✓ All 2 figures verified" in text
    assert "fake-model" in text
    # Same data + focus + model: a second click is served from the cache (no new call).
    at.button(key="summary_generate").click().run()
    assert len(client.calls) == 1


def test_ask_example_question_runs_tools_and_shows_trace() -> None:
    client = FakeLLMClient(
        [
            reply(None, tool_call("top_items", {"metric": "carb_g", "category": "drink", "n": 3})),
            reply("Highest-carb drink: Starbucks® Signature Hot Chocolate."),
        ]
    )
    at = _app(client)
    at.button(key="example_0").click().run()
    assert not at.exception
    assert any(e.label == "How I got this" for e in at.expander)
    assert "top_items" in "\n".join(m.value for m in at.markdown)
    assert client.calls[0]["messages"][-1]["content"] == EXAMPLES[0]


def test_chat_input_rate_limited_shows_friendly_warning() -> None:
    at = _app(FakeLLMClient([LLMRateLimited("x", retry_after=7)]))
    at.chat_input(key="chat_box").set_value("Which food is highest in fiber?").run()
    assert not at.exception
    assert any("rate-limited" in w.value and "7 seconds" in w.value for w in at.warning)


def test_unexpected_error_in_ask_shows_friendly_message() -> None:
    at = _app(FakeLLMClient([RuntimeError("bug")]))
    at.chat_input(key="chat_box").set_value("Which food is highest in fiber?").run()
    assert not at.exception
    assert any("Something went wrong" in e.value for e in at.error)


def test_clear_chat() -> None:
    at = _app(FakeLLMClient())
    at.chat_input(key="chat_box").set_value("hello?").run()
    assert at.session_state["chat_history"]
    at.button(key="chat_clear").click().run()
    assert at.session_state["chat_history"] == []


def test_follow_up_question_keeps_input_and_history() -> None:
    client = FakeLLMClient(
        [
            reply(None, tool_call("top_items", {"metric": "carb_g", "category": "drink", "n": 1})),
            reply("Drinks: Cinnamon Dolce Frappuccino® Blended Coffee has the most carbs."),
            reply(None, tool_call("top_items", {"metric": "carb_g", "category": "food", "n": 1})),
            reply("Food: see the lookup."),
        ]
    )
    at = _app(client)
    at.chat_input(key="chat_box").set_value("Which drinks have the most carbs?").run()
    assert not at.exception
    assert len(at.session_state["chat_history"]) == 1
    assert len(at.chat_input) == 1  # input still there after the first answer
    assert any(e.label == "Example questions" for e in at.expander)  # examples tucked away

    at.chat_input(key="chat_box").set_value("and for food?").run()
    assert not at.exception
    history = at.session_state["chat_history"]
    assert [t.question for t in history] == ["Which drinks have the most carbs?", "and for food?"]
    assert len(at.chat_input) == 1

    follow_up_messages = client.calls[2]["messages"]  # first model call of question 2
    assert [m["role"] for m in follow_up_messages[:4]] == ["system", "user", "assistant", "user"]
    assert follow_up_messages[1]["content"] == "Which drinks have the most carbs?"
    assert "Cinnamon Dolce" in follow_up_messages[2]["content"]
    assert follow_up_messages[3]["content"] == "and for food?"


def test_example_button_uses_same_path_and_input_stays() -> None:
    at = _app(FakeLLMClient())
    at.button(key="example_1").click().run()
    assert [t.question for t in at.session_state["chat_history"]] == [EXAMPLES[1]]
    assert len(at.chat_input) == 1
