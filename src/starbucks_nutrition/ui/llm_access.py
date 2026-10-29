"""Shared plumbing for the AI tabs: client creation, call budget, friendly errors.

The client lives in session_state (one per browser session, so no global state). Tests
inject a fake by setting ``st.session_state[CLIENT_KEY]`` before the first run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import streamlit as st

from starbucks_nutrition.llm.client import (
    KEY_ENV,
    MODEL_ENV,
    BudgetedClient,
    CallBudget,
    GroqClient,
    LLMBudgetExceeded,
    LLMClient,
    LLMError,
    LLMNotConfigured,
    LLMRateLimited,
    resolve_model,
    resolve_setting,
)
from starbucks_nutrition.llm.grounding import GroundingResult

logger = logging.getLogger(__name__)

CLIENT_KEY = "llm_client"
BUDGET_KEY = "llm_budget"

SETUP_MESSAGE = """\
**AI features need a Groq API key** (free): the rest of the app works without one.

1. Create a key at [console.groq.com/keys](https://console.groq.com/keys).
2. Local run: copy `.env.example` to `.env` and set `GROQ_API_KEY=...`, then restart
   `streamlit run streamlit_app.py`.
3. Streamlit Cloud: add `GROQ_API_KEY = "..."` under *App settings → Secrets*.

Optional: `GROQ_MODEL` picks another Groq model (default `llama-3.3-70b-versatile`).
"""


def _secrets() -> Mapping[str, Any] | None:
    """Only the two LLM settings from st.secrets; None when no secrets file exists."""
    try:
        return {k: st.secrets[k] for k in (KEY_ENV, MODEL_ENV) if k in st.secrets}
    except Exception:  # no secrets.toml: Streamlit raises; env/.env is used instead
        return None


def get_client() -> LLMClient:
    """The session's LLM client, wrapped in the per-session call budget.

    Raises LLMNotConfigured when no key is set (the tab then shows SETUP_MESSAGE).
    """
    budget = st.session_state.setdefault(BUDGET_KEY, CallBudget())
    inner = st.session_state.get(CLIENT_KEY)
    if inner is None:
        secrets = _secrets()
        key = resolve_setting(KEY_ENV, secrets)
        if not key:
            raise LLMNotConfigured("No Groq API key configured.")
        inner = GroqClient(api_key=key, model=resolve_model(secrets))
        st.session_state[CLIENT_KEY] = inner
    return BudgetedClient(inner, budget)


def remaining_calls() -> int:
    return st.session_state.setdefault(BUDGET_KEY, CallBudget()).remaining


def show_setup_message() -> None:
    st.info(SETUP_MESSAGE)


def show_error(exc: LLMError) -> None:
    """Turn a typed LLM error into a message a user can act on (never a traceback)."""
    if isinstance(exc, LLMNotConfigured):
        st.error(f"{exc} Check GROQ_API_KEY.")
        show_setup_message()
    elif isinstance(exc, LLMRateLimited):
        wait = f" in about {exc.retry_after:.0f} seconds" if exc.retry_after else " shortly"
        st.warning(f"The free Groq tier is rate-limited right now. Please try again{wait}.")
    elif isinstance(exc, LLMBudgetExceeded):
        st.warning(str(exc))
    else:
        logger.warning("LLM error shown to user: %s", exc)
        st.error(f"The AI request failed: {exc} The rest of the app still works.")


def grounding_badge(result: GroundingResult) -> None:
    """✓ when every figure in the text appears in the data the model was given."""
    if not result.checked:
        st.caption("No figures to verify in this answer.")
    elif result.ok:
        st.success(f"✓ All {result.checked} figures verified against the data", icon=None)
    else:
        st.warning(
            "⚠ Some figures were not found in the data the model was given: "
            + ", ".join(result.unverified)
            + ". Treat them with caution."
        )
