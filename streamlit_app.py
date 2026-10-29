"""Streamlit entry point: ``streamlit run streamlit_app.py``.

Kept short on purpose: the sidebar and each tab live in src/starbucks_nutrition/ui/.
"""

from __future__ import annotations

import streamlit as st

from starbucks_nutrition.textnorm import use_utf8_console
from starbucks_nutrition.ui import ai_summary, ask, charts, data_quality, explore, overview
from starbucks_nutrition.ui.data_source import sidebar_data_source, sidebar_include_unreliable

use_utf8_console()  # server log lines (stderr) keep ™/®/accents on Windows
st.set_page_config(page_title="Menu Nutrition Explorer", layout="wide")
st.title("Starbucks menu nutrition explorer")
st.caption(
    "Every number is computed with pandas from the menu files; limitations of the data "
    "are shown next to the numbers they affect."
)

data = sidebar_data_source()
if data is None:
    st.stop()
include_unreliable = sidebar_include_unreliable(data)

TABS = {
    "Overview": overview.render,
    "AI summary": ai_summary.render,
    "Ask the menu": ask.render,
    "Explore & filter": explore.render,
    "Charts": charts.render,
    "Data quality": data_quality.render,
}
for tab, render in zip(st.tabs(list(TABS)), TABS.values(), strict=True):
    with tab:
        render(data, include_unreliable)
