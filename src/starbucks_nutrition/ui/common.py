"""Small widgets shared by several tabs."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from starbucks_nutrition.ui.data_source import CATEGORIES, MenuData, source_label

CAFFEINE_PROXY_HELP = (
    "Caffeine status is a **name-based proxy**: the data has no caffeine column, so it is "
    "inferred from the drink name (espresso, cold brew, black/green tea, Refreshers → "
    "likely; decaf, herbal, crème, juice → unlikely; a tea whose type the name does not "
    "say → unknown). Food is not assessed (unknown). A *Caffeine (mg)* column in an "
    "uploaded file replaces the proxy."
)


def source_caption(data: MenuData) -> None:
    """One line saying where each category's numbers come from."""
    st.caption(
        "Source: "
        + " · ".join(
            f"{CATEGORIES[cat]} = **{source_label(src)}**" for cat, src in data.sources.items()
        )
    )


def caveats_box(caveats: list[str], title: str = "Read these numbers with care") -> None:
    if caveats:
        st.warning(f"**{title}**\n\n" + "\n".join(f"- {c}" for c in caveats))


def chart(fig: go.Figure, key: str) -> None:
    # theme=None keeps the chart's own colours (one fixed colour per category).
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


def how_to_read(text: str) -> None:
    st.caption(f"**How to read this:** {text}")
