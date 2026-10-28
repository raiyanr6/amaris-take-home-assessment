"""Tab 3: the charts, each in its own container with a one-line reading guide."""

from __future__ import annotations

import streamlit as st

from starbucks_nutrition import analysis, viz
from starbucks_nutrition.ui.common import CAFFEINE_PROXY_HELP, chart, how_to_read, source_caption
from starbucks_nutrition.ui.data_source import CATEGORIES, MenuData

BOTH = "both"


def _top_n(data: MenuData, include_unreliable: bool) -> None:
    controls = st.columns([3, 2, 2, 2])
    category = controls[1].selectbox(
        "Category",
        [*CATEGORIES, BOTH],
        format_func=lambda c: "Drinks and food" if c == BOTH else CATEGORIES[c],
        key="top_category",
    )
    scope = None if category == BOTH else category
    metrics = analysis.available_metrics(data.menu, scope, include_unreliable)
    if not metrics:
        st.info("No metric has data for this category.")
        return
    metric = controls[0].selectbox(
        "Metric", metrics, format_func=analysis.metric_label, key="top_metric"
    )
    n = controls[2].slider("Number of items", 3, 20, 10, key="top_n")
    direction = controls[3].radio("Show", ["Highest", "Lowest"], horizontal=True, key="top_dir")
    fig = viz.top_n_chart(
        data.menu,
        metric,
        n,
        scope,
        ascending=direction == "Lowest",
        include_unreliable=include_unreliable,
    )
    chart(fig, key="chart_top_n")
    how_to_read(
        "Longer bar = more of the metric. Only items with a value are ranked; items "
        "without data are never treated as 0."
    )


def render(data: MenuData, include_unreliable: bool) -> None:
    source_caption(data)
    menu = data.menu

    with st.container(border=True):
        _top_n(data, include_unreliable)

    with st.container(border=True):
        chart(viz.macro_energy_chart(menu, include_unreliable), key="chart_macro")
        how_to_read(
            "Each bar is 100% of the energy from fat, carbs and protein (fat 9 kcal/g, "
            "carbs and protein 4 kcal/g), summed over items with data. Compare segment "
            "lengths between drinks and food."
        )

    with st.container(border=True):
        chart(viz.calories_vs_protein_scatter(menu, include_unreliable), key="chart_scatter")
        how_to_read(
            "One dot per item. Dots high and to the left give more protein per calorie; "
            "hover a dot to see the item."
        )

    with st.container(border=True):
        chart(viz.caffeine_status_chart(menu, "drink", include_unreliable), key="chart_caffeine")
        how_to_read("Counts of drinks per status, including drinks without nutrition data.")
        st.caption(CAFFEINE_PROXY_HELP)
