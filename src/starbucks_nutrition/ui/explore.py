"""Tab 2: filter the menu with FilterSpec and browse / download the matching items.

Slider rule: a slider left at its full range applies no filter (filters.active_ranges),
so the default view lists every item, including rows without nutrition data. Once a
range is narrowed, rows with no value for that nutrient are hidden and a caption says
so (filters.range_filter_notes).
"""

from __future__ import annotations

import math

import streamlit as st

from starbucks_nutrition import analysis, config
from starbucks_nutrition.filters import (
    FilterSpec,
    active_ranges,
    apply_filters,
    range_filter_notes,
)
from starbucks_nutrition.ui.common import CAFFEINE_PROXY_HELP, source_caption
from starbucks_nutrition.ui.data_source import CATEGORIES, MenuData

UNDER, UP_TO = "under (<)", "up to (≤)"
NO_MATCH_MESSAGE = "No items match these filters. Try widening a range or clearing the search."
# Coarse steps keep the sliders usable; grams get half-gram precision.
_STEP = {"kcal": 10.0, "mg": 5.0, "g": 0.5}


def _limits(low: float, high: float, step: float) -> tuple[float, float]:
    """Slider end points snapped outward to the step, so every value is reachable."""
    return math.floor(low / step) * step, math.ceil(high / step) * step


def _range_sliders(
    data: MenuData, categories: tuple[str, ...]
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Render one slider per nutrient; returns (selected, limits) for active_ranges."""
    bounds = analysis.metric_bounds(data.menu, categories)
    overall = analysis.metric_bounds(data.menu)
    selected: dict[str, tuple[float, float]] = {}
    limits: dict[str, tuple[float, float]] = {}
    cols = st.columns(3)
    # One slider per nutrient that exists anywhere in the menu; it is disabled (with a
    # caption) when the selected categories have no values, e.g. sodium for food only.
    shown = [m for m in overall.index if overall.loc[m, "n_values"]]
    for i, metric in enumerate(shown):
        row = bounds.loc[metric]
        label = analysis.metric_label(metric)
        with cols[i % 3]:
            if row["n_values"] == 0 or row["min"] == row["max"]:
                st.slider(label, 0.0, 1.0, (0.0, 1.0), disabled=True, key=f"off_{metric}")
                who = " and ".join(CATEGORIES[c].lower() for c in categories) or "items"
                st.caption(f"No {label.split(' (')[0].lower()} data for the selected {who}.")
                continue
            step = _STEP.get(str(row["unit"]), 1.0)
            low, high = _limits(float(row["min"]), float(row["max"]), step)
            # The key includes the categories so the slider resets when its limits change.
            selected[metric] = st.slider(
                label,
                min_value=low,
                max_value=high,
                value=(low, high),
                step=step,
                format="%.1f" if step < 1 else "%d",
                key=f"range_{metric}_{'-'.join(categories)}",
            )
            limits[metric] = (low, high)
    return selected, limits


def render(data: MenuData, include_unreliable: bool) -> None:
    # include_unreliable is unused on purpose: the table lists every row, flagged ones
    # included (Reliable column), so users can always find an item they searched for.
    del include_unreliable
    menu = data.menu
    source_caption(data)

    with st.expander("Filters", expanded=True):
        top = st.columns([2, 3, 2])
        categories = tuple(
            top[0].multiselect(
                "Category",
                list(CATEGORIES),
                default=list(CATEGORIES),
                format_func=CATEGORIES.get,
                key="filter_categories",
            )
        )
        text = top[1].text_input(
            "Search item names", key="filter_text", placeholder="e.g. latte, ombre, panini"
        )
        statuses = tuple(
            top[2].multiselect(
                "Caffeine status (proxy)",
                list(config.CAFFEINE_STATUSES),
                key="filter_caffeine",
                help=CAFFEINE_PROXY_HELP,
            )
        )
        options = st.columns([2, 3])
        upper = options[0].radio(
            "Upper bounds mean",
            [UNDER, UP_TO],
            horizontal=True,
            key="filter_upper",
            help="'Under 500' excludes items of exactly 500; 'up to 500' keeps them. "
            "Lower bounds always include the value (≥).",
        )
        with_data_only = options[1].checkbox(
            "Only items with nutrition data", key="filter_with_data"
        )
        selected, limits = _range_sliders(data, categories)

    ranges = active_ranges(selected, limits)
    spec = FilterSpec(
        categories=categories or tuple(CATEGORIES),
        text=text,
        ranges=ranges,
        inclusive=upper == UP_TO,
        lower_inclusive=True,
        caffeine_statuses=statuses,
        with_data_only=with_data_only,
    )
    scope = apply_filters(menu, FilterSpec(categories=spec.categories))
    for note in range_filter_notes(scope, ranges):
        st.caption(note)

    result = apply_filters(menu, spec)
    st.markdown(f"**{len(result)}** of {len(menu)} items match")
    if result.empty:
        st.info(NO_MATCH_MESSAGE)
        return

    table = analysis.display_table(result)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Caffeine status (proxy)": st.column_config.TextColumn(help=CAFFEINE_PROXY_HELP),
            "Quality flag": st.column_config.TextColumn(width="medium"),
        },
    )
    st.caption(
        "Click a column header to sort. *None* means no data (never 0). Caffeine "
        "status is a name-based proxy; unreliable rows are always listed here, marked "
        "Reliable = false."
    )
    st.download_button(
        "Download these rows as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="menu_filtered.csv",
        mime="text/csv",
        key="download_filtered",
    )
