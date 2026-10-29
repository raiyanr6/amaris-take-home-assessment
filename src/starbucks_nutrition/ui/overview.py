"""Tab 1: headline numbers, drinks vs food, fat-to-protein and the caveats."""

from __future__ import annotations

import math

import streamlit as st

from starbucks_nutrition import analysis, viz
from starbucks_nutrition.ui.common import caveats_box, chart, source_caption
from starbucks_nutrition.ui.data_source import CATEGORIES, MenuData

_NUM = st.column_config.NumberColumn


def _kpi_tiles(data: MenuData, include_unreliable: bool) -> None:
    table = analysis.kpis(data.menu, include_unreliable)
    # One row per category keeps labels and "based on n items" readable at any width.
    for category, row in table.iterrows():
        name = CATEGORIES.get(str(category), str(category).capitalize())
        tiles = st.columns(3)
        tiles[0].metric(f"{name}: items", f"{int(row['items']):,}", border=True)
        # delta_arrow="off": these deltas are context ("91% of items"), not changes.
        tiles[1].metric(
            f"{name}: with nutrition data",
            f"{int(row['with_data']):,}",
            delta=f"{row['coverage_pct']:.0f}% of items",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        mean = row["mean_calories"]
        tiles[2].metric(
            f"{name}: mean calories",
            "n/a" if math.isnan(mean) else f"{mean:,.0f} kcal",
            delta=f"based on {int(row['n_calories'])} items",
            delta_color="off",
            delta_arrow="off",
            border=True,
            help="Mean over items with data; unreliable rows excluded unless included "
            "in the sidebar.",
        )
        st.caption(analysis.coverage_sentence(data.menu, str(category), include_unreliable))


def render(data: MenuData, include_unreliable: bool) -> None:
    menu = data.menu
    source_caption(data)
    _kpi_tiles(data, include_unreliable)

    chart(viz.compare_means_chart(menu, include_unreliable), key="overview_compare")
    caveats_box(analysis.comparison_caveats(menu, include_unreliable=include_unreliable))

    st.subheader("Drinks vs food, metric by metric")
    comparison = analysis.compare_categories(menu, include_unreliable=include_unreliable)
    comparison.index = [analysis.metric_label(m, with_unit=False) for m in comparison.index]
    config = {
        "unit": "Unit",
        "abs_diff": _NUM("Food − drink", format="%.2f"),
        "pct_diff": _NUM("Difference vs drink (%)", format="%.0f%%"),
    }
    for cat in ("drink", "food"):
        label = CATEGORIES[cat]
        config |= {
            f"n_total_{cat}": _NUM(f"{label}: items"),
            f"n_with_data_{cat}": _NUM(f"{label}: with value"),
            f"coverage_pct_{cat}": _NUM(f"{label}: coverage", format="%.0f%%"),
            f"n_analysed_{cat}": _NUM(f"{label}: n in mean"),
            f"mean_{cat}": _NUM(f"{label}: mean", format="%.2f"),
        }
    st.dataframe(comparison, column_config=config)
    st.caption(
        "Only metrics both categories have are compared. Coverage = share of all items "
        "with a value; the mean uses 'n in mean' items."
    )

    st.subheader("Fat-to-protein ratio, three views")
    ratios = analysis.ratio_summary(menu, include_unreliable=include_unreliable)
    ratios.index = [CATEGORIES.get(str(c), str(c)) for c in ratios.index]
    st.dataframe(
        ratios,
        column_config={
            "n_items_with_ratio": _NUM("Items with a ratio"),
            "mean_of_item_ratios": _NUM("Mean of item ratios", format="%.2f"),
            "median_of_item_ratios": _NUM("Median item ratio", format="%.2f"),
            "n_items_aggregate": _NUM("Items in aggregate"),
            "aggregate_ratio": _NUM("Aggregate (Σ fat / Σ protein)", format="%.2f"),
        },
    )
    st.caption(
        "Grams of fat per gram of protein. **Mean** averages each item's ratio (items "
        "with 0 g protein have no ratio and are skipped) and is pulled up by outliers. "
        "**Median** is the typical item. **Aggregate** divides total fat by total protein "
        "over all items with both values: 'for every gram of protein on the menu, how "
        "much fat'."
    )
