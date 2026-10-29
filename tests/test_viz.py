"""Chart functions: trace counts, colours, empty input and unreliable-row handling."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from starbucks_nutrition import viz

CHARTS = [
    viz.compare_means_chart,
    viz.top_n_chart,
    viz.macro_energy_chart,
    viz.calories_vs_protein_scatter,
    viz.caffeine_status_chart,
    viz.data_coverage_chart,
]


def _annotations(fig: go.Figure) -> list[str]:
    return [a.text for a in fig.layout.annotations]


@pytest.mark.parametrize("chart", CHARTS)
def test_every_chart_returns_a_figure(chart, small_menu) -> None:
    fig = chart(small_menu)
    assert isinstance(fig, go.Figure)
    assert fig.data and viz.NO_DATA_MESSAGE not in _annotations(fig)


@pytest.mark.parametrize("chart", CHARTS)
def test_empty_dataframe_gives_no_data_figure(chart, small_menu) -> None:
    for empty in (small_menu.iloc[:0], pd.DataFrame()):
        fig = chart(empty)
        assert isinstance(fig, go.Figure)
        assert viz.NO_DATA_MESSAGE in _annotations(fig)


def test_compare_means_two_categories_two_panels(small_menu) -> None:
    fig = viz.compare_means_chart(small_menu)
    assert len(fig.data) == 4  # (drink, food) x (kcal panel, gram panel)
    assert {t.marker.color for t in fig.data if t.legendgroup == "drink"} == {
        viz.CATEGORY_COLORS["drink"]
    }
    assert "Based on 4 of 6 items" in fig.data[0].hovertext[0]  # coverage on hover
    assert "5 of 6 drink items (83%) have nutrition data; 1 excluded as unreliable" in (
        fig.layout.title.text
    )


def test_top_n_order_and_unreliable_excluded(small_menu) -> None:
    fig = viz.top_n_chart(small_menu, "calories", n=10, category="drink", ascending=True)
    (bars,) = fig.data
    # First-ranked item at the top: plotly draws y in reverse, so it is the last element.
    assert list(bars.y)[-1] == "Espresso"
    assert "Mystery Cold Brew with Milk" not in bars.y
    with_unreliable = viz.top_n_chart(small_menu, "calories", 10, "drink", True, True)
    assert "Mystery Cold Brew with Milk" in with_unreliable.data[0].y


def test_macro_energy_three_macros_two_categories(small_menu) -> None:
    fig = viz.macro_energy_chart(small_menu)
    assert [t.name for t in fig.data] == ["Fat", "Carbs", "Protein"]
    assert all(len(t.x) == 2 for t in fig.data)
    assert fig.layout.barmode == "stack"


def test_scatter_one_trace_per_category_and_skips_unreliable(small_menu) -> None:
    fig = viz.calories_vs_protein_scatter(small_menu)
    assert [t.marker.color for t in fig.data] == [
        viz.CATEGORY_COLORS["drink"],
        viz.CATEGORY_COLORS["food"],
    ]
    drink_items = list(fig.data[0].text)
    assert "Mystery Cold Brew with Milk" not in drink_items  # unreliable
    assert "Pink Drink" not in drink_items  # no data
    included = viz.calories_vs_protein_scatter(small_menu, include_unreliable=True)
    assert "Mystery Cold Brew with Milk" in list(included.data[0].text)


def test_caffeine_chart_three_statuses_and_proxy_note(small_menu) -> None:
    fig = viz.caffeine_status_chart(small_menu)
    assert list(fig.data[0].x) == ["Likely", "Unlikely", "Unknown"]
    assert "proxy" in fig.layout.title.text


def test_data_coverage_with_and_without(small_menu) -> None:
    fig = viz.data_coverage_chart(small_menu)
    assert [t.name for t in fig.data] == ["With data", "Without data"]
    assert sum(fig.data[0].x) + sum(fig.data[1].x) == 6


def test_axis_titles_carry_units(small_menu) -> None:
    assert viz.top_n_chart(small_menu, "sodium_mg").layout.xaxis.title.text == "Sodium (mg)"
    scatter = viz.calories_vs_protein_scatter(small_menu)
    assert (scatter.layout.xaxis.title.text, scatter.layout.yaxis.title.text) == (
        "Calories (kcal)",
        "Protein (g)",
    )
