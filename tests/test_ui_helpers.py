"""Pure package helpers added for the UI: active_ranges, kpis, bounds, shares, counts."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import DRINK_HEADER, make_csv
from starbucks_nutrition import analysis
from starbucks_nutrition.filters import FilterSpec, active_ranges, apply_filters
from starbucks_nutrition.loader import SchemaError, load_menu


# --------------------------------------------------------------------------------------
# active_ranges + lower_inclusive: untouched sliders must not filter anything
# --------------------------------------------------------------------------------------
def test_active_ranges_drops_untouched_bounds() -> None:
    limits = {"calories": (0.0, 650.0), "sodium_mg": (0.0, 400.0)}
    selected = {"calories": (0.0, 500.0), "sodium_mg": (0.0, 400.0)}
    assert active_ranges(selected, limits) == {"calories": (None, 500.0)}
    assert active_ranges({"calories": (100.0, 650.0)}, limits) == {"calories": (100.0, None)}
    assert active_ranges(limits, limits) == {}


def test_default_ranges_keep_every_row(real_menu) -> None:
    menu, _ = real_menu
    limits = {
        m: (row["min"], row["max"])
        for m, row in analysis.metric_bounds(menu).iterrows()
        if row["n_values"]
    }
    spec = FilterSpec(ranges=active_ranges(limits, limits), inclusive=False)
    assert len(apply_filters(menu, spec)) == 267  # incl. food (no sodium) + no-data drinks


def test_lower_bound_stays_inclusive_when_upper_is_strict(small_menu) -> None:
    spec = FilterSpec(ranges={"calories": (90, 190)}, inclusive=False, lower_inclusive=True)
    items = set(apply_filters(small_menu, spec)["item"])
    assert "Starbucks Refreshers™ Strawberry Lemonade" in items  # 90 kept (>=)
    assert "Caffè Latte" not in items  # 190 excluded (<)


def test_food_under_500_via_sliders(real_menu) -> None:
    menu, _ = real_menu
    limits = {"calories": (0.0, 650.0)}
    spec = FilterSpec(
        categories=("food",),
        ranges=active_ranges({"calories": (0.0, 500.0)}, limits),
        inclusive=False,
        lower_inclusive=True,
    )
    assert len(apply_filters(menu, spec)) == 99


# --------------------------------------------------------------------------------------
# analysis helpers
# --------------------------------------------------------------------------------------
def test_kpis_small_menu(small_menu) -> None:
    k = analysis.kpis(small_menu)
    assert k.loc["drink", "items"] == 6 and k.loc["drink", "with_data"] == 5
    assert k.loc["drink", "n_calories"] == 4  # all-zero row excluded as unreliable
    assert k.loc["drink", "mean_calories"] == pytest.approx((190 + 140 + 90 + 10) / 4)
    assert k.loc["food", "coverage_pct"] == 100


def test_metric_bounds_per_category(small_menu) -> None:
    food = analysis.metric_bounds(small_menu, ("food",))
    assert (food.loc["calories", "min"], food.loc["calories", "max"]) == (170, 650)
    assert food.loc["sodium_mg", "n_values"] == 0 and pd.isna(food.loc["sodium_mg", "max"])
    assert analysis.metric_bounds(small_menu).loc["sodium_mg", "n_values"] == 5


def test_macro_energy_share_sums_to_100(small_menu) -> None:
    shares = analysis.macro_energy_share(small_menu)
    assert shares.groupby("category")["pct"].sum().round(6).eq(100).all()
    food_fat = shares[(shares["category"] == "food") & (shares["macro"] == "fat_g")]
    # fat 52 g * 9 = 468 kcal; carbs 119*4 = 476; protein 40*4 = 160 -> 468 / 1104
    assert food_fat["pct"].item() == pytest.approx(468 / 1104 * 100)
    assert set(shares["n_items"]) == {4, 3}


def test_macro_energy_share_empty() -> None:
    assert analysis.macro_energy_share(
        pd.DataFrame(columns=["category", "has_nutrition", "is_reliable", *analysis.MACROS])
    ).empty


def test_caffeine_counts_includes_rows_without_data(small_menu) -> None:
    counts = analysis.caffeine_counts(small_menu).set_index("caffeine_status")["n_items"]
    # Latte, Ombré Pink Drink, Refresher, Espresso, Pink Drink (no data) -> likely;
    # the all-zero Cold Brew is unreliable and excluded by default.
    assert counts.to_dict() == {"likely": 5, "unlikely": 0, "unknown": 0}
    with_unreliable = analysis.caffeine_counts(small_menu, include_unreliable=True)
    assert with_unreliable["n_items"].sum() == 6


def test_family_coverage_real(real_menu) -> None:
    menu, _ = real_menu
    fam = analysis.family_coverage(menu).set_index("family")
    assert fam.loc["Frappuccino", "without_data"] == 33
    assert fam["total"].sum() == 154 and fam["with_data"].sum() == 74
    assert fam.index[0] == "Frappuccino"  # sorted by items without data


def test_display_table_units_in_headers(small_menu) -> None:
    table = analysis.display_table(small_menu)
    assert {"Item", "Calories (kcal)", "Fat (g)", "Sodium (mg)", "Reliable"} <= set(table.columns)
    assert "Caffeine status (proxy)" in table.columns
    assert "Sugar (g)" not in table.columns  # no values anywhere -> hidden
    assert len(table) == len(small_menu)


def test_load_error_names_the_file() -> None:
    bad = make_csv("Code,Colour", "A1,red")
    with pytest.raises(SchemaError, match=r"^Food file: Could not find the item-name column"):
        load_menu(make_csv(DRINK_HEADER, "Latte,190,7,19,0,13,170"), bad)
