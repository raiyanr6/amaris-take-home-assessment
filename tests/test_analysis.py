"""Statistics, derived metrics, comparisons, rankings and availability."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from conftest import DRINK_HEADER, make_csv
from starbucks_nutrition import analysis
from starbucks_nutrition.loader import load_menu


def test_derived_ratio_is_nan_when_protein_is_zero(small_menu: pd.DataFrame) -> None:
    derived = analysis.add_derived_metrics(small_menu).set_index("item")
    assert math.isnan(derived.loc["Starbucks Refreshers™ Strawberry Lemonade", "fat_to_protein"])
    assert derived.loc["Caffè Latte", "fat_to_protein"] == pytest.approx(7 / 13)
    assert not np.isinf(derived.select_dtypes("number")).any().any()


def test_per_100kcal_is_nan_when_calories_is_zero(small_menu: pd.DataFrame) -> None:
    derived = analysis.add_derived_metrics(small_menu).set_index("item")
    assert math.isnan(derived.loc["Mystery Cold Brew with Milk", "protein_per_100kcal"])
    assert derived.loc["Butter Croissant", "fiber_per_100kcal"] == pytest.approx(100 / 260)


def test_describe_uses_only_reliable_rows_with_data(small_menu: pd.DataFrame) -> None:
    stats = analysis.describe(small_menu)
    calories = stats.loc[("drink", "calories")]
    # 6 drinks: 1 without data, 1 all-zero (unreliable) -> 4 analysed.
    assert calories["count"] == 4
    assert calories["total"] == 190 + 140 + 90 + 10
    assert calories["mean"] == pytest.approx(430 / 4)
    assert calories["unit"] == "kcal"
    assert math.isnan(stats.loc[("drink", "fat_to_protein"), "total"])


def test_describe_can_include_unreliable_rows(small_menu: pd.DataFrame) -> None:
    stats = analysis.describe(small_menu, include_unreliable=True)
    assert stats.loc[("drink", "calories"), "count"] == 5
    assert stats.loc[("drink", "calories"), "min"] == 0


def test_describe_omits_metrics_a_category_lacks(small_menu: pd.DataFrame) -> None:
    stats = analysis.describe(small_menu)
    assert ("food", "sodium_mg") not in stats.index
    assert ("drink", "sodium_mg") in stats.index
    assert ("drink", "sugar_g") not in stats.index


def test_describe_on_empty_frame_returns_empty(small_menu: pd.DataFrame) -> None:
    assert analysis.describe(small_menu.iloc[0:0]).empty


def test_coverage_reports_excluded_rows(small_menu: pd.DataFrame) -> None:
    cov = analysis.coverage(small_menu)
    assert cov.loc["drink"].to_dict() == {
        "rows": 6,
        "with_data": 5,
        "excluded_unreliable": 1,
        "analysed": 4,
    }
    assert cov.loc["food", "analysed"] == 3


def test_compare_uses_shared_metrics_only(small_menu: pd.DataFrame) -> None:
    comp = analysis.compare_categories(small_menu)
    assert "sodium_mg" not in comp.index  # food has no sodium
    assert "sugar_g" not in comp.index
    row = comp.loc["calories"]
    assert row["mean_drink"] == pytest.approx(107.5)
    assert row["mean_food"] == pytest.approx(360.0)
    assert row["abs_diff"] == pytest.approx(252.5)
    assert row["pct_diff"] == pytest.approx(252.5 / 107.5 * 100)


def test_compare_pct_diff_nan_when_baseline_is_zero(small_menu: pd.DataFrame) -> None:
    menu = small_menu.copy()
    menu.loc[menu["category"] == "drink", "fiber_g"] = 0.0
    comp = analysis.compare_categories(menu)
    assert math.isnan(comp.loc["fiber_g", "pct_diff"])


def test_top_n_ascending_excludes_unreliable_by_default(small_menu: pd.DataFrame) -> None:
    lowest = analysis.top_n(small_menu, "calories", 1, category="drink", ascending=True)
    assert lowest.loc[0, "item"] == "Espresso"
    with_bad = analysis.top_n(
        small_menu, "calories", 1, category="drink", ascending=True, include_unreliable=True
    )
    assert with_bad.loc[0, "item"] == "Mystery Cold Brew with Milk"


def test_top_n_derived_metric_and_category_filter(small_menu: pd.DataFrame) -> None:
    top = analysis.top_n(small_menu, "protein_per_100kcal", 2, category="food")
    assert top["item"].tolist() == ["Egg Bites", "Lentil Bowl"]
    assert set(top["category"]) == {"food"}


def test_top_n_unknown_metric_raises(small_menu: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="Unknown metric"):
        analysis.top_n(small_menu, "vitamin_c", 3)


def test_availability_reports_missing_metrics(small_menu: pd.DataFrame) -> None:
    avail = analysis.availability(small_menu).set_index(["category", "metric"])
    assert not avail.loc[("food", "sodium_mg"), "available"]
    assert avail.loc[("drink", "sodium_mg"), "available"]
    for category in ("drink", "food"):
        assert not avail.loc[(category, "sugar_g"), "available"]
        assert not avail.loc[(category, "caffeine_mg"), "available"]
    assert "proxy" in avail.loc[("drink", "caffeine_mg"), "note"]
    assert avail.loc[("drink", "calories"), "n_values"] == 5


def test_compare_includes_coverage_columns(small_menu: pd.DataFrame) -> None:
    row = analysis.compare_categories(small_menu).loc["calories"]
    # 6 drinks: 5 have calories (incl. the unreliable all-zero row), 4 are analysed.
    assert (row["n_total_drink"], row["n_with_data_drink"], row["n_analysed_drink"]) == (6, 5, 4)
    assert row["coverage_pct_drink"] == pytest.approx(5 / 6 * 100)
    assert (row["n_total_food"], row["n_with_data_food"], row["coverage_pct_food"]) == (3, 3, 100)


def test_ratio_summary_mean_median_and_aggregate(small_menu: pd.DataFrame) -> None:
    summary = analysis.ratio_summary(small_menu)
    drink = summary.loc["drink"]
    # Analysed drinks (fat, protein): Latte (7, 13), Ombré (2.5, 1), Refreshers (0, 0),
    # Espresso (0, 1). Refreshers has 0 g protein: skipped for item ratios but its fat
    # still counts in the aggregate.
    assert drink["n_items_with_ratio"] == 3
    assert drink["mean_of_item_ratios"] == pytest.approx((7 / 13 + 2.5 + 0) / 3)
    assert drink["median_of_item_ratios"] == pytest.approx(7 / 13)
    assert drink["n_items_aggregate"] == 4
    assert drink["aggregate_ratio"] == pytest.approx(9.5 / 15)
    assert summary.loc["food", "aggregate_ratio"] == pytest.approx((15 + 8 + 29) / (5 + 12 + 23))


def test_ratio_summary_all_zero_denominator_is_nan() -> None:
    df = pd.DataFrame(
        {
            "category": ["drink", "drink"],
            "fat_g": [1.0, 2.0],
            "protein_g": [0.0, 0.0],
            "has_nutrition": [True, True],
            "is_reliable": [True, True],
        }
    )
    row = analysis.ratio_summary(df).loc["drink"]
    assert math.isnan(row["aggregate_ratio"]) and math.isnan(row["mean_of_item_ratios"])


def test_product_family_first_match_wins() -> None:
    keys = pd.Series(["starbucks bottled mocha frappuccino coffee drink", "espresso", "ginger ale"])
    assert analysis.product_family(keys).tolist() == ["Frappuccino", "Coffee & espresso", "Other"]


def test_caveats_on_small_menu(small_menu: pd.DataFrame) -> None:
    caveats = analysis.comparison_caveats(small_menu)
    text = "\n".join(caveats)
    assert "Only 5 of 6 drink items (83%) have nutrition data" in text
    assert "Refreshers 1" in text  # Pink Drink has no data
    assert "excluded as unreliable: Mystery Cold Brew with Milk" in text
    assert "only drink has data: sodium_mg" in text
    assert "either dataset: sugar_g, caffeine_mg" in text
    assert not any("food items" in c and "have nutrition data" in c for c in caveats)


def test_caveats_estimate_bias_from_family_means() -> None:
    drinks = make_csv(
        DRINK_HEADER,
        "Mocha Frappuccino,300,10,50,0,5,200",
        "Caramel Frappuccino,300,10,50,0,5,200",
        "Java Chip Frappuccino,300,10,50,0,5,200",
        "Espresso,10,0,2,0,1,0",
        "Americano,10,0,2,0,1,0",
        "Pike Place Roast,10,0,2,0,1,0",
        "Vanilla Frappuccino,-,-,-,-,-,-",
        "Coffee Frappuccino,-,-,-,-,-,-",
    )
    menu, _ = load_menu(drinks, None)
    (bias,) = [c for c in analysis.comparison_caveats(menu) if "not random" in c]
    # Observed mean 155; with the 2 missing Frappuccinos at 300: (930 + 600) / 8 = 191.25.
    assert "likely understated" in bias
    assert "~191 kcal instead of 155 kcal" in bias
    assert "Frappuccino items that do have data average 300 kcal" in bias
