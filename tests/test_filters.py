"""Range/search filters, FilterSpec, and the caffeine classification."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_csv
from starbucks_nutrition.filters import (
    FilterSpec,
    apply_filters,
    classify_caffeine,
    filter_caffeinated,
    filter_range,
    is_likely_caffeinated,
    search,
)
from starbucks_nutrition.loader import load_menu

# --------------------------------------------------------------------------------------
# Range / search
# --------------------------------------------------------------------------------------


def test_filter_range_inclusive_and_skips_nan(small_menu: pd.DataFrame) -> None:
    food = small_menu[small_menu["category"] == "food"]
    under = filter_range(food, "calories", max_value=260)
    assert set(under["item"]) == {"Butter Croissant", "Egg Bites"}
    # Pink Drink has no calories: it must not count as "under 500".
    assert "Pink Drink" not in filter_range(small_menu, "calories", max_value=500)["item"].tolist()


def test_filter_range_returning_nothing_keeps_columns(small_menu: pd.DataFrame) -> None:
    result = filter_range(small_menu, "calories", min_value=10_000)
    assert result.empty
    assert list(result.columns) == list(small_menu.columns)


def test_filter_range_unknown_column_raises(small_menu: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        filter_range(small_menu, "vitamin_c", max_value=1)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ombre", "Ombré Pink Drink"),
        ("OMBRÉ", "Ombré Pink Drink"),
        ("caffe latte", "Caffè Latte"),
        ("refreshers strawberry", "Starbucks Refreshers™ Strawberry Lemonade"),
    ],
)
def test_search_is_case_and_accent_insensitive(
    small_menu: pd.DataFrame, query: str, expected: str
) -> None:
    assert search(small_menu, query)["item"].tolist() == [expected]


def test_search_with_no_match_and_regex_chars(small_menu: pd.DataFrame) -> None:
    assert search(small_menu, "unicorn").empty
    assert search(small_menu, "(2 pack").empty  # treated literally, no regex error
    assert len(search(small_menu, "  ")) == len(small_menu)  # blank search = no filter


# --------------------------------------------------------------------------------------
# Caffeine proxy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "status"),
    [
        # likely
        ("Espresso", "likely"),
        ("Caffè Americano", "likely"),
        ("Nitro Cold Brew", "likely"),
        ("Pike Place® Roast", "likely"),
        ("Cool Lime Starbucks Refreshers™ Beverage", "likely"),  # green-coffee extract
        ("Pink Drink", "likely"),
        ("Chai Crème Frappuccino® Blended Crème", "likely"),  # tea overrides crème
        ("Green Tea Crème Frappuccino® Blended Crème", "likely"),
        ("Teavana® Shaken Iced Black Tea", "likely"),
        ("Java Chip Frappuccino® Blended Coffee", "likely"),
        # unlikely
        ("Decaf Pike Place® Roast", "unlikely"),  # decaf beats "roast"
        ("Teavana® Passion Tango® Herbal Tea", "unlikely"),  # herbal beats "tea"
        ("Vanilla Bean Crème Frappuccino® Blended Crème", "unlikely"),
        ("Midnight Mint Mocha Frappuccino® Blended Crème", "unlikely"),  # crème beats mocha
        ("Hot Chocolate", "unlikely"),
        ("Strawberry Smoothie", "unlikely"),
        ("Evolution Fresh™ Defense Up", "unlikely"),
        ("Ginger Ale", "unlikely"),
        ("Orange Cream Soda", "unlikely"),
        # unknown: a tea whose base the name does not reveal
        ("Tazo® Bottled Giant Peach", "unknown"),
        ("Tazo® Bottled Brambleberry", "unknown"),
        ("Tazo® Bottled Berry Blossom White", "unknown"),
        ("Tazo® Bottled Plum Pomegranate", "unknown"),
    ],
)
def test_caffeine_proxy_three_states(name: str, status: str) -> None:
    assert classify_caffeine(name) == status
    assert is_likely_caffeinated(name) is (status == "likely")


def test_filter_caffeinated_returns_likely_only(small_menu: pd.DataFrame) -> None:
    caffeinated = filter_caffeinated(small_menu)
    assert set(caffeinated["caffeine_status"]) == {"likely"}
    assert "Caffè Latte" in caffeinated["item"].tolist()
    # Food is never assessed by name ("Coffee Cake" has no caffeine).
    food = small_menu[small_menu["category"] == "food"]
    assert set(food["caffeine_status"]) == {"unknown"}
    assert set(food["caffeine_basis"]) == {"not_assessed"}


def test_measured_caffeine_overrides_name_proxy() -> None:
    drinks = make_csv(
        ",Calories,Caffeine (mg)",
        "Herbal Wake-Up Tea,0,40",  # name says unlikely, measurement says caffeine
        "Espresso,10,0",  # name says likely, measurement says none
        "Tazo® Bottled Giant Peach,100,",  # no measurement -> name proxy
    )
    menu, _ = load_menu(drinks, None)
    status = dict(zip(menu["item"], menu["caffeine_status"], strict=True))
    basis = dict(zip(menu["item"], menu["caffeine_basis"], strict=True))
    assert status == {
        "Herbal Wake-Up Tea": "likely",
        "Espresso": "unlikely",
        "Tazo® Bottled Giant Peach": "unknown",
    }
    assert basis["Espresso"] == "measured" and basis["Tazo® Bottled Giant Peach"] == "name_proxy"
    assert filter_caffeinated(menu)["item"].tolist() == ["Herbal Wake-Up Tea"]


# --------------------------------------------------------------------------------------
# FilterSpec
# --------------------------------------------------------------------------------------


def test_apply_filters_combines_criteria(small_menu: pd.DataFrame) -> None:
    spec = FilterSpec(
        categories=("drink",), caffeinated_only=True, ranges={"calories": (None, 150)}
    )
    result = apply_filters(small_menu, spec)
    assert set(result["item"]) == {
        "Ombré Pink Drink",
        "Starbucks Refreshers™ Strawberry Lemonade",
        "Espresso",
        "Mystery Cold Brew with Milk",
    }
    reliable = apply_filters(small_menu, FilterSpec(**{**spec.__dict__, "reliable_only": True}))
    assert "Mystery Cold Brew with Milk" not in reliable["item"].tolist()


def test_apply_filters_food_under_500(small_menu: pd.DataFrame) -> None:
    spec = FilterSpec(categories=("FOOD",), ranges={"calories": (None, 500)})
    assert set(apply_filters(small_menu, spec)["item"]) == {"Butter Croissant", "Egg Bites"}


def test_apply_filters_status_and_data_flags(small_menu: pd.DataFrame) -> None:
    unknown = apply_filters(small_menu, FilterSpec(caffeine_statuses=("unknown",)))
    assert set(unknown["category"]) == {"food"}
    with_data = apply_filters(small_menu, FilterSpec(with_data_only=True))
    assert "Pink Drink" not in with_data["item"].tolist()


def test_apply_filters_empty_result_keeps_columns(small_menu: pd.DataFrame) -> None:
    result = apply_filters(small_menu, FilterSpec(text="unicorn", categories=("drink",)))
    assert result.empty
    assert list(result.columns) == list(small_menu.columns)


def test_apply_filters_default_spec_is_identity(small_menu: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(apply_filters(small_menu, FilterSpec()), small_menu)
