"""End-to-end checks against the provided data/raw files (expected counts from profiling)."""

from __future__ import annotations

import hashlib

import pytest

from starbucks_nutrition import analysis, config
from starbucks_nutrition.__main__ import main
from starbucks_nutrition.filters import filter_range


def test_drinks_counts(real_menu) -> None:
    _, reports = real_menu
    drinks = reports["drink"]
    assert drinks.encoding_detected == "utf-8"
    assert drinks.rows_read == 177
    assert drinks.exact_duplicates_removed == 22
    assert len(drinks.conflicts_resolved) == 23  # 22 exact + Iced Coffee conflict
    assert drinks.rows_final == 154
    assert drinks.rows_without_data == 80
    assert drinks.columns_unavailable == ["sugar_g", "caffeine_mg"]


def test_food_counts(real_menu) -> None:
    _, reports = real_menu
    food = reports["food"]
    assert food.encoding_detected == "utf-16"
    assert (food.rows_read, food.rows_final, food.rows_without_data) == (113, 113, 0)
    assert food.exact_duplicates_removed == 0 and not food.conflicts_resolved
    assert food.columns_unavailable == ["sodium_mg", "sugar_g", "caffeine_mg"]


def test_one_row_per_item_per_category(real_menu) -> None:
    menu, _ = real_menu
    assert not menu.duplicated(["category", "item_key"]).any()
    assert len(menu) == 154 + 113


def test_iced_coffee_conflict_resolved_to_5_kcal(real_menu) -> None:
    menu, reports = real_menu
    iced = menu[(menu["category"] == "drink") & (menu["item"] == "Iced Coffee")]
    assert len(iced) == 1
    assert (iced["calories"].item(), iced["sodium_mg"].item()) == (5, 5)
    (conflict,) = [
        d for d in reports["drink"].conflicts_resolved if d["rule"] != "1_exact_duplicate"
    ]
    assert conflict["item"] == "Iced Coffee" and conflict["calories"] == 0


def test_narino_with_milk_is_unreliable_and_not_lowest(real_menu) -> None:
    menu, reports = real_menu
    assert [s["item"] for s in reports["drink"].suspicious] == ["Nariño 70 Cold Brew with Milk"]
    lowest = analysis.top_n(menu, "calories", 1, category="drink", ascending=True)
    assert lowest.loc[0, "item"] != "Nariño 70 Cold Brew with Milk"
    assert lowest.loc[0, "calories"] == 5


def test_names_cleaned(real_menu) -> None:
    menu, _ = real_menu
    items = set(menu["item"])
    assert "Ancho Chipotle Chicken Panini" in items
    assert "Spicy Chorizo Monterey Jack & Egg Breakfast Sandwich" in items
    assert "Starbucks® Bottled Cold Brew Black, Unsweetened" in items
    assert not menu["item"].str.contains(r"^\s|\s$|\s{2}", regex=True).any()


def test_caffeine_unknowns_are_the_unspecified_teas(real_menu) -> None:
    menu, _ = real_menu
    unknown = menu[(menu["category"] == "drink") & (menu["caffeine_status"] == "unknown")]
    assert len(unknown) == 8
    assert unknown["item"].str.contains("Tazo|Tea", regex=True).all()


@pytest.mark.parametrize("path", [config.DEFAULT_DRINKS_PATH, config.DEFAULT_FOOD_PATH])
def test_raw_files_are_not_modified(path) -> None:
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    main([])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_cli_runs(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Drinks vs food" in out and "NAME-BASED PROXY" in out


def test_cli_reports_bad_file(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("Code,Colour\nA1,red\n", encoding="utf-8")
    assert main(["--drinks", str(bad)]) == 1
    assert "item-name column" in capsys.readouterr().err


def test_food_under_500_strict_vs_inclusive(real_menu) -> None:
    menu, _ = real_menu
    food = menu[menu["category"] == "food"]
    assert len(filter_range(food, "calories", max_value=500, inclusive=False)) == 99
    assert len(filter_range(food, "calories", max_value=500)) == 103


def test_real_caveats_flag_understated_drink_mean(real_menu) -> None:
    menu, _ = real_menu
    text = "\n".join(analysis.comparison_caveats(menu))
    assert "Only 74 of 154 drink items (48%)" in text
    assert "Frappuccino 33" in text
    assert "drink calorie mean is likely understated" in text
