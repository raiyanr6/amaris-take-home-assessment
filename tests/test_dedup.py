"""Duplicate-handling rules 1-3 (see loader._deduplicate)."""

from __future__ import annotations

from conftest import DRINK_HEADER, FOOD_HEADER, clean_bytes, make_csv
from starbucks_nutrition.loader import load_menu


def test_rule1_exact_duplicates_keep_first() -> None:
    df, report = clean_bytes(
        make_csv(
            DRINK_HEADER,
            "Espresso,10,0,2,0,1,0",
            "Latte,190,7,19,0,13,170",
            "Espresso,10,0,2,0,1,0",
        )
    )
    assert df["item"].tolist() == ["Espresso", "Latte"]
    assert report.exact_duplicates_removed == 1
    (drop,) = report.conflicts_resolved
    assert drop["rule"] == "1_exact_duplicate"
    assert (drop["source_row"], drop["source_line"], drop["kept_line"]) == (2, 4, 2)


def test_rule1_compares_cleaned_values_and_names() -> None:
    # Same item after whitespace/case normalisation and unit stripping.
    df, report = clean_bytes(
        make_csv(DRINK_HEADER, "Espresso,10,0,2,0,1,0", "  espresso ,10g,0,2,0,1,0")
    )
    assert len(df) == 1 and report.exact_duplicates_removed == 1


def test_rule2_row_with_data_beats_all_missing_row_regardless_of_order() -> None:
    df, report = clean_bytes(
        make_csv(DRINK_HEADER, "Pink Drink,-,-,-,-,-,-", "Pink Drink,140,2.5,27,1,1,65")
    )
    assert len(df) == 1
    assert df.loc[0, "calories"] == 140
    (drop,) = report.conflicts_resolved
    assert drop["rule"] == "2_missing_vs_data"
    assert drop["source_line"] == 2 and drop["kept_line"] == 3
    assert "has nutrition data" in drop["reason"]


def test_rule3_more_non_null_values_wins() -> None:
    df, report = clean_bytes(
        make_csv(DRINK_HEADER, "Latte,190,-,-,-,-,-", "Latte,190,7,19,0,13,170")
    )
    assert df.loc[0, "fat_g"] == 7
    assert report.conflicts_resolved[0]["rule"] == "3_conflicting_values"
    assert "fewer non-null" in report.conflicts_resolved[0]["reason"]


def test_rule3_energy_consistent_row_wins_tie() -> None:
    df, report = clean_bytes(
        make_csv(
            DRINK_HEADER,
            "Latte,900,7,19,0,13,170",  # 900 kcal vs ~191 estimate -> inconsistent
            "Latte,190,7,19,0,13,170",
        )
    )
    assert df.loc[0, "calories"] == 190
    assert "energy check" in report.conflicts_resolved[0]["reason"]


def test_rule3_iced_coffee_all_zeros_loses_to_5_kcal_row() -> None:
    """Real case from drinks.csv: both rows pass the energy check; not-all-zeros decides."""
    df, report = clean_bytes(
        make_csv(DRINK_HEADER, "Iced Coffee,0,0,0,0,0,0", "Iced Coffee,5,0,0,0,0,5")
    )
    assert len(df) == 1
    assert (df.loc[0, "calories"], df.loc[0, "sodium_mg"]) == (5, 5)
    assert df.loc[0, "is_reliable"]
    (drop,) = report.conflicts_resolved
    assert drop["rule"] == "3_conflicting_values"
    assert drop["calories"] == 0
    assert "all values reported as 0" in drop["reason"]


def test_rule3_full_tie_keeps_first_occurrence() -> None:
    df, report = clean_bytes(
        make_csv(DRINK_HEADER, "Latte,190,7,19,0,13,170", "Latte,200,7,19,0,13,170")
    )
    assert df.loc[0, "calories"] == 190
    assert "kept first occurrence" in report.conflicts_resolved[0]["reason"]


def test_every_dropped_row_is_recorded() -> None:
    _, report = clean_bytes(
        make_csv(
            DRINK_HEADER,
            "A,1,0,0,0,0,0",
            "A,1,0,0,0,0,0",
            "B,-,-,-,-,-,-",
            "B,50,0,12,0,0,0",
            "C,5,0,0,0,0,5",
            "C,0,0,0,0,0,0",
        )
    )
    rules = sorted(d["rule"] for d in report.conflicts_resolved)
    assert rules == ["1_exact_duplicate", "2_missing_vs_data", "3_conflicting_values"]
    assert report.rows_read == 6 and report.rows_final == 3


def test_same_name_across_categories_is_not_deduplicated() -> None:
    drinks = make_csv(DRINK_HEADER, "Pumpkin Spice,380,14,52,0,14,240")
    food = make_csv(FOOD_HEADER, "Pumpkin Spice,410,15,64,2,6")
    menu, reports = load_menu(drinks, food)
    assert len(menu[menu["item"] == "Pumpkin Spice"]) == 2
    assert all(not r.conflicts_resolved for r in reports.values())
