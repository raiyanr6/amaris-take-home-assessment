"""Reading, header mapping, value coercion and quality flags."""

from __future__ import annotations

import codecs
import io
import json
import math

import pandas as pd
import pytest

from conftest import DRINK_HEADER, FOOD_HEADER, clean_bytes, make_csv
from starbucks_nutrition.filters import search
from starbucks_nutrition.loader import (
    OUTPUT_COLUMNS,
    DataLoadError,
    SchemaError,
    read_csv_robust,
    validate_schema,
)

# --------------------------------------------------------------------------------------
# Encodings and source types
# --------------------------------------------------------------------------------------


def test_reads_utf8() -> None:
    df = read_csv_robust(make_csv(DRINK_HEADER, "Ombré Pink Drink,140,2.5,27,1,1,65"))
    assert df.attrs["encoding"] == "utf-8"
    assert df.iloc[0, 0] == "Ombré Pink Drink"


def test_reads_utf16_with_bom() -> None:
    data = make_csv(FOOD_HEADER, "Volpi™ Foldover,340,16,32,1,17", encoding="utf-16")
    assert data.startswith(codecs.BOM_UTF16_LE)
    df = read_csv_robust(data)
    assert df.attrs["encoding"] == "utf-16"
    assert df.iloc[0, 0] == "Volpi™ Foldover"


def test_reads_utf16_without_bom() -> None:
    data = make_csv(FOOD_HEADER, "Croissant,260,15,28,1,5", encoding="utf-16-le")
    df = read_csv_robust(data)
    assert df.attrs["encoding"] == "utf-16-le"
    assert df.shape == (1, 6)


def test_reads_utf8_sig_and_cp1252() -> None:
    sig = codecs.BOM_UTF8 + make_csv(DRINK_HEADER, "Espresso,10,0,2,0,1,0")
    assert read_csv_robust(sig).attrs["encoding"] == "utf-8-sig"
    # ™ is byte 0x99 in cp1252, which is invalid UTF-8 -> falls through to cp1252.
    win = make_csv(DRINK_HEADER, "Refreshers™,90,0,22,0,0,10", encoding="cp1252")
    df = read_csv_robust(win)
    assert df.attrs["encoding"] == "cp1252"
    assert df.iloc[0, 0] == "Refreshers™"


def test_accepts_path_bytes_and_file_like(tmp_path) -> None:
    data = make_csv(DRINK_HEADER, "Espresso,10,0,2,0,1,0")
    path = tmp_path / "drinks.csv"
    path.write_bytes(data)

    buffer = io.BytesIO(data)
    buffer.read()  # simulate a Streamlit upload that was already consumed
    sources = [path, str(path), data, bytearray(data), buffer, io.StringIO(data.decode())]
    for source in sources:
        df = read_csv_robust(source)
        assert df.iloc[0, 0] == "Espresso", type(source)


def test_empty_file_raises_schema_error() -> None:
    with pytest.raises(SchemaError, match="empty"):
        read_csv_robust(b"")


def test_malformed_csv_raises_data_load_error() -> None:
    with pytest.raises(DataLoadError):
        read_csv_robust(b'a,b\n"unterminated,1\n2,3,4,5\n')


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


def test_header_only_file_gives_empty_canonical_frame() -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER))
    assert df.empty
    assert list(df.columns) == OUTPUT_COLUMNS
    assert report.rows_read == 0 and report.rows_final == 0


def test_missing_item_column_raises() -> None:
    data = make_csv("Product Code,Calories", "A1,100")
    with pytest.raises(SchemaError, match=r"item-name column.*Product Code"):
        clean_bytes(data)


def test_no_nutrient_columns_raises() -> None:
    with pytest.raises(SchemaError, match="No nutrition columns"):
        clean_bytes(make_csv(",Colour,Size", "Latte,brown,tall"))


def test_validate_schema_accepts_canonical_frame() -> None:
    validate_schema(pd.DataFrame(columns=["item", "calories"]))


def test_unknown_extra_column_is_ignored_and_reported() -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER + ",Size", "Espresso,10,0,2,0,1,0,Solo"))
    assert "Size" in report.unknown_columns
    assert "Size" not in df.columns
    assert df.loc[0, "calories"] == 10


def test_named_item_column_and_reordered_columns() -> None:
    df, _ = clean_bytes(make_csv("Protein (g),Name,CALORIES", "13,Caffè Latte,190"))
    assert df.loc[0, "item"] == "Caffè Latte"
    assert df.loc[0, "calories"] == 190 and df.loc[0, "protein_g"] == 13


def test_header_variants_map_to_canonical_names() -> None:
    df, _ = clean_bytes(make_csv(FOOD_HEADER, "Croissant,260,15,28,1,5"), "food")
    row = df.iloc[0]
    assert (row["calories"], row["fat_g"], row["carb_g"], row["fiber_g"], row["protein_g"]) == (
        260,
        15,
        28,
        1,
        5,
    )


def test_food_without_sodium_gets_nan_column_and_reports_it() -> None:
    df, report = clean_bytes(make_csv(FOOD_HEADER, "Croissant,260,15,28,1,5"), "food")
    assert df["sodium_mg"].isna().all()
    assert "sodium_mg" in report.columns_unavailable
    assert (df["category"] == "food").all()


def test_optional_sugar_and_caffeine_columns_are_kept() -> None:
    data = make_csv(
        ",Calories,Fat (g),Carb. (g),Protein,Sugars (g),Caffeine (mg)",
        "Caffè Latte,190,7,19,13,17,150",
        "Vanilla Crème,200,8,25,10,24,0",
    )
    df, report = clean_bytes(data)
    assert df["sugar_g"].tolist() == [17, 24]
    assert df["caffeine_mg"].tolist() == [150, 0]
    assert "sugar_g" not in report.columns_unavailable
    assert "caffeine_mg" not in report.columns_unavailable


# --------------------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["-", "", "N/A", "na", "--", " - "])
def test_missing_tokens_become_nan_not_zero(token: str) -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER, f"Espresso,10,{token},2,0,1,0"))
    assert math.isnan(df.loc[0, "fat_g"])
    assert report.invalid_values == {}


def test_all_missing_rows_are_kept_and_flagged() -> None:
    df, report = clean_bytes(
        make_csv(DRINK_HEADER, "Pink Drink,-,-,-,-,-,-", "Espresso,10,0,2,0,1,0")
    )
    assert len(df) == 2
    pink = df[df["item"] == "Pink Drink"].iloc[0]
    assert not pink["has_nutrition"]
    assert pink[["calories", "fat_g", "sodium_mg"]].isna().all()
    assert report.rows_without_data == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5g", 5.0), ("5 g", 5.0), ("120 kcal", 120.0), ("2.5", 2.5), ("1,200", 1200.0), (".5", 0.5)],
)
def test_numbers_with_units_are_parsed(raw: str, expected: float) -> None:
    df, _ = clean_bytes(make_csv(DRINK_HEADER, f'Espresso,"{raw}",0,2,0,1,0'))
    assert df.loc[0, "calories"] == expected


@pytest.mark.parametrize(("raw", "expected"), [("1 200", 1200.0), ("12 345", 12345.0)])
def test_space_thousands_separator(raw: str, expected: float) -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER, f"Espresso,{raw},0,2,0,1,0"))
    assert df.loc[0, "calories"] == expected
    assert report.invalid_values == {}


@pytest.mark.parametrize("raw", ["<1", "< 1 g", "<1g", " <1 "])
def test_less_than_values_are_halved_and_counted_as_approximate(raw: str) -> None:
    df, report = clean_bytes(
        make_csv(DRINK_HEADER, f"Espresso,10,{raw},2,{raw},1,0", "Latte,190,7,19,0,13,170")
    )
    assert df.loc[0, "fat_g"] == 0.5 and df.loc[0, "fiber_g"] == 0.5
    assert report.approximate_values == {"fat_g": 1, "fiber_g": 1}
    assert report.invalid_values == {}


def test_unparseable_value_becomes_nan_and_is_counted() -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER, "Espresso,lots,0,2,0,1,0"))
    assert math.isnan(df.loc[0, "calories"])
    assert report.invalid_values == {"calories": 1}


def test_negative_value_becomes_nan_and_is_counted() -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER, "Espresso,10,-3,2,0,1,0"))
    assert math.isnan(df.loc[0, "fat_g"])
    assert df.loc[0, "calories"] == 10
    assert report.invalid_values == {"fat_g": 1}


def test_non_latin_names_are_not_collapsed_into_one_row() -> None:
    df, report = clean_bytes(
        make_csv(
            DRINK_HEADER,
            "抹茶ラテ,200,5,30,1,6,100",
            "ほうじ茶ラテ,180,4,28,0,6,90",
            "カフェモカ,300,10,40,2,8,120",
        )
    )
    assert list(df["item"]) == ["抹茶ラテ", "ほうじ茶ラテ", "カフェモカ"]
    assert df["item_key"].is_unique
    assert report.conflicts_resolved == []
    assert len(search(df, "抹茶")) == 1


def test_item_names_are_normalised_and_keyed() -> None:
    df, _ = clean_bytes(
        make_csv(
            DRINK_HEADER,
            "  Spicy Chorizo  Monterey Jack ,10,0,2,0,1,0",
            "Starbucks Refreshers™ Raspberry,90,0,22,0,0,10",
            '"Starbucks® Bottled Cold Brew Black, Unsweetened",10,0,2,0,1,0',
            "Ombré Pink Drink,140,2.5,27,1,1,65",
        )
    )
    assert df["item"].tolist() == [
        "Spicy Chorizo Monterey Jack",
        "Starbucks Refreshers™ Raspberry",  # display name keeps the symbol
        "Starbucks® Bottled Cold Brew Black, Unsweetened",
        "Ombré Pink Drink",
    ]
    assert df["item_key"].tolist() == [
        "spicy chorizo monterey jack",
        "starbucks refreshers raspberry",  # ™ dropped, not expanded to "tm"
        "starbucks bottled cold brew black, unsweetened",
        "ombre pink drink",
    ]


def test_rows_without_name_are_dropped_and_reported() -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER, ",10,0,2,0,1,0", "Espresso,10,0,2,0,1,0"))
    assert df["item"].tolist() == ["Espresso"]
    assert report.rows_without_name == 1


# --------------------------------------------------------------------------------------
# Quality flags
# --------------------------------------------------------------------------------------


def test_all_zero_row_kept_but_unreliable() -> None:
    df, report = clean_bytes(make_csv(DRINK_HEADER, "Cold Brew with Milk,0,0,0,0,0,0"))
    row = df.iloc[0]
    assert row["has_nutrition"] and not row["is_reliable"]
    assert row["quality_flag"] == "all nutrients reported as 0"
    assert report.suspicious[0]["item"] == "Cold Brew with Milk"


def test_zero_kcal_with_macros_is_flagged() -> None:
    df, _ = clean_bytes(make_csv(DRINK_HEADER, "Latte,0,7,19,0,13,170"))
    assert df.loc[0, "quality_flag"] == "0 kcal but non-zero macronutrients"
    assert not df.loc[0, "is_reliable"]


def test_energy_inconsistency_is_flagged_but_small_gaps_are_not() -> None:
    df, _ = clean_bytes(
        make_csv(
            DRINK_HEADER,
            "Wild,500,1,10,0,1,0",  # 4*10 + 4*1 + 9*1 = 53 kcal vs 500 reported
            "Rounded,200,7,19,0,13,170",  # estimate 191: normal label rounding
            "Tiny,5,0,0,0,0,5",  # estimate 0 but within the absolute tolerance
        )
    )
    flags = dict(zip(df["item"], df["quality_flag"], strict=True))
    assert flags["Wild"].startswith("calories (500 kcal) far from")
    assert flags["Rounded"] == "" and flags["Tiny"] == ""


def test_report_is_json_serialisable() -> None:
    _, report = clean_bytes(
        make_csv(DRINK_HEADER, "Pink Drink,-,-,-,-,-,-", "Pink Drink,-,-,-,-,-,-")
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["exact_duplicates_removed"] == 1
    assert payload["conflicts_resolved"][0]["calories"] is None
