"""Shared fixtures: tiny in-memory CSVs shaped like the real files, plus the real menu."""

from __future__ import annotations

import pandas as pd
import pytest

from starbucks_nutrition import config
from starbucks_nutrition.loader import clean, load_menu, read_csv_robust

# Same header quirks as the provided files: unnamed first column, drinks without units on
# Protein/Sodium, food with leading spaces and no sodium.
DRINK_HEADER = ",Calories,Fat (g),Carb. (g),Fiber (g),Protein,Sodium"
FOOD_HEADER = ", Calories, Fat (g), Carb. (g), Fiber (g), Protein (g)"


def make_csv(header: str, *rows: str, encoding: str = "utf-8") -> bytes:
    """Build CSV bytes from a header and row strings."""
    return ("\n".join([header, *rows]) + "\n").encode(encoding)


def clean_bytes(data: bytes, category: str = "drink"):
    """read_csv_robust + clean in one step; returns (df, report)."""
    return clean(read_csv_robust(data), category)


@pytest.fixture
def small_menu() -> pd.DataFrame:
    """A combined menu with known values for analysis/filter tests.

    Includes a zero-protein drink, a no-data drink, an all-zero (unreliable) drink, and
    accented / trademarked names.
    """
    drinks = make_csv(
        DRINK_HEADER,
        "Caffè Latte,190,7,19,0,13,170",
        "Ombré Pink Drink,140,2.5,27,1,1,65",
        "Starbucks Refreshers™ Strawberry Lemonade,90,0,22,0,0,10",  # protein 0
        "Espresso,10,0,2,0,1,0",
        "Pink Drink,-,-,-,-,-,-",  # no data
        "Mystery Cold Brew with Milk,0,0,0,0,0,0",  # all zeros -> unreliable
    )
    food = make_csv(
        FOOD_HEADER,
        "Butter Croissant,260,15,28,1,5",
        "Egg Bites,170,8,11,0,12",
        "Lentil Bowl,650,29,80,21,23",
        encoding="utf-16",
    )
    menu, _ = load_menu(drinks, food)
    return menu


@pytest.fixture(scope="session")
def real_menu():
    """The provided files, loaded once per test session: (menu, reports)."""
    return load_menu(config.DEFAULT_DRINKS_PATH, config.DEFAULT_FOOD_PATH)
