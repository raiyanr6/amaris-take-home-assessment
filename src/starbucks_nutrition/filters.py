"""Row filters for the menu table, plus the caffeine classification.

All filters return a (possibly empty) DataFrame with the input's columns — never raise on
"no matches" — so a UI can render the result directly.

Caffeine: the provided datasets have no caffeine column, so `caffeine_status` falls back
to a NAME-BASED PROXY (rules in config.CAFFEINE_RULE_TIERS). It is a heuristic, not a
measurement, and has three outcomes — "likely", "unlikely", "unknown" — so callers (and
the LLM) can say "unclear from the name" rather than guess. If a CSV provides a real
caffeine_mg value for a row, that value is used instead (> 0 mg -> "likely").
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd

from starbucks_nutrition import config
from starbucks_nutrition.config import CATEGORY_COL, ITEM_KEY_COL
from starbucks_nutrition.textnorm import fold_key, fold_text

logger = logging.getLogger(__name__)

CAFFEINE_STATUS_COL = "caffeine_status"
CAFFEINE_BASIS_COL = "caffeine_basis"  # "measured" | "name_proxy" | "not_assessed"


# --------------------------------------------------------------------------------------
# Caffeine
# --------------------------------------------------------------------------------------
@cache
def _tier_regexes() -> tuple[tuple[str, str], ...]:
    """Compile each tier's keyword list into one word-bounded regex: (status, regex)."""
    return tuple(
        (status, r"\b(?:" + "|".join(patterns) + r")\b")
        for _name, status, patterns in config.CAFFEINE_RULE_TIERS
    )


def caffeine_status_from_name(item_keys: pd.Series) -> pd.Series:
    """Classify drinks by name only (vectorised). Expects folded keys (see fold_key).

    np.select picks the first matching tier, which is exactly the precedence documented
    in config (hard exclusions > tea override > soft exclusions > coffee words > tea).
    """
    conditions = [item_keys.str.contains(regex, regex=True) for _, regex in _tier_regexes()]
    choices = [status for status, _ in _tier_regexes()]
    return pd.Series(
        np.select(conditions, choices, default=config.CAFFEINE_DEFAULT_STATUS),
        index=item_keys.index,
    )


def classify_caffeine(item_name: str) -> str:
    """Name-based caffeine status for a single drink name: likely/unlikely/unknown."""
    return str(caffeine_status_from_name(pd.Series([fold_text(item_name)])).iloc[0])


def is_likely_caffeinated(item_name: str) -> bool:
    """True only when the name clearly indicates caffeine ("unknown" counts as False).

    PROXY: based on the drink name, because the dataset has no caffeine column.
    """
    return classify_caffeine(item_name) == "likely"


def add_caffeine_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add caffeine_status and caffeine_basis columns.

    Per row: a measured caffeine_mg wins; otherwise drinks use the name proxy; food is
    "unknown"/"not_assessed" because food names ("Coffee Cake") are not a caffeine signal.
    """
    measured = (
        df["caffeine_mg"].notna()
        if "caffeine_mg" in df.columns
        else pd.Series(False, index=df.index)
    )
    has_caffeine = measured & df.get("caffeine_mg", pd.Series(np.nan, index=df.index)).gt(0)
    is_drink = df[CATEGORY_COL].eq("drink")
    by_name = caffeine_status_from_name(df[ITEM_KEY_COL].fillna(""))

    status = np.select(
        [has_caffeine, measured, is_drink],
        ["likely", "unlikely", by_name],
        default=config.CAFFEINE_FOOD_STATUS,
    )
    basis = np.select([measured, is_drink], ["measured", "name_proxy"], default="not_assessed")
    return df.assign(**{CAFFEINE_STATUS_COL: status, CAFFEINE_BASIS_COL: basis})


def _ensure_caffeine(df: pd.DataFrame) -> pd.DataFrame:
    return df if CAFFEINE_STATUS_COL in df.columns else add_caffeine_status(df)


def filter_caffeinated(df: pd.DataFrame) -> pd.DataFrame:
    """Items whose caffeine status is "likely" (measured > 0 mg, or name proxy)."""
    df = _ensure_caffeine(df)
    return df[df[CAFFEINE_STATUS_COL].eq("likely")]


# --------------------------------------------------------------------------------------
# Generic filters
# --------------------------------------------------------------------------------------
def _range_mask(
    df: pd.DataFrame,
    column: str,
    min_value: float | None,
    max_value: float | None,
    inclusive: bool = True,
) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Unknown column {column!r}; available: {list(df.columns)}")
    values = df[column]
    # Rows with NaN never match a range: an item with unknown calories is not "under 500".
    mask = values.notna()
    if min_value is not None:
        mask &= values.ge(min_value) if inclusive else values.gt(min_value)
    if max_value is not None:
        mask &= values.le(max_value) if inclusive else values.lt(max_value)
    return mask


def filter_range(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
    inclusive: bool = True,
) -> pd.DataFrame:
    """Rows with ``df[column]`` between the bounds (either bound optional).

    ``inclusive=True`` keeps values equal to a bound (``<=``/``>=``); ``inclusive=False``
    is strict (``<``/``>``). The distinction matters on real data: four food items are
    exactly 500 kcal, so "under 500" (strict) gives 99 items and "up to 500" gives 103.

    Example: food under 500 kcal ->
    ``filter_range(food, "calories", max_value=500, inclusive=False)``.
    """
    return df[_range_mask(df, column, min_value, max_value, inclusive)]


def _search_mask(df: pd.DataFrame, text: str) -> pd.Series:
    needle = fold_text(text)
    if not needle:
        return pd.Series(True, index=df.index)
    keys = df[ITEM_KEY_COL] if ITEM_KEY_COL in df.columns else fold_key(df[config.ITEM_COL])
    return keys.str.contains(needle, regex=False)


def search(df: pd.DataFrame, text: str) -> pd.DataFrame:
    """Case-, accent- and ™/®-insensitive substring search on item names.

    "ombre" finds "Ombré Pink Drink"; "REFRESHERS" finds "Starbucks Refreshers™ ...".
    Plain substring (regex=False) so user input like "(2 Pack)" is not a regex.
    """
    return df[_search_mask(df, text)]


@dataclass
class FilterSpec:
    """A combination of criteria, all ANDed. Defaults mean "no restriction".

    Designed so a UI can bind widgets directly to fields, e.g.
    ``FilterSpec(categories=("food",), ranges={"calories": (None, 500)}, inclusive=False)``
    for "food under 500 kcal". ``inclusive`` applies to every bound in ``ranges``.
    """

    categories: tuple[str, ...] = ()
    text: str = ""
    ranges: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    inclusive: bool = True
    caffeinated_only: bool = False
    caffeine_statuses: tuple[str, ...] = ()
    with_data_only: bool = False
    reliable_only: bool = False


def apply_filters(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """Apply every criterion in ``spec`` in one boolean mask.

    Returns an empty frame with the same columns when nothing matches.
    """
    needs_caffeine = spec.caffeinated_only or spec.caffeine_statuses
    if needs_caffeine:
        df = _ensure_caffeine(df)

    mask = pd.Series(True, index=df.index)
    if spec.categories:
        mask &= df[CATEGORY_COL].isin([c.lower() for c in spec.categories])
    if spec.text:
        mask &= _search_mask(df, spec.text)
    for column, (low, high) in spec.ranges.items():
        mask &= _range_mask(df, column, low, high, spec.inclusive)
    if spec.caffeinated_only:
        mask &= df[CAFFEINE_STATUS_COL].eq("likely")
    if spec.caffeine_statuses:
        mask &= df[CAFFEINE_STATUS_COL].isin(spec.caffeine_statuses)
    if spec.with_data_only:
        mask &= df["has_nutrition"]
    if spec.reliable_only:
        mask &= df["is_reliable"]

    result = df[mask]
    logger.debug("apply_filters(%s) -> %d rows", spec, len(result))
    return result
