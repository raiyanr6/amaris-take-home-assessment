"""Descriptive statistics and comparisons over the cleaned menu table.

Pure functions: DataFrame in, DataFrame out; no I/O, no printing. These are the numbers
the LLM layer will later be given to narrate — it must never compute them itself.

Conventions:
- Only rows with nutrition data are analysed (all-missing rows would otherwise be NaN
  noise at best and, if filled, zeros that bias every average).
- Rows flagged unreliable (is_reliable == False) are excluded by default; pass
  include_unreliable=True to see them. coverage() reports how many were excluded.
- A metric absent for a category (e.g. food sodium) yields NaN/missing rows, never 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from starbucks_nutrition.config import (
    CATEGORY_COL,
    DERIVED_UNITS,
    ITEM_COL,
    NUTRIENT_COLUMNS,
    NUTRIENT_UNITS,
)

METRIC_UNITS: dict[str, str] = {**NUTRIENT_UNITS, **DERIVED_UNITS}
_STATS = ["count", "mean", "median", "min", "max", "std", "sum"]


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-item ratios. Division by zero yields NaN (not inf, not an error).

    - fat_to_protein: g fat per g protein; NaN when protein is 0 (45 drinks have 0 g).
    - protein_per_100kcal / fiber_per_100kcal: nutrient density; NaN when calories is 0.
    """
    protein = df["protein_g"].where(df["protein_g"] > 0)
    calories = df["calories"].where(df["calories"] > 0)
    return df.assign(
        fat_to_protein=df["fat_g"] / protein,
        protein_per_100kcal=df["protein_g"] / calories * 100,
        fiber_per_100kcal=df["fiber_g"] / calories * 100,
    )


def _analysable(df: pd.DataFrame, include_unreliable: bool) -> pd.DataFrame:
    """Rows that statistics should use."""
    mask = df["has_nutrition"]
    if not include_unreliable:
        mask &= df["is_reliable"]
    return df[mask]


def _with_derived(df: pd.DataFrame) -> pd.DataFrame:
    return df if "fat_to_protein" in df.columns else add_derived_metrics(df)


def _available_metrics(df: pd.DataFrame) -> list[str]:
    """Metrics (in canonical order) that have at least one value in ``df``."""
    return [m for m in METRIC_UNITS if m in df.columns and df[m].notna().any()]


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per category: total rows, rows with data, rows excluded as unreliable, rows analysed.

    Shown next to every statistic so readers know what the numbers are based on.
    """
    grouped = df.groupby(CATEGORY_COL, sort=True)
    has_data = df["has_nutrition"]
    unreliable = has_data & ~df["is_reliable"]
    return pd.DataFrame(
        {
            "rows": grouped.size(),
            "with_data": has_data.groupby(df[CATEGORY_COL]).sum(),
            "excluded_unreliable": unreliable.groupby(df[CATEGORY_COL]).sum(),
            "analysed": (has_data & ~unreliable).groupby(df[CATEGORY_COL]).sum(),
        }
    ).astype(int)


def describe(df: pd.DataFrame, include_unreliable: bool = False) -> pd.DataFrame:
    """Per-category summary statistics for every available nutrient and derived metric.

    Returns a tidy frame indexed by (category, metric) with columns count, mean, median,
    min, max, std, total (+ unit). ``count`` is the number of items with a value. ``total``
    is only meaningful for additive nutrients (e.g. total calories), so it is NaN for
    ratios. Metrics a category doesn't have (food sodium) are omitted for that category.
    """
    data = _with_derived(_analysable(df, include_unreliable))
    metrics = _available_metrics(data)
    if data.empty or not metrics:
        return pd.DataFrame(
            columns=["count", "mean", "median", "min", "max", "std", "total", "unit"],
            index=pd.MultiIndex.from_tuples([], names=[CATEGORY_COL, "metric"]),
        )

    stats = data.groupby(CATEGORY_COL)[metrics].agg(_STATS)
    tidy = stats.stack(level=0, future_stack=True)
    tidy.index.names = [CATEGORY_COL, "metric"]
    tidy = tidy.rename(columns={"sum": "total"})
    tidy = tidy[tidy["count"] > 0]  # drop metrics a category has no values for
    # Summing a ratio is meaningless; blank it rather than show a misleading number.
    is_ratio = tidy.index.get_level_values("metric").isin(list(DERIVED_UNITS))
    tidy.loc[is_ratio, "total"] = np.nan
    tidy["count"] = tidy["count"].astype(int)
    tidy["unit"] = tidy.index.get_level_values("metric").map(METRIC_UNITS)
    return tidy


def compare_categories(
    df: pd.DataFrame,
    a: str = "drink",
    b: str = "food",
    include_unreliable: bool = False,
) -> pd.DataFrame:
    """Compare mean values of category ``a`` vs ``b`` on metrics BOTH categories have.

    Only shared metrics are compared — comparing drink sodium to a food column that does
    not exist would be inventing data. Columns: unit, n_a, mean_a, n_b, mean_b,
    abs_diff (b - a) and pct_diff ((b - a) / a * 100; NaN when a's mean is 0).
    """
    data = _with_derived(_analysable(df, include_unreliable))
    left = data[data[CATEGORY_COL] == a]
    right = data[data[CATEGORY_COL] == b]
    shared = [m for m in _available_metrics(left) if m in _available_metrics(right)]

    out = pd.DataFrame(
        {
            "unit": [METRIC_UNITS[m] for m in shared],
            f"n_{a}": left[shared].notna().sum().to_numpy(),
            f"mean_{a}": left[shared].mean().to_numpy(),
            f"n_{b}": right[shared].notna().sum().to_numpy(),
            f"mean_{b}": right[shared].mean().to_numpy(),
        },
        index=pd.Index(shared, name="metric"),
    )
    out["abs_diff"] = out[f"mean_{b}"] - out[f"mean_{a}"]
    base = out[f"mean_{a}"].where(out[f"mean_{a}"] != 0)
    out["pct_diff"] = out["abs_diff"] / base * 100
    return out


def top_n(
    df: pd.DataFrame,
    metric: str,
    n: int = 5,
    category: str | None = None,
    ascending: bool = False,
    include_unreliable: bool = False,
) -> pd.DataFrame:
    """The ``n`` highest (or lowest, with ascending=True) items by ``metric``.

    Items with no value for the metric are excluded rather than ranked as 0. Unreliable
    rows are excluded by default so e.g. a drink wrongly reported as 0 kcal is not
    presented as the lowest-calorie drink.
    """
    data = _with_derived(_analysable(df, include_unreliable))
    if metric not in data.columns:
        raise KeyError(f"Unknown metric {metric!r}; choose from {list(METRIC_UNITS)}")
    if category is not None:
        data = data[data[CATEGORY_COL] == category]
    ranked = data[data[metric].notna()].sort_values(metric, ascending=ascending, kind="stable")
    return ranked[[CATEGORY_COL, ITEM_COL, metric]].head(n).reset_index(drop=True)


def availability(df: pd.DataFrame) -> pd.DataFrame:
    """Which metrics have data in each category.

    Tidy columns: category, metric, unit, n_values, available. Callers (and later the LLM
    prompt) use this to state that e.g. sugar, caffeine or food sodium are *unavailable*
    instead of inventing numbers. Caffeine has an extra name-based proxy (see filters),
    which is reported in `note`.
    """
    rows = []
    for category, group in df.groupby(CATEGORY_COL, sort=True):
        with_data = group[group["has_nutrition"]]
        for metric in NUTRIENT_COLUMNS:
            n_values = int(with_data[metric].notna().sum()) if metric in group else 0
            note = ""
            if metric == "caffeine_mg" and n_values == 0 and category == "drink":
                note = "no measurements; name-based proxy in caffeine_status"
            rows.append(
                {
                    CATEGORY_COL: category,
                    "metric": metric,
                    "unit": NUTRIENT_UNITS[metric],
                    "n_values": n_values,
                    "available": n_values > 0,
                    "note": note,
                }
            )
    return pd.DataFrame(
        rows, columns=[CATEGORY_COL, "metric", "unit", "n_values", "available", "note"]
    )
