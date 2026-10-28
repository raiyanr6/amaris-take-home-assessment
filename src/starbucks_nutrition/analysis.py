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

from starbucks_nutrition import config
from starbucks_nutrition.config import (
    CATEGORY_COL,
    DERIVED_UNITS,
    ITEM_COL,
    ITEM_KEY_COL,
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


def _shared_metrics(df: pd.DataFrame, a: str, b: str, include_unreliable: bool) -> list[str]:
    data = _with_derived(_analysable(df, include_unreliable))
    left_metrics = _available_metrics(data[data[CATEGORY_COL] == a])
    right_metrics = set(_available_metrics(data[data[CATEGORY_COL] == b]))
    return [m for m in left_metrics if m in right_metrics]


def compare_categories(
    df: pd.DataFrame,
    a: str = "drink",
    b: str = "food",
    include_unreliable: bool = False,
) -> pd.DataFrame:
    """Compare mean values of category ``a`` vs ``b`` on metrics BOTH categories have.

    Only shared metrics are compared — comparing drink sodium to a food column that does
    not exist would be inventing data. Per category ``c`` the columns are:

    - n_total_c: all items in the category;
    - n_with_data_c / coverage_pct_c: items with a value for the metric (and the share);
    - n_analysed_c / mean_c: items the mean is based on (unreliable rows excluded).

    plus abs_diff (b - a) and pct_diff ((b - a) / a * 100; NaN when a's mean is 0).
    Coverage is shown next to the means because a mean over 47% of the drinks is not a
    mean over all drinks; see comparison_caveats() for what that implies.
    """
    shared = _shared_metrics(df, a, b, include_unreliable)
    everything = _with_derived(df)
    analysed = _with_derived(_analysable(df, include_unreliable))

    columns: dict[str, object] = {"unit": [METRIC_UNITS[m] for m in shared]}
    for cat in (a, b):
        all_rows = everything[everything[CATEGORY_COL] == cat]
        used = analysed[analysed[CATEGORY_COL] == cat]
        n_total = len(all_rows)
        n_with_data = all_rows[shared].notna().sum().to_numpy()
        columns[f"n_total_{cat}"] = n_total
        columns[f"n_with_data_{cat}"] = n_with_data
        columns[f"coverage_pct_{cat}"] = n_with_data / n_total * 100 if n_total else np.nan
        columns[f"n_analysed_{cat}"] = used[shared].notna().sum().to_numpy()
        columns[f"mean_{cat}"] = used[shared].mean().to_numpy()

    out = pd.DataFrame(columns, index=pd.Index(shared, name="metric"))
    out["abs_diff"] = out[f"mean_{b}"] - out[f"mean_{a}"]
    base = out[f"mean_{a}"].where(out[f"mean_{a}"] != 0)
    out["pct_diff"] = out["abs_diff"] / base * 100
    return out


def ratio_summary(
    df: pd.DataFrame,
    numerator: str = "fat_g",
    denominator: str = "protein_g",
    include_unreliable: bool = False,
) -> pd.DataFrame:
    """Three views of a ratio per category, because they answer different questions.

    - mean_of_item_ratios: the average item's ratio (items with denominator 0 skipped).
      Sensitive to outliers such as a 6 g fat / 1 g protein drink.
    - median_of_item_ratios: the typical item's ratio; robust to those outliers.
    - aggregate_ratio: sum(numerator) / sum(denominator) over items with both values —
      "for every gram of protein on the menu, how much fat". Items with 0 g protein still
      contribute their fat, so nothing is silently dropped.
    """
    data = _analysable(df, include_unreliable)
    both = data[data[numerator].notna() & data[denominator].notna()]
    item_ratio = both[numerator] / both[denominator].where(both[denominator] > 0)
    grouped = both.assign(_ratio=item_ratio).groupby(CATEGORY_COL, sort=True)
    sums = grouped[[numerator, denominator]].sum()
    return pd.DataFrame(
        {
            "n_items_with_ratio": grouped["_ratio"].count(),
            "mean_of_item_ratios": grouped["_ratio"].mean(),
            "median_of_item_ratios": grouped["_ratio"].median(),
            "n_items_aggregate": grouped.size(),
            "aggregate_ratio": sums[numerator] / sums[denominator].where(sums[denominator] > 0),
        }
    )


def product_family(item_keys: pd.Series) -> pd.Series:
    """Label each item with its product family from config.PRODUCT_FAMILIES (first match)."""
    conditions = [
        item_keys.str.contains(pattern, regex=True) for _, pattern in config.PRODUCT_FAMILIES
    ]
    labels = [label for label, _ in config.PRODUCT_FAMILIES]
    return pd.Series(
        np.select(conditions, labels, default=config.PRODUCT_FAMILY_OTHER),
        index=item_keys.index,
    )


def _missing_data_caveats(df: pd.DataFrame, category: str, include_unreliable: bool) -> list[str]:
    """Explain how incomplete coverage in one category may bias its calorie mean."""
    group = df[df[CATEGORY_COL] == category]
    missing = group[~group["has_nutrition"]]
    if group.empty or missing.empty:
        return []

    n_total, n_missing = len(group), len(missing)
    missing_family = product_family(missing[ITEM_KEY_COL])
    breakdown = ", ".join(f"{fam} {n}" for fam, n in missing_family.value_counts().items())
    caveats = [
        f"Only {n_total - n_missing} of {n_total} {category} items "
        f"({(n_total - n_missing) / n_total:.0%}) have nutrition data, so {category} averages "
        f"describe those items only. Items without data by family: {breakdown}."
    ]

    # Rough bias estimate: stand in each missing item with the calorie mean of analysed
    # items from the same family (only families with enough analysed items).
    analysed = _analysable(group, include_unreliable)
    analysed = analysed[analysed["calories"].notna()]
    if analysed.empty:
        return caveats
    observed = analysed["calories"].mean()
    family_stats = (
        analysed["calories"].groupby(product_family(analysed[ITEM_KEY_COL])).agg(["mean", "count"])
    )
    usable = family_stats[family_stats["count"] >= config.MIN_FAMILY_SAMPLE]["mean"]
    stand_in = missing_family.map(usable)
    if stand_in.notna().any():
        estimate = (analysed["calories"].sum() + stand_in.sum()) / (
            len(analysed) + stand_in.notna().sum()
        )
        direction = "understated" if estimate > observed else "overstated"
        largest = missing_family.value_counts().index[0]
        detail = ""
        if largest in usable.index:
            detail = (
                f" {largest} items that do have data average {usable[largest]:.0f} kcal vs "
                f"{observed:.0f} kcal for all analysed {category} items."
            )
        skipped = sorted(set(missing_family[stand_in.isna()]))
        skipped_note = (
            f" {int(stand_in.isna().sum())} missing items ({', '.join(skipped)}) have too "
            f"few same-family items with data and are left out of this estimate."
            if skipped
            else ""
        )
        caveats.append(
            f"Missing {category} data is not random, so the {category} calorie mean is "
            f"likely {direction}.{detail} Rough estimate if missing items resembled their "
            f"family: ~{estimate:.0f} kcal instead of {observed:.0f} kcal.{skipped_note}"
        )
    return caveats


def comparison_caveats(
    df: pd.DataFrame,
    a: str = "drink",
    b: str = "food",
    include_unreliable: bool = False,
) -> list[str]:
    """Plain-language caveats to show next to compare_categories() output.

    Derived from the data (not hard-coded), so they stay correct for uploaded CSVs. Meant
    to be surfaced verbatim by the UI and passed to the LLM prompt.
    """
    caveats: list[str] = []
    for cat in (a, b):
        caveats += _missing_data_caveats(df, cat, include_unreliable)
        unreliable = df[(df[CATEGORY_COL] == cat) & df["has_nutrition"] & ~df["is_reliable"]]
        if not include_unreliable and not unreliable.empty:
            caveats.append(
                f"{len(unreliable)} {cat} item(s) excluded as unreliable: "
                f"{', '.join(unreliable[ITEM_COL])}."
            )

    shared = set(_shared_metrics(df, a, b, include_unreliable))
    present = {
        cat: set(NUTRIENT_COLUMNS) & set(_available_metrics(df[df[CATEGORY_COL] == cat]))
        for cat in (a, b)
    }
    for cat, other in ((a, b), (b, a)):
        only_here = [m for m in NUTRIENT_COLUMNS if m in present[cat] - present[other] - shared]
        if only_here:
            caveats.append(f"Not compared because only {cat} has data: {', '.join(only_here)}.")
    nowhere = [m for m in NUTRIENT_COLUMNS if m not in present[a] | present[b]]
    if nowhere:
        caveats.append(f"Not available in either dataset: {', '.join(nowhere)}.")
    return caveats


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
