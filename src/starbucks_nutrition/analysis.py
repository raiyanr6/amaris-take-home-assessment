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
    ATWATER_KCAL_PER_G,
    CATEGORY_COL,
    DERIVED_UNITS,
    ITEM_COL,
    ITEM_KEY_COL,
    NUTRIENT_COLUMNS,
    NUTRIENT_UNITS,
)
from starbucks_nutrition.filters import CAFFEINE_STATUS_COL, add_caffeine_status

METRIC_UNITS: dict[str, str] = {**NUTRIENT_UNITS, **DERIVED_UNITS}
_STATS = ["count", "mean", "median", "min", "max", "std", "sum"]


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-item ratios. Division by zero yields NaN (not inf, not an error).

    - fat_to_protein: g fat per g protein; NaN when protein is 0 (31 drinks have 0 g, 30
      of them analysed: Nariño 70 Cold Brew with Milk is excluded as unreliable).
    - protein_per_100kcal / fiber_per_100kcal: nutrient density; NaN when calories is 0.
    """
    protein = df["protein_g"].where(df["protein_g"] > 0)
    calories = df["calories"].where(df["calories"] > 0)
    return df.assign(
        fat_to_protein=df["fat_g"] / protein,
        protein_per_100kcal=df["protein_g"] / calories * 100,
        fiber_per_100kcal=df["fiber_g"] / calories * 100,
    )


def analysable(df: pd.DataFrame, include_unreliable: bool = False) -> pd.DataFrame:
    """Rows that statistics and charts should use: with data, reliable unless asked.

    Public so charts apply exactly the same row rules as the statistics they sit next to.
    """
    mask = df["has_nutrition"]
    if not include_unreliable:
        mask &= df["is_reliable"]
    return df[mask]


def _with_derived(df: pd.DataFrame) -> pd.DataFrame:
    return df if "fat_to_protein" in df.columns else add_derived_metrics(df)


def _available_metrics(df: pd.DataFrame) -> list[str]:
    """Metrics (in canonical order) that have at least one value in ``df``."""
    return [m for m in METRIC_UNITS if m in df.columns and df[m].notna().any()]


def available_metrics(
    df: pd.DataFrame, category: str | None = None, include_unreliable: bool = False
) -> list[str]:
    """Nutrients and derived metrics with at least one analysable value (canonical order).

    Lets a UI offer only metrics that can actually be ranked, e.g. no sodium for food.
    """
    data = analysable(df, include_unreliable)
    if category is not None:
        data = data[data[CATEGORY_COL] == category]
    return _available_metrics(_with_derived(data))


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


def coverage_summary(
    df: pd.DataFrame, category: str, include_unreliable: bool = False
) -> dict[str, float]:
    """THE definition of "have data", used by every tile, subtitle, caveat and the facts.

    - coverage_pct = items with any nutrition data / all items (unreliable rows count as
      "with data": they have values, they are just not trusted);
    - excluded_unreliable / analysed say separately what the means are based on.
    Keeping one definition stops the UI from showing 91% in one place and 82% in another.
    """
    group = df[df[CATEGORY_COL] == category]
    items = len(group)
    with_data = int(group["has_nutrition"].sum())
    flagged = int((group["has_nutrition"] & ~group["is_reliable"]).sum())
    excluded = 0 if include_unreliable else flagged
    return {
        "items": items,
        "with_data": with_data,
        "coverage_pct": with_data / items * 100 if items else float("nan"),
        "flagged_unreliable": flagged,
        "excluded_unreliable": excluded,
        "analysed": with_data - excluded,
    }


def coverage_sentence(df: pd.DataFrame, category: str, include_unreliable: bool = False) -> str:
    """E.g. "10 of 11 drink items (91%) have nutrition data; 1 excluded as unreliable, so
    means use 9." Coverage and the unreliable exclusion are stated separately."""
    s = coverage_summary(df, category, include_unreliable)
    if not s["items"]:
        return f"There are no {category} items."
    text = (
        f"{s['with_data']} of {s['items']} {category} items ({s['coverage_pct']:.0f}%) "
        "have nutrition data"
    )
    if s["excluded_unreliable"]:
        return (
            f"{text}; {s['excluded_unreliable']} excluded as unreliable, so means use "
            f"{s['analysed']}."
        )
    if s["flagged_unreliable"]:
        return (
            f"{text}; {s['flagged_unreliable']} flagged as unreliable but included, so means "
            f"use {s['analysed']}."
        )
    return f"{text}."


def describe(df: pd.DataFrame, include_unreliable: bool = False) -> pd.DataFrame:
    """Per-category summary statistics for every available nutrient and derived metric.

    Returns a tidy frame indexed by (category, metric) with columns count, mean, median,
    min, max, std, total (+ unit). ``count`` is the number of items with a value. ``total``
    is only meaningful for additive nutrients (e.g. total calories), so it is NaN for
    ratios. Metrics a category doesn't have (food sodium) are omitted for that category.
    """
    data = _with_derived(analysable(df, include_unreliable))
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
    data = _with_derived(analysable(df, include_unreliable))
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
    analysed = _with_derived(analysable(df, include_unreliable))

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
    data = analysable(df, include_unreliable)
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


def missing_data_caveats(
    df: pd.DataFrame, category: str = "drink", include_unreliable: bool = False
) -> list[str]:
    """Explain how incomplete coverage in one category may bias its calorie mean."""
    group = df[df[CATEGORY_COL] == category]
    missing = group[~group["has_nutrition"]]
    if group.empty or missing.empty:
        return []

    summary = coverage_summary(df, category, include_unreliable)
    missing_family = product_family(missing[ITEM_KEY_COL])
    breakdown = ", ".join(f"{fam} {n}" for fam, n in missing_family.value_counts().items())
    caveats = [
        f"Only {summary['with_data']} of {summary['items']} {category} items "
        f"({summary['coverage_pct']:.0f}%) have nutrition data, so {category} averages "
        f"describe those items only. Items without data by family: {breakdown}."
    ]

    # Rough bias estimate: stand in each missing item with the calorie mean of analysed
    # items from the same family (only families with enough analysed items).
    analysed = analysable(group, include_unreliable)
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
        caveats += missing_data_caveats(df, cat, include_unreliable)
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
    data = _with_derived(analysable(df, include_unreliable))
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


# --------------------------------------------------------------------------------------
# Helpers for the UI and charts (pure; the UI must not compute numbers itself)
# --------------------------------------------------------------------------------------
METRIC_LABELS: dict[str, str] = {
    **config.NUTRIENT_LABELS,
    "fat_to_protein": "Fat-to-protein",
    "protein_per_100kcal": "Protein density",
    "fiber_per_100kcal": "Fiber density",
}


def metric_label(metric: str, with_unit: bool = True) -> str:
    """Human label for a metric, e.g. "Fat (g)": used for axes and table headers."""
    label = METRIC_LABELS.get(metric, metric)
    unit = METRIC_UNITS.get(metric)
    return f"{label} ({unit})" if with_unit and unit else label


def availability_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """availability() as a metric x category table of value counts (0 = not in the data)."""
    avail = availability(df)
    matrix = avail.pivot(index="metric", columns=CATEGORY_COL, values="n_values")
    matrix = matrix.reindex(list(NUTRIENT_COLUMNS))
    matrix.insert(0, "unit", [NUTRIENT_UNITS[m] for m in matrix.index])
    matrix.columns.name = None
    return matrix


def kpis(df: pd.DataFrame, include_unreliable: bool = False) -> pd.DataFrame:
    """Headline numbers per category: coverage (see coverage_summary) and mean calories.

    The mean is over analysed rows with a calorie value (n_calories), so it is shown with
    its coverage and with how many unreliable rows were excluded.
    """
    data = analysable(df, include_unreliable)
    grouped = data.groupby(CATEGORY_COL)["calories"]
    categories = sorted(df[CATEGORY_COL].unique())
    table = pd.DataFrame(
        [coverage_summary(df, cat, include_unreliable) for cat in categories],
        index=pd.Index(categories, name=CATEGORY_COL),
    )
    table["n_calories"] = grouped.count().reindex(table.index, fill_value=0).astype(int)
    table["mean_calories"] = grouped.mean().reindex(table.index)
    return table


def metric_bounds(df: pd.DataFrame, categories: tuple[str, ...] = ()) -> pd.DataFrame:
    """Min/max and value count per nutrient over the selected categories (all if empty).

    Drives the UI sliders: a nutrient with n_values == 0 (e.g. sodium when only food is
    selected) has NaN bounds and its slider should be disabled.
    """
    data = df[df[CATEGORY_COL].isin(categories)] if categories else df
    present = [m for m in NUTRIENT_COLUMNS if m in data.columns]
    return pd.DataFrame(
        {
            "min": data[present].min(),
            "max": data[present].max(),
            "n_values": data[present].notna().sum().astype(int),
            "unit": [NUTRIENT_UNITS[m] for m in present],
        }
    ).rename_axis("metric")


MACROS: tuple[str, ...] = ("fat_g", "carb_g", "protein_g")


def macro_energy_share(df: pd.DataFrame, include_unreliable: bool = False) -> pd.DataFrame:
    """Share of energy from fat / carbs / protein per category, using Atwater 9/4/4.

    Summed over items that report all three macros, so each category's shares add to
    100% of *macro-derived* energy. That differs from the label calorie total (rounding,
    fiber), which is why shares are relative to the Atwater sum, not to calories.
    Tidy output: category, macro, kcal, pct, n_items.
    """
    columns = [CATEGORY_COL, "macro", "kcal", "pct", "n_items"]
    data = analysable(df, include_unreliable)
    data = data[data[list(MACROS)].notna().all(axis=1)]
    if data.empty:
        return pd.DataFrame(columns=columns)
    factors = pd.Series({m: ATWATER_KCAL_PER_G[m] for m in MACROS})
    kcal = data[list(MACROS)].mul(factors).groupby(data[CATEGORY_COL]).sum()
    totals = kcal.sum(axis=1)
    pct = kcal.div(totals.where(totals > 0), axis=0) * 100
    tidy = pd.DataFrame({"kcal": kcal.stack(), "pct": pct.stack()})
    tidy.index.names = [CATEGORY_COL, "macro"]
    tidy = tidy.reset_index()
    tidy["n_items"] = tidy[CATEGORY_COL].map(data.groupby(CATEGORY_COL).size()).astype(int)
    return tidy[columns]


def caffeine_counts(
    df: pd.DataFrame, category: str = "drink", include_unreliable: bool = False
) -> pd.DataFrame:
    """Items per caffeine status (likely / unlikely / unknown) for one category.

    Rows WITHOUT nutrition data are counted: the status comes from the name (or a
    measured caffeine_mg), not from the nutrients. Only rows flagged unreliable are
    excluded by default, consistent with every other chart. Columns: caffeine_status,
    n_items, basis (how the status was decided, e.g. "name_proxy").
    """
    data = df if CAFFEINE_STATUS_COL in df.columns else add_caffeine_status(df)
    data = data[data[CATEGORY_COL] == category]
    if not include_unreliable:
        data = data[data["is_reliable"]]
    counts = (
        data[CAFFEINE_STATUS_COL]
        .value_counts()
        .reindex(list(config.CAFFEINE_STATUSES), fill_value=0)
        .astype(int)
    )
    basis = ", ".join(sorted(data["caffeine_basis"].unique()))
    return pd.DataFrame(
        {CAFFEINE_STATUS_COL: counts.index, "n_items": counts.to_numpy(), "basis": basis}
    )


def family_coverage(df: pd.DataFrame, category: str = "drink") -> pd.DataFrame:
    """Per product family: items with nutrition data vs without (all rows counted).

    Makes the non-random missing data visible: a family with many items and no data
    (Frappuccino) biases the category mean. Sorted by items without data, descending.
    """
    columns = ["family", "with_data", "without_data", "total", "coverage_pct"]
    group = df[df[CATEGORY_COL] == category]
    if group.empty:
        return pd.DataFrame(columns=columns)
    family = product_family(group[ITEM_KEY_COL])
    counts = group["has_nutrition"].groupby(family).agg(["sum", "size"])
    out = pd.DataFrame(
        {
            "family": counts.index,
            "with_data": counts["sum"].astype(int).to_numpy(),
            "total": counts["size"].astype(int).to_numpy(),
        }
    )
    out["without_data"] = out["total"] - out["with_data"]
    out["coverage_pct"] = out["with_data"] / out["total"] * 100
    out = out.sort_values(["without_data", "total"], ascending=False, kind="stable")
    return out[columns].reset_index(drop=True)


DISPLAY_COLUMNS: tuple[str, ...] = (
    ITEM_COL,
    CATEGORY_COL,
    *NUTRIENT_COLUMNS,
    "caffeine_status",
    "has_nutrition",
    "is_reliable",
    "quality_flag",
)


def display_table(df: pd.DataFrame, drop_empty: bool = True) -> pd.DataFrame:
    """Columns for showing items to people, with units in the headers ("Fat (g)").

    ``drop_empty`` hides nutrient columns with no values at all (sugar/caffeine in the
    provided files); availability() and the caveats already report them as missing.
    """
    columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
    if drop_empty:
        columns = [c for c in columns if c not in NUTRIENT_COLUMNS or df[c].notna().any()]
    names = {
        ITEM_COL: "Item",
        CATEGORY_COL: "Category",
        "caffeine_status": "Caffeine status (proxy)",
        "has_nutrition": "Has nutrition data",
        "is_reliable": "Reliable",
        "quality_flag": "Quality flag",
        **{m: metric_label(m) for m in NUTRIENT_COLUMNS},
    }
    return df[columns].rename(columns=names).reset_index(drop=True)
