"""build_facts: the compact, JSON-safe summary of computed numbers the model narrates.

Built ONLY from existing package functions, so every number has a tested origin. The
model gets this instead of raw CSVs: it is smaller (about 2-3k tokens), carries units,
coverage and caveats, and cannot be misread the way a raw table can.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from starbucks_nutrition import analysis
from starbucks_nutrition.config import CATEGORY_COL, DERIVED_UNITS, ITEM_COL, NUTRIENT_UNITS
from starbucks_nutrition.llm.safety import json_number, safe_text
from starbucks_nutrition.loader import QualityReport

FACTS_SCHEMA = "facts/v1"
RANKING_SIZE = 5
FOCUS_RANKING_SIZE = 10
# (key, metric, ascending). "highest_carb_g" doubles as the sugar proxy list.
RANKINGS: tuple[tuple[str, str, bool], ...] = (
    ("highest_calories", "calories", False),
    ("lowest_calories", "calories", True),
    ("highest_fat_g", "fat_g", False),
    ("highest_carb_g", "carb_g", False),
    ("highest_protein_g", "protein_g", False),
    ("highest_fiber_g", "fiber_g", False),
    ("highest_protein_per_100kcal", "protein_per_100kcal", False),
)
_STAT_FIELDS = ("count", "mean", "median", "min", "max", "total")
_RATIO_METRICS = frozenset(DERIVED_UNITS)


def _digits(metric: str) -> int:
    return 2 if metric in _RATIO_METRICS else 1


def _stats(menu: pd.DataFrame, include_unreliable: bool) -> dict[str, dict[str, Any]]:
    table = analysis.describe(menu, include_unreliable)
    out: dict[str, dict[str, Any]] = {}
    for (category, metric), row in table.iterrows():
        entry = {
            ("n" if f == "count" else f): json_number(row[f], _digits(metric)) for f in _STAT_FIELDS
        }
        if entry["total"] is None:
            entry.pop("total")  # ratios have no meaningful total
        out.setdefault(str(category), {})[str(metric)] = entry
    return out


def _rankings(
    menu: pd.DataFrame, include_unreliable: bool, expand: Iterable[str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    expanded = set(expand)
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for category in sorted(menu[CATEGORY_COL].unique()):
        lists: dict[str, list[dict[str, Any]]] = {}
        available = set(analysis.available_metrics(menu, category, include_unreliable))
        for key, metric, ascending in RANKINGS:
            if metric not in available:
                continue
            n = FOCUS_RANKING_SIZE if key in expanded else RANKING_SIZE
            ranked = analysis.top_n(menu, metric, n, category, ascending, include_unreliable)
            lists[key] = [
                {"item": safe_text(item), "value": json_number(value, _digits(metric))}
                for item, value in zip(ranked[ITEM_COL], ranked[metric], strict=True)
            ]
        out[str(category)] = lists
    return out


def _comparison(menu: pd.DataFrame, include_unreliable: bool) -> list[dict[str, Any]]:
    table = analysis.compare_categories(menu, include_unreliable=include_unreliable)
    rows = []
    for metric, row in table.iterrows():
        digits = _digits(str(metric))
        rows.append(
            {
                "metric": metric,
                "unit": row["unit"],
                "drink_mean": json_number(row["mean_drink"], digits),
                "drink_n": json_number(row["n_analysed_drink"], 0),
                "drink_coverage_pct": json_number(row["coverage_pct_drink"], 0),
                "food_mean": json_number(row["mean_food"], digits),
                "food_n": json_number(row["n_analysed_food"], 0),
                "food_coverage_pct": json_number(row["coverage_pct_food"], 0),
                "diff_food_minus_drink": json_number(row["abs_diff"], digits),
                "pct_diff_vs_drink": json_number(row["pct_diff"], 0),
            }
        )
    return rows


def _ratio(menu: pd.DataFrame, include_unreliable: bool) -> dict[str, dict[str, Any]]:
    table = analysis.ratio_summary(menu, include_unreliable=include_unreliable)
    return {
        str(cat): {
            "mean_of_item_ratios": json_number(row["mean_of_item_ratios"], 2),
            "median_of_item_ratios": json_number(row["median_of_item_ratios"], 2),
            "aggregate_ratio": json_number(row["aggregate_ratio"], 2),
            "n_items_with_ratio": json_number(row["n_items_with_ratio"], 0),
            "n_items_aggregate": json_number(row["n_items_aggregate"], 0),
        }
        for cat, row in table.iterrows()
    }


def _macro(menu: pd.DataFrame, include_unreliable: bool) -> dict[str, dict[str, Any]]:
    shares = analysis.macro_energy_share(menu, include_unreliable)
    out: dict[str, dict[str, Any]] = {}
    for row in shares.itertuples(index=False):
        entry = out.setdefault(str(row.category), {"n_items": int(row.n_items)})
        entry[analysis.metric_label(row.macro, with_unit=False).lower() + "_pct"] = json_number(
            row.pct, 0
        )
    return out


def _caffeine(menu: pd.DataFrame, include_unreliable: bool) -> dict[str, Any]:
    if not (menu[CATEGORY_COL] == "drink").any():
        return {}
    counts = analysis.caffeine_counts(menu, "drink", include_unreliable)
    entry: dict[str, Any] = dict(
        zip(counts["caffeine_status"], (int(n) for n in counts["n_items"]), strict=True)
    )
    entry["basis"] = counts["basis"].iloc[0] if len(counts) else ""
    return {"drink": entry}


def _availability(menu: pd.DataFrame) -> dict[str, Any]:
    avail = analysis.availability(menu)
    unavailable = {
        str(cat): list(group.loc[~group["available"], "metric"])
        for cat, group in avail.groupby(CATEGORY_COL)
    }
    # Also keyed by metric, in words: models misread "which category lacks what" from the
    # category-keyed lists alone (e.g. claimed drinks lack sodium).
    by_metric = {
        str(metric): ", ".join(
            f"{cat}: {'available' if ok else 'NOT available'}"
            for cat, ok in zip(group[CATEGORY_COL], group["available"], strict=True)
        )
        for metric, group in avail.groupby("metric", sort=False)
    }
    return {
        "unavailable": unavailable,
        "by_metric": by_metric,
        "notes": [
            "Sugar is not in the data; carbohydrates (carb_g, which include sugars) are the "
            "closest available proxy and must be labelled as such.",
            "Caffeine in mg is not in the data; caffeine_status_counts is an estimate from "
            "drink names (name-based proxy), not a measurement.",
        ],
    }


def _quality(reports: Mapping[str, QualityReport]) -> dict[str, dict[str, Any]]:
    return {
        category: {
            "rows_read": report.rows_read,
            "rows_final": report.rows_final,
            "exact_duplicates_removed": report.exact_duplicates_removed,
            "other_duplicates_resolved": len(report.conflicts_resolved)
            - report.exact_duplicates_removed,
            "rows_without_data": report.rows_without_data,
            "unreliable_items": [safe_text(s["item"]) for s in report.suspicious],
        }
        for category, report in reports.items()
    }


def build_facts(
    menu: pd.DataFrame,
    reports: Mapping[str, QualityReport],
    include_unreliable: bool = False,
    sources: Mapping[str, str] | None = None,
    expand_rankings: Iterable[str] = (),
) -> dict[str, Any]:
    """Compact, JSON-safe facts for the summariser (no NaN/inf: missing is null).

    ``expand_rankings`` lists ranking keys (see RANKINGS) to send with 10 items instead
    of 5, so a summary focus gets more depth without sending everything.
    """
    cov = analysis.coverage(menu)
    coverage = {
        str(cat): {
            "items": int(row["rows"]),
            "with_data": int(row["with_data"]),
            "coverage_pct": json_number(row["with_data"] / row["rows"] * 100, 0),
            "excluded_unreliable": 0 if include_unreliable else int(row["excluded_unreliable"]),
            "analysed": int(row["with_data"] if include_unreliable else row["analysed"]),
        }
        for cat, row in cov.iterrows()
    }
    return {
        "schema": FACTS_SCHEMA,
        "source": dict(sources or {}),
        "include_unreliable": include_unreliable,
        "units": {**NUTRIENT_UNITS, **{k: v for k, v in DERIVED_UNITS.items()}},
        "coverage": coverage,
        "stats": _stats(menu, include_unreliable),
        "comparison": _comparison(menu, include_unreliable),
        "fat_to_protein": _ratio(menu, include_unreliable),
        "macro_energy_pct": _macro(menu, include_unreliable),
        "rankings": _rankings(menu, include_unreliable, expand_rankings),
        "caffeine_status_counts": _caffeine(menu, include_unreliable),
        "availability": _availability(menu),
        "caveats": [
            safe_text(c, 600)
            for c in analysis.comparison_caveats(menu, include_unreliable=include_unreliable)
        ],
        "data_quality": _quality(reports),
    }


def item_names(*sources: Any) -> set[str]:
    """Every string under an "item"/"items"/"unreliable_items" key, at any depth.

    Grounding removes these names before extracting numbers ("Nariño 70 Cold Brew").
    """
    names: set[str] = set()

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                walk(v, str(k))
        elif isinstance(node, list | tuple):
            for v in node:
                walk(v, key)
        elif isinstance(node, str) and key in ("item", "items", "unreliable_items", "name"):
            names.add(node)

    for source in sources:
        walk(source)
    return names
