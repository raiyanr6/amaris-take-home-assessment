"""Read-only tools the Q&A model can call, plus a validating executor.

Each tool is a thin wrapper over a tested analysis/filters function. Arguments are
validated against whitelists; anything invalid comes back to the model as an error object
(with the allowed values) instead of raising, so the model can correct itself. Results
are capped (rows, columns) and item names are sanitised: tool results are data the model
reads, and uploaded files are untrusted.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from starbucks_nutrition import analysis, config
from starbucks_nutrition.config import CATEGORY_COL, ITEM_COL, NUTRIENT_COLUMNS
from starbucks_nutrition.filters import (
    FilterSpec,
    apply_filters,
    range_filter_notes,
    search,
)
from starbucks_nutrition.llm.safety import json_number, safe_text

CATEGORIES = ("drink", "food")
METRICS = tuple(analysis.METRIC_UNITS)  # nutrients + derived ratios
MAX_N = 20
MAX_ROWS = 25
MAX_SEARCH_CHARS = 80

logger = logging.getLogger(__name__)


class ToolArgumentError(ValueError):
    """Invalid tool argument; turned into an error object for the model."""


def _is_number(value: Any) -> bool:
    """A finite int/float. Python's json accepts NaN/Infinity, so finiteness is checked."""
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _schema(name: str, description: str, properties: dict[str, Any], required=()) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
            },
        },
    }


_METRIC = {
    "type": "string",
    "enum": list(METRICS),
    "description": "Metric name. Units: "
    + ", ".join(f"{m}={u}" for m, u in analysis.METRIC_UNITS.items()),
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "get_stats",
        "Summary statistics (n, mean, median, min, max, total) of one metric for one "
        "category, with coverage (how many items have a value).",
        {"category": {"type": "string", "enum": list(CATEGORIES)}, "metric": _METRIC},
        ("category", "metric"),
    ),
    _schema(
        "top_items",
        "The items with the highest or lowest value of a metric. Items without a value "
        "are never ranked.",
        {
            "metric": _METRIC,
            "category": {"type": "string", "enum": [*CATEGORIES, "any"]},
            "n": {"type": "integer", "minimum": 1, "maximum": MAX_N},
            "order": {"type": "string", "enum": ["highest", "lowest"]},
        },
        ("metric",),
    ),
    _schema(
        "filter_items",
        "Items matching nutrient ranges, caffeine status and/or a name search. Returns "
        "total_matches and up to `limit` rows.",
        {
            "category": {"type": "string", "enum": [*CATEGORIES, "any"]},
            "ranges": {
                "type": "object",
                "description": 'Nutrient ranges, e.g. {"calories": {"max": 500}, '
                '"protein_g": {"min": 15}}. Keys: ' + ", ".join(NUTRIENT_COLUMNS),
                "additionalProperties": {
                    "type": "object",
                    "properties": {"min": {"type": "number"}, "max": {"type": "number"}},
                },
            },
            "upper_inclusive": {
                "type": "boolean",
                "description": "How to treat each max: under / below / less than -> false; "
                "up to / at most / or less -> true. Default false. Minimums are always "
                "inclusive (>=).",
            },
            "caffeine_status": {
                "type": "string",
                "enum": list(config.CAFFEINE_STATUSES),
                "description": "Name-based caffeine estimate (drinks only; food is unknown).",
            },
            "search": {"type": "string", "description": "Substring of the item name."},
            "sort_by": {
                **_METRIC,
                "description": "Metric to sort by, e.g. protein_g for 'high in protein'. "
                "Default: the first range metric, else calories.",
            },
            "order": {"type": "string", "enum": ["highest", "lowest"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS},
        },
    ),
    _schema(
        "get_item",
        "Look up items by (part of) their name; returns all nutrients for up to 5 matches.",
        {"name": {"type": "string"}},
        ("name",),
    ),
    _schema(
        "compare_drinks_food",
        "Mean of each shared metric for drinks vs food, with coverage, the fat-to-protein "
        "ratio three ways, and the caveats that must accompany the comparison.",
        {},
    ),
    _schema(
        "data_availability",
        "Which metrics exist for each category, and the proxies for missing ones (sugar, "
        "caffeine). Call this before answering about sugar or caffeine.",
        {},
    ),
]


@dataclass
class ToolExecutor:
    """Runs tools against the currently loaded menu with the UI's reliability setting."""

    menu: pd.DataFrame
    include_unreliable: bool = False

    # ---------------------------------------------------------------- validation
    @staticmethod
    def _choice(value: Any, allowed: tuple[str, ...], name: str, default: str | None = None) -> str:
        if value is None and default is not None:
            return default
        if value not in allowed:
            raise ToolArgumentError(f"Invalid {name} {value!r}; allowed: {list(allowed)}")
        return str(value)

    @staticmethod
    def _int(value: Any, name: str, default: int, maximum: int) -> int:
        if value is None:
            return default
        if not _is_number(value) or value != int(value):
            raise ToolArgumentError(f"{name} must be an integer between 1 and {maximum}")
        if not 1 <= int(value) <= maximum:
            raise ToolArgumentError(f"{name} must be between 1 and {maximum}")
        return int(value)

    def _category(self, value: Any, allow_any: bool = False) -> str | None:
        allowed = (*CATEGORIES, "any") if allow_any else CATEGORIES
        category = self._choice(value, allowed, "category", "any" if allow_any else None)
        return None if category == "any" else category

    # -------------------------------------------------------------------- helpers
    def _rows(self, df: pd.DataFrame, metrics: list[str]) -> list[dict[str, Any]]:
        rows = []
        for record in df[[ITEM_COL, CATEGORY_COL, *metrics]].head(MAX_ROWS).to_dict("records"):
            row: dict[str, Any] = {
                "item": safe_text(record[ITEM_COL]),
                "category": record[CATEGORY_COL],
            }
            for metric in metrics:
                value = record[metric]
                row[metric] = value if isinstance(value, str) else json_number(value, 2)
            rows.append(row)
        return rows

    def _coverage(self, category: str, metric: str) -> dict[str, Any]:
        group = self.menu[self.menu[CATEGORY_COL] == category]
        with_value = analysis.analysable(group, include_unreliable=True)
        n_value = int(with_value[metric].notna().sum()) if metric in with_value else 0
        return {
            "items_in_category": len(group),
            "items_with_value": n_value,
            "coverage_pct": json_number(n_value / len(group) * 100, 0) if len(group) else None,
        }

    # ---------------------------------------------------------------------- tools
    def get_stats(self, category: Any = None, metric: Any = None) -> dict[str, Any]:
        category = self._category(category)
        metric = self._choice(metric, METRICS, "metric")
        stats = analysis.describe(self.menu, self.include_unreliable)
        key = (category, metric)
        if key not in stats.index:
            return {
                "category": category,
                "metric": metric,
                "available": False,
                "message": f"{metric} is not available for {category} in this data.",
            }
        row = stats.loc[key]
        digits = 2 if metric in config.DERIVED_UNITS else 1
        result = {
            "category": category,
            "metric": metric,
            "unit": analysis.METRIC_UNITS[metric],
            "available": True,
            **{f: json_number(row[f], digits) for f in ("mean", "median", "min", "max")},
            "n_in_stats": int(row["count"]),
            "coverage": self._coverage(category, metric),
            "unreliable_rows_included": self.include_unreliable,
        }
        if metric in config.NUTRIENT_UNITS:
            result["total"] = json_number(row["total"], 1)
        return result

    def top_items(
        self, metric: Any = None, category: Any = None, n: Any = None, order: Any = None
    ) -> dict[str, Any]:
        metric = self._choice(metric, METRICS, "metric")
        scope = self._category(category, allow_any=True)
        n = self._int(n, "n", 5, MAX_N)
        order = self._choice(order, ("highest", "lowest"), "order", "highest")
        ranked = analysis.top_n(
            self.menu, metric, n, scope, order == "lowest", self.include_unreliable
        )
        return {
            "metric": metric,
            "unit": analysis.METRIC_UNITS[metric],
            "category": scope or "any",
            "order": order,
            "items": self._rows(ranked, [metric]),
            "note": "Only items with a value are ranked."
            if len(ranked)
            else f"No items have {metric} data for this selection.",
        }

    def filter_items(
        self,
        category: Any = None,
        ranges: Any = None,
        upper_inclusive: Any = None,
        caffeine_status: Any = None,
        search: Any = None,
        sort_by: Any = None,
        order: Any = None,
        limit: Any = None,
    ) -> dict[str, Any]:
        scope = self._category(category, allow_any=True)
        limit = self._int(limit, "limit", 10, MAX_ROWS)
        order = self._choice(order, ("highest", "lowest"), "order", "highest")
        if upper_inclusive is not None and not isinstance(upper_inclusive, bool):
            raise ToolArgumentError("upper_inclusive must be true or false")
        if ranges is not None and not isinstance(ranges, Mapping):
            raise ToolArgumentError('ranges must be an object like {"calories": {"max": 500}}')
        parsed: dict[str, tuple[float | None, float | None]] = {}
        for metric, bounds in (ranges or {}).items():
            self._choice(metric, NUTRIENT_COLUMNS, "range metric")
            if not isinstance(bounds, Mapping) or not set(bounds) <= {"min", "max"}:
                raise ToolArgumentError(f'Range for {metric} must look like {{"min": 1, "max": 2}}')
            low, high = bounds.get("min"), bounds.get("max")
            for bound in (low, high):
                if bound is not None and not _is_number(bound):
                    raise ToolArgumentError(f"Range bounds for {metric} must be finite numbers")
            parsed[metric] = (low, high)
        statuses = ()
        if caffeine_status is not None:
            statuses = (self._choice(caffeine_status, config.CAFFEINE_STATUSES, "caffeine_status"),)
        text = ""
        if search is not None:
            if not isinstance(search, str) or len(search) > MAX_SEARCH_CHARS:
                raise ToolArgumentError(f"search must be text of at most {MAX_SEARCH_CHARS} chars")
            text = search
        sort_metric = self._choice(sort_by, METRICS, "sort_by", next(iter(parsed), "calories"))

        spec = FilterSpec(
            categories=(scope,) if scope else (),
            text=text,
            ranges=parsed,
            inclusive=bool(upper_inclusive),
            lower_inclusive=True,
            caffeine_statuses=statuses,
            reliable_only=not self.include_unreliable,
        )
        result = analysis.add_derived_metrics(apply_filters(self.menu, spec))
        result = result.sort_values(sort_metric, ascending=order == "lowest", na_position="last")
        metrics = list(dict.fromkeys([*parsed, sort_metric, "calories"]))
        rows = self._rows(result.head(limit), metrics)
        if statuses:
            for row, status in zip(rows, result["caffeine_status"], strict=False):
                row["caffeine_status"] = status
        scope_df = apply_filters(self.menu, FilterSpec(categories=spec.categories))
        return {
            "total_matches": len(result),
            "returned": len(rows),
            "upper_bounds": "<= (up to)" if upper_inclusive else "< (under)",
            "sorted_by": f"{sort_metric} ({order} first)",
            "items": rows,
            "notes": range_filter_notes(scope_df, parsed)
            + (["caffeine_status is a name-based estimate, not mg."] if statuses else []),
        }

    def get_item(self, name: Any = None) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip() or len(name) > MAX_SEARCH_CHARS:
            raise ToolArgumentError(f"name must be text of 1-{MAX_SEARCH_CHARS} chars")
        matches = search(self.menu, name)
        rows = []
        for record in matches.head(5).to_dict("records"):
            rows.append(
                {
                    "item": safe_text(record[ITEM_COL]),
                    "category": record[CATEGORY_COL],
                    "has_nutrition_data": bool(record["has_nutrition"]),
                    "reliable": bool(record["is_reliable"]),
                    "caffeine_status": record["caffeine_status"],
                    **{
                        m: json_number(record[m], 1)
                        for m in NUTRIENT_COLUMNS
                        if json_number(record[m]) is not None
                    },
                }
            )
        return {"query": safe_text(name), "total_matches": len(matches), "items": rows}

    def compare_drinks_food(self) -> dict[str, Any]:
        table = analysis.compare_categories(self.menu, include_unreliable=self.include_unreliable)
        ratios = analysis.ratio_summary(self.menu, include_unreliable=self.include_unreliable)
        return {
            "comparison": [
                {
                    "metric": metric,
                    "unit": row["unit"],
                    "drink_mean": json_number(row["mean_drink"], 2),
                    "drink_n": int(row["n_analysed_drink"]),
                    "drink_coverage_pct": json_number(row["coverage_pct_drink"], 0),
                    "food_mean": json_number(row["mean_food"], 2),
                    "food_n": int(row["n_analysed_food"]),
                    "food_coverage_pct": json_number(row["coverage_pct_food"], 0),
                    "pct_diff_vs_drink": json_number(row["pct_diff"], 0),
                }
                for metric, row in table.iterrows()
            ],
            "fat_to_protein": {
                str(cat): {k: json_number(v, 2) for k, v in row.items()}
                for cat, row in ratios.iterrows()
            },
            "caveats": [
                safe_text(c, 600)
                for c in analysis.comparison_caveats(
                    self.menu, include_unreliable=self.include_unreliable
                )
            ],
        }

    def data_availability(self) -> dict[str, Any]:
        avail = analysis.availability(self.menu)
        return {
            "by_category": {
                str(cat): {
                    "available": list(group.loc[group["available"], "metric"]),
                    "not_available": list(group.loc[~group["available"], "metric"]),
                }
                for cat, group in avail.groupby(CATEGORY_COL)
            },
            "proxies": {
                "sugar_g": "Not in the data. Closest proxy: carb_g (carbohydrates include "
                "sugars); label it as carbohydrates, not sugar.",
                "caffeine_mg": "Not in the data. caffeine_status (likely/unlikely/unknown) "
                "is estimated from drink names; it is not a measurement and there is no "
                "average caffeine content.",
            },
        }

    # ------------------------------------------------------------------- dispatch
    def _tools(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "get_stats": self.get_stats,
            "top_items": self.top_items,
            "filter_items": self.filter_items,
            "get_item": self.get_item,
            "compare_drinks_food": self.compare_drinks_food,
            "data_availability": self.data_availability,
        }

    def execute(self, name: str, arguments: str | Mapping[str, Any] | None) -> dict[str, Any]:
        """Run one tool call. Never raises: problems become {"error": ...} for the model.

        Validation covers the argument shapes we know of; the final catch-all keeps an
        argument shape nobody anticipated from crashing the whole answer.
        """
        tools = self._tools()
        if name not in tools:
            return {"error": f"Unknown tool {name!r}", "allowed": list(tools)}
        try:
            args = (
                json.loads(arguments or "{}")
                if isinstance(arguments, str)
                else dict(arguments or {})
            )
        except json.JSONDecodeError:
            return {"error": "Arguments were not valid JSON."}
        if not isinstance(args, dict):
            return {"error": "Arguments must be a JSON object."}
        try:
            return tools[name](**args)
        except ToolArgumentError as exc:
            return {"error": str(exc)}
        except TypeError as exc:  # unexpected argument names
            return {"error": f"Invalid arguments for {name}: {exc}"}
        except Exception:
            logger.exception("Tool %s failed on arguments %r", name, args)
            return {"error": f"{name} could not run with these arguments; try different ones."}
