"""Command-line report: ``python -m starbucks_nutrition [--drinks PATH] [--food PATH]``.

This is the "Python script" from the brief: it loads both CSVs, prints the data-quality
report, per-category statistics, the drinks-vs-food comparison and some example filters.
It only formats results; every number comes from loader/analysis/filters.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from starbucks_nutrition import analysis, config
from starbucks_nutrition.filters import (
    FilterSpec,
    apply_filters,
    filter_caffeinated,
    filter_range,
)
from starbucks_nutrition.loader import DataLoadError, QualityReport, load_menu


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _print_report(report: QualityReport) -> None:
    print(f"\n[{report.category}] encoding={report.encoding_detected}")
    print(
        f"  rows read: {report.rows_read} -> final: {report.rows_final} | exact duplicates "
        f"removed: {report.exact_duplicates_removed} | other rows dropped: "
        f"{len(report.conflicts_resolved) - report.exact_duplicates_removed} | "
        f"rows without data (kept): {report.rows_without_data}"
    )
    if report.rows_without_name:
        print(f"  rows without a name (dropped): {report.rows_without_name}")
    print(f"  columns unavailable: {', '.join(report.columns_unavailable) or 'none'}")
    if report.unknown_columns:
        print(f"  columns ignored: {', '.join(report.unknown_columns)}")
    if report.invalid_values:
        print(f"  invalid values set to NaN: {report.invalid_values}")
    for drop in report.conflicts_resolved:
        if drop["rule"] != "1_exact_duplicate":
            print(
                f"  resolved {drop['rule']}: line {drop['source_line']} "
                f"'{drop['item']}' dropped — {drop['reason']}"
            )
    for row in report.suspicious:
        print(
            f"  suspicious (kept, marked unreliable): line {row['source_line']} "
            f"'{row['item']}' — {row['quality_flag']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drinks", type=Path, default=config.DEFAULT_DRINKS_PATH)
    parser.add_argument("--food", type=Path, default=config.DEFAULT_FOOD_PATH)
    parser.add_argument("-v", "--verbose", action="store_true", help="show INFO logs")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Windows consoles default to a legacy code page that cannot print ™/®/é.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:,.2f}".format)

    try:
        menu, reports = load_menu(args.drinks, args.food)
    except (DataLoadError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _section("1. Data quality")
    for report in reports.values():
        _print_report(report)

    _section("2. Coverage (statistics use 'analysed' rows only)")
    print(analysis.coverage(menu).to_string())

    _section("3. Metric availability")
    avail = analysis.availability(menu)
    print(
        avail.pivot(index="metric", columns="category", values="n_values")
        .reindex(list(config.NUTRIENT_COLUMNS))
        .to_string()
    )
    print("(n_values = items with a value; 0 = not in the dataset.)")

    _section("4. Descriptive statistics per category")
    print(analysis.describe(menu).to_string())

    _section("5. Drinks vs food (shared metrics only; pct_diff relative to drinks)")
    print(analysis.compare_categories(menu).to_string())
    print(
        "(n_total = all items; n_with_data / coverage_pct = items with a value; "
        "mean is over n_analysed = with data, unreliable rows excluded.)"
    )
    print("\nFat-to-protein ratio, three views (g fat per g protein):")
    print("  mean_of_item_ratios   = average of each item's ratio (protein 0 g items skipped)")
    print("  median_of_item_ratios = typical item's ratio")
    print("  aggregate_ratio       = total fat / total protein over items with both values")
    print(analysis.ratio_summary(menu).to_string())
    print("\nCaveats:")
    for caveat in analysis.comparison_caveats(menu):
        print(f"  - {caveat}")

    _section("6. Rankings (unreliable rows excluded)")
    print("Highest-calorie drinks:")
    print(analysis.top_n(menu, "calories", 5, category="drink").to_string())
    print("\nLowest-calorie drinks:")
    print(analysis.top_n(menu, "calories", 5, category="drink", ascending=True).to_string())
    print("\nMost protein per 100 kcal (food):")
    print(analysis.top_n(menu, "protein_per_100kcal", 5, category="food").to_string())

    _section("7. Example filters")
    cols = ["item", "calories", "protein_g"]
    food = menu[menu["category"] == "food"]
    light_food = filter_range(food, "calories", max_value=500, inclusive=False)
    up_to = filter_range(food, "calories", max_value=500)
    print(
        f"Food items under 500 kcal (< 500): {len(light_food)} of {len(food)} "
        f"({len(up_to)} with <= 500; {len(up_to) - len(light_food)} are exactly 500 kcal)"
    )
    print(light_food.nsmallest(5, "calories")[cols].to_string(index=False))

    drinks = menu[menu["category"] == "drink"]
    print("\nDrink caffeine status (NAME-BASED PROXY — the dataset has no caffeine column):")
    print(drinks["caffeine_status"].value_counts().to_string())
    caffeinated = filter_caffeinated(drinks)
    print(
        f"\nLikely-caffeinated drinks with data and <= 100 kcal: "
        f"{len(apply_filters(caffeinated, FilterSpec(ranges={'calories': (None, 100)})))}"
    )
    unknown = drinks[drinks["caffeine_status"] == "unknown"]["item"].tolist()
    print(f"Caffeine unclear from name ({len(unknown)}): {', '.join(unknown)}")

    none = apply_filters(menu, FilterSpec(text="unicorn"))
    print(f"\nSearch 'unicorn': {len(none)} rows (empty result, columns kept: {len(none.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
