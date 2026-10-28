# Starbucks Nutrition Analysis

A prototype LLM-powered nutrition analysis tool for the Starbucks menu. It loads the drinks
and food CSVs, cleans them, computes statistics and comparisons with pandas, and (in later
steps) uses the Groq LLM API to narrate the results and a Streamlit UI to explore them.

> Current status: **Part 1 — data layer** (loading, cleaning, statistics, comparison,
> filtering) with tests and a command-line report.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` also installs this project in editable mode (`-e .`), so the package
can be run from any directory.

## Run the report

```bash
python -m starbucks_nutrition                 # uses data/raw/drinks.csv and data/raw/food.csv
python -m starbucks_nutrition -v              # also show cleaning decisions as INFO logs
python -m starbucks_nutrition --drinks my_drinks.csv --food my_food.csv
```

The report prints: the data-quality report for each file, coverage (rows analysed vs
excluded), which metrics exist per category, descriptive statistics, a drinks-vs-food
comparison, rankings and example filters.

## Use as a library

```python
from starbucks_nutrition import load_menu
from starbucks_nutrition import analysis
from starbucks_nutrition.filters import FilterSpec, apply_filters

# Sources may be paths, bytes or file objects (e.g. Streamlit uploads).
menu, reports = load_menu()

analysis.describe(menu)  # per-category stats
analysis.compare_categories(menu)  # drinks vs food, shared metrics + coverage
analysis.comparison_caveats(menu)  # plain-language caveats for that comparison
analysis.ratio_summary(menu)  # fat:protein as mean, median and aggregate ratio
analysis.top_n(menu, "calories", 5, category="drink")

# Food strictly under 500 kcal (inclusive=True, the default, would mean <= 500).
spec = FilterSpec(categories=("food",), ranges={"calories": (None, 500)}, inclusive=False)
apply_filters(menu, spec)

# What cleaning found and changed, as a JSON-friendly dict.
reports["drink"].to_dict()
```

## Tests and lint

```bash
pytest
ruff check .
ruff format --check .
```

## Data notes

- The raw files in `data/raw/` are never modified; all cleaning happens in code.
- Missing values (`-`, `N/A`, blank…) become NaN, never 0.
- Duplicates are resolved to one row per item per category; every dropped row is listed
  in the quality report with the rule applied and the reason.
- Only 48% of drinks have nutrition data and the gaps are concentrated in Frappuccinos
  and Doubleshot drinks, so drink averages are likely understated. Comparisons report
  coverage next to every mean, plus caveats generated from the data.
- Suspicious rows (e.g. a milk drink reported as 0 kcal) are kept but marked unreliable
  and excluded from statistics and rankings by default.
- The datasets contain **no sugar or caffeine data**. These metrics are reported as
  unavailable. "Has caffeine" uses a **name-based proxy** (`likely` / `unlikely` /
  `unknown`). Refreshers count as `likely` because they contain green-coffee extract. If
  an uploaded CSV has a `Caffeine (mg)` column, the measured value is used instead.
