# Starbucks Nutrition Analysis

A prototype LLM-powered nutrition analysis tool for the Starbucks menu. It loads the drinks
and food CSVs, cleans them, computes statistics and comparisons with pandas, and (in later
steps) uses the Groq LLM API to narrate the results and a Streamlit UI to explore them.

> Current status: **data layer** (loading, cleaning, statistics, comparison, filtering),
> **charts** (plotly) and a **Streamlit web UI** with CSV upload. The Groq LLM summaries
> come next.

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

## Run the web app

```bash
streamlit run streamlit_app.py
```

It opens at http://localhost:8501 with the bundled sample files. In the sidebar you can
switch to **Upload my CSVs** and upload a drinks file, a food file, or both. A category
you don't upload keeps using the sample, and every tab says which is which (sample vs
uploaded). One global toggle, **Include unreliable rows**, adds rows flagged as
implausible (e.g. every nutrient reported as 0) back into statistics and charts.

| Tab | What it shows |
|---|---|
| Overview | KPI tiles per category (items, items with data, mean calories and what it is based on), drinks-vs-food chart, caveats, comparison table with coverage, fat-to-protein ratio three ways |
| Explore & filter | Filters (category, name search, caffeine status, nutrient ranges with an "under (<) / up to (≤)" toggle); sortable table with units in headers; CSV download |
| Charts | Top/bottom N items for any metric, where calories come from (fat/carbs/protein), calories vs protein, caffeine status |
| Data quality | Per file: encoding, rows read → kept, duplicates resolved (line, item, rule, reason), unreliable rows, unavailable columns; nutrients per file; missing data by drink family |

Range sliders start at the full range and filter nothing until you move them. While a
range is active, items with no value for that nutrient are hidden and a caption says so
(e.g. all food when filtering on sodium).

### Upload format

One row per menu item, as CSV (UTF-8 or UTF-16, up to 5 MB per file):

- an **item-name column**: an unnamed first column, or a header such as `Item` or `Name`;
- plus any of `Calories`, `Fat (g)`, `Carb. (g)`, `Fiber (g)`, `Protein (g)`, `Sodium`,
  `Sugars (g)`, `Caffeine (mg)`. Header spelling and spacing are flexible (`Total Fat`,
  `Carbohydrates`, `Protein`...); unrecognised columns are ignored and listed.
- Missing values may be blank, `-` or `N/A`. They stay missing and are never treated as 0.

A file that cannot be used shows an error naming the file and what was expected.

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

# Charts are pure functions returning plotly figures (no Streamlit needed).
from starbucks_nutrition import viz

viz.compare_means_chart(menu).show()
```

## Tests and lint

```bash
pytest                 # everything, incl. Streamlit AppTest smoke tests
pytest -m "not ui"     # skip the slower UI smoke tests
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
