# Starbucks Nutrition Analysis

A prototype LLM-powered nutrition analysis tool for the Starbucks menu. It loads the drinks
and food CSVs (or your own uploads), cleans them, and computes statistics, comparisons and
filters with pandas. A Streamlit web UI shows the results with plotly charts, and the Groq
LLM API writes a plain-language summary (Part 2) and answers natural-language questions
through read-only lookups (bonus). A CLI report gives the same numbers in a terminal.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements-dev.txt   # app + test/lint tools
# or, to run the app only:  pip install -r requirements.txt
```

`requirements.txt` holds the runtime dependencies (what Streamlit Cloud installs) and
also installs this project in editable mode (`-e .`), so the package can be run from any
directory. `requirements-dev.txt` adds pytest, pytest-cov and ruff.

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
- Trace amounts written as `<1` or `< 1 g` are stored as half the limit (`0.5`) and
  counted as approximate in the Data quality tab. Thousands separators may be a comma or
  a space (`1,200`, `1 200`).

A file that cannot be used shows an error naming the file and what was expected.

## AI features (Groq)

**Design: pandas computes, the LLM narrates.** The model never sees the CSVs and never
runs code:

- **AI summary** sends a compact JSON of figures computed by the package (stats,
  comparison with coverage, rankings, caveats, which metrics are missing) and asks the
  model to describe them under fixed headings.
- **Ask the menu** uses **tool calling**: the model picks one of six read-only lookups
  (`get_stats`, `top_items`, `filter_items`, `get_item`, `compare_drinks_food`,
  `data_availability`). Arguments are validated against whitelists, results are capped
  at 25 rows, and the model phrases the answer.
- **Grounding check:** every figure in the output is checked against the figures the
  model was given. The app shows "✓ All N figures verified against the data" or lists
  the figures it could not find.
- **Missing data is stated, not guessed:** sugar and caffeine (mg) are not in the data,
  and the answer says so. Carbohydrates are offered as a labelled sugar proxy, and the
  name-based caffeine status as a caffeine estimate.

### Setup

1. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys).
2. Local: `copy .env.example .env` (macOS/Linux: `cp .env.example .env`) and set
   `GROQ_API_KEY=...`. `.env` is git-ignored and the key is never logged.
3. Streamlit Cloud: add the key under *App settings → Secrets*:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Optional: `GROQ_MODEL` picks another model. The default is `openai/gpt-oss-120b`, a
   production model on Groq, run with low reasoning effort.

Without a key, the two AI tabs explain how to add one and the rest of the app works as
before. The app is protected for a shared free-tier key in three ways:
- rate limits (429) are retried with backoff, honouring `retry-after`;
- each browser session is capped at 30 AI requests (one per summary or question, however
  many tool rounds it takes);
- summaries are cached per (data, focus, model), so clicking Generate again doesn't re-bill.

### The two AI tabs

| Tab | What it does |
|---|---|
| AI summary | Pick a focus (overview, highest-carb drinks as a sugar proxy, most calorie-dense food, best protein per calorie, lighter options) and generate. Shows the grounding badge, model / tokens / latency, and an expander with the exact system prompt and facts the model was given. |
| Ask the menu | Chat with example questions. Each answer has a grounding badge and a "How I got this" expander with every tool call, its arguments and the (truncated) result. Follow-ups such as "and for food?" use the last 3 turns. |

Example questions: *Which drinks have the most carbs?* · *Which food items are under 500
calories and high in protein?* · *What's the average caffeine content for drinks?* (the
answer: not in the data) · *Which drinks have the most sugar?* (the answer: not in the
data; carbohydrates offered as a proxy).

Off-topic requests and prompt injection ("ignore your instructions…") get a short
refusal. Uploaded item names are treated as data, never instructions: they are
sanitised and truncated, and both system prompts say so.

### From Python

```python
from starbucks_nutrition import load_menu
from starbucks_nutrition.llm.client import GroqClient
from starbucks_nutrition.llm.context import build_facts
from starbucks_nutrition.llm.qa import answer_question
from starbucks_nutrition.llm.summarize import generate_summary

menu, reports = load_menu()
client = GroqClient()  # reads GROQ_API_KEY / GROQ_MODEL from the environment or .env
print(generate_summary(client, build_facts(menu, reports), "overview").text)
result = answer_question(client, "Which drinks have the most carbs?", menu)
print(result.answer, result.grounding)
```

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

Needs the dev tools (`pip install -r requirements-dev.txt`).

```bash
pytest                 # everything except live API tests, incl. AppTest smoke tests
pytest -m "not ui"     # skip the slower UI smoke tests
pytest -m live -s      # real Groq calls; skipped unless GROQ_API_KEY is set
ruff check .
ruff format --check .
```

## Data notes

- The raw files in `data/raw/` are never modified; all cleaning happens in code.
- Missing values (`-`, `N/A`, blank…) become NaN, never 0. `<x` values become `x / 2`
  and are reported as approximate (`config.LESS_THAN_VALUE_FACTOR`).
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
