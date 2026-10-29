# CLAUDE.md

## Project goal
Take-home technical test: an LLM-powered nutrition analysis tool for the Starbucks menu
(brief: `docs/INSTRUCTIONS.md`). Built in steps:
1. **Data layer** (done): loading, cleaning, stats, comparison, filtering + tests.
2. **Visualisations + Streamlit UI** (done): `viz.py`, `ui/`, `streamlit_app.py`.
3. Groq LLM summarisation / natural-language Q&A.
4. CI/CD and deployment to Streamlit Cloud.

## Layout
```
streamlit_app.py             Streamlit entry point (sidebar + tabs only)
.streamlit/config.toml       neutral theme, 5 MB upload limit (secrets.toml stays gitignored)
data/raw/{drinks,food}.csv   provided data — READ-ONLY
docs/                        brief
src/starbucks_nutrition/
  config.py     canonical schema, header aliases, missing tokens, tolerances, caffeine rules
  textnorm.py   name/header normalisation and folded search keys
  loader.py     read_csv_robust, clean, validate_schema, load_menu, QualityReport, SchemaError
  analysis.py   describe, coverage, add_derived_metrics, compare_categories, comparison_caveats,
                ratio_summary, product_family, top_n, availability, plus UI helpers
                (kpis, metric_bounds, macro_energy_share, caffeine_counts,
                family_coverage, availability_matrix, display_table, metric_label)
  filters.py    filter_range, search, caffeine_status proxy, FilterSpec, apply_filters,
                active_ranges, range_filter_notes
  viz.py        plotly charts: DataFrame in -> Figure out (no Streamlit imports)
  ui/           data_source.py (sidebar, uploads, cache), common.py, one module per tab:
                overview.py, explore.py, charts.py, data_quality.py
  __main__.py   CLI report: python -m starbucks_nutrition
tests/          pytest (no __init__.py); conftest.py has fixture CSV builders;
                test_app.py = Streamlit AppTest smoke tests (marker `ui`)
```

## Rules
- **`data/raw/` is read-only.** Never modify, re-save or re-encode those files; all
  cleaning happens in code. `.gitattributes` marks them `-text` to keep them byte-exact.
- **pandas computes numbers; the LLM only narrates.** The LLM layer must receive numbers
  computed by `analysis`/`filters` (plus `availability()` and the QualityReport) and must
  never calculate or invent values. Unavailable metrics (sugar, caffeine, food sodium)
  must be stated as unavailable.
- Missing values are NaN, never 0. Unreliable rows (`is_reliable == False`) are excluded
  from stats/rankings by default (`include_unreliable=False`).
- Any category comparison shown to users or the LLM must carry its coverage columns and
  `analysis.comparison_caveats()` — drink means cover only 48% of drinks.
- The loader must accept paths, bytes and file-like objects (Streamlit uploads) and must
  not assume the exact layout of the provided files (use the alias map in config).
- Rules and thresholds live in `config.py` as data, not as if-chains in code.
- Use `logging`, not `print` (except `__main__.py`). Vectorised pandas; no row-wise apply.
- Type hints everywhere; docstrings explain *why*.

## UI rules
- The UI is a THIN layer: no pandas calculations in `ui/` or `streamlit_app.py`. If the UI
  needs a number or table the package doesn't expose, add a pure, tested function to
  `analysis`/`filters` and call it.
- Charts live in `viz.py` as pure functions (DataFrame in, plotly Figure out). They never
  import Streamlit and return a "No data for this selection" figure for empty input.
- One fixed colour per category in every chart (drink `#2a78d6`, food `#eb6834`,
  colour-blind safe); units in every axis label; titles state the question answered.
- Unreliable rows are excluded from stats and charts unless the global sidebar toggle
  "Include unreliable rows" is on. Rows without data never appear in numeric charts but
  do appear in tables/search; tables always list unreliable rows (Reliable column).
- Coverage and `comparison_caveats()` / `missing_data_caveats()` sit next to the numbers
  they qualify. The caffeine proxy is explained wherever `caffeine_status` is shown.
- Each category is labelled "sample" or "uploaded"; uploading one file falls back to the
  sample for the other. Load errors show a friendly `st.error` (which file, what was
  expected), never a stack trace. Loading uses `st.cache_data` keyed on the file bytes.
- Range sliders: a slider at its full range applies no filter (`active_ranges`); "under /
  up to" changes the upper bound only (`lower_inclusive=True`); an active range hides NaN
  rows, and `range_filter_notes` says so.

## Commands
```bash
python -m venv .venv
.venv/Scripts/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt   # also installs this package in editable mode
python -m starbucks_nutrition     # CLI report (-v for INFO logs)
streamlit run streamlit_app.py    # web UI
pytest                            # tests (-m "not ui" skips the AppTest smoke tests)
ruff check . && ruff format --check .
```

## Platform
Developed on Windows (PowerShell); runs on Linux in CI and Streamlit Cloud. Use
`pathlib`, lowercase file names, no OS-specific code. Paths are resolved from the package
location (`config.PROJECT_ROOT`), not the working directory. Text files are LF.

## Data quirks (verified by profiling)
- `food.csv` is UTF-16 LE with BOM; `drinks.csv` is UTF-8 (no BOM). Encoding is detected
  (BOM -> NUL-byte UTF-16 -> utf-8 -> cp1252 -> latin-1), not hard-coded per file.
- First column has no header (`Unnamed: 0`) — it is the item name.
- Food headers have leading spaces (`" Calories"`); drinks has `Protein`/`Sodium` without
  units; food has `Protein (g)` and **no sodium**.
- Drinks uses `-` as missing placeholder; 85/177 rows are entirely `-`.
- Drinks has 22 exact duplicate rows (repeated blocks) and one conflict: `Iced Coffee` all
  zeros (line 11) vs 5 kcal / 5 mg sodium (line 102) -> keep the 5 kcal row.
- Final: drinks 177 -> 154 rows (74 with data, 80 without); food 113 rows, all with data.
- Names: trailing/double spaces (food), ™/®/accents, one quoted name with a comma.
- Missing drink data is NOT random: of 80 drinks without data, 33 are Frappuccinos
  (blended + bottled), 22 coffee/espresso, 11 Doubleshot. Frappuccinos with data average
  220 kcal vs 141 for all analysed drinks, so the drink calorie mean is likely understated
  (family-based estimate ~162 kcal). `comparison_caveats()` derives this from the data.
- Four food items are exactly 500 kcal: "under 500" (strict, `inclusive=False`) = 99,
  "up to 500" = 103.
- Fat-to-protein: report mean of item ratios, median, AND aggregate (sum fat / sum
  protein) via `analysis.ratio_summary()`; 30 analysed drinks have 0 g protein.
- `Nariño 70 Cold Brew with Milk` is all zeros -> kept, flagged, `is_reliable=False`.
- No sugar or caffeine columns in either file. Caffeine uses a tri-state name-based proxy
  (`likely`/`unlikely`/`unknown`); Refreshers count as `likely` (green-coffee extract);
  food is `unknown` (not assessed). A real `caffeine_mg` column overrides the proxy.
