# CLAUDE.md

## Project goal
Take-home technical test: an LLM-powered nutrition analysis tool for the Starbucks menu
(brief: `docs/INSTRUCTIONS.md`). Built in steps:
1. **Data layer** (done): loading, cleaning, stats, comparison, filtering + tests.
2. Visualisations.
3. Groq LLM summarisation / natural-language Q&A.
4. Streamlit UI with CSV upload (deployed to Streamlit Cloud).

## Layout
```
data/raw/{drinks,food}.csv   provided data — READ-ONLY
docs/                        brief
src/starbucks_nutrition/
  config.py     canonical schema, header aliases, missing tokens, tolerances, caffeine rules
  textnorm.py   name/header normalisation and folded search keys
  loader.py     read_csv_robust, clean, validate_schema, load_menu, QualityReport, SchemaError
  analysis.py   describe, coverage, add_derived_metrics, compare_categories, top_n, availability
  filters.py    filter_range, search, caffeine_status proxy, FilterSpec, apply_filters
  __main__.py   CLI report: python -m starbucks_nutrition
tests/          pytest (no __init__.py); conftest.py has fixture CSV builders
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
- The loader must accept paths, bytes and file-like objects (Streamlit uploads) and must
  not assume the exact layout of the provided files (use the alias map in config).
- Rules and thresholds live in `config.py` as data, not as if-chains in code.
- Use `logging`, not `print` (except `__main__.py`). Vectorised pandas; no row-wise apply.
- Type hints everywhere; docstrings explain *why*.

## Commands
```bash
python -m venv .venv
.venv/Scripts/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt   # also installs this package in editable mode
python -m starbucks_nutrition     # CLI report (-v for INFO logs)
pytest                            # tests
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
- `Nariño 70 Cold Brew with Milk` is all zeros -> kept, flagged, `is_reliable=False`.
- No sugar or caffeine columns in either file. Caffeine uses a tri-state name-based proxy
  (`likely`/`unlikely`/`unknown`); Refreshers count as `likely` (green-coffee extract);
  food is `unknown` (not assessed). A real `caffeine_mg` column overrides the proxy.
