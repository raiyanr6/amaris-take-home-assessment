# CLAUDE.md

## Project goal
Take-home technical test: an LLM-powered nutrition analysis tool for the Starbucks menu
(brief: `docs/INSTRUCTIONS.md`). Built in steps:
1. **Data layer** (done): loading, cleaning, stats, comparison, filtering + tests.
2. **Visualisations + Streamlit UI** (done): `viz.py`, `ui/`, `streamlit_app.py`.
3. **Groq LLM** (done): summary (`llm/summarize.py`) and tool-calling Q&A (`llm/qa.py`).
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
  llm/          client.py (GroqClient, typed errors, retries, BudgetedClient),
                context.py (build_facts), grounding.py (verify_numbers), safety.py,
                prompts.py (versioned), summarize.py, tools.py (ToolExecutor), qa.py
  ui/           data_source.py (sidebar, uploads, cache), common.py, llm_access.py, one
                module per tab: overview, ai_summary, ask, explore, charts, data_quality
  __main__.py   CLI report: python -m starbucks_nutrition
tests/          pytest (no __init__.py); conftest.py has fixture CSV builders; fakes.py has
                FakeLLMClient / FakeGroqSDK; test_app*.py = AppTest (marker `ui`);
                test_live_groq.py = real API (marker `live`, deselected by default)
.env.example    GROQ_API_KEY / GROQ_MODEL template (.env itself is git-ignored)
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

## LLM rules
- **Secrets:** `GROQ_API_KEY` comes from Streamlit secrets, the environment or `.env`.
  Never print, log, commit or put it in an exception message; `.env` and
  `.streamlit/secrets.toml` stay git-ignored.
- **Model:** from `GROQ_MODEL`, else `config.LLM_DEFAULT_MODEL`; never hard-code it at
  call sites. The default is `openai/gpt-oss-120b`: the project key got 404 for
  `llama-3.3-70b-versatile`. gpt-oss runs with `reasoning_effort="low"` and
  `include_reasoning=False`. Use temperature 0.2 and capped max tokens.
- **Summaries:** the model gets `build_facts()` output only (never raw CSVs). Every value
  in it comes from package functions, JSON-safe (NaN becomes null), with units, coverage,
  caveats and availability.
- **Q&A:** tool calling only (no table in context, no code generation / exec).
  - Tools are read-only wrappers in `llm/tools.py` with whitelist validation. Bad
    arguments return `{"error": ...}` and never raise.
  - Limits: 25 rows per result, at most 4 tool rounds then a forced answer, questions
    capped at 500 chars.
  - A failed or malformed tool call is retried once, then the user gets a friendly
    "please rephrase" answer.
  - Q&A uses the loaded menu and the UI's `include_unreliable` setting.
- **Grounding:** `verify_numbers()` checks every figure in the output against the sources:
  the facts, or ALL tool results plus the tool arguments and the question. Item names are
  removed first, since they can contain digits. The UI shows the badge.
- **Untrusted input:** both the user's question and uploaded item names are untrusted.
  - Both system prompts include `DATA_NOT_INSTRUCTIONS`.
  - Names pass through `safety.safe_text`: control characters stripped, 100 chars max.
  - Off-topic requests and injection attempts get a short refusal.
- **Unavailable data:** sugar, caffeine (mg) and food sodium are reported as unavailable.
  Carbs are offered as a labelled sugar proxy; caffeine uses the name-based status.
- **Caffeine counts:** one reliability rule everywhere (`analysis.caffeine_counts`,
  respects `include_unreliable`); CLI, Charts tab, facts and tools agree (tested).
- **Prompt changes:** bump `prompts.PROMPT_VERSION`.
- **Tests:** never call the network. Inject `FakeLLMClient`; the UI picks it up from
  `st.session_state["llm_client"]`.
- **Protecting the free key:** 30 AI calls per session (`BudgetedClient`); summaries are
  cached on (facts, focus, model).

## Commands
```bash
python -m venv .venv
.venv/Scripts/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt   # also installs this package in editable mode
python -m starbucks_nutrition     # CLI report (-v for INFO logs)
streamlit run streamlit_app.py    # web UI
pytest                            # tests (-m "not ui" skips the AppTest smoke tests)
pytest -m live -s                 # real Groq calls (needs GROQ_API_KEY)
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
