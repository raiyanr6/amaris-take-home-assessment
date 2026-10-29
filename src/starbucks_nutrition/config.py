"""Canonical schema and tunable rules, kept in one place.

Everything the cleaning/filtering code needs to *know* about the data (column names,
header spellings, placeholder tokens, thresholds, caffeine keywords) lives here as plain
data. The code in loader/filters only interprets these tables, so supporting a new header
variant or keyword is a one-line change here rather than a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
# Resolved from this file, not the current working directory, so the CLI/tests behave the
# same no matter where they are launched from (repo root, IDE, CI, Streamlit Cloud).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
DEFAULT_DRINKS_PATH: Final[Path] = RAW_DATA_DIR / "drinks.csv"
DEFAULT_FOOD_PATH: Final[Path] = RAW_DATA_DIR / "food.csv"

# --------------------------------------------------------------------------------------
# Canonical columns
# --------------------------------------------------------------------------------------
ITEM_COL: Final = "item"
ITEM_KEY_COL: Final = "item_key"
CATEGORY_COL: Final = "category"

# Canonical nutrient columns -> display unit. Order here is the output column order.
# sugar_g and caffeine_mg are absent from the provided files but are part of the schema so
# that an uploaded CSV which *does* have them is used instead of silently dropped.
NUTRIENT_UNITS: Final[dict[str, str]] = {
    "calories": "kcal",
    "fat_g": "g",
    "carb_g": "g",
    "fiber_g": "g",
    "protein_g": "g",
    "sodium_mg": "mg",
    "sugar_g": "g",
    "caffeine_mg": "mg",
}
NUTRIENT_COLUMNS: Final[tuple[str, ...]] = tuple(NUTRIENT_UNITS)
# Human labels for charts, tables and UI messages (units are added from the dicts above).
NUTRIENT_LABELS: Final[dict[str, str]] = {
    "calories": "Calories",
    "fat_g": "Fat",
    "carb_g": "Carbs",
    "fiber_g": "Fiber",
    "protein_g": "Protein",
    "sodium_mg": "Sodium",
    "sugar_g": "Sugar",
    "caffeine_mg": "Caffeine",
}

# Derived per-item metrics computed in analysis.add_derived_metrics.
DERIVED_UNITS: Final[dict[str, str]] = {
    "fat_to_protein": "ratio",
    "protein_per_100kcal": "g/100 kcal",
    "fiber_per_100kcal": "g/100 kcal",
}

# Header aliases. Keys are *normalised* headers: lowercased, accents folded, punctuation
# turned into spaces, and a trailing unit token ("g", "mg", "kcal") removed — so
# " Carb. (g)" -> "carb", "Protein (g)" -> "protein", "Caffeine (mg)" -> "caffeine".
# See loader._normalise_header.
COLUMN_ALIASES: Final[dict[str, str]] = {
    # item name
    "item": ITEM_COL,
    "item name": ITEM_COL,
    "name": ITEM_COL,
    "menu item": ITEM_COL,
    "product": ITEM_COL,
    "beverage": ITEM_COL,
    "drink": ITEM_COL,
    "food": ITEM_COL,
    # calories
    "calories": "calories",
    "calorie": "calories",
    "energy": "calories",
    "kcal": "calories",
    # fat
    "fat": "fat_g",
    "total fat": "fat_g",
    # carbohydrates
    "carb": "carb_g",
    "carbs": "carb_g",
    "carbohydrate": "carb_g",
    "carbohydrates": "carb_g",
    "total carbohydrate": "carb_g",
    "total carbohydrates": "carb_g",
    # fiber
    "fiber": "fiber_g",
    "fibre": "fiber_g",
    "dietary fiber": "fiber_g",
    # protein
    "protein": "protein_g",
    # sodium (drinks.csv has a bare "Sodium" header; values are mg per standard labelling)
    "sodium": "sodium_mg",
    # sugar (optional)
    "sugar": "sugar_g",
    "sugars": "sugar_g",
    "total sugar": "sugar_g",
    "total sugars": "sugar_g",
    # caffeine (optional)
    "caffeine": "caffeine_mg",
}

# Unit tokens stripped from the end of a normalised header before alias lookup.
HEADER_UNIT_TOKENS: Final[tuple[str, ...]] = ("g", "mg", "kcal")

# Cell values meaning "no data". Compared case-insensitively after stripping whitespace.
# They become NaN — never 0 — because "unknown" and "zero" mean different things
# (a 0 would drag averages down and make items look healthier than they are).
MISSING_TOKENS: Final[frozenset[str]] = frozenset({"-", "", "n/a", "na", "--"})

# --------------------------------------------------------------------------------------
# Plausibility checks
# --------------------------------------------------------------------------------------
# Atwater factors: kcal per gram of macro. Used to sanity-check reported calories.
ATWATER_KCAL_PER_G: Final[dict[str, float]] = {"carb_g": 4.0, "protein_g": 4.0, "fat_g": 9.0}
# A row is "energy inconsistent" when |calories - Atwater estimate| exceeds BOTH an
# absolute and a relative tolerance. The absolute part stops tiny items (5 kcal coffee,
# which labels round) from being flagged; the relative part scales for large items.
# Label rounding + fiber/sugar-alcohol effects routinely produce ~10-20% gaps, so only
# gaps well beyond that are treated as data errors.
ENERGY_ABS_TOLERANCE_KCAL: Final[float] = 15.0
ENERGY_REL_TOLERANCE: Final[float] = 0.35

# --------------------------------------------------------------------------------------
# Caffeine proxy rules (name-based)
# --------------------------------------------------------------------------------------
# The provided datasets have NO caffeine column. When caffeine_mg is missing for a row,
# filters.caffeine_status estimates caffeine from the drink NAME. It is a PROXY, not a
# measurement. Statuses:
#   "likely"   - the name names a caffeinated base (coffee, espresso, black/green tea...)
#   "unlikely" - the name names a caffeine-free base (decaf, herbal, crème, juice...)
#   "unknown"  - the name does not reveal the base (e.g. "Tazo® Bottled Giant Peach")
#
# Tiers are evaluated top-down; the FIRST tier that matches wins. Patterns are regexes
# matched with word boundaries against item_key (lowercase, accent/™/® folded), so write
# them folded: "creme" not "crème".
CAFFEINE_RULE_TIERS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    # 1. Explicitly caffeine-free bases always win, even over tea/coffee words.
    (
        "hard_exclude",
        "unlikely",
        (r"decaf", r"herbal", r"passion"),  # Passion Tango / Iced Passion are herbal infusions
    ),
    # 2. Real tea bases override the crème exclusion below: Chai Crème and Green Tea
    #    (matcha) Crème Frappuccinos contain caffeine even though they are crème-based.
    (
        "tea_override",
        "likely",
        (r"chai", r"green tea", r"matcha", r"black tea", r"white tea"),
    ),
    # 3. Caffeine-free product families. Crème Frappuccinos/steamers have no coffee;
    #    Midnight Mint Mocha "Blended Crème" is listed separately from its "Blended Coffee"
    #    twin precisely because it has no coffee. Hot chocolate has only trace caffeine.
    (
        "soft_exclude",
        "unlikely",
        (
            r"creme",
            r"blended beverage",  # e.g. Pokémon GO Frappuccino — crème base
            r"juice",
            r"evolution fresh",
            r"smoothie",
            r"soda",
            r"ale",
            r"hot chocolate",
            r"chocolate milk",
        ),
    ),
    # 4. Caffeinated bases. Refreshers (and the Pink/Violet/Ombré drinks built on Refresher
    #    bases) are deliberately "likely": they are made with green-coffee extract (~45 mg
    #    per grande), even though they taste like fruit drinks.
    (
        "caffeinated_base",
        "likely",
        (
            r"coffee",
            r"espresso",
            r"latte",
            r"mocha",
            r"macchiato",
            r"americano",
            r"cappuccino",
            r"flat white",
            r"cold brew",
            r"roast",
            r"misto",
            r"doubleshot",
            r"refreshers?",
            r"pink drink",
            r"violet drink",
            r"black",  # Tazo® Bottled Black Mango / Black with Lemon — black tea base
        ),
    ),
    # 5. It is a tea, but the name does not say which kind (could be herbal or not).
    (
        "tea_unspecified",
        "unknown",
        (r"tea", r"tazo", r"teavana"),
    ),
)
# --------------------------------------------------------------------------------------
# Product families (for explaining missing data)
# --------------------------------------------------------------------------------------
# Missing nutrition data is not random: in drinks.csv it is concentrated in Frappuccinos
# and Doubleshot cans (typically high-calorie), but also includes espresso shots and sodas.
# analysis.comparison_caveats groups rows without data by family to say WHICH items are
# missing and to estimate the bias. First matching family wins; patterns match item_key.
PRODUCT_FAMILIES: Final[tuple[tuple[str, str], ...]] = (
    ("Frappuccino", r"frappuccino"),
    ("Doubleshot", r"doubleshot"),
    ("Refreshers", r"refreshers?|pink drink|violet drink"),
    ("Tea", r"\btea\b|tazo|teavana|chai"),
    (
        "Coffee & espresso",
        r"coffee|cold brew|espresso|americano|latte|mocha|macchiato|misto|cappuccino"
        r"|flat white|roast",
    ),
)
PRODUCT_FAMILY_OTHER: Final = "Other"
# A family's analysed mean is only used to estimate its missing items when it is based on
# at least this many items (one canned Doubleshot on Ice says little about Doubleshot Energy).
MIN_FAMILY_SAMPLE: Final = 3

# Status for drinks no tier matches.
CAFFEINE_DEFAULT_STATUS: Final = "unlikely"
# Food names are not a usable caffeine signal ("Classic Coffee Cake" has none, chocolate
# items have a little), so the proxy is not applied to food: it reports "unknown".
CAFFEINE_FOOD_STATUS: Final = "unknown"
CAFFEINE_STATUSES: Final[tuple[str, ...]] = ("likely", "unlikely", "unknown")

# --------------------------------------------------------------------------------------
# LLM (Groq)
# --------------------------------------------------------------------------------------
# Production model on Groq (console.groq.com/docs/models, checked 2026-10-29) with tool
# calling incl. parallel calls. Override with the GROQ_MODEL env var / secret.
LLM_DEFAULT_MODEL: Final = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: Final = 0.2  # factual narration, not creativity
LLM_MAX_TOKENS: Final = 900
LLM_TIMEOUT_S: Final = 30.0
LLM_MAX_ATTEMPTS: Final = 3  # 1 call + 2 retries on 429 / 5xx / timeouts
LLM_MAX_BACKOFF_S: Final = 20.0
# Protects a free-tier key once deployed: API calls per browser session.
LLM_MAX_CALLS_PER_SESSION: Final = 30
# Untrusted text (item names from uploaded CSVs) is cut to this length before it is
# shown to the model.
LLM_MAX_NAME_CHARS: Final = 100
