"""Load and clean Starbucks nutrition CSVs into one canonical, analysis-ready table.

Pipeline (per file): read_csv_robust -> clean -> (load_menu concatenates categories).

The loader never trusts the exact layout of its input: Streamlit users will upload their
own CSVs, so encoding, header spelling, column order, extra columns and placeholder values
are all detected rather than assumed. Every row that is altered or removed is recorded in
a QualityReport so nothing is changed silently.
"""

from __future__ import annotations

import codecs
import io
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from starbucks_nutrition import config
from starbucks_nutrition.config import CATEGORY_COL, ITEM_COL, ITEM_KEY_COL, NUTRIENT_COLUMNS
from starbucks_nutrition.filters import add_caffeine_status
from starbucks_nutrition.textnorm import clean_display_text, fold_key

logger = logging.getLogger(__name__)

# Anything we can turn into CSV text: a path, raw bytes, or an open/uploaded file object
# (e.g. Streamlit's UploadedFile, which is a BytesIO subclass).
CsvSource = str | Path | bytes | bytearray | IO[bytes] | IO[str]

# Bookkeeping columns carried through cleaning so reports can point at the original row.
SOURCE_ROW_COL = "source_row"  # 0-based data-row index in the original file
SOURCE_LINE_COL = "source_line"  # 1-based line number in the file (header is line 1)


class DataLoadError(Exception):
    """The input could not be read as CSV at all (empty, undecodable, malformed)."""


class SchemaError(DataLoadError):
    """The CSV was read but does not contain the columns we need.

    Messages are written for end users (they will be shown in the Streamlit UI).
    """


@dataclass
class QualityReport:
    """What cleaning found and changed for one file.

    Kept as plain data (see to_dict) so the UI can render it and the LLM prompt can cite
    it — e.g. to say "80 drinks have no nutrition data" instead of guessing.
    """

    category: str
    encoding_detected: str = "unknown"
    rows_read: int = 0
    rows_final: int = 0
    rows_without_name: int = 0
    exact_duplicates_removed: int = 0
    rows_without_data: int = 0
    conflicts_resolved: list[dict[str, Any]] = field(default_factory=list)
    suspicious: list[dict[str, Any]] = field(default_factory=list)
    missing_counts: dict[str, int] = field(default_factory=dict)
    invalid_values: dict[str, int] = field(default_factory=dict)
    columns_unavailable: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (NaN already converted to None in the row records)."""
        return asdict(self)


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------
def _read_raw(source: CsvSource) -> bytes | str:
    """Get the raw content of any supported source without assuming its type."""
    if isinstance(source, bytes | bytearray):
        return bytes(source)
    if isinstance(source, str | Path):
        # Strings are treated as paths; pass CSV *content* as bytes or a file object.
        return Path(source).read_bytes()
    # File-like. Prefer getvalue() (BytesIO/UploadedFile) because it ignores the current
    # position — Streamlit reruns can leave an uploaded file already read to the end.
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "seek") and getattr(source, "seekable", lambda: False)():
        source.seek(0)
    return source.read()


def _decode(raw: bytes) -> tuple[str, str]:
    """Decode bytes, returning (text, encoding name).

    Order: BOM sniff (unambiguous) -> BOM-less UTF-16 (detected by NUL bytes, because
    decoding arbitrary bytes as UTF-16 almost never *fails*, so it cannot simply be
    "tried") -> UTF-8 -> cp1252 (Excel on Windows) -> latin-1 (never fails; last resort).
    """
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16"), "utf-16"

    candidates: list[str] = []
    sample = raw[:4096]
    if b"\x00" in sample:
        # In UTF-16 text that is mostly ASCII, every other byte is 0. Zeros at odd
        # offsets mean little-endian, at even offsets big-endian.
        little = sample[1::2].count(0) >= sample[0::2].count(0)
        candidates.append("utf-16-le" if little else "utf-16-be")
    candidates += ["utf-8", "cp1252", "latin-1"]

    for encoding in candidates:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    # Unreachable in practice (latin-1 accepts any byte) but keeps the type checker honest.
    raise DataLoadError("Could not decode the file with any supported encoding.")


def read_csv_robust(source: CsvSource) -> pd.DataFrame:
    """Read a CSV from a path, bytes or file-like object, detecting its encoding.

    All cells are read as strings with pandas' NA detection switched off: deciding what
    counts as "missing" is a cleaning decision (config.MISSING_TOKENS), not a parsing one.
    The detected encoding is stored in ``df.attrs["encoding"]``.
    """
    raw = _read_raw(source)
    if isinstance(raw, str):
        text, encoding = raw, "text (already decoded)"
    else:
        text, encoding = _decode(raw)

    if not text.strip():
        raise SchemaError("The file is empty — expected a header row and nutrition data.")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"The file is not a valid CSV: {exc}") from exc

    logger.info("Read %d rows x %d columns (encoding: %s)", len(df), df.shape[1], encoding)
    df.attrs["encoding"] = encoding
    return df


# --------------------------------------------------------------------------------------
# Header mapping
# --------------------------------------------------------------------------------------
_UNNAMED_HEADER = re.compile(r"^(unnamed \d+)?$")


def _normalise_header(header: object) -> str:
    """Reduce a raw header to a lookup key: ' Carb. (g)' -> 'carb', 'Protein (g)' -> 'protein'."""
    key = fold_key(pd.Series([str(header)])).iloc[0]
    key = re.sub(r"[^a-z0-9]+", " ", key).strip()
    words = key.split()
    # Drop a trailing unit token, but only if something remains ("kcal" alone is a name).
    if len(words) > 1 and words[-1] in config.HEADER_UNIT_TOKENS:
        words = words[:-1]
    return " ".join(words)


def _map_columns(columns: list[object]) -> tuple[dict[object, str], list[str]]:
    """Map raw headers to canonical names. Returns (mapping, unknown/ignored headers)."""
    mapping: dict[object, str] = {}
    unknown: list[str] = []
    for col in columns:
        canonical = config.COLUMN_ALIASES.get(_normalise_header(col))
        if canonical is None or canonical in mapping.values():
            # Unknown columns are ignored (not an error) so extra columns in uploads
            # don't break loading. A second column mapping to the same canonical name is
            # ignored too: first one wins.
            unknown.append(str(col))
            continue
        mapping[col] = canonical

    # The provided files leave the item-name header blank (pandas calls it "Unnamed: 0").
    # Use a blank/unnamed FIRST column as the item name only if nothing was named item.
    first = columns[0] if columns else None
    if (
        ITEM_COL not in mapping.values()
        and first is not None
        and _UNNAMED_HEADER.match(_normalise_header(first))
    ):
        mapping[first] = ITEM_COL
        unknown.remove(str(first))
    return mapping, unknown


def validate_schema(df: pd.DataFrame, original_columns: list[object] | None = None) -> None:
    """Raise SchemaError unless ``df`` (canonical names) has an item column and a nutrient.

    The message lists the headers that were found, so a user who uploaded the wrong file
    can see why it was rejected.
    """
    found = ", ".join(repr(str(c)) for c in (original_columns or list(df.columns))) or "none"
    if ITEM_COL not in df.columns:
        raise SchemaError(
            "Could not find the item-name column. Expected an unnamed first column or a "
            f"header such as 'Item' or 'Name'. Columns found: {found}."
        )
    if not any(col in df.columns for col in NUTRIENT_COLUMNS):
        raise SchemaError(
            "No nutrition columns were recognised. Expected headers such as 'Calories', "
            f"'Fat (g)', 'Carb. (g)', 'Fiber (g)', 'Protein', 'Sodium'. Columns found: {found}."
        )


# --------------------------------------------------------------------------------------
# Value cleaning
# --------------------------------------------------------------------------------------
# A number optionally followed by a unit, e.g. "5", "5.5", "5g", "5 mg", "120 kcal".
_NUMBER_WITH_UNIT = r"^([-+]?(?:\d+\.?\d*|\.\d+))\s*(?:kcal|cal|mg|g)?$"


def _coerce_numeric(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Convert raw strings to floats. Returns (numbers, invalid_mask, negative_mask).

    - Missing tokens ("-", "N/A", blank...) -> NaN. Never 0: unknown is not zero.
    - Units are stripped ("5g" -> 5) since the column already defines the unit.
    - Thousands separators are removed ("1,200" -> 1200).
    - Anything else unparseable (e.g. "lots") -> NaN and counted as invalid.
    - Negative values are physically impossible for nutrients -> NaN and counted.
    """
    text = values.astype("object").where(values.notna(), "").astype(str).str.strip()
    is_missing = text.str.lower().isin(config.MISSING_TOKENS)
    text = text.str.replace(r"(?<=\d),(?=\d{3}(?!\d))", "", regex=True)
    number_text = text.str.extract(_NUMBER_WITH_UNIT, flags=re.IGNORECASE)[0]
    numbers = pd.to_numeric(number_text, errors="coerce").astype("float64")

    invalid = numbers.isna() & ~is_missing
    negative = numbers < 0
    return numbers.mask(negative), invalid, negative


def _quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row data-quality columns (vectorised).

    Adds: has_nutrition, energy_consistent, quality_flag (reason text, "" if fine) and
    is_reliable. Suspicious rows are KEPT (they are real menu items) but marked unreliable
    so statistics and rankings can exclude them — e.g. "Nariño 70 Cold Brew with Milk"
    reported as 0 kcal would otherwise rank as the lowest-calorie drink.
    """
    nutrients = df[list(NUTRIENT_COLUMNS)]
    has_nutrition = nutrients.notna().any(axis=1)
    all_zero = has_nutrition & (nutrients.eq(0) | nutrients.isna()).all(axis=1)

    macros = df[list(config.ATWATER_KCAL_PER_G)]
    zero_kcal_with_macros = df["calories"].eq(0) & macros.gt(0).any(axis=1)

    # Energy check only where calories and all three macros are known.
    estimate = sum(df[col] * factor for col, factor in config.ATWATER_KCAL_PER_G.items())
    gap = (df["calories"] - estimate).abs()
    tolerance = np.maximum(
        config.ENERGY_ABS_TOLERANCE_KCAL, config.ENERGY_REL_TOLERANCE * df["calories"]
    )
    checkable = df["calories"].notna() & macros.notna().all(axis=1)
    energy_inconsistent = checkable & gap.gt(tolerance)

    energy_reason = (
        "calories ("
        + df["calories"].round(0).astype("Int64").astype(str)
        + " kcal) far from 4/4/9 macro estimate ("
        + estimate.round(0).astype("Int64").astype(str)
        + " kcal)"
    )
    flag = np.select(
        [all_zero, zero_kcal_with_macros, energy_inconsistent],
        ["all nutrients reported as 0", "0 kcal but non-zero macronutrients", energy_reason],
        default="",
    )
    return df.assign(
        has_nutrition=has_nutrition,
        energy_consistent=checkable & ~energy_inconsistent,
        quality_flag=flag,
        is_reliable=flag == "",
    )


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Rows as dicts with NaN -> None, so reports serialise cleanly to JSON."""
    out = df[columns].to_dict("records")
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in out
    ]


def _deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Keep one row per item_key (within a single category). Returns (kept, drops, n_exact).

    Rule 1: exact duplicates (same key + same values) -> keep first occurrence.
    Rule 2: same key, one row all-missing, other has data -> keep the row with data.
    Rule 3: same key, conflicting values -> rank by (most non-null nutrients, passes the
            energy check, not all zeros, earliest row) and keep the best.
    Rules 2 and 3 are one vectorised sort: a row with data always has more non-null
    values than an all-missing row, so rule 2 falls out of rule 3's first criterion.
    Every dropped row is returned with the rule and a human-readable reason.
    """
    nutrients = list(NUTRIENT_COLUMNS)
    value_cols = [ITEM_COL, *nutrients]

    # --- Rule 1: exact duplicates (compared after cleaning, so "5" == "5g") ---
    subset = [ITEM_KEY_COL, *nutrients]
    exact = df.duplicated(subset=subset, keep="first")
    first_line = df.groupby(subset, dropna=False, sort=False)[SOURCE_LINE_COL].transform("first")
    exact_drops = df[exact].assign(
        rule="1_exact_duplicate",
        kept_line=first_line[exact],
        reason="identical to line " + first_line[exact].astype(str) + " (kept first occurrence)",
    )
    remaining = df[~exact]

    # --- Rules 2 & 3: rank the remaining rows sharing a key ---
    ranked = remaining.assign(
        _non_null=remaining[nutrients].notna().sum(axis=1),
        _non_zero=remaining["has_nutrition"]
        & remaining["quality_flag"].ne("all nutrients reported as 0"),
    ).sort_values(
        ["_non_null", "energy_consistent", "_non_zero", SOURCE_ROW_COL],
        ascending=[False, False, False, True],
        kind="stable",
    )
    loser = ranked.duplicated(ITEM_KEY_COL, keep="first")
    winners = ranked[~loser]
    rank_cols = ["_non_null", "energy_consistent", "_non_zero", "has_nutrition"]
    losers = ranked[loser].merge(
        winners[[ITEM_KEY_COL, SOURCE_LINE_COL, *rank_cols]].rename(
            columns={SOURCE_LINE_COL: "kept_line", **{c: f"{c}_kept" for c in rank_cols}}
        ),
        on=ITEM_KEY_COL,
        how="left",
    )
    kept_line = losers["kept_line"].astype(str)
    is_rule2 = ~losers["has_nutrition"] & losers["has_nutrition_kept"]
    losers = losers.assign(
        rule=np.where(is_rule2, "2_missing_vs_data", "3_conflicting_values"),
        reason=np.select(
            [
                is_rule2,
                losers["_non_null_kept"] > losers["_non_null"],
                losers["energy_consistent_kept"] & ~losers["energy_consistent"],
                losers["_non_zero_kept"] & ~losers["_non_zero"],
            ],
            [
                "all values missing; line " + kept_line + " has nutrition data",
                "fewer non-null nutrients than line " + kept_line,
                "fails the 4/4/9 energy check; line " + kept_line + " passes",
                "all values reported as 0; line " + kept_line + " has non-zero values",
            ],
            default="tied with line " + kept_line + " on all criteria; kept first occurrence",
        ),
    )

    drops = pd.concat([exact_drops, losers], ignore_index=True).sort_values(SOURCE_ROW_COL)
    records = _records(
        drops, [SOURCE_ROW_COL, SOURCE_LINE_COL, *value_cols, "rule", "reason", "kept_line"]
    )
    for rec in records:
        logger.info(
            "Dropped line %s (%s): %s — %s",
            rec[SOURCE_LINE_COL],
            rec[ITEM_COL],
            rec["rule"],
            rec["reason"],
        )
    kept = winners.drop(columns=["_non_null", "_non_zero"]).sort_values(SOURCE_ROW_COL)
    return kept, records, int(exact.sum())


# --------------------------------------------------------------------------------------
# Public cleaning API
# --------------------------------------------------------------------------------------
OUTPUT_COLUMNS: list[str] = [
    CATEGORY_COL,
    ITEM_COL,
    ITEM_KEY_COL,
    *NUTRIENT_COLUMNS,
    "has_nutrition",
    "is_reliable",
    "quality_flag",
    SOURCE_ROW_COL,
    SOURCE_LINE_COL,
]


def clean(df: pd.DataFrame, category: str) -> tuple[pd.DataFrame, QualityReport]:
    """Turn a raw CSV frame into the canonical schema for one category.

    Returns one row per item (see _deduplicate) with columns OUTPUT_COLUMNS, plus a
    QualityReport describing every decision taken.
    """
    category = category.strip().lower()
    report = QualityReport(
        category=category,
        encoding_detected=str(df.attrs.get("encoding", "unknown")),
        rows_read=len(df),
    )

    # 1. Headers -> canonical names; unknown columns are reported and ignored.
    original_columns = list(df.columns)
    mapping, unknown = _map_columns(original_columns)
    report.unknown_columns = unknown
    if unknown:
        logger.info("[%s] ignoring unrecognised columns: %s", category, unknown)
    out = df[list(mapping)].rename(columns=mapping)
    validate_schema(out, original_columns)

    # 2. Keep the original position so reports can point back at the source file.
    out = out.assign(**{SOURCE_ROW_COL: np.arange(len(out))})
    out[SOURCE_LINE_COL] = out[SOURCE_ROW_COL] + 2  # +1 for 1-based, +1 for the header

    # 3. Item names: tidy whitespace for display (keep ™/®), plus a folded search key.
    out[ITEM_COL] = clean_display_text(out[ITEM_COL])
    out[ITEM_KEY_COL] = fold_key(out[ITEM_COL])
    nameless = out[ITEM_COL].eq("")
    report.rows_without_name = int(nameless.sum())
    if nameless.any():
        # A row without a name can't be shown, searched or compared; drop and report it.
        logger.warning("[%s] dropping %d rows with no item name", category, nameless.sum())
    out = out[~nameless]

    # 4. Nutrients -> floats. Columns absent from this file are added as NaN so both
    #    categories share one schema (e.g. food has no sodium_mg) — never filled with 0.
    for col in NUTRIENT_COLUMNS:
        if col not in out.columns:
            report.columns_unavailable.append(col)
            out[col] = np.nan
            continue
        numbers, invalid, negative = _coerce_numeric(out[col])
        out[col] = numbers
        if invalid.any() or negative.any():
            report.invalid_values[col] = int(invalid.sum() + negative.sum())
            logger.warning(
                "[%s] %s: %d unparseable and %d negative values set to NaN",
                category,
                col,
                invalid.sum(),
                negative.sum(),
            )

    # 5. Quality flags, then de-duplication (which uses them to rank conflicting rows).
    out = _quality_flags(out.assign(**{CATEGORY_COL: category}))
    out, drops, n_exact = _deduplicate(out)
    report.exact_duplicates_removed = n_exact
    report.conflicts_resolved = drops

    out = out[OUTPUT_COLUMNS].reset_index(drop=True)
    report.rows_final = len(out)
    report.rows_without_data = int((~out["has_nutrition"]).sum())
    report.missing_counts = {
        col: int(out[col].isna().sum())
        for col in NUTRIENT_COLUMNS
        if col not in report.columns_unavailable
    }
    report.suspicious = _records(
        out[~out["is_reliable"]],
        [SOURCE_ROW_COL, SOURCE_LINE_COL, ITEM_COL, *NUTRIENT_COLUMNS, "quality_flag"],
    )
    for rec in report.suspicious:
        logger.warning("[%s] suspicious: %s — %s", category, rec[ITEM_COL], rec["quality_flag"])
    return out, report


def load_menu(
    drinks_src: CsvSource | None = config.DEFAULT_DRINKS_PATH,
    food_src: CsvSource | None = config.DEFAULT_FOOD_PATH,
) -> tuple[pd.DataFrame, dict[str, QualityReport]]:
    """Load, clean and combine both menus.

    Either source may be None (e.g. the user uploaded only one file). Rows are never
    de-duplicated across categories: a drink and a food item may share a name.
    The combined frame also carries caffeine_status/caffeine_basis (see filters).
    """
    sources = {"drink": drinks_src, "food": food_src}
    if all(src is None for src in sources.values()):
        raise ValueError("At least one of drinks_src / food_src must be provided.")

    frames: list[pd.DataFrame] = []
    reports: dict[str, QualityReport] = {}
    for category, src in sources.items():
        if src is None:
            continue
        try:
            cleaned, report = clean(read_csv_robust(src), category)
        except DataLoadError as exc:
            # Say WHICH file failed: with two uploads the user needs to know which to fix.
            raise type(exc)(f"{category.capitalize()} file: {exc}") from exc
        frames.append(cleaned)
        reports[category] = report

    combined = pd.concat(frames, ignore_index=True)
    return add_caffeine_status(combined), reports
