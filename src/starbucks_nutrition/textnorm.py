"""Text normalisation shared by the loader (item names, headers) and filters (search).

Kept separate so `loader` and `filters` can both use it without importing each other.
"""

from __future__ import annotations

import pandas as pd

# Trademark-style symbols are dropped from search keys. They must be removed BEFORE
# Unicode NFKD folding, because NFKD expands "™" to the letters "TM", which would turn
# "Refreshers™" into "refresherstm" and break keyword matching.
_SYMBOLS_PATTERN = r"[™®©℠]"
# Typographic punctuation that NFKD does not fold to ASCII, mapped to the ASCII form so
# "Devil’s" and "Devil's" produce the same key.
_PUNCT_TRANSLATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def clean_display_text(values: pd.Series) -> pd.Series:
    """Tidy names for display while keeping ™/®/accents (they are part of the brand name).

    Collapses any run of whitespace (incl. non-breaking spaces and tabs) to one space and
    strips the ends, and applies NFC so visually identical names compare equal.
    """
    text = values.astype("object").where(values.notna(), "").astype(str)
    return text.str.normalize("NFC").str.replace(r"\s+", " ", regex=True).str.strip()


def fold_key(values: pd.Series) -> pd.Series:
    """Build a lowercase, ASCII-only key for matching and de-duplication.

    "Ombré Pink Drink" -> "ombre pink drink", "Refreshers™" -> "refreshers". Used for
    case/accent-insensitive search and so that names differing only in spacing, case or
    accents are recognised as the same item.
    """
    text = clean_display_text(values)
    return (
        text.str.translate(_PUNCT_TRANSLATION)
        .str.replace(_SYMBOLS_PATTERN, "", regex=True)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def fold_text(text: str) -> str:
    """Scalar version of :func:`fold_key`, for user-typed search strings."""
    return str(fold_key(pd.Series([text])).iloc[0])
