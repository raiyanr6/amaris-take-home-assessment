"""Sanitising untrusted text before it reaches the model.

Item names come from CSVs, and uploaded CSVs are untrusted: a "name" could carry
instructions (indirect prompt injection), control characters or a huge payload. The
system prompts tell the model that data is never instructions; this module additionally
strips control characters and truncates names so they stay plain, short labels.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from starbucks_nutrition import config

_WHITESPACE = re.compile(r"\s+")


def safe_text(value: object, max_chars: int = config.LLM_MAX_NAME_CHARS) -> str:
    """Control/format characters -> space, whitespace collapsed, truncated with "…"."""
    text = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in str(value)
    )
    text = _WHITESPACE.sub(" ", text).strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def json_number(value: Any, digits: int = 1) -> float | int | None:
    """Round for the model; NaN/inf -> None; whole numbers become ints ("430" not "430.0")."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    rounded = round(number, digits)
    return int(rounded) if rounded == int(rounded) else rounded
