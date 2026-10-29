"""A cheap hallucination guard: is every figure in the answer present in the inputs?

verify_numbers extracts the numbers from the model's text and checks each one against
the numbers in its sources (the facts JSON, or every tool result in a Q&A exchange),
allowing for rounding. It cannot prove an answer right (a real number in the wrong
sentence passes), but it reliably catches invented or self-calculated figures.

Noise removed before extraction, so it is not mistaken for a claim:
- item names, which can contain digits ("Nariño 70 Cold Brew", "8-Grain Roll",
  "Bantam Bagels (2 Pack)"), taken from the sources;
- markdown list markers ("1. ") and headings, "top 5", "per 100 kcal";
- integers 0-3 (counts like "both", "2 categories" are not worth flagging).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from starbucks_nutrition.llm.context import item_names

SMALL_INT_MAX = 3
REL_TOLERANCE = 0.005  # 0.5% for numbers quoted with less precision than the source

_NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")
_NOISE = (
    re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE),  # list markers
    re.compile(r"^\s*#+\s.*$", re.MULTILINE),  # markdown headings
    re.compile(r"\btop[\s-]+\d+\b", re.IGNORECASE),  # "top 5"
    re.compile(r"\b(?:per|/)\s*100\s*kcal\b", re.IGNORECASE),  # unit "g per 100 kcal"
)


@dataclass(frozen=True)
class GroundingResult:
    verified: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    checked: int = field(default=0)

    @property
    def ok(self) -> bool:
        return not self.unverified


# Models restyle names: "8‑Grain Roll" with a non-breaking hyphen (U+2011), curly
# quotes, no-break spaces, dropped ™/®. Both the answer and the names are normalised the
# same way before matching, so the digit in "8-Grain" is never read as a claim.
# Code points, so the invisible characters are readable in the source.
_DASHES = "".join(map(chr, (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)))
_SPACES = "".join(map(chr, (0x00A0, 0x2007, 0x2009, 0x202F)))  # no-break / thin spaces
_TYPOGRAPHY = str.maketrans(
    {
        **dict.fromkeys(_DASHES, "-"),
        **dict.fromkeys(_SPACES, " "),
        chr(0x2018): "'",
        chr(0x2019): "'",
        chr(0x201C): '"',
        chr(0x201D): '"',
        **dict.fromkeys("™®©℠", ""),
    }
)


def _normalise(text: str) -> str:
    return text.translate(_TYPOGRAPHY)


def _strip_names(text: str, names: Iterable[str]) -> str:
    text = _normalise(text)
    # Longest first, so "Nariño 70 Cold Brew with Milk" goes before "Nariño 70 Cold Brew".
    for name in sorted({_normalise(n).strip() for n in names if n}, key=len, reverse=True):
        if name:
            text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return text


def extract_numbers(text: str, names: Iterable[str] = ()) -> list[str]:
    """Numbers (as written) that count as claims, after removing names and noise."""
    cleaned = _strip_names(text, names)
    for pattern in _NOISE:
        cleaned = pattern.sub(" ", cleaned)
    return [m.group(0) for m in _NUMBER.finditer(cleaned)]


def _source_numbers(node: Any, names: set[str]) -> list[float]:
    """All numbers in a JSON-like structure, incl. numbers written inside strings."""
    found: list[float] = []
    if isinstance(node, bool) or node is None:
        return found
    if isinstance(node, int | float):
        found.append(float(node))
    elif isinstance(node, str):
        found += [_to_float(n) for n in extract_numbers(node, names)]
    elif isinstance(node, Mapping):
        for value in node.values():
            found += _source_numbers(value, names)
    elif isinstance(node, list | tuple):
        for value in node:
            found += _source_numbers(value, names)
    return found


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def _matches(claim: str, sources: list[float]) -> bool:
    value = _to_float(claim)
    decimals = len(claim.split(".")[1]) if "." in claim else 0
    half_step = 0.5 * 10**-decimals + 1e-9
    for source in sources:
        # Rounded to the precision the answer used ("141" for 140.62, "0.6" for 0.56)...
        if abs(value - source) <= half_step:
            return True
        # ...or a close match to a source given with less precision than the claim.
        if source and abs(value - source) <= abs(source) * REL_TOLERANCE:
            return True
        # Percentages may be written as "-15%" or "15% lower" for a pct_diff of -15.3.
        if abs(abs(value) - abs(source)) <= half_step:
            return True
    return False


def verify_numbers(answer: str, *sources: Any) -> GroundingResult:
    """Check each figure in ``answer`` against the numbers in ``sources``."""
    names = item_names(*sources)
    pool = [n for s in sources for n in _source_numbers(s, names)]
    verified: list[str] = []
    unverified: list[str] = []
    for claim in extract_numbers(answer, names):
        value = _to_float(claim)
        if value == int(value) and 0 <= value <= SMALL_INT_MAX and "." not in claim:
            continue
        (verified if _matches(claim, pool) else unverified).append(claim)
    return GroundingResult(tuple(verified), tuple(unverified), len(verified) + len(unverified))
