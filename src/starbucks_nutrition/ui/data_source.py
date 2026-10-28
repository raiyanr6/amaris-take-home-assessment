"""Sidebar data source: bundled sample files or user uploads, loaded once per content.

Each category falls back to the sample file on its own, so a user can upload just their
drinks CSV and still compare against the sample food. The source of every category
("sample" or "uploaded") travels with the data so each tab can label it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from starbucks_nutrition import config
from starbucks_nutrition.loader import DataLoadError, QualityReport, load_menu

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 5
SAMPLE = "sample"
CATEGORIES: dict[str, str] = {"drink": "Drinks", "food": "Food"}
SAMPLE_PATHS: dict[str, Path] = {
    "drink": config.DEFAULT_DRINKS_PATH,
    "food": config.DEFAULT_FOOD_PATH,
}
EXPECTED_FORMAT = (
    "Expected a CSV with one row per menu item: an item-name column (an unnamed first "
    "column, or a header such as *Item* or *Name*) plus any of **Calories, Fat (g), "
    "Carb. (g), Fiber (g), Protein (g), Sodium, Sugars (g), Caffeine (mg)**. Missing "
    "values may be blank or `-`. UTF-8 or UTF-16 encoding."
)


@dataclass(frozen=True)
class MenuData:
    """The loaded menu plus where each category came from."""

    menu: pd.DataFrame
    reports: dict[str, QualityReport]
    sources: dict[str, str]  # category -> "sample" | "uploaded: <file name>"


@st.cache_data(show_spinner="Loading and cleaning the menu…", max_entries=16)
def load_cached(drinks: bytes, food: bytes) -> tuple[pd.DataFrame, dict[str, QualityReport]]:
    """Parse + clean once per distinct file content.

    st.cache_data hashes the bytes, so the cache key is the content itself: reruns
    (every widget change) skip parsing, and re-uploading an edited file with the same
    name still reloads. The returned objects are copies, so callers cannot mutate the
    cached menu.
    """
    return load_menu(drinks, food)


@st.cache_data(show_spinner=False)
def _sample_bytes(category: str) -> bytes:
    return SAMPLE_PATHS[category].read_bytes()  # read-only: never written back


def source_label(source: str) -> str:
    return "sample data" if source == SAMPLE else source


def _uploaded(category: str) -> tuple[bytes, str] | None:
    """Read one uploader; returns (bytes, label) or None (falls back to sample)."""
    upload = st.sidebar.file_uploader(
        f"{CATEGORIES[category]} CSV",
        type=["csv", "txt"],
        key=f"upload_{category}",
        help=f"Optional. Without it, the sample {CATEGORIES[category].lower()} file is used.",
    )
    if upload is None:
        return None
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.sidebar.error(
            f"{upload.name} is {upload.size / 1024 / 1024:.1f} MB; the limit is "
            f"{MAX_UPLOAD_MB} MB. Using the sample {CATEGORIES[category].lower()} file."
        )
        return None
    return upload.getvalue(), f"uploaded: {upload.name}"


def sidebar_data_source() -> MenuData | None:
    """Render the data-source controls and return the loaded menu (None on error).

    Errors are shown as a friendly message (what was expected, which file failed),
    never as a stack trace.
    """
    st.sidebar.header("Data source")
    mode = st.sidebar.radio(
        "Menu data",
        ["Sample data", "Upload my CSVs"],
        key="source_mode",
        help="The sample is the provided Starbucks drinks and food files.",
    )
    files = {cat: _sample_bytes(cat) for cat in CATEGORIES}
    sources = dict.fromkeys(CATEGORIES, SAMPLE)
    if mode == "Upload my CSVs":
        with st.sidebar.expander("Expected format"):
            st.markdown(EXPECTED_FORMAT)
        for category in CATEGORIES:
            uploaded = _uploaded(category)
            if uploaded is not None:
                files[category], sources[category] = uploaded

    st.sidebar.markdown(
        "  \n".join(f"**{CATEGORIES[cat]}:** {source_label(src)}" for cat, src in sources.items())
    )

    try:
        menu, reports = load_cached(files["drink"], files["food"])
    except DataLoadError as exc:
        st.error(f"**Could not load the data.** {exc}\n\n{EXPECTED_FORMAT}")
        return None
    except Exception:  # anything unexpected: log it for us, keep the page friendly
        logger.exception("Unexpected error while loading uploaded files")
        st.error(
            "**Could not read the uploaded file.** It does not look like a CSV this tool "
            f"understands.\n\n{EXPECTED_FORMAT}"
        )
        return None
    return MenuData(menu=menu, reports=reports, sources=sources)


def sidebar_include_unreliable(data: MenuData) -> bool:
    """The one global toggle for rows flagged unreliable (e.g. all nutrients 0)."""
    n_flagged = sum(len(report.suspicious) for report in data.reports.values())
    return st.sidebar.toggle(
        "Include unreliable rows",
        value=False,
        key="include_unreliable",
        help=(
            f"{n_flagged} row(s) are flagged as implausible (for example every nutrient "
            "reported as 0 for a milk drink). They are excluded from statistics and "
            "charts by default; tables always list them with Reliable = false."
        ),
    )
