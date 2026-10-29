"""Tab 4: what cleaning found and changed in each file, and where data is missing."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from starbucks_nutrition import analysis, viz
from starbucks_nutrition.loader import QualityReport
from starbucks_nutrition.ui.common import caveats_box, chart, how_to_read
from starbucks_nutrition.ui.data_source import CATEGORIES, MenuData, source_label

EXACT = "1_exact_duplicate"
RULE_LABELS = {
    EXACT: "Exact duplicate",
    "2_missing_vs_data": "Duplicate without data (a copy with data was kept)",
    "3_conflicting_values": "Conflicting values (most plausible copy kept)",
}
_DROP_COLUMNS = {
    "source_line": "Line",
    "item": "Item",
    "rule": "Rule",
    "reason": "Reason",
    "kept_line": "Kept line",
}


def _drops_table(drops: list[dict]) -> pd.DataFrame:
    """Records -> display frame (selection and renaming only, no calculations)."""
    table = pd.DataFrame(drops, columns=list(_DROP_COLUMNS))
    table["rule"] = table["rule"].map(RULE_LABELS).fillna(table["rule"])
    return table.rename(columns=_DROP_COLUMNS)


def _report_card(category: str, report: QualityReport, source: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {CATEGORIES.get(category, category)}")
        st.caption(
            f"Source: **{source_label(source)}** · encoding detected: `{report.encoding_detected}`"
        )
        tiles = st.columns(3)
        tiles[0].metric("Rows read → kept", f"{report.rows_read} → {report.rows_final}")
        tiles[1].metric("Exact duplicates removed", report.exact_duplicates_removed)
        tiles[2].metric("Rows without data (kept)", report.rows_without_data)

        exact = [d for d in report.conflicts_resolved if d["rule"] == EXACT]
        other = [d for d in report.conflicts_resolved if d["rule"] != EXACT]
        facts = [
            f"Other duplicate rows resolved: **{len(other)}**",
            f"Rows without an item name dropped: **{report.rows_without_name}**",
            "Values that were not numbers or negative (set to empty): "
            + (
                ", ".join(
                    f"{analysis.metric_label(c)} {n}" for c, n in report.invalid_values.items()
                )
                if report.invalid_values
                else "**none**"
            ),
            "Values given as '<x' (stored as half of x, e.g. <1 -> 0.5): "
            + (
                ", ".join(
                    f"{analysis.metric_label(c)} {n}" for c, n in report.approximate_values.items()
                )
                if report.approximate_values
                else "**none**"
            ),
            "Columns not in this file: "
            + (", ".join(analysis.metric_label(c) for c in report.columns_unavailable) or "none"),
        ]
        if report.unknown_columns:
            facts.append(f"Columns ignored (not recognised): {', '.join(report.unknown_columns)}")
        st.markdown("\n".join(f"- {f}" for f in facts))

        if other:
            st.markdown("**Duplicates resolved by a rule**")
            st.dataframe(_drops_table(other), hide_index=True)
        if exact:
            with st.expander(f"{len(exact)} exact duplicate rows removed"):
                st.dataframe(_drops_table(exact), hide_index=True)
        if report.suspicious:
            st.markdown("**Kept but marked unreliable** (excluded from statistics by default)")
            suspicious = pd.DataFrame(report.suspicious)
            st.dataframe(
                suspicious[["source_line", "item", "quality_flag"]].rename(
                    columns={"source_line": "Line", "item": "Item", "quality_flag": "Why"}
                ),
                hide_index=True,
            )


def render(data: MenuData, include_unreliable: bool) -> None:
    st.markdown(
        "Every cleaning decision is listed here. The raw files are never modified; "
        "line numbers refer to the original CSV (line 1 = header)."
    )
    columns = st.columns(len(data.reports))
    for column, (category, report) in zip(columns, data.reports.items(), strict=True):
        with column:
            _report_card(category, report, data.sources[category])

    st.subheader("Which nutrients each file has")
    matrix = analysis.availability_matrix(data.menu)
    matrix.index = [analysis.metric_label(m, with_unit=False) for m in matrix.index]
    st.dataframe(
        matrix,
        column_config={
            c: st.column_config.NumberColumn(f"{CATEGORIES.get(c, c)}: items with a value")
            for c in data.reports
        },
    )
    st.caption("0 = the column is not in that file; those metrics are reported as unavailable.")

    with st.container(border=True):
        chart(viz.data_coverage_chart(data.menu, "drink"), key="chart_coverage")
        how_to_read(
            "Grey = drinks listed without any nutrition data. The gaps are concentrated in "
            "some families, so drink averages describe the remaining items only."
        )
        caveats_box(
            analysis.missing_data_caveats(data.menu, "drink", include_unreliable),
            title="What the missing drink data means",
        )
