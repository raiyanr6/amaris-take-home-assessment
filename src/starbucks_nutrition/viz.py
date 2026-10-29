"""Plotly charts: DataFrame in, Figure out.

No Streamlit imports here, so every chart is testable and reusable (CLI, deck exports).
Charts take the full menu table and get their numbers from `analysis`, so they apply the
same row rules as the statistics: rows without data never appear in numeric charts and
unreliable rows are excluded unless ``include_unreliable=True``.

Design rules applied throughout:
- One fixed colour per category (drink = blue, food = orange) in every chart. The pair
  is colour-blind safe (validated: CVD ΔE 24.7, both >= 3:1 contrast on the surface).
- Units in every axis label; titles phrase the question the chart answers.
- Empty input never raises: it returns a figure saying "No data for this selection".
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from starbucks_nutrition import analysis
from starbucks_nutrition.config import CATEGORY_COL, ITEM_COL

CATEGORY_COLORS: dict[str, str] = {"drink": "#2a78d6", "food": "#eb6834"}
OTHER_CATEGORY_COLOR = "#6250d6"  # an uploaded file could only add drink/food, but be safe
# Macros are a different dimension from category, so they get hues of their own (never
# the drink/food colours). Their segments carry % labels, which the validator requires
# for the two lighter hues (< 3:1 contrast on white).
MACRO_COLORS: dict[str, str] = {"fat_g": "#eda100", "carb_g": "#1baf7a", "protein_g": "#6250d6"}
NO_DATA_COLOR = "#c3c2b7"  # neutral grey: "without data" is an absence, not a series

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
NO_DATA_MESSAGE = "No data for this selection"


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, OTHER_CATEGORY_COLOR)


def _title(question: str, subtitle: str = "") -> dict[str, object]:
    text = (
        f"{question}<br><sup style='color:{INK_SECONDARY}'>{subtitle}</sup>"
        if subtitle
        else question
    )
    return {"text": text, "x": 0, "xanchor": "left", "font": {"size": 17, "color": INK}}


def _base_layout(fig: go.Figure, question: str, subtitle: str = "") -> go.Figure:
    """Shared look: recessive grid, legend on top, rounded bar ends, room for a subtitle."""
    fig.update_layout(
        title=_title(question, subtitle),
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": INK},
        margin={"l": 10, "r": 10, "t": 90 if subtitle else 60, "b": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
            "traceorder": "normal",
        },
        hoverlabel={"bgcolor": "white", "font_color": INK},
        barcornerradius=4,
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=INK_SECONDARY, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=INK_SECONDARY, automargin=True)
    return fig


def empty_figure(question: str = "", message: str = NO_DATA_MESSAGE) -> go.Figure:
    """A blank chart carrying a message, so the UI never shows an unexplained void."""
    fig = _base_layout(go.Figure(), question)
    fig.add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font={"size": 15, "color": INK_MUTED},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _is_empty(df: pd.DataFrame) -> bool:
    return df.empty or "has_nutrition" not in df.columns


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def compare_means_chart(
    df: pd.DataFrame,
    include_unreliable: bool = False,
    a: str = "drink",
    b: str = "food",
) -> go.Figure:
    """Grouped bars of mean calories and macros, drinks vs food.

    Calories (kcal) and grams sit in two panels because one shared y-axis would flatten
    the gram bars. Hover shows what each mean is based on (n analysed / n total) and the
    subtitle states the drink coverage, because missing drink data is not random.
    """
    question = "How do drinks and food compare on average?"
    if _is_empty(df):
        return empty_figure(question)
    table = analysis.compare_categories(df, a, b, include_unreliable)
    table = table[table.index.isin(analysis.NUTRIENT_COLUMNS)]
    if table.empty:
        return empty_figure(question, "No nutrient is available for both categories")

    panels = (
        [m for m in table.index if m == "calories"],
        [m for m in table.index if m != "calories"],
    )
    panels = [p for p in panels if p]
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        column_widths=[0.25, 0.75][: len(panels)] if len(panels) == 2 else None,
        horizontal_spacing=0.08,
    )
    for col, metrics in enumerate(panels, start=1):
        for cat in (a, b):
            rows = table.loc[metrics]
            hover = [
                f"<b>{cat.capitalize()}</b> · {analysis.metric_label(m, with_unit=False)}<br>"
                f"Mean: {mean:,.1f} {unit}<br>"
                f"Based on {n_an} of {n_tot} items ({cov:.0f}% have data)"
                for m, mean, unit, n_an, n_tot, cov in zip(
                    metrics,
                    rows[f"mean_{cat}"],
                    rows["unit"],
                    rows[f"n_analysed_{cat}"],
                    rows[f"n_total_{cat}"],
                    rows[f"coverage_pct_{cat}"],
                    strict=True,
                )
            ]
            fig.add_trace(
                go.Bar(
                    name=cat.capitalize(),
                    legendgroup=cat,
                    showlegend=col == 1,
                    x=[analysis.metric_label(m, with_unit=False) for m in metrics],
                    y=rows[f"mean_{cat}"],
                    marker_color=category_color(cat),
                    hovertext=hover,
                    hoverinfo="text",
                    text=rows[f"mean_{cat}"].map(
                        "{:,.0f}".format if col == 1 else "{:,.1f}".format
                    ),
                    textposition="outside",
                    cliponaxis=False,
                    textfont={"color": INK_SECONDARY, "size": 11},
                ),
                row=1,
                col=col,
            )
        unit = "kcal" if metrics == ["calories"] else "g"
        fig.update_yaxes(title_text=f"Mean per item ({unit})", row=1, col=col)

    # Category-level coverage from the shared definition (not the per-metric minimum,
    # which dropped whenever one nutrient had a blank cell).
    notes = [
        analysis.coverage_sentence(df, cat, include_unreliable)
        for cat in (a, b)
        if (s := analysis.coverage_summary(df, cat, include_unreliable))["coverage_pct"] < 100
        or s["flagged_unreliable"]
    ]
    subtitle = " ".join(["Means over items with data.", *notes, "See caveats." if notes else ""])
    _base_layout(fig, question, subtitle)
    fig.update_layout(barmode="group")
    return fig


def top_n_chart(
    df: pd.DataFrame,
    metric: str = "calories",
    n: int = 10,
    category: str | None = "drink",
    ascending: bool = False,
    include_unreliable: bool = False,
) -> go.Figure:
    """Horizontal bars of the N highest (or lowest) items for a metric.

    Horizontal because item names are long. The first-ranked item is drawn at the top.
    """
    label = analysis.metric_label(metric, with_unit=False).lower()
    scope = f"{category} items" if category else "items"
    question = f"Which {scope} are {'lowest' if ascending else 'highest'} in {label}?"
    if _is_empty(df):
        return empty_figure(question)
    ranked = analysis.top_n(df, metric, n, category, ascending, include_unreliable)
    if ranked.empty:
        return empty_figure(question)
    unit = analysis.METRIC_UNITS.get(metric, "")

    ranked = ranked.iloc[::-1]  # plotly draws the first y category at the bottom
    value_format = "{:,.0f}".format if unit in ("kcal", "mg") else "{:,.1f}".format
    fig = go.Figure()
    for cat, rows in ranked.groupby(CATEGORY_COL, sort=False):
        fig.add_trace(
            go.Bar(
                name=str(cat).capitalize(),
                orientation="h",
                x=rows[metric],
                y=rows[ITEM_COL],
                marker_color=category_color(str(cat)),
                text=rows[metric].map(value_format),
                textposition="outside",
                cliponaxis=False,
                textfont={"color": INK_SECONDARY, "size": 11},
                hovertemplate=f"<b>%{{y}}</b><br>%{{x:,.1f}} {unit}<extra></extra>",
            )
        )
    _base_layout(fig, question, f"Top {len(ranked)} by {analysis.metric_label(metric)}")
    fig.update_layout(showlegend=category is None, height=max(320, 40 + 32 * len(ranked)))
    fig.update_xaxes(title_text=analysis.metric_label(metric))
    fig.update_yaxes(categoryorder="array", categoryarray=list(ranked[ITEM_COL]))
    return fig


def macro_energy_chart(df: pd.DataFrame, include_unreliable: bool = False) -> go.Figure:
    """Share of energy from fat, carbs and protein per category, as 100% stacked bars.

    The brief suggests a pie chart; a 100% stacked bar shows the same part-to-whole split
    but puts drinks and food on one shared scale side by side, and lengths along a common
    baseline are easier to compare than pie angles. Shares use Atwater factors (fat 9,
    carbs and protein 4 kcal/g) over items that report all three macros.
    """
    question = "Where do the calories come from?"
    if _is_empty(df):
        return empty_figure(question)
    shares = analysis.macro_energy_share(df, include_unreliable)
    if shares.empty:
        return empty_figure(question)

    fig = go.Figure()
    for macro in analysis.MACROS:
        rows = shares[shares["macro"] == macro]
        name = analysis.metric_label(macro, with_unit=False)
        fig.add_trace(
            go.Bar(
                name=name,
                orientation="h",
                y=[
                    f"{c.capitalize()} (n={n})"
                    for c, n in zip(rows[CATEGORY_COL], rows["n_items"], strict=True)
                ],
                x=rows["pct"],
                marker={"color": MACRO_COLORS[macro], "line": {"color": SURFACE, "width": 2}},
                text=rows["pct"].map("{:.0f}%".format),
                textposition="inside",
                textangle=0,
                insidetextanchor="middle",
                textfont={"color": INK if macro == "fat_g" else "white"},
                customdata=rows["kcal"],
                hovertemplate=f"<b>%{{y}}</b> · {name}<br>%{{x:.1f}}% of macro energy"
                "<br>%{customdata:,.0f} kcal summed<extra></extra>",
            )
        )
    _base_layout(
        fig, question, "Share of energy from each macronutrient (fat 9, carbs/protein 4 kcal per g)"
    )
    fig.update_layout(barmode="stack", barcornerradius=0, height=300)
    fig.update_xaxes(title_text="Share of macro energy (%)", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return fig


def calories_vs_protein_scatter(df: pd.DataFrame, include_unreliable: bool = False) -> go.Figure:
    """One dot per item: calories vs protein, coloured by category, name on hover.

    Items towards the upper left give more protein for their calories.
    """
    question = "Which items give the most protein for their calories?"
    if _is_empty(df):
        return empty_figure(question)
    data = analysis.analysable(df, include_unreliable)
    data = data[data["calories"].notna() & data["protein_g"].notna()]
    if data.empty:
        return empty_figure(question)

    fig = go.Figure()
    for cat, rows in data.groupby(CATEGORY_COL, sort=True):
        fig.add_trace(
            go.Scatter(
                name=f"{str(cat).capitalize()} (n={len(rows)})",
                mode="markers",
                x=rows["calories"],
                y=rows["protein_g"],
                text=rows[ITEM_COL],
                marker={
                    "color": category_color(str(cat)),
                    "size": 9,
                    "opacity": 0.85,
                    "line": {"color": SURFACE, "width": 1.5},
                },
                hovertemplate=(
                    "<b>%{text}</b><br>%{x:,.0f} kcal · %{y:,.1f} g protein<extra></extra>"
                ),
            )
        )
    _base_layout(
        fig, question, "Each dot is one item with data; upper left = more protein per kcal"
    )
    fig.update_xaxes(title_text=analysis.metric_label("calories"), rangemode="tozero")
    fig.update_yaxes(title_text=analysis.metric_label("protein_g"), rangemode="tozero")
    return fig


CAFFEINE_PROXY_NOTE = (
    "Name-based proxy: the data has no caffeine column, so status is inferred from the "
    "drink name (e.g. espresso → likely, herbal/crème → unlikely, unspecified tea → unknown)."
)


def caffeine_status_chart(
    df: pd.DataFrame, category: str = "drink", include_unreliable: bool = False
) -> go.Figure:
    """Item counts per caffeine status, annotated as a name-based proxy.

    All items count (with or without nutrition data), since status comes from the name.
    """
    question = f"How many {category} items likely contain caffeine?"
    if df.empty or "is_reliable" not in df.columns:
        return empty_figure(question)
    counts = analysis.caffeine_counts(df, category, include_unreliable)
    if counts["n_items"].sum() == 0:
        return empty_figure(question)

    measured = counts["basis"].iloc[0] == "measured"
    fig = go.Figure(
        go.Bar(
            x=counts["caffeine_status"].str.capitalize(),
            y=counts["n_items"],
            marker_color=category_color(category),
            text=counts["n_items"],
            textposition="outside",
            cliponaxis=False,
            textfont={"color": INK_SECONDARY},
            hovertemplate="<b>%{x}</b><br>%{y} items<extra></extra>",
            name=category.capitalize(),
        )
    )
    subtitle = (
        "Based on measured caffeine (mg) in the file"
        if measured
        else "Name-based proxy: the data has no caffeine column"
    )
    _base_layout(fig, question, subtitle)
    fig.update_layout(showlegend=False, height=360)
    fig.update_xaxes(title_text="Caffeine status")
    fig.update_yaxes(title_text="Items (count)")
    return fig


def data_coverage_chart(df: pd.DataFrame, category: str = "drink") -> go.Figure:
    """Per product family: items with nutrition data vs without (stacked horizontal bars).

    Makes the missing-data bias visible: when the families missing data are high-calorie
    ones (Frappuccino), the category mean over the remaining items is understated.
    """
    question = f"Which {category} families are missing nutrition data?"
    if df.empty or "has_nutrition" not in df.columns:
        return empty_figure(question)
    fam = analysis.family_coverage(df, category)
    if fam.empty:
        return empty_figure(question)

    fam = fam.iloc[::-1]
    fig = go.Figure()
    for column, name, color in (
        ("with_data", "With data", category_color(category)),
        ("without_data", "Without data", NO_DATA_COLOR),
    ):
        fig.add_trace(
            go.Bar(
                name=name,
                orientation="h",
                y=fam["family"],
                x=fam[column],
                marker={"color": color, "line": {"color": SURFACE, "width": 2}},
                text=fam[column].where(fam[column] > 0, ""),
                textposition="inside",
                textangle=0,
                textfont={"color": "white" if column == "with_data" else INK},
                customdata=fam["coverage_pct"],
                hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x}} items"
                "<br>Coverage: %{customdata:.0f}%<extra></extra>",
            )
        )
    total, with_data = int(fam["total"].sum()), int(fam["with_data"].sum())
    _base_layout(
        fig,
        question,
        f"{with_data} of {total} {category} items ({with_data / total:.0%}) have nutrition data",
    )
    fig.update_layout(barmode="stack", barcornerradius=0, height=max(300, 60 + 40 * len(fam)))
    fig.update_xaxes(title_text="Items (count)")
    return fig
