"""Streamlit smoke tests with AppTest (headless). Skip with ``pytest -m "not ui"``."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from starbucks_nutrition import config
from starbucks_nutrition.ui.explore import NO_MATCH_MESSAGE

pytestmark = pytest.mark.ui

APP = str(config.PROJECT_ROOT / "streamlit_app.py")
TABS = ["Overview", "Explore & filter", "Charts", "Data quality"]


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _match_count(at: AppTest) -> str:
    return next(m.value for m in at.markdown if "items match" in m.value)


def _slider(at: AppTest, prefix: str):
    return next(s for s in at.slider if s.key and s.key.startswith(prefix))


def test_app_runs_with_sample_data(app: AppTest) -> None:
    assert [t.label for t in app.tabs] == TABS
    assert not app.error
    assert "sample data" in app.sidebar.markdown[0].value


def test_each_tab_renders_its_content(app: AppTest) -> None:
    overview, explore, charts, quality = app.tabs
    assert len(overview.metric) == 6 and overview.warning  # KPI tiles + caveats box
    assert "likely understated" in overview.warning[0].value
    assert explore.expander and explore.dataframe
    assert len(charts.get("plotly_chart")) == 4
    assert quality.dataframe and quality.get("plotly_chart")


def test_explore_defaults_show_every_item(app: AppTest) -> None:
    assert _match_count(app) == "**267** of 267 items match"


def test_food_under_500_with_upper_bound_only(app: AppTest) -> None:
    app.multiselect(key="filter_categories").set_value(["food"]).run()
    _slider(app, "range_calories_").set_range(90.0, 500.0).run()
    assert _match_count(app) == "**99** of 267 items match"
    app.radio(key="filter_upper").set_value("up to (≤)").run()
    assert _match_count(app) == "**103** of 267 items match"


def test_sodium_filter_hides_food_and_says_so(app: AppTest) -> None:
    low, high = _slider(app, "range_sodium_mg_").value
    _slider(app, "range_sodium_mg_").set_range(low, high - 50).run()
    assert "all 113 food" in " ".join(c.value for c in app.caption)
    assert int(_match_count(app).split("**")[1]) < 74  # only drinks with sodium remain


def test_sodium_slider_disabled_for_food_only(app: AppTest) -> None:
    app.multiselect(key="filter_categories").set_value(["food"]).run()
    assert app.slider(key="off_sodium_mg").disabled
    assert any("No sodium data" in c.value for c in app.caption)


def test_no_match_shows_info(app: AppTest) -> None:
    app.text_input(key="filter_text").input("unicorn frappé").run()
    assert _match_count(app) == "**0** of 267 items match"
    assert any(NO_MATCH_MESSAGE in i.value for i in app.info)


def test_include_unreliable_toggle(app: AppTest) -> None:
    before = app.tabs[0].metric[2].delta  # drink mean calories tile: "based on 73 items"
    app.toggle(key="include_unreliable").set_value(True).run()
    assert not app.exception
    assert before == "based on 73 items"
    assert app.tabs[0].metric[2].delta == "based on 74 items"
    assert not any("excluded as unreliable" in w.value for w in app.warning)
