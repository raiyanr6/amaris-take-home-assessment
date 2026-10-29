"""The deployed app installs requirements.txt only: keep dev tools out of it."""

from __future__ import annotations

from starbucks_nutrition import config

DEV_TOOLS = ("pytest", "pytest-cov", "ruff")


def _packages(name: str) -> list[str]:
    lines = (config.PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _name(requirement: str) -> str:
    for separator in ("==", ">=", "<=", "~=", ">", "<", "["):
        requirement = requirement.split(separator)[0]
    return requirement.strip().lower()


def test_runtime_requirements_have_no_dev_tools() -> None:
    names = {_name(r) for r in _packages("requirements.txt")}
    assert names.isdisjoint(DEV_TOOLS)
    assert {"streamlit", "groq", "pandas", "plotly", "-e ."} <= names


def test_dev_requirements_extend_runtime_with_dev_tools() -> None:
    packages = _packages("requirements-dev.txt")
    assert packages[0] == "-r requirements.txt"
    assert set(DEV_TOOLS) <= {_name(r) for r in packages[1:]}
