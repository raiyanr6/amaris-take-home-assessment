"""Starbucks menu nutrition analysis: data layer (loading, cleaning, stats, filtering).

Design principle for the whole project: pandas computes every number; the LLM added in
later steps only narrates numbers it is given. Nothing in this package calls an LLM.
"""

from starbucks_nutrition.loader import DataLoadError, QualityReport, SchemaError, load_menu

__all__ = ["DataLoadError", "QualityReport", "SchemaError", "load_menu"]
__version__ = "0.1.0"
