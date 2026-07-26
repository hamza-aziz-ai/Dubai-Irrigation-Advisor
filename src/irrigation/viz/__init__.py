"""Shared presentation layer for the notebooks and the dashboard.

Kept in one place so a chart means the same thing wherever it appears. If
physics is blue in the notebook and blue in the dashboard, a reader carries
their understanding between them; if the colours are assigned per-figure by
whatever the plotting library happened to cycle to, they cannot.
"""
from .style import (
    ACTUAL,
    GRID,
    OKABE_ITO,
    PREDICTOR_COLOURS,
    PREDICTOR_DASHES,
    PREDICTOR_MARKERS,
    SEQUENCE_COLOURS,
    apply_style,
    colour_for,
    dashes_for,
    marker_for,
)

__all__ = [
    "ACTUAL",
    "GRID",
    "OKABE_ITO",
    "PREDICTOR_COLOURS",
    "PREDICTOR_DASHES",
    "PREDICTOR_MARKERS",
    "SEQUENCE_COLOURS",
    "apply_style",
    "colour_for",
    "dashes_for",
    "marker_for",
]
