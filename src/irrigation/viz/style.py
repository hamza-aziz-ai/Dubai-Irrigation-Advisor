"""Accessible, consistent chart styling.

THE COLOUR CHOICE IS NOT DECORATIVE

Roughly one man in twelve has some form of colour vision deficiency, and the
default matplotlib cycle puts red and green adjacent - the single most common
confusion. This module uses the Okabe-Ito palette, which was designed to stay
distinguishable under deuteranopia, protanopia and tritanopia.

Colour is never the only channel. Every predictor also gets a distinct dash
pattern and marker, so the charts survive being read by someone with colour
vision deficiency, printed in greyscale, or projected badly in a meeting room.
That last case is the one that actually happens.

Reference: Okabe, M. & Ito, K. (2008). "Color Universal Design (CUD)."
<https://jfly.uni-koeln.de/color/>
"""
from __future__ import annotations

from typing import Any

# The eight-colour Okabe-Ito qualitative palette, in its published order.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Ground truth is always black: it is the reference every other line is
# measured against, and black reads as authoritative in any medium.
ACTUAL = OKABE_ITO["black"]
GRID = "#D9D9D9"

# Stable assignment per predictor. Yellow is excluded throughout - it has
# poor contrast against white and is the palette's weakest line colour.
PREDICTOR_COLOURS = {
    "Physics (FAO-56 balance)": OKABE_ITO["blue"],
    "Sensor only": OKABE_ITO["vermillion"],
    "Physics + sensor fusion": OKABE_ITO["orange"],
    "Random forest": OKABE_ITO["bluish_green"],
    "Gradient boosting": OKABE_ITO["reddish_purple"],
    "XGBoost": OKABE_ITO["sky_blue"],
}

# Second channel: dash patterns, so the lines separate in greyscale.
PREDICTOR_DASHES = {
    "Physics (FAO-56 balance)": (None, None),
    "Sensor only": (4, 2),
    "Physics + sensor fusion": (6, 2, 1, 2),
    "Random forest": (1, 1.5),
    "Gradient boosting": (8, 2),
    "XGBoost": (3, 1, 1, 1),
}

# Third channel: markers, for scatter and for sparse series.
PREDICTOR_MARKERS = {
    "Physics (FAO-56 balance)": "o",
    "Sensor only": "s",
    "Physics + sensor fusion": "^",
    "Random forest": "D",
    "Gradient boosting": "v",
    "XGBoost": "P",
}

# Sequence-model series, kept clear of the predictor colours so a combined
# figure never reuses one hue for two different things.
SEQUENCE_COLOURS = {
    "Observed (NASA GWETROOT)": ACTUAL,
    "Persistence": OKABE_ITO["orange"],
    "Climatology": OKABE_ITO["vermillion"],
    "LSTM": OKABE_ITO["blue"],
    "GRU": OKABE_ITO["bluish_green"],
}


def colour_for(name: str) -> str:
    """Colour for a named series, falling back to a neutral grey."""
    return PREDICTOR_COLOURS.get(name, SEQUENCE_COLOURS.get(name, "#666666"))


def dashes_for(name: str) -> tuple[Any, ...]:
    return PREDICTOR_DASHES.get(name, (None, None))


def marker_for(name: str) -> str:
    return PREDICTOR_MARKERS.get(name, "o")


def apply_style(base_font_size: int = 12) -> None:
    """Set matplotlib defaults for a client-facing figure.

    Larger type than matplotlib's default and a much lighter grid. The
    audience for these charts is a grounds manager or a project sponsor, not a
    reviewer with the figure at full screen width - so the failure mode to
    design against is a 9-point tick label in a PDF on a laptop.
    """
    import matplotlib as mpl
    # From the `cycler` package rather than `mpl.cycler`. Matplotlib re-exports
    # it at runtime but does not declare it in its type stubs, so the shorter
    # form reads as an unknown attribute to a checker. `cycler` is a hard
    # matplotlib dependency, so this adds nothing to install.
    from cycler import cycler

    mpl.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 3,
        "axes.labelsize": base_font_size + 1,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,

        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,

        # Only the axes a reader needs. Removing the top and right spines
        # takes ink off the page without removing information.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#4D4D4D",

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,

        "lines.linewidth": 2.0,
        "lines.markersize": 5,

        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=[
            OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"],
            OKABE_ITO["orange"], OKABE_ITO["reddish_purple"], OKABE_ITO["sky_blue"],
        ]),
    })
