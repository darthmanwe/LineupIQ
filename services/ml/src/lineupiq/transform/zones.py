"""Shot-zone taxonomy.

Coordinates are NBA shot-chart units: tenths of a foot, origin at the centre of
the basket, ``LOC_Y`` increasing away from the baseline. The hoop sits at
(0, 0); the three-point arc is 23.75 ft at the top and 22 ft in the corners.

Zones are derived from geometry rather than taken from the feed's
``SHOT_ZONE_BASIC``, for two reasons: the model needs a zone id that is stable
across seasons and sources, and deriving it produces an independent value to
check the feed against. Agreement between the two is a data-quality gate.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = ["ZONES", "ZONE_IDS", "derive_zone", "is_three_expr"]

#: Distances in tenths of a foot, matching LOC_X / LOC_Y.
_RIM_RADIUS: Final = 40  # 4 ft -- restricted area
_PAINT_HALF_WIDTH: Final = 80  # 8 ft -- half the 16 ft lane
_PAINT_DEPTH: Final = 190  # 19 ft from the baseline
_CORNER_THREE_X: Final = 220  # 22 ft
_ARC_THREE: Final = 237.5  # 23.75 ft
#: Above this Y a shot is beyond the corner, so the arc distance applies.
_CORNER_MAX_Y: Final = 92.5

ZONES: Final[tuple[tuple[str, str], ...]] = (
    ("restricted_area", "Within 4 ft of the rim"),
    ("paint_non_ra", "In the lane, outside the restricted area"),
    ("mid_baseline", "Mid-range, baseline side"),
    ("mid_wing", "Mid-range, wing"),
    ("mid_top", "Mid-range, top of the key"),
    ("corner_three_left", "Left corner three"),
    ("corner_three_right", "Right corner three"),
    ("wing_three", "Above-the-break three, wing"),
    ("top_three", "Above-the-break three, top of the key"),
)

ZONE_IDS: Final[tuple[str, ...]] = tuple(z for z, _ in ZONES)


def is_three_expr(x: str = "loc_x", y: str = "loc_y") -> pl.Expr:
    """True when the shot is behind the three-point line.

    The arc is not a circle: it is truncated to straight lines 22 ft from the
    centre in the corners, below y = 9.25 ft. Treating it as a pure radius
    misclassifies every corner three, which is the highest-value shot on the
    floor and the one the spacing model cares most about.
    """
    radius = (pl.col(x).pow(2) + pl.col(y).pow(2)).sqrt()
    return (
        pl.when(pl.col(y) <= _CORNER_MAX_Y)
        .then(pl.col(x).abs() >= _CORNER_THREE_X)
        .otherwise(radius >= _ARC_THREE)
    )


def derive_zone(x: str = "loc_x", y: str = "loc_y") -> pl.Expr:
    """Assign one of :data:`ZONE_IDS` from coordinates."""
    radius = (pl.col(x).pow(2) + pl.col(y).pow(2)).sqrt()
    three = is_three_expr(x, y)
    in_paint = (pl.col(x).abs() <= _PAINT_HALF_WIDTH) & (pl.col(y) <= _PAINT_DEPTH)

    return (
        pl.when(three & (pl.col(y) <= _CORNER_MAX_Y) & (pl.col(x) < 0))
        .then(pl.lit("corner_three_left"))
        .when(three & (pl.col(y) <= _CORNER_MAX_Y) & (pl.col(x) >= 0))
        .then(pl.lit("corner_three_right"))
        # Above the break: split wing from top by angle, not by raw x, so the
        # boundary follows the arc instead of cutting across it.
        .when(three & (pl.col(x).abs() > pl.col(y)))
        .then(pl.lit("wing_three"))
        .when(three)
        .then(pl.lit("top_three"))
        .when(radius <= _RIM_RADIUS)
        .then(pl.lit("restricted_area"))
        .when(in_paint)
        .then(pl.lit("paint_non_ra"))
        .when(pl.col(y) <= _CORNER_MAX_Y)
        .then(pl.lit("mid_baseline"))
        .when(pl.col(x).abs() > pl.col(y))
        .then(pl.lit("mid_wing"))
        .otherwise(pl.lit("mid_top"))
        .alias("zone_id")
    )
