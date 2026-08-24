"""The drawn court and the model's taxonomy must agree.

This is the test that makes "one vocabulary, two consumers" a fact rather than an
intention. The court heatmap fills nine regions; the model scores nine zones. If
the outlines are restated in TypeScript they drift, and the chart then colours a
region the model never scored -- silently, because a coloured polygon looks
correct whatever it means.

So the outlines are generated from the same constants as ``derive_zone``, and
this walks a dense grid asserting that every point inside a zone's outline is a
point ``derive_zone`` puts in that zone.
"""

from __future__ import annotations

import math

import polars as pl

from lineupiq.transform.zone_geometry import (
    COURT_VIEWBOX,
    ZONE_OUTLINES,
    point_in_polygon,
    zone_containing,
    zone_svg_path,
)
from lineupiq.transform.zones import ZONE_IDS, derive_zone

#: How close to a boundary a sample may be before disagreement is acceptable.
#:
#: Arcs are polygons at one degree per segment, so a chord sits at most
#: ``r * (1 - cos(0.5 deg))`` inside the true arc -- about 0.009 court units at
#: the three-point radius, a hundredth of a foot. A point closer than this to a
#: boundary can land on either side of it, and that is a rendering hairline
#: rather than a taxonomy error.
BOUNDARY_TOLERANCE = 0.05

_ARC_THREE = 237.5
_RIM_RADIUS = 40.0
_CORNER_MAX_Y = 92.5
_CORNER_THREE_X = 220.0
_PAINT_HALF_WIDTH = 80.0
_PAINT_DEPTH = 190.0


def _near_a_boundary(x: float, y: float, tolerance: float = BOUNDARY_TOLERANCE) -> bool:
    """True when a point sits on one of the taxonomy's dividing lines."""
    radius = math.hypot(x, y)
    return any(
        abs(distance) < tolerance
        for distance in (
            radius - _ARC_THREE,
            radius - _RIM_RADIUS,
            y - _CORNER_MAX_Y,
            abs(x) - _CORNER_THREE_X,
            abs(x) - _PAINT_HALF_WIDTH,
            y - _PAINT_DEPTH,
            abs(x) - y,
        )
    )


def _grid(
    step: float = 2.5, offset: tuple[float, float] = (0.37, 0.23)
) -> list[tuple[float, float]]:
    """A dense grid, deliberately offset so samples avoid exact boundaries."""
    dx, dy = offset
    xs = [x * step + dx for x in range(int(-250 / step), int(250 / step))]
    ys = [y * step + dy for y in range(int(-47.5 / step), int(422.5 / step))]
    return [(x, y) for x in xs for y in ys]


def test_zone_geometry_agrees_with_derive_zone() -> None:
    """Every point inside an outline is a point the model puts in that zone."""
    points = _grid()
    truth = (
        pl.DataFrame({"loc_x": [p[0] for p in points], "loc_y": [p[1] for p in points]})
        .with_columns(derive_zone())["zone_id"]
        .to_list()
    )

    mismatches: list[tuple[tuple[float, float], str, str | None]] = []
    for (x, y), expected in zip(points, truth, strict=True):
        got = zone_containing((x, y), ZONE_OUTLINES)
        if got != expected and not _near_a_boundary(x, y):
            mismatches.append(((x, y), expected, got))

    assert mismatches[:5] == []
    assert not mismatches


def test_every_court_point_belongs_to_exactly_one_zone() -> None:
    """No gaps and no overlaps.

    A gap renders as a hole in the court; an overlap renders as two fills on the
    same pixel. An earlier version had 2,245 grid points' worth of gap in the
    upper corners, because the top-three outline cut straight from the arc to
    the corner of the viewBox instead of following the 45-degree lines out.
    """
    for x, y in _grid(step=5.0):
        if _near_a_boundary(x, y):
            continue
        hits = [
            zone
            for zone in ZONE_IDS
            if sum(1 for polygon in ZONE_OUTLINES[zone] if point_in_polygon((x, y), polygon)) % 2
            == 1
        ]
        assert hits, f"({x}, {y}) is in no zone"
        assert len(hits) == 1, f"({x}, {y}) is in {hits}"


def test_the_restricted_area_is_a_hole_in_the_lane() -> None:
    """`paint_non_ra` must exclude the restricted area, not contain it."""
    at_the_rim = (5.0, 5.0)
    assert zone_containing(at_the_rim, ZONE_OUTLINES) == "restricted_area"
    # Even-odd winding: inside both rings means outside `paint_non_ra`.
    rings = ZONE_OUTLINES["paint_non_ra"]
    inside_count = sum(1 for polygon in rings if point_in_polygon(at_the_rim, polygon))
    assert inside_count == 2


def test_every_zone_has_an_outline() -> None:
    assert set(ZONE_OUTLINES) == set(ZONE_IDS)
    for zone, polygons in ZONE_OUTLINES.items():
        assert polygons, zone
        for polygon in polygons:
            assert len(polygon) >= 3, zone


def test_svg_paths_are_well_formed_and_flipped() -> None:
    """Court y grows away from the baseline; SVG y grows down."""
    path = zone_svg_path(ZONE_OUTLINES["restricted_area"])
    assert path.startswith("M ")
    assert path.endswith(" Z")
    assert "L" in path

    # The hoop is at court y = 0, which must render near the *bottom* of the
    # viewBox rather than the top. Without the flip the whole court is inverted
    # and every label lands in the wrong half.
    ys = [
        float(pair.split(",")[1]) for pair in path.replace("M ", "").replace(" Z", "").split(" L ")
    ]
    assert min(ys) > 300.0
    assert max(ys) < 450.0


def test_viewbox_matches_the_court_bounds() -> None:
    assert COURT_VIEWBOX == "-250 -47.5 500 470"
    minimum_x, minimum_y, width, height = (float(v) for v in COURT_VIEWBOX.split())
    assert minimum_x + width == 250.0
    assert minimum_y + height == 422.5


def test_every_label_anchor_sits_inside_its_own_zone() -> None:
    """A label in the wrong region is a chart that lies about itself.

    Anchors are hand-placed -- a polygon centroid for the lane-minus-restricted-
    area ring lands inside the hole -- so nothing but this test stops one from
    drifting into a neighbour. It has already caught one: the wing-three anchor
    sat at court (-215, 225), where |x| < y, which is top-of-the-key territory.
    """
    from lineupiq.serve.export import _ZONE_LABEL_ANCHORS

    assert set(_ZONE_LABEL_ANCHORS) == set(ZONE_IDS)
    for zone, (x, svg_y) in _ZONE_LABEL_ANCHORS.items():
        # Anchors are stored in SVG space; the outlines are in court space.
        court_point = (x, 375.0 - svg_y)
        crossings = sum(
            1 for polygon in ZONE_OUTLINES[zone] if point_in_polygon(court_point, polygon)
        )
        assert crossings % 2 == 1, f"{zone} label at {court_point} is outside {zone}"
