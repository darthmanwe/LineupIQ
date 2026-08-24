"""Court geometry for the zones, generated from the same constants as the model.

The court heatmap and the model must agree about what "corner three" means. The
usual way that breaks is restating the geometry in TypeScript: the two drift, and
the chart then colours a region the model never scored.

So the outlines are generated here, from the constants in
:mod:`lineupiq.transform.zones`, exported as JSON, and merely *rendered* by the
web app. There is one definition of the taxonomy and two consumers of it.

**Outlines are polygons, not arc paths.** An analytic SVG arc path would be
shorter, but point-in-arc-path is awkward to test, and the property that actually
matters -- that every point inside the ``corner_three_left`` outline is a point
``derive_zone`` calls ``corner_three_left`` -- is only checkable if containment
is checkable. Arcs are discretised finely enough to be visually
indistinguishable, and
``test_zone_geometry_agrees_with_derive_zone`` walks a dense grid asserting
exactly that agreement.
"""

from __future__ import annotations

import math
from typing import Final

from lineupiq.transform.zones import ZONE_IDS

__all__ = [
    "COURT_VIEWBOX",
    "ZONE_OUTLINES",
    "point_in_polygon",
    "zone_outlines",
    "zone_svg_path",
]

#: The SVG viewBox the outlines are drawn for. Court units are tenths of a foot.
#:
#: ``LOC_Y`` grows away from the baseline while SVG y grows downward, so every
#: point is emitted through :func:`_svg_y`. Without that flip the court renders
#: upside down and every zone label lands in the wrong half.
COURT_VIEWBOX: Final = "-250 -47.5 500 470"

#: y of the hoop in SVG space. Court y = 0 is the hoop; the baseline sits below.
_HOOP_SVG_Y: Final = 375.0

# Geometry, re-imported rather than re-typed. These mirror `zones.py` exactly.
_RIM_RADIUS: Final = 40.0
_PAINT_HALF_WIDTH: Final = 80.0
_PAINT_DEPTH: Final = 190.0
_CORNER_THREE_X: Final = 220.0
_ARC_THREE: Final = 237.5
_CORNER_MAX_Y: Final = 92.5

#: Court bounds, matching the viewBox.
_X_MIN, _X_MAX = -250.0, 250.0
_Y_MIN, _Y_MAX = -47.5, 422.5

#: Degrees per segment when discretising an arc. At this radius one degree is
#: about four tenths of a foot, well below a rendered pixel.
_ARC_STEP_DEGREES: Final = 1.0

Point = tuple[float, float]
Polygon = list[Point]


def _svg_y(court_y: float) -> float:
    return _HOOP_SVG_Y - court_y


def _arc(radius: float, from_degrees: float, to_degrees: float) -> Polygon:
    """Points along a circle centred on the hoop, angles measured from +x."""
    steps = max(int(abs(to_degrees - from_degrees) / _ARC_STEP_DEGREES), 1)
    out: Polygon = []
    for i in range(steps + 1):
        angle = math.radians(from_degrees + (to_degrees - from_degrees) * i / steps)
        out.append((radius * math.cos(angle), radius * math.sin(angle)))
    return out


def _arc_y_at_x(x: float) -> float:
    """Height of the three-point arc at a given x, or 0 outside it."""
    inside = _ARC_THREE**2 - x**2
    return math.sqrt(inside) if inside > 0 else 0.0


def _arc_x_at_y(y: float) -> float:
    inside = _ARC_THREE**2 - y**2
    return math.sqrt(inside) if inside > 0 else 0.0


def _restricted_area() -> Polygon:
    return _arc(_RIM_RADIUS, 0.0, 360.0)


def _paint_non_ra() -> list[Polygon]:
    """The lane, with the restricted-area circle as a hole.

    Two rings, rendered with ``fill-rule: evenodd`` so the inner one subtracts.
    """
    outer: Polygon = [
        (-_PAINT_HALF_WIDTH, _Y_MIN),
        (_PAINT_HALF_WIDTH, _Y_MIN),
        (_PAINT_HALF_WIDTH, _PAINT_DEPTH),
        (-_PAINT_HALF_WIDTH, _PAINT_DEPTH),
    ]
    return [outer, _restricted_area()]


def _corner_three(side: int) -> Polygon:
    """A corner three: beyond 22 ft laterally, below the corner cutoff."""
    near = side * _CORNER_THREE_X
    far = side * _X_MAX
    return [(near, _Y_MIN), (far, _Y_MIN), (far, _CORNER_MAX_Y), (near, _CORNER_MAX_Y)]


def _above_break_three(kind: str) -> list[Polygon]:
    """Above-the-break threes, split into wing and top by the 45-degree lines.

    ``|x| > y`` is the wing side. The boundary meets the arc at
    ``x = y = 237.5 / sqrt(2)``, and the arc itself sits *inside* the 22 ft
    corner line at the cutoff height -- ``sqrt(237.5^2 - 92.5^2)`` is about
    218.7, not 220 -- which is the real geometry rather than a simplification.
    """
    diagonal = _ARC_THREE / math.sqrt(2.0)

    if kind == "top":
        # Between the two 45-degree lines, outside the arc, up to the top.
        #
        # The 45-degree lines have to be followed all the way out to the side
        # boundary before turning up. Cutting the corner straight from the arc
        # to the top-right of the viewBox leaves two wedges above y = 250
        # belonging to no zone at all -- 2,245 grid points' worth, all of which
        # `derive_zone` calls `top_three`.
        arc = _arc(_ARC_THREE, 45.0, 135.0)
        return [
            [
                (-diagonal, diagonal),
                *arc,
                (diagonal, diagonal),
                (_X_MAX, _X_MAX),
                (_X_MAX, _Y_MAX),
                (_X_MIN, _Y_MAX),
                (_X_MIN, _X_MAX),
            ]
        ]

    wings: list[Polygon] = []
    for side in (1, -1):
        # From the corner cutoff, up the arc to the 45-degree line, then out.
        start_angle = math.degrees(math.atan2(_CORNER_MAX_Y, _arc_x_at_y(_CORNER_MAX_Y)))
        arc = _arc(_ARC_THREE, start_angle, 45.0)
        polygon: Polygon = [(side * _arc_x_at_y(_CORNER_MAX_Y), _CORNER_MAX_Y)]
        polygon += [(side * x, y) for x, y in arc]
        polygon += [
            (side * diagonal, diagonal),
            (side * _X_MAX, _X_MAX),
            (side * _X_MAX, _CORNER_MAX_Y),
        ]
        wings.append(polygon)
    return wings


def _mid_baseline() -> list[Polygon]:
    """Mid-range, baseline side: below the cutoff, outside the paint, inside the arc."""
    out: list[Polygon] = []
    for side in (1, -1):
        near = side * _PAINT_HALF_WIDTH
        far = side * _CORNER_THREE_X
        out.append([(near, _Y_MIN), (far, _Y_MIN), (far, _CORNER_MAX_Y), (near, _CORNER_MAX_Y)])
    return out


def _mid_wing() -> list[Polygon]:
    """Mid-range wing: |x| > y, inside the arc, above the corner cutoff.

    The inner boundary is the lane where the lane is present, and the
    45-degree line above it.
    """
    diagonal = _ARC_THREE / math.sqrt(2.0)
    out: list[Polygon] = []
    for side in (1, -1):
        start_angle = math.degrees(math.atan2(_CORNER_MAX_Y, _arc_x_at_y(_CORNER_MAX_Y)))
        arc = list(reversed(_arc(_ARC_THREE, start_angle, 45.0)))
        # The inner corner is where the 45-degree line meets the corner cutoff,
        # at (92.5, 92.5) -- not the lane edge at (80, 92.5). Starting at the
        # lane would overlap `mid_baseline`, and while the first-match order
        # hides that from the agreement test, two fills would still overlap on
        # screen.
        polygon: Polygon = [
            (side * _CORNER_MAX_Y, _CORNER_MAX_Y),
            (side * _arc_x_at_y(_CORNER_MAX_Y), _CORNER_MAX_Y),
        ]
        polygon += [(side * x, y) for x, y in reversed(arc)]
        polygon += [(side * diagonal, diagonal)]
        out.append(polygon)
    return out


def _mid_top() -> list[Polygon]:
    """Mid-range top of the key: inside the arc, between the 45-degree lines."""
    diagonal = _ARC_THREE / math.sqrt(2.0)
    arc = list(reversed(_arc(_ARC_THREE, 45.0, 135.0)))
    polygon: Polygon = [(-diagonal, diagonal)]
    polygon += list(arc)
    polygon += [
        (diagonal, diagonal),
        (_CORNER_MAX_Y, _CORNER_MAX_Y),
        (_PAINT_HALF_WIDTH, _CORNER_MAX_Y),
        (_PAINT_HALF_WIDTH, _PAINT_DEPTH),
        (-_PAINT_HALF_WIDTH, _PAINT_DEPTH),
        (-_PAINT_HALF_WIDTH, _CORNER_MAX_Y),
        (-_CORNER_MAX_Y, _CORNER_MAX_Y),
    ]
    return [polygon]


def zone_outlines() -> dict[str, list[Polygon]]:
    """Every zone as one or more polygons, in court coordinates."""
    return {
        "restricted_area": [_restricted_area()],
        "paint_non_ra": _paint_non_ra(),
        "mid_baseline": _mid_baseline(),
        "mid_wing": _mid_wing(),
        "mid_top": _mid_top(),
        "corner_three_left": [_corner_three(-1)],
        "corner_three_right": [_corner_three(1)],
        "wing_three": _above_break_three("wing"),
        "top_three": _above_break_three("top"),
    }


def zone_svg_path(polygons: list[Polygon]) -> str:
    """Render polygons as one SVG path, flipped into SVG's y-down space."""
    parts: list[str] = []
    for polygon in polygons:
        if not polygon:
            continue
        points = " ".join(f"{x:.2f},{_svg_y(y):.2f}" for x, y in polygon)
        head, *rest = points.split(" ")
        parts.append("M " + head + "".join(f" L {p}" for p in rest) + " Z")
    return " ".join(parts)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray casting. Exposed so the agreement test can use it."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing:
                inside = not inside
    return inside


def zone_containing(point: Point, outlines: dict[str, list[Polygon]]) -> str | None:
    """Which zone's outline contains this court point, by even-odd winding."""
    for zone in ZONE_IDS:
        polygons = outlines.get(zone, [])
        crossings = sum(1 for polygon in polygons if point_in_polygon(point, polygon))
        # Even-odd: the lane minus the restricted-area circle means a point
        # inside both rings is *not* in `paint_non_ra`.
        if crossings % 2 == 1:
            return zone
    return None


ZONE_OUTLINES: Final = zone_outlines()
