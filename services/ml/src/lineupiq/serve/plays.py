"""Ranking a lineup's zones by what the lineup is worth in each, with intervals.

This is the endpoint the plan calls "top-k actions ranked by their priced
contribution", and the whole difficulty is in the word *ranked*.

The point estimate is already available: :func:`lineupiq.serve.score.score_selection`
returns the shot mix with and without the lineup terms, and pricing the
difference at league conversion rates gives each zone's contribution in points
per 100 attempts. Sorting nine numbers is not a problem. The problem is that a
sorted list of nine numbers *looks like* a claim that the first beats the second,
and at this effect size that claim is usually false -- the whole shift prices at
a standard deviation of 0.19 points per 100, spread over nine zones.

**So the ranking is only served as an ordering where the ordering is supported,
and as an explicitly unordered band where it is not.**

Why the full covariance, and not the standard errors
----------------------------------------------------
The obvious test is to draw an interval around each contribution and check
whether they overlap. That test is wrong, and it is wrong in the direction that
looks responsible, which is what makes it worth writing down.

Zone shares come out of a softmax, so they sum to one: share that appears at the
rim came from somewhere else. Two contributions are therefore strongly
*negatively* correlated, and::

    Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b)

With a large negative covariance the difference is far better determined than
either endpoint. Comparing marginal intervals ignores that term entirely and
concludes "indistinguishable" for pairs that separate decisively -- refusing to
rank things the model genuinely can rank, which is its own kind of dishonesty.
:attr:`PlayRanking.diagonal_would_refuse` counts how often that happens on the
request being served, so the claim in this docstring is a measurement rather than
an argument.

That is the reason ``selection_model.json`` carries a 20x20 matrix rather than
20 numbers.

The gradient
------------
Each contribution is a smooth scalar function of the twenty coefficients, so the
delta method applies: ``Var(f) = grad(f)' Sigma grad(f)``. The gradient is taken
by central differences **on the served scorer itself**, not on a hand-derived
expression. A softmax difference times a constant is differentiable by hand, but
a hand-derived gradient of the *served* function is a second implementation of
the model that can drift from the first, and it would drift silently -- the
intervals would simply be the wrong width. Differencing the scorer costs forty
scorer calls, each a few hundred flops.

The same choice is what makes the TypeScript mirror possible at all: the Worker
finite-differences its own scorer, so parity on the intervals is a real check on
both implementations rather than a check that two transcriptions of one formula
match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lineupiq.serve.score import RateOverrides, ScoreRequest, score_selection

__all__ = [
    "Play",
    "PlayRanking",
    "contrast_shares",
    "contribution_gradients",
    "quadratic_form",
    "rank_plays",
    "standard_error",
]

#: Relative step for the central difference, identical in the TypeScript mirror.
#:
#: 1e-4 rather than the 1e-5 used for the observed information. That one
#: differences an analytic gradient, where truncation error is O(h^2) on a
#: quantity of order 1; this differences a probability difference of order 1e-3,
#: so the cancellation is far worse and a larger step trades truncation error for
#: the round-off that would otherwise dominate.
GRADIENT_STEP = 1e-4


@dataclass(frozen=True)
class Play:
    """One zone's priced contribution, with the interval that governs its rank."""

    zone: str
    #: Predicted share of this shooter's attempts with the lineup on the floor.
    share: float
    #: The same shooter with every lineup term at the league average.
    baseline_share: float
    #: ``100 * (share - baseline_share) * league_points_per_attempt(zone)``.
    #: Summing this over zones is exactly ``ScoreResult.points_per_100``.
    points_per_100: float
    standard_error: float
    interval: tuple[float, float]
    #: 1-based band index. Zones the data cannot separate share a rank, and a
    #: shared rank is the honest rendering of a tie -- not a coin flip resolved
    #: by float comparison.
    rank: int


@dataclass(frozen=True)
class PlayRanking:
    plays: tuple[Play, ...]
    #: The bands, in order. A band with more than one member is an unordered set.
    bands: tuple[tuple[str, ...], ...]
    #: False when every eligible zone landed in one band -- i.e. the model has a
    #: point estimate for each zone and no basis for putting them in any order.
    #: The UI renders that as a set, and the API says so rather than implying an
    #: order by list position.
    ordered: bool
    #: Zones below the pre-registered share floor, excluded before ranking.
    excluded: tuple[str, ...]
    confidence: float
    critical_value: float
    #: Ranked pairs that the marginal intervals would call indistinguishable but
    #: the difference test separates. The measured cost of shipping only the
    #: diagonal, on this request.
    diagonal_would_refuse: int
    #: Total pairs compared, so the count above has a denominator.
    pairs_compared: int
    #: Indistinguishable pairs that landed in different bands, because bands are
    #: contiguous runs of the ranked list and the pair was not adjacent enough to
    #: share one. This is the information contiguity throws away, counted instead
    #: of argued about.
    ties_spanning_bands: int


def contrast_shares(
    left: ScoreRequest,
    right: ScoreRequest | None,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
    *,
    rate_overrides: RateOverrides | None = None,
) -> list[float]:
    """Per-zone difference in predicted shot share between two lineups.

    ``right=None`` means "the same shooter with every lineup term at the league
    average", which is :attr:`ScoreResult.baseline_mix` and needs no second
    scoring call -- every lineup term is a centred deviation, so dropping them
    *is* the league-average lineup.

    That is not a special case bolted on for the comparison endpoint: it is the
    quantity :func:`rank_plays` has always ranked. Writing both through one
    function is what makes ``compare(L, league_average)`` and
    ``/lineups/score``'s per-zone ``delta`` the same number by construction
    rather than by two implementations agreeing.
    """
    scored = score_selection(
        left, profiles, coefficients, term_names, rate_overrides=rate_overrides
    )
    if right is None:
        reference = scored.baseline_mix
    else:
        reference = score_selection(
            right, profiles, coefficients, term_names, rate_overrides=rate_overrides
        ).mix
    return [m - b for m, b in zip(scored.mix, reference, strict=True)]


def _contribution(
    request: ScoreRequest,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
) -> list[float]:
    """Per-zone priced contribution, in points per 100 attempts."""
    deltas = contrast_shares(request, None, profiles, coefficients, term_names)
    zone_points = [float(v) for v in profiles["zone_points"]]
    return [100.0 * d * p for d, p in zip(deltas, zone_points, strict=True)]


def contribution_gradients(
    request: ScoreRequest,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
) -> list[list[float]]:
    """``d contribution[zone] / d theta[j]``, by central differences.

    Returns one gradient per zone, each of length ``len(coefficients)``. Indexed
    ``[zone][term]``.
    """
    n_terms = len(coefficients)
    n_zones = len(profiles["zones"])
    gradients = [[0.0] * n_terms for _ in range(n_zones)]

    for j in range(n_terms):
        step = GRADIENT_STEP * max(1.0, abs(coefficients[j]))
        forward = list(coefficients)
        backward = list(coefficients)
        forward[j] += step
        backward[j] -= step
        high = _contribution(request, profiles, forward, term_names)
        low = _contribution(request, profiles, backward, term_names)
        for z in range(n_zones):
            gradients[z][j] = (high[z] - low[z]) / (2.0 * step)
    return gradients


def quadratic_form(left: list[float], covariance: list[list[float]], right: list[float]) -> float:
    """``left' Sigma right``, accumulated in a fixed order.

    Written as an explicit double loop rather than as two matrix products,
    because the TypeScript mirror has to sum the same terms in the same order to
    agree at 1e-9. A BLAS `dot` is free to reassociate; this is not.
    """
    total = 0.0
    for i, row in enumerate(covariance):
        inner = 0.0
        for j, value in enumerate(row):
            inner += value * right[j]
        total += left[i] * inner
    return total


def standard_error(quadratic: float) -> float:
    # A tiny negative comes out of the delta method when the true variance is
    # near zero and the covariance is only numerically positive semi-definite.
    # Clamping at zero is right; taking an absolute value would turn a flat
    # direction into a confident-looking interval.
    return math.sqrt(quadratic) if quadratic > 0.0 else 0.0


def rank_plays(
    request: ScoreRequest,
    profiles: dict[str, Any],
    model: dict[str, Any],
    *,
    confidence: float,
    critical_value: float,
    min_zone_share: float,
) -> PlayRanking:
    """Rank the zones this lineup moves, refusing to order what it cannot.

    ``confidence``, ``critical_value`` and ``min_zone_share`` are passed in
    rather than read here: they come from the pre-registered, hash-pinned
    thresholds file, and a serving module that reached for its own defaults would
    be a second place the contract lives.
    """
    coefficients = [float(c) for c in model["coefficients"]]
    term_names = list(model["term_names"])
    covariance = [[float(v) for v in row] for row in model["covariance"]]

    zones: list[str] = list(profiles["zones"])
    scored = score_selection(request, profiles, coefficients, term_names)
    contributions = _contribution(request, profiles, coefficients, term_names)
    gradients = contribution_gradients(request, profiles, coefficients, term_names)

    errors = [
        standard_error(quadratic_form(gradients[z], covariance, gradients[z]))
        for z in range(len(zones))
    ]

    eligible = [z for z in range(len(zones)) if scored.mix[z] >= min_zone_share]
    excluded = tuple(zones[z] for z in range(len(zones)) if z not in set(eligible))

    # Sorted by contribution descending, with the zone index as a total
    # tiebreaker. Two zones can price identically -- a zone the lineup does not
    # move at all contributes exactly 0.0, and there can be several -- and an
    # untiebroken sort would order those by whatever the sort happened to do.
    order = sorted(eligible, key=lambda z: (-contributions[z], z))

    # Which ranked pairs the data separates. Both tests are computed: the one
    # that decides the ranking, and the naive marginal-overlap test, so the
    # difference between them is reported instead of asserted.
    distinguishable: set[tuple[int, int]] = set()
    diagonal_would_refuse = 0
    pairs_compared = 0
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1 :]:
            pairs_compared += 1
            difference = contributions[a] - contributions[b]
            delta = [gradients[a][j] - gradients[b][j] for j in range(len(coefficients))]
            joint = standard_error(quadratic_form(delta, covariance, delta))
            separates = abs(difference) > critical_value * joint
            if separates:
                distinguishable.add((a, b))
            # The naive test: do the marginal intervals overlap? Equivalent to
            # comparing the difference against `z * (se_a + se_b)`, which is
            # always at least as large as `z * se(a - b)` and much larger when
            # the two are negatively correlated.
            naive_separates = abs(difference) > critical_value * (errors[a] + errors[b])
            if separates and not naive_separates:
                diagonal_would_refuse += 1

    # Bands are maximal **contiguous** runs of the sorted list, extended while
    # the incoming zone is indistinguishable from at least one zone already in
    # the run. Single linkage, restricted to contiguity. A Tukey letter display.
    #
    # Two decisions are packed in here and both were made the hard way.
    #
    # *Single* linkage rather than breaking at the first separated adjacent pair:
    # a chain of individually-indistinguishable gaps can add up to a gap that is
    # not, and breaking on adjacency would report that sum as a real ordering.
    # Extending the band while *any* member ties refuses more often, which is the
    # right direction for a mechanism whose entire job is to not invent an order.
    #
    # *Contiguous* rather than true connected components: the difference test has
    # a per-pair standard error, so a wider gap can separate while a narrower one
    # inside it does not, and unrestricted components can therefore interleave --
    # zone 6 landing in the same component as zone 1 while zones 2 through 5 form
    # their own. That is not renderable as a ranked list and it is not coherent as
    # one either: the first version of this function did exactly that and produced
    # rank sequences like 1, 2, 2, 1. Contiguity is what makes `rank` monotone in
    # list position, which is the property anyone reading the output will assume.
    #
    # What contiguity costs is counted rather than assumed: `ties_spanning_bands`
    # is the number of indistinguishable pairs that ended up in different bands.
    band_of: dict[int, int] = {}
    bands: list[list[int]] = []
    for zone_index in order:
        if bands and any((member, zone_index) not in distinguishable for member in bands[-1]):
            bands[-1].append(zone_index)
        else:
            bands.append([zone_index])
        band_of[zone_index] = len(bands) - 1

    ties_spanning_bands = 0
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1 :]:
            if (a, b) not in distinguishable and band_of[a] != band_of[b]:
                ties_spanning_bands += 1

    plays = tuple(
        Play(
            zone=zones[z],
            share=scored.mix[z],
            baseline_share=scored.baseline_mix[z],
            points_per_100=contributions[z],
            standard_error=errors[z],
            interval=(
                contributions[z] - critical_value * errors[z],
                contributions[z] + critical_value * errors[z],
            ),
            rank=band_of[z] + 1,
        )
        for z in order
    )
    return PlayRanking(
        plays=plays,
        bands=tuple(tuple(zones[z] for z in band) for band in bands),
        ordered=len(bands) > 1,
        excluded=excluded,
        confidence=confidence,
        critical_value=critical_value,
        diagonal_would_refuse=diagonal_would_refuse,
        pairs_compared=pairs_compared,
        ties_spanning_bands=ties_spanning_bands,
    )
