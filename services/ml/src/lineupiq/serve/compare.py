"""Comparing two lineups for one shooter, with an interval on the difference.

The product question is "what does swapping this player for that one do?", and
the honest answer to it is narrower than the question sounds.

**What this measures, and what it deliberately does not.** The trade projection
endpoint -- a change in the receiving team's points per 100 possessions -- is
withheld, because its own pre-committed power analysis says the smallest effect
the sample could detect is the same size as the effects it projects, and its
backtest says the projection does not beat assuming no change. Nothing here
revisits that. This module answers a different question with a different model:
given that a shot is taken, *where* is it taken from, and how do the other four
players change that. The selection model has a measured out-of-sample gain on
unseen five-man combinations, so the question has an answer; the conversion
model does not, so "does the shot go in" stays unanswerable and is not asked.

Two lineups, one covariance
---------------------------
Both predictions are functions of the same twenty coefficients, so they are
correlated and::

    Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b)

is much smaller than the sum of the two variances. Drawing an interval around
each lineup and checking whether they overlap is therefore the wrong test, and
wrong in the direction that looks careful: it refuses differences the model can
actually resolve. :mod:`lineupiq.serve.plays` already makes this argument for
two zones of one lineup; this is the same argument for one zone of two lineups,
running through the same :func:`quadratic_form`.

The variance that is not in the covariance
------------------------------------------
That would be the whole story if the coefficients were the only estimated thing.
They are not, and for this question they are not even the important part.

A one-player swap moves exactly three numbers -- ``spacing``, ``spacing_min``
and ``teammate_rim`` -- and all three are built from the two players' own
shooting rates. Those rates are fitted quantities held in the profile tables,
and the coefficient covariance says nothing whatsoever about them. So a
coefficient-only interval would be *technically correct about the wrong
quantity*: 671,251 attempts is overwhelming evidence about twenty parameters and
almost none about any particular pair of players. It would report a confident
answer whose confidence came from somewhere else.

So the variance carries two terms, and both are published per zone::

    Var = grad_theta' Sigma grad_theta  +  sum_k (d/d r_k)^2 SE(r_k)^2

The second sums over each distinct player whose rate enters either lineup,
treating different players' rates as independent -- they are estimated from
disjoint sets of shots, which is an argument for independence and not a proof of
it, and it is recorded here as an assumption rather than asserted as a fact.
``variance_profiles`` is returned beside ``variance_coefficients`` so a reader
can see which one is doing the work rather than being told.

Both gradients are central differences **on the served scorer**, the coefficient
one by perturbing a coefficient and the profile one by perturbing a rate through
:data:`lineupiq.serve.score.RateOverrides`. Neither is hand-derived, for the
reason :mod:`lineupiq.serve.plays` gives: a hand-derived gradient is a second
implementation of the model that drifts silently, because a wrong gradient does
not raise -- it just makes every interval the wrong width.

Why an omnibus test comes first, and why it has two degrees of freedom
----------------------------------------------------------------------
There are nine zones. Testing each at 80% and reporting whichever separated
would manufacture a difference on most comparisons, which is exactly the failure
the ranking mechanism was built to avoid one level down. So the headline is a
single Wald statistic on the whole shift, and the per-zone numbers are read
underneath it.

The obvious form of that test is eight-dimensional -- nine shares on a simplex,
one of them the reference -- and it is wrong. **Every one of the five lineup
terms multiplies either the rim indicator or the three indicator and nothing
else**, so a lineup's entire contribution to the nine utilities is::

    a * rim[z] + b * three[z]

with ``a = theta_teammate_rim * f_teammate_rim + theta_opp_rim * f_opp_rim`` and
``b`` the analogous sum over the three three-terms. A lineup has exactly two
knobs. It can pull a shooter toward the rim and it can pull him toward the arc;
it cannot move mid-range baseline independently of mid-range wing, because
nothing in the model connects it to either.

So to first order the difference between two lineups lies on a
two-dimensional manifold, and its zone-level covariance is very nearly rank two.
Measured across the parity corpus, the top two eigenvalues of that matrix carry
between 99.94% and 99.997% of its trace, the third is three to five orders below
the first, and the condition number runs from 3e4 to 1.3e6. The remaining
directions are second-order leakage rather than exact zeros, which is why an
eight-dimensional Wald statistic does not fail outright -- it just spends six of
its degrees of freedom inverting rounding error.

This version did exactly that until the Python and TypeScript statistics stopped
agreeing to 1e-9 on one case out of ninety-eight. The tempting repair was a
looser tolerance on the omnibus; the actual fault was testing an
eight-dimensional hypothesis that the model cannot express.

The test is therefore on ``(delta_a, delta_b)`` directly, with a two-by-two
covariance from the same delta method. That is better conditioned, more
powerful -- the critical value falls from 11.03 to 3.22 against an alternative
that was always two-dimensional -- and more interpretable, since the two numbers
are the rim pull and the three pull rather than eight zone contrasts.
:func:`test_the_lineup_effect_really_is_two_dimensional` asserts the
decomposition against the served scorer, so a sixth lineup term with a different
zone structure fails the suite rather than silently invalidating the degrees of
freedom.

The critical value is **pre-registered** in the hash-pinned thresholds file
rather than computed from a distribution at request time. That is partly the
same discipline the rest of the refusal contract runs on, and partly practical:
it means the Worker needs no incomplete gamma function, which would otherwise
have been the only special function anywhere in the serving path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from lineupiq.serve.plays import GRADIENT_STEP, contrast_shares, quadratic_form, standard_error
from lineupiq.serve.score import RateOverrides, ScoreRequest, score_selection

__all__ = [
    "LINEUP_TERMS",
    "RATE_TABLES",
    "LineupComparison",
    "MechanismTerm",
    "Omnibus",
    "UnprofiledPlayerError",
    "ZoneComparison",
    "compare_lineups",
    "profile_share_gradients",
    "share_gradients",
]

#: The rate tables a swap can move, each paired with the table holding its
#: standard error. Fixed here rather than taken from a dict's iteration order,
#: because the profile variance is a sum over these and the TypeScript mirror
#: has to accumulate it in the same sequence to agree at 1e-9.
RATE_TABLES: tuple[tuple[str, str], ...] = (
    ("player_three_rate", "player_three_rate_se"),
    ("player_rim_rate", "player_rim_rate_se"),
    ("opp_three_allowed", "opp_three_allowed_se"),
    ("opp_rim_allowed", "opp_rim_allowed_se"),
)

#: The five lineup features, in :attr:`ScoreResult.lineup_features` order,
#: paired with the zone attribute each one's coefficient multiplies.
#:
#: That second element is what makes the omnibus two-dimensional rather than
#: eight, so it is data rather than a comment. A new lineup term multiplying
#: something other than ``rim`` or ``three`` would have to be added here, and
#: adding it would immediately fail the test that checks this decomposition
#: against the served scorer -- which is the intended way to be forced to
#: revisit the degrees of freedom.
LINEUP_TERM_ATTRIBUTE: tuple[tuple[str, str], ...] = (
    ("spacing_x_three", "three"),
    ("spacing_min_x_three", "three"),
    ("teammate_rim_x_rim", "rim"),
    ("opp_rim_allowed_x_rim", "rim"),
    ("opp_three_allowed_x_three", "three"),
)

#: Just the names, in the same order.
LINEUP_TERMS: tuple[str, ...] = tuple(term for term, _ in LINEUP_TERM_ATTRIBUTE)

#: Which league rate each table falls back to for a player it does not hold.
_LEAGUE_FALLBACK: dict[str, str] = {
    "player_three_rate": "league_three_rate",
    "player_rim_rate": "league_rim_rate",
    "opp_three_allowed": "league_three_rate",
    "opp_rim_allowed": "league_rim_rate",
}


class UnprofiledPlayerError(ValueError):
    """A lineup contains a player with no fitted shooting rate.

    Such a player silently inherits the league rate, so a comparison involving
    him returns a difference of exactly zero -- a number that looks like a
    finding and is actually a missing row. Raising is the same choice
    :func:`lineupiq.serve.plays.rank_plays` makes when the model was exported
    without a covariance: the serving layer turns it into a refusal that names
    the cause, rather than this module inventing a fallback.
    """

    def __init__(self, players: tuple[int, ...]) -> None:
        self.players = players
        listed = ", ".join(str(p) for p in players)
        super().__init__(f"no fitted shooting rate for player(s): {listed}")


@dataclass(frozen=True)
class ZoneComparison:
    zone: str
    #: ``P(zone | shooter, left) - P(zone | shooter, right)``. Sums to zero
    #: across zones, because both sides are simplices.
    delta_share: float
    share_left: float
    share_right: float
    #: ``100 * delta_share * league_points_per_attempt(zone)``.
    points_per_100: float
    standard_error: float
    interval: tuple[float, float]
    #: The two variance components of ``standard_error ** 2``, in the same
    #: units. Published rather than summed away -- which of the two dominates is
    #: the most informative thing this endpoint returns.
    variance_coefficients: float
    variance_profiles: float


@dataclass(frozen=True)
class MechanismTerm:
    """One lineup feature the swap moved, with its pre-registered verdict."""

    term: str
    feature_left: float
    feature_right: float
    feature_delta: float
    coefficient: float
    expected_sign: int | None
    #: ``agrees`` / ``DISAGREES`` / ``indeterminate``, copied from the fitted
    #: sign audit. It travels with the mechanism because one of the three terms
    #: an offensive swap moves is the one whose pre-registered sign came back
    #: wrong, and a reader is owed that beside the number rather than in a
    #: footnote somewhere else.
    verdict: str


@dataclass(frozen=True)
class Omnibus:
    """Did the shot mix move at all, tested on the two parameters a lineup has.

    ``rim_shift`` and ``three_shift`` are those two parameters: how far the left
    lineup pulls this shooter toward the rim and toward the arc, relative to the
    right one, in utility units. They are jointly sufficient for the whole
    difference -- the nine-zone delta is zero if and only if both are.
    """

    statistic: float
    degrees_of_freedom: int
    critical_value: float
    distinguishable: bool
    #: The change in the rim pull and the three pull, with their errors.
    rim_shift: float
    three_shift: float
    rim_shift_error: float
    three_shift_error: float
    #: True when the two lineups produce identical predictions, so the
    #: covariance of the difference is exactly zero and there is no statistic to
    #: compute. Comparing a lineup with itself is the obvious case, and it is
    #: the serving equivalent of the backtest's placebo arm: swapping a player
    #: for himself must return exactly nothing, and if it ever drifts off zero
    #: every other number here is measuring a bug.
    degenerate: bool


@dataclass(frozen=True)
class LineupComparison:
    zones: tuple[ZoneComparison, ...]
    omnibus: Omnibus
    mechanism: tuple[MechanismTerm, ...]
    confidence: float
    critical_value: float
    #: Share of the total variance carried by the profile term, pooled over
    #: zones. One number for the headline; the per-zone split rides on each zone.
    profile_variance_share: float
    #: True when two teammates' three-point rates are within a perturbation step
    #: of each other, so ``spacing_min`` sits on the kink of its own `min` and
    #: the finite difference is a subgradient rather than a derivative.
    argmin_unstable: bool


def _rate_keys(left: ScoreRequest, right: ScoreRequest | None) -> tuple[tuple[str, str, int], ...]:
    """Every ``(table, se_table, player)`` whose rate enters either lineup.

    Sorted on a total key. The profile variance is a sum over these, and float
    addition is not associative, so an unordered iteration would make the answer
    depend on set or dictionary ordering -- reproducible on one machine and not
    across two, which is the exact bug shape this repository has already found
    in five other places.
    """
    keys: set[tuple[str, str, int]] = set()
    for request in (left, right):
        if request is None:
            continue
        for player in request.offense:
            if player == request.shooter_id:
                continue
            keys.add(("player_three_rate", "player_three_rate_se", player))
            keys.add(("player_rim_rate", "player_rim_rate_se", player))
        for player in request.defense:
            keys.add(("opp_three_allowed", "opp_three_allowed_se", player))
            keys.add(("opp_rim_allowed", "opp_rim_allowed_se", player))
    order = {table: index for index, (table, _) in enumerate(RATE_TABLES)}
    return tuple(sorted(keys, key=lambda key: (order[key[0]], key[2])))


def contrast_offsets(
    left: ScoreRequest,
    right: ScoreRequest | None,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
    *,
    rate_overrides: RateOverrides | None = None,
) -> list[float]:
    """``[delta_rim_pull, delta_three_pull]`` between two lineups.

    The two numbers that jointly determine the entire difference. ``right=None``
    gives the league-average lineup, whose two offsets are exactly zero by
    construction -- every lineup feature is a centred deviation, so the
    league-average lineup pulls in neither direction.

    Read off :attr:`ScoreResult.lineup_features` and the fitted coefficients
    rather than re-derived: this is the model's linear index, which the scorer
    already computes, and the only arithmetic here is the five multiply-adds
    that group it by zone attribute.
    """
    theta = dict(zip(term_names, coefficients, strict=True))

    def offsets(request: ScoreRequest | None) -> tuple[float, float]:
        if request is None:
            return 0.0, 0.0
        features = score_selection(
            request, profiles, coefficients, term_names, rate_overrides=rate_overrides
        ).lineup_features
        rim = 0.0
        three = 0.0
        for index, (term, attribute) in enumerate(LINEUP_TERM_ATTRIBUTE):
            contribution = theta.get(term, 0.0) * features[index]
            if attribute == "rim":
                rim += contribution
            else:
                three += contribution
        return rim, three

    left_rim, left_three = offsets(left)
    right_rim, right_three = offsets(right)
    return [left_rim - right_rim, left_three - right_three]


def offset_gradients(
    left: ScoreRequest,
    right: ScoreRequest | None,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
) -> list[list[float]]:
    """``d offset[i] / d theta[j]``, indexed ``[offset][term]``."""
    n_terms = len(coefficients)
    gradients = [[0.0] * n_terms for _ in range(2)]
    for j in range(n_terms):
        step = GRADIENT_STEP * max(1.0, abs(coefficients[j]))
        forward = list(coefficients)
        backward = list(coefficients)
        forward[j] += step
        backward[j] -= step
        high = contrast_offsets(left, right, profiles, forward, term_names)
        low = contrast_offsets(left, right, profiles, backward, term_names)
        for i in range(2):
            gradients[i][j] = (high[i] - low[i]) / (2.0 * step)
    return gradients


def profile_offset_gradients(
    left: ScoreRequest,
    right: ScoreRequest | None,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
    keys: tuple[tuple[str, str, int], ...],
) -> list[list[float]]:
    """``d offset[i] / d rate[k]``, indexed ``[key][offset]``."""
    gradients: list[list[float]] = []
    for table, _, player in keys:
        fallback = float(profiles[_LEAGUE_FALLBACK[table]])
        base = float(profiles[table].get(str(player), fallback))
        step = GRADIENT_STEP * max(1.0, abs(base))
        high = contrast_offsets(
            left,
            right,
            profiles,
            coefficients,
            term_names,
            rate_overrides={table: {str(player): base + step}},
        )
        low = contrast_offsets(
            left,
            right,
            profiles,
            coefficients,
            term_names,
            rate_overrides={table: {str(player): base - step}},
        )
        gradients.append([(high[i] - low[i]) / (2.0 * step) for i in range(2)])
    return gradients


def share_gradients(
    left: ScoreRequest,
    right: ScoreRequest | None,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
) -> list[list[float]]:
    """``d delta_share[zone] / d theta[j]``, indexed ``[zone][term]``."""
    n_terms = len(coefficients)
    n_zones = len(profiles["zones"])
    gradients = [[0.0] * n_terms for _ in range(n_zones)]
    for j in range(n_terms):
        step = GRADIENT_STEP * max(1.0, abs(coefficients[j]))
        forward = list(coefficients)
        backward = list(coefficients)
        forward[j] += step
        backward[j] -= step
        high = contrast_shares(left, right, profiles, forward, term_names)
        low = contrast_shares(left, right, profiles, backward, term_names)
        for z in range(n_zones):
            gradients[z][j] = (high[z] - low[z]) / (2.0 * step)
    return gradients


def profile_share_gradients(
    left: ScoreRequest,
    right: ScoreRequest | None,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
    keys: tuple[tuple[str, str, int], ...],
) -> list[list[float]]:
    """``d delta_share[zone] / d rate[k]``, indexed ``[key][zone]``.

    The perturbation goes through the scorer's override map rather than through
    a rewritten copy of the profiles, so what is differentiated is the served
    function itself and not a reconstruction of it.
    """
    n_zones = len(profiles["zones"])
    gradients: list[list[float]] = []
    for table, _, player in keys:
        fallback = float(profiles[_LEAGUE_FALLBACK[table]])
        base = float(profiles[table].get(str(player), fallback))
        step = GRADIENT_STEP * max(1.0, abs(base))
        high_override: RateOverrides = {table: {str(player): base + step}}
        low_override: RateOverrides = {table: {str(player): base - step}}
        high = contrast_shares(
            left, right, profiles, coefficients, term_names, rate_overrides=high_override
        )
        low = contrast_shares(
            left, right, profiles, coefficients, term_names, rate_overrides=low_override
        )
        gradients.append([(high[z] - low[z]) / (2.0 * step) for z in range(n_zones)])
    return gradients


def _cholesky_quadratic(matrix: list[list[float]], vector: list[float]) -> float | None:
    """``v' M^-1 v`` for symmetric positive-definite ``M``, or None if it is not.

    Factor once and solve, rather than inverting: ``v' (L L')^-1 v`` is
    ``||L^-1 v||^2``, so the forward substitution is the whole computation and
    the inverse is never formed. Written as explicit loops for the same reason
    :func:`quadratic_form` is -- a library factorisation is free to pivot and
    reassociate, and the TypeScript mirror has to reproduce this to 1e-9.

    Returns None when a pivot is not positive. For this matrix that does not
    mean something went wrong: it means the two lineups differ by nothing the
    model can resolve, which is a state the caller has to render rather than a
    failure it has to handle.
    """
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j]
            for k in range(j):
                total -= lower[i][k] * lower[j][k]
            if i == j:
                if total <= 0.0:
                    return None
                lower[i][j] = math.sqrt(total)
            else:
                lower[i][j] = total / lower[j][j]
    solved = [0.0] * n
    for i in range(n):
        total = vector[i]
        for k in range(i):
            total -= lower[i][k] * solved[k]
        solved[i] = total / lower[i][i]
    accumulated = 0.0
    for value in solved:
        accumulated += value * value
    return accumulated


def compare_lineups(
    left: ScoreRequest,
    right: ScoreRequest | None,
    profiles: dict[str, Any],
    model: dict[str, Any],
    *,
    confidence: float,
    critical_value: float,
    omnibus_critical_value: float,
) -> LineupComparison:
    """Compare two lineups for one shooter.

    ``right=None`` compares against the league-average lineup, which is the same
    quantity ``/lineups/score`` already returns per zone -- deliberately routed
    through :func:`contrast_shares` so the two cannot drift.

    Thresholds are passed in rather than read here, for the reason
    :func:`lineupiq.serve.plays.rank_plays` gives: they come from the
    pre-registered, hash-pinned file, and a serving module reaching for its own
    defaults would be a second place the contract lives.
    """
    coefficients = [float(c) for c in model["coefficients"]]
    term_names = list(model["term_names"])
    covariance = [[float(v) for v in row] for row in model["covariance"]]
    zones: list[str] = list(profiles["zones"])
    zone_points = [float(v) for v in profiles["zone_points"]]
    sign_audit: dict[str, Any] = model.get("sign_audit", {})

    keys = _rate_keys(left, right)

    # Every player whose rate enters the answer needs both a rate and an error
    # for it. Missing either means the league fallback is doing the work, and
    # the difference would come back as a zero that looks like a measurement.
    unprofiled = sorted(
        {
            player
            for table, se_table, player in keys
            if str(player) not in profiles.get(table, {})
            or str(player) not in profiles.get(se_table, {})
        }
    )
    if unprofiled:
        raise UnprofiledPlayerError(tuple(unprofiled))

    scored_left = score_selection(left, profiles, coefficients, term_names)
    scored_right = (
        score_selection(right, profiles, coefficients, term_names) if right is not None else None
    )
    reference = scored_right.mix if scored_right is not None else scored_left.baseline_mix
    deltas = [m - b for m, b in zip(scored_left.mix, reference, strict=True)]

    theta_gradients = share_gradients(left, right, profiles, coefficients, term_names)
    rate_gradients = profile_share_gradients(left, right, profiles, coefficients, term_names, keys)
    rate_errors = [float(profiles[se_table][str(player)]) for _, se_table, player in keys]

    comparisons: list[ZoneComparison] = []
    total_coefficient_variance = 0.0
    total_profile_variance = 0.0
    for z, zone in enumerate(zones):
        variance_theta = quadratic_form(theta_gradients[z], covariance, theta_gradients[z])
        variance_rates = 0.0
        for k in range(len(keys)):
            gradient = rate_gradients[k][z]
            error = rate_errors[k]
            variance_rates += gradient * gradient * error * error
        scale = 100.0 * zone_points[z]
        squared = scale * scale
        variance_coefficients = squared * variance_theta
        variance_profiles = squared * variance_rates
        points = 100.0 * deltas[z] * zone_points[z]
        error = standard_error(variance_coefficients + variance_profiles)
        total_coefficient_variance += variance_coefficients
        total_profile_variance += variance_profiles
        comparisons.append(
            ZoneComparison(
                zone=zone,
                delta_share=deltas[z],
                share_left=scored_left.mix[z],
                share_right=reference[z],
                points_per_100=points,
                standard_error=error,
                interval=(
                    points - critical_value * error,
                    points + critical_value * error,
                ),
                variance_coefficients=variance_coefficients,
                variance_profiles=variance_profiles,
            )
        )

    # The omnibus, on the two parameters a lineup actually has: how far it pulls
    # this shooter toward the rim, and how far toward the arc. See the module
    # docstring for why this is not an eight-dimensional test over zones.
    offsets = contrast_offsets(left, right, profiles, coefficients, term_names)
    offset_theta = offset_gradients(left, right, profiles, coefficients, term_names)
    offset_rates = profile_offset_gradients(left, right, profiles, coefficients, term_names, keys)
    matrix = [[0.0, 0.0], [0.0, 0.0]]
    for a in range(2):
        for b in range(2):
            value = quadratic_form(offset_theta[a], covariance, offset_theta[b])
            for k in range(len(keys)):
                value += offset_rates[k][a] * offset_rates[k][b] * rate_errors[k] * rate_errors[k]
            matrix[a][b] = value
    statistic = _cholesky_quadratic(matrix, offsets)
    omnibus = Omnibus(
        statistic=0.0 if statistic is None else statistic,
        degrees_of_freedom=2,
        critical_value=omnibus_critical_value,
        distinguishable=statistic is not None and statistic > omnibus_critical_value,
        degenerate=statistic is None,
        rim_shift=offsets[0],
        three_shift=offsets[1],
        rim_shift_error=standard_error(matrix[0][0]),
        three_shift_error=standard_error(matrix[1][1]),
    )

    theta = dict(zip(term_names, coefficients, strict=True))
    right_features = (
        scored_right.lineup_features if scored_right is not None else (0.0, 0.0, 0.0, 0.0, 0.0)
    )
    mechanism = tuple(
        MechanismTerm(
            term=term,
            feature_left=scored_left.lineup_features[index],
            feature_right=right_features[index],
            feature_delta=scored_left.lineup_features[index] - right_features[index],
            coefficient=theta.get(term, 0.0),
            expected_sign=sign_audit.get(term, {}).get("expected_sign"),
            verdict=str(sign_audit.get(term, {}).get("verdict", "unaudited")),
        )
        for index, term in enumerate(LINEUP_TERMS)
    )

    total = total_coefficient_variance + total_profile_variance
    left_three_rates = sorted(
        float(profiles["player_three_rate"][str(player)])
        for player in left.offense
        if player != left.shooter_id and str(player) in profiles["player_three_rate"]
    )
    argmin_unstable = (
        len(left_three_rates) > 1
        and (left_three_rates[1] - left_three_rates[0]) < 2.0 * GRADIENT_STEP
    )

    return LineupComparison(
        zones=tuple(comparisons),
        omnibus=omnibus,
        mechanism=mechanism,
        confidence=confidence,
        critical_value=critical_value,
        profile_variance_share=(total_profile_variance / total) if total > 0.0 else 0.0,
        argmin_unstable=argmin_unstable,
    )
