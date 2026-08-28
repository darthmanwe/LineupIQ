"""The served selection scorer, in Python, reading the exported contract.

This is the model the Worker runs, and it is deliberately written against the
**exported JSON** rather than against the fitted :class:`SelectionProfiles`
object. That choice is the whole point.

A scorer written against the in-memory profiles would use full float64
precision; the Worker can only ever see what was serialised, which is rounded.
Compare those two and the parity test either fails for a reason that has nothing
to do with either implementation, or -- worse -- gets a tolerance loosened until
it passes. Both implementations reading the same rounded contract means a parity
failure can only be a real disagreement about the arithmetic.

**Why a closed form is servable at all.** A conditional logit's prediction is
``softmax(Xθ)`` over nine alternatives. Every term is either a constant per zone,
a shot-level scalar times a zone indicator, or a per-player vector -- so scoring
one lineup is a handful of multiply-adds and one exponential per zone. No matrix
factorisation, no tree traversal, nothing to load beyond a 210 KB JSON file.
That is what buys the counterfactual: any five of ~450 players can be scored on
demand, and ``C(450, 5)`` is 1.5e11, so nothing could have been precomputed.

The gradient-boosted reference model cannot do this, and the log-loss gap between
them is published rather than hidden. This is the cost of serving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["RateOverrides", "ScoreRequest", "ScoreResult", "score_selection"]

#: Per-table, per-player replacements for the four rate tables.
#:
#: ``{table_name: {player_id_as_str: rate}}``. This exists so that
#: :mod:`lineupiq.serve.compare` can take a derivative of the **served**
#: scorer with respect to a player's shooting rate, in exactly the way
#: :mod:`lineupiq.serve.plays` already takes one with respect to a
#: coefficient. The alternative -- differentiating a hand-derived softmax
#: expression -- would be a second implementation of the model that could
#: drift from this one, and it would drift silently, because a wrong
#: gradient produces intervals of the wrong width rather than an error.
#:
#: With no overrides the arithmetic is unchanged down to the last bit, which
#: is why the existing parity fixture is untouched by this parameter.
RateOverrides = dict[str, dict[str, float]]

#: The zone whose alternative-specific constant is fixed at zero. A conditional
#: logit is invariant to adding a constant to every utility, so one alternative
#: must be pinned or the coefficients are not identified.
REFERENCE_ZONE = "restricted_area"


@dataclass(frozen=True)
class ScoreRequest:
    """One counterfactual: this shooter, on this floor, against these five.

    Context defaults to the league-average possession rather than to zero,
    because zero is not neutral -- ``seconds_into_possession = 0`` is a
    fast-break, which is a strong statement about shot selection. The mean is
    the honest "no information" default, and it makes the standardised driver
    exactly zero.
    """

    shooter_id: int
    offense: tuple[int, ...]
    defense: tuple[int, ...]
    team_id: int | None = None
    season: int | None = None
    seconds_into_possession: float | None = None
    live_ball: bool = False
    second_chance: bool = False
    clutch: bool = False


@dataclass(frozen=True)
class ScoreResult:
    zones: tuple[str, ...]
    #: Predicted share of this shooter's attempts, by zone. Sums to 1.
    mix: tuple[float, ...]
    #: The same shooter with every lineup term at the league average. The
    #: difference between the two is the lineup effect, which is the only thing
    #: this model claims to measure.
    baseline_mix: tuple[float, ...]
    #: Utilities before the softmax, exposed because they are what the parity
    #: fixture compares -- a softmax is a contraction, so agreeing after it is a
    #: weaker check than agreeing before it.
    utilities: tuple[float, ...]
    #: Shooters with no profile fall back to the league mix. Serving that
    #: silently would be a confident answer about a player the model has never
    #: seen.
    shooter_known: bool
    #: How much of the shooter's mix is evidence rather than prior, from the
    #: Dirichlet-multinomial shrinkage. Low means the answer is mostly a prior.
    shooter_weight: float
    #: The five lineup features, in the order their coefficients appear in
    #: ``SELECTION_TERMS``: spacing, spacing_min, teammate_rim, opp_rim,
    #: opp_three. Every one is a centred deviation from the league rate, which
    #: is why dropping them all gives the league-average lineup.
    #:
    #: Exposed because :mod:`lineupiq.serve.compare` reports *which* of them a
    #: swap moved, and recomputing them there would be a second implementation
    #: of the aggregation -- including the `min`, which is the one most likely
    #: to be got subtly wrong twice.
    lineup_features: tuple[float, ...]
    #: The shot-mix shift, priced in points per 100 attempts.
    #:
    #: ``sum(delta_share * league_points_per_attempt)``, scaled by 100. This is
    #: the number the product is actually about -- "0.27 percentage points more
    #: corner threes" is not a quantity anyone can act on, and this is the same
    #: fact in units that mean something.
    #:
    #: Priced at **league** conversion rates rather than the shooter's own, which
    #: is the estimand and not a shortcut: using his rates would fold the two
    #: channels back together, so part of the answer would be "he shoots better
    #: from there" and part "the lineup got him there". At fixed conversion, all
    #: of it is selection.
    points_per_100: float


def _softmax(values: list[float]) -> list[float]:
    # Shifted by the max before exponentiating. The utilities here are small
    # enough that it does not matter numerically, but an unshifted softmax is a
    # latent overflow that only shows up on the input nobody tested.
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rate(
    profiles: dict[str, Any],
    overrides: RateOverrides | None,
    table: str,
    player: int,
    fallback: float,
) -> float:
    """One player's rate from ``table``, with an override taking precedence.

    The fallback is the league rate, and it is the reason a player below the
    profile fit's attempt floor scores as exactly league-average rather than
    raising. That is correct for scoring one lineup and wrong for comparing two,
    so the comparison endpoint refuses such a player instead of reporting the
    zero this fallback would produce.
    """
    key = str(player)
    if overrides is not None:
        table_overrides = overrides.get(table)
        if table_overrides is not None and key in table_overrides:
            return float(table_overrides[key])
    return float(profiles[table].get(key, fallback))


def score_selection(
    request: ScoreRequest,
    profiles: dict[str, Any],
    coefficients: list[float],
    term_names: list[str],
    *,
    rate_overrides: RateOverrides | None = None,
) -> ScoreResult:
    """Evaluate the conditional logit for one lineup.

    ``coefficients`` and ``term_names`` come from ``selection_model.json``;
    ``profiles`` from ``selection_profiles.json``. Terms are looked up by name
    rather than by position: the coefficient order is a documented contract, but
    a contract enforced by a dictionary lookup fails loudly if it is broken,
    while one enforced by counting fails silently with plausible numbers.
    """
    zones: list[str] = profiles["zones"]
    rim: list[float] = profiles["rim"]
    three: list[float] = profiles["three"]
    theta = dict(zip(term_names, coefficients, strict=True))

    def coef(name: str) -> float:
        return theta.get(name, 0.0)

    shooter_key = str(request.shooter_id)
    shooter_ratio = profiles["shooter_log_ratio"].get(shooter_key)
    shooter_known = shooter_ratio is not None
    if shooter_ratio is None:
        # Exactly zero, not an arbitrary player: the log ratio of the league mix
        # to itself. The model falls back to its alternative-specific constants.
        shooter_ratio = [0.0] * len(zones)
    shooter_weight = float(profiles["shooter_weight"].get(shooter_key, 0.0))

    team_ratio = [0.0] * len(zones)
    if request.team_id is not None and request.season is not None:
        key = f"{request.team_id}:{request.season % 100:02d}"
        team_ratio = profiles["team_log_ratio"].get(key, team_ratio)

    league_three = float(profiles["league_three_rate"])
    league_rim = float(profiles["league_rim_rate"])

    teammates = [p for p in request.offense if p != request.shooter_id]
    spacing = spacing_min = teammate_rim = 0.0
    if teammates:
        three_rates = [
            _rate(profiles, rate_overrides, "player_three_rate", p, league_three) for p in teammates
        ]
        rim_rates = [
            _rate(profiles, rate_overrides, "player_rim_rate", p, league_rim) for p in teammates
        ]
        spacing = _mean(three_rates) - league_three
        spacing_min = min(three_rates) - league_three
        teammate_rim = _mean(rim_rates) - league_rim

    opp_rim = opp_three = 0.0
    if request.defense:
        opp_rim = (
            _mean(
                [
                    _rate(profiles, rate_overrides, "opp_rim_allowed", p, league_rim)
                    for p in request.defense
                ]
            )
            - league_rim
        )
        opp_three = (
            _mean(
                [
                    _rate(profiles, rate_overrides, "opp_three_allowed", p, league_three)
                    for p in request.defense
                ]
            )
            - league_three
        )

    seconds = (
        float(profiles["seconds_mean"])
        if request.seconds_into_possession is None
        else request.seconds_into_possession
    )
    seconds_z = (seconds - float(profiles["seconds_mean"])) / float(profiles["seconds_std"])
    live = 1.0 if request.live_ball else 0.0
    second_chance = 1.0 if request.second_chance else 0.0
    clutch = 1.0 if request.clutch else 0.0

    def utilities(*, with_lineup: bool) -> list[float]:
        out: list[float] = []
        for z, zone in enumerate(zones):
            u = 0.0 if zone == REFERENCE_ZONE else coef(f"alt_{zone}")
            u += coef("shooter_mix") * shooter_ratio[z]
            u += coef("team_mix") * team_ratio[z]
            u += coef("into_possession_x_rim") * seconds_z * rim[z]
            u += coef("into_possession_x_three") * seconds_z * three[z]
            u += coef("live_ball_x_rim") * live * rim[z]
            u += coef("second_chance_x_rim") * second_chance * rim[z]
            u += coef("clutch_x_three") * clutch * three[z]
            if with_lineup:
                u += coef("spacing_x_three") * spacing * three[z]
                u += coef("spacing_min_x_three") * spacing_min * three[z]
                u += coef("teammate_rim_x_rim") * teammate_rim * rim[z]
                u += coef("opp_rim_allowed_x_rim") * opp_rim * rim[z]
                u += coef("opp_three_allowed_x_three") * opp_three * three[z]
            out.append(u)
        return out

    full = utilities(with_lineup=True)
    mix = _softmax(full)
    baseline = _softmax(utilities(with_lineup=False))

    zone_points: list[float] = [float(v) for v in profiles.get("zone_points", [0.0] * len(zones))]
    points_per_100 = 100.0 * sum(
        (m - b) * p for m, b, p in zip(mix, baseline, zone_points, strict=True)
    )

    return ScoreResult(
        zones=tuple(zones),
        mix=tuple(mix),
        # The lineup terms are all deviations from the league average, so
        # dropping them *is* the league-average lineup. No second profile needed.
        baseline_mix=tuple(baseline),
        utilities=tuple(full),
        lineup_features=(spacing, spacing_min, teammate_rim, opp_rim, opp_three),
        shooter_known=shooter_known,
        shooter_weight=shooter_weight,
        points_per_100=points_per_100,
    )
