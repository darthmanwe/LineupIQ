"""Properties of a lineup comparison, asserted rather than its values.

None of these tests asserts a number the code produced. A delta-method interval
never raises, and one that is wrong by a factor of two still looks exactly like
a standard error -- so the only useful tests are the ones that would fail if the
arithmetic were wrong in a way a reader could not see. That is the same reason
the covariance tests in ``test_selection.py`` assert properties.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import pytest

from lineupiq.models.support import load_thresholds
from lineupiq.serve.compare import (
    UnprofiledPlayerError,
    compare_lineups,
    profile_share_gradients,
    share_gradients,
)
from lineupiq.serve.plays import contrast_shares, quadratic_form, standard_error
from lineupiq.serve.score import REFERENCE_ZONE, ScoreRequest, score_selection

_ASSETS = pathlib.Path(__file__).resolve().parents[3] / "apps" / "web" / "public" / "data"


def _load(name: str) -> dict[str, Any]:
    path = _ASSETS / name
    if not path.exists():  # pragma: no cover - only when the export has not run
        pytest.skip(f"{name} has not been exported")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profiles() -> dict[str, Any]:
    return _load("selection_profiles.json")


@pytest.fixture(scope="module")
def model() -> dict[str, Any]:
    return _load("selection_model.json")


@pytest.fixture(scope="module")
def contract() -> dict[str, float]:
    thresholds = load_thresholds()
    return {
        "confidence": thresholds.ranking_confidence,
        "critical_value": 1.2815515655446004,
        "omnibus_critical_value": thresholds.comparison_omnibus_critical_value,
    }


def _five(profiles: dict[str, Any], start: int = 0) -> tuple[int, ...]:
    """Five profiled players, taken in a fixed order so the test is stable."""
    known = sorted(int(p) for p in profiles["player_three_rate"])
    return tuple(known[start : start + 5])


def _request(profiles: dict[str, Any], start: int = 0) -> ScoreRequest:
    five = _five(profiles, start)
    return ScoreRequest(shooter_id=five[0], offense=five, defense=())


def test_a_lineup_compared_with_itself_returns_exactly_nothing(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """The placebo identity, at serving time.

    The trade backtest's placebo arm projects a player swapped for himself and
    requires exactly +0.000, on the grounds that a placebo which drifts off zero
    means every other number in the run is measuring a pipeline bug. The same
    argument applies here and the same standard is used: not "close to zero",
    exactly zero, including the standard error -- a difference of a quantity
    with itself has no sampling variability at all, and any float that appears
    there came from an asymmetry in the finite differences.
    """
    request = _request(profiles)
    result = compare_lineups(request, request, profiles, model, **contract)

    assert all(zone.delta_share == 0.0 for zone in result.zones)
    assert all(zone.points_per_100 == 0.0 for zone in result.zones)
    assert all(zone.standard_error == 0.0 for zone in result.zones)
    assert all(zone.variance_profiles == 0.0 for zone in result.zones)
    assert all(zone.variance_coefficients == 0.0 for zone in result.zones)
    assert result.omnibus.degenerate is True
    assert result.omnibus.distinguishable is False


def test_comparing_the_other_way_round_flips_every_sign(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """``compare(A, B) == -compare(B, A)``, intervals included.

    An asymmetry here would mean the variance depends on which lineup was
    nominated as the left one, which is not a property any covariance has.
    """
    left = _request(profiles, 0)
    right = ScoreRequest(left.shooter_id, (left.shooter_id, *_five(profiles, 10)[:4]), ())

    forward = compare_lineups(left, right, profiles, model, **contract)
    backward = compare_lineups(right, left, profiles, model, **contract)

    for a, b in zip(forward.zones, backward.zones, strict=True):
        assert a.zone == b.zone
        assert a.delta_share == pytest.approx(-b.delta_share, abs=1e-15)
        assert a.points_per_100 == pytest.approx(-b.points_per_100, abs=1e-13)
        assert a.standard_error == pytest.approx(b.standard_error, rel=1e-9)
        assert a.interval[0] == pytest.approx(-b.interval[1], abs=1e-13)
        assert a.interval[1] == pytest.approx(-b.interval[0], abs=1e-13)
    assert forward.omnibus.statistic == pytest.approx(backward.omnibus.statistic, rel=1e-9)


def test_the_deltas_sum_to_zero_because_shares_live_on_a_simplex(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    left = _request(profiles, 0)
    right = ScoreRequest(left.shooter_id, (left.shooter_id, *_five(profiles, 20)[:4]), ())
    result = compare_lineups(left, right, profiles, model, **contract)
    assert sum(zone.delta_share for zone in result.zones) == pytest.approx(0.0, abs=1e-12)


def test_league_average_mode_is_the_same_number_the_score_route_publishes(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """The two endpoints must agree where they overlap, or one of them is wrong.

    ``/lineups/score`` already returns ``mix - baseline_mix`` per zone, and
    comparing against the league average is definitionally that same quantity.
    They are routed through one function precisely so this cannot drift, and
    this test is what would notice if somebody ever un-routed it.
    """
    request = _request(profiles, 3)
    coefficients = [float(c) for c in model["coefficients"]]
    scored = score_selection(request, profiles, coefficients, list(model["term_names"]))
    compared = compare_lineups(request, None, profiles, model, **contract)

    for index, zone in enumerate(compared.zones):
        expected = scored.mix[index] - scored.baseline_mix[index]
        assert zone.delta_share == expected
        assert zone.share_right == scored.baseline_mix[index]


def test_the_covariance_term_is_actually_being_used(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """``Var(a - b)`` must beat ``Var(a) + Var(b)`` somewhere.

    If the two lineups' variances were simply added, this would hold nowhere.
    That is the specific mistake worth a test: adding variances is what you get
    by computing each side's interval separately and combining them, it is
    always larger than the truth for correlated estimates, and being too wide
    looks responsible enough that nobody would question it.
    """
    left = _request(profiles, 0)
    right = ScoreRequest(left.shooter_id, (left.shooter_id, *_five(profiles, 30)[:4]), ())
    coefficients = [float(c) for c in model["coefficients"]]
    terms = list(model["term_names"])
    covariance = [[float(v) for v in row] for row in model["covariance"]]

    joint = compare_lineups(left, right, profiles, model, **contract)
    separate_left = share_gradients(left, None, profiles, coefficients, terms)
    separate_right = share_gradients(right, None, profiles, coefficients, terms)

    # Compared on the coefficient term alone. The profile term is a sum of
    # squares either way and carries no cross-lineup covariance, so including it
    # would compare a two-component quantity against a one-component one and
    # measure the wrong thing.
    tighter = 0
    for index, zone in enumerate(joint.zones):
        naive = standard_error(
            quadratic_form(separate_left[index], covariance, separate_left[index])
        ) + standard_error(quadratic_form(separate_right[index], covariance, separate_right[index]))
        scale = abs(100.0 * float(profiles["zone_points"][index]))
        if math.sqrt(zone.variance_coefficients) < naive * scale:
            tighter += 1
    assert tighter == len(joint.zones)


def test_the_profile_variance_is_a_real_term_and_not_a_rounding_artefact(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """Zeroing the rate errors must strictly narrow every non-degenerate zone.

    This is the test that would fail if the profile gradients were silently
    zero -- which is exactly what a typo in the override key would produce, and
    it would produce it without raising anything.
    """
    left = _request(profiles, 0)
    right = ScoreRequest(left.shooter_id, (left.shooter_id, *_five(profiles, 40)[:4]), ())

    with_profiles = compare_lineups(left, right, profiles, model, **contract)
    zeroed = {
        **profiles,
        "player_three_rate_se": dict.fromkeys(profiles["player_three_rate_se"], 0.0),
        "player_rim_rate_se": dict.fromkeys(profiles["player_rim_rate_se"], 0.0),
        "opp_three_allowed_se": dict.fromkeys(profiles["opp_three_allowed_se"], 0.0),
        "opp_rim_allowed_se": dict.fromkeys(profiles["opp_rim_allowed_se"], 0.0),
    }
    without = compare_lineups(left, right, zeroed, model, **contract)

    assert with_profiles.profile_variance_share > 0.0
    assert without.profile_variance_share == 0.0
    moved = 0
    for a, b in zip(with_profiles.zones, without.zones, strict=True):
        assert a.standard_error >= b.standard_error
        if a.standard_error > b.standard_error:
            moved += 1
    assert moved == len(with_profiles.zones)


def test_the_lineup_effect_really_is_two_dimensional(
    profiles: dict[str, Any], model: dict[str, Any]
) -> None:
    """The omnibus has two degrees of freedom because of this identity.

    Every lineup term multiplies either the rim indicator or the three
    indicator, so turning the lineup terms on must shift the nine utilities by
    exactly ``a * rim[z] + b * three[z]`` for two scalars. That is not an
    approximation and it is not asserted to a tolerance -- the utilities are
    sums of the same floats in the same order either way, so the identity is
    exact.

    This is the guard on the pre-registered critical value. A sixth lineup term
    multiplying anything else would break this test, and breaking it is the
    intended way to be forced to revisit the degrees of freedom rather than
    quietly testing a two-parameter hypothesis with an eight-parameter
    threshold.
    """
    from lineupiq.serve.compare import LINEUP_TERM_ATTRIBUTE

    request = _request(profiles, 0)
    coefficients = [float(c) for c in model["coefficients"]]
    terms = list(model["term_names"])
    theta = dict(zip(terms, coefficients, strict=True))

    scored = score_selection(request, profiles, coefficients, terms)
    stripped = score_selection(
        request,
        profiles,
        [
            0.0 if name in {t for t, _ in LINEUP_TERM_ATTRIBUTE} else c
            for name, c in zip(terms, coefficients, strict=True)
        ],
        terms,
    )

    rim_pull = 0.0
    three_pull = 0.0
    for index, (term, attribute) in enumerate(LINEUP_TERM_ATTRIBUTE):
        contribution = theta[term] * scored.lineup_features[index]
        if attribute == "rim":
            rim_pull += contribution
        else:
            three_pull += contribution

    for z in range(len(profiles["zones"])):
        expected = rim_pull * profiles["rim"][z] + three_pull * profiles["three"][z]
        assert scored.utilities[z] - stripped.utilities[z] == pytest.approx(expected, abs=1e-12)


def test_the_zone_level_covariance_would_have_been_rank_deficient(
    profiles: dict[str, Any], model: dict[str, Any]
) -> None:
    """Why the omnibus is not eight-dimensional, recorded as a measurement.

    The first version of this test built the eight-by-eight covariance of the
    nine-zone difference and inverted it. It worked, in the sense that it
    produced a number; it was found out only when the Python and TypeScript
    statistics stopped agreeing to 1e-9 on one case in ninety-eight.

    The cause was not a rounding tolerance that needed loosening. To first order
    the difference between two lineups lives on a two-dimensional manifold, so
    that matrix is very nearly rank two and six of its eight directions carry
    second-order leakage rather than signal. This asserts the concentration
    directly, so the reasoning behind the two-degree-of-freedom test is a
    measurement in the suite rather than a claim in a docstring.

    Note what is *not* asserted: that the small eigenvalues are zero. They are
    not, which is exactly why the eight-dimensional version produced a plausible
    number instead of failing.
    """
    import numpy as np

    from lineupiq.serve.compare import _rate_keys

    left = _request(profiles, 0)
    right = ScoreRequest(left.shooter_id, (left.shooter_id, *_five(profiles, 15)[:4]), ())
    coefficients = [float(c) for c in model["coefficients"]]
    terms = list(model["term_names"])
    covariance = [[float(v) for v in row] for row in model["covariance"]]
    zones = list(profiles["zones"])

    keys = _rate_keys(left, right)
    theta = share_gradients(left, right, profiles, coefficients, terms)
    rates = profile_share_gradients(left, right, profiles, coefficients, terms, keys)
    errors = [float(profiles[se_table][str(player)]) for _, se_table, player in keys]
    free = [z for z, zone in enumerate(zones) if zone != REFERENCE_ZONE]

    matrix = np.array(
        [
            [
                quadratic_form(theta[a], covariance, theta[b])
                + sum(rates[k][a] * rates[k][b] * errors[k] ** 2 for k in range(len(keys)))
                for b in free
            ]
            for a in free
        ]
    )
    eigenvalues = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    # Two directions carry essentially all of the variance.
    assert float(eigenvalues[:2].sum() / eigenvalues.sum()) > 0.99
    # And the third is orders below the first, rather than merely smaller.
    assert float(eigenvalues[2] / eigenvalues[0]) < 1e-3
    # Which is what makes inverting all eight an amplification of rounding.
    assert float(np.linalg.cond(matrix)) > 1e4


def test_the_omnibus_is_exactly_zero_when_the_two_offsets_are(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """The two offsets are sufficient for the whole difference.

    The nine-zone delta is zero if and only if both the rim pull and the three
    pull are, which is what makes a two-parameter test equivalent to the
    nine-zone one it replaced rather than merely cheaper than it.
    """
    request = _request(profiles, 0)
    identical = compare_lineups(request, request, profiles, model, **contract)
    assert identical.omnibus.rim_shift == 0.0
    assert identical.omnibus.three_shift == 0.0
    assert identical.omnibus.degenerate is True

    right = ScoreRequest(request.shooter_id, (request.shooter_id, *_five(profiles, 25)[:4]), ())
    moved = compare_lineups(request, right, profiles, model, **contract)
    assert moved.omnibus.degrees_of_freedom == 2
    assert (moved.omnibus.rim_shift, moved.omnibus.three_shift) != (0.0, 0.0)
    assert any(zone.delta_share != 0.0 for zone in moved.zones)


def test_a_player_with_no_fitted_rate_is_refused_rather_than_scored_as_zero(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """The silent-zero trap, made loud.

    A player below the profile fit's attempt floor inherits the league rate, so
    swapping him in changes the lineup aggregates by nothing and the comparison
    returns exactly 0.000 -- indistinguishable, to a reader, from a real finding
    that the swap does not matter. There is no honest way to serve that, so it
    raises and the route turns it into a refusal that names the player.
    """
    left = _request(profiles, 0)
    unprofiled = 99_999_999
    assert str(unprofiled) not in profiles["player_three_rate"]
    right = ScoreRequest(
        left.shooter_id, (left.shooter_id, *_five(profiles, 0)[1:4], unprofiled), ()
    )

    with pytest.raises(UnprofiledPlayerError) as raised:
        compare_lineups(left, right, profiles, model, **contract)
    assert unprofiled in raised.value.players


def test_the_profile_gradient_matches_a_coarser_difference(
    profiles: dict[str, Any], model: dict[str, Any]
) -> None:
    """An independent check on the override plumbing.

    The gradient is taken at a 1e-4 step; this re-takes it at 1e-2 through a
    different code path (a rewritten profiles table rather than the override
    map) and requires the two to agree to the truncation error of the coarser
    one. A gradient that was silently reading the wrong table would pass every
    other test in this file and fail this one.
    """
    from lineupiq.serve.compare import _rate_keys

    left = _request(profiles, 0)
    coefficients = [float(c) for c in model["coefficients"]]
    terms = list(model["term_names"])
    keys = _rate_keys(left, None)
    analytic = profile_share_gradients(left, None, profiles, coefficients, terms, keys)

    table, _, player = keys[0]
    base = float(profiles[table][str(player)])
    coarse_step = 1e-2
    high = contrast_shares(
        left,
        None,
        {**profiles, table: {**profiles[table], str(player): base + coarse_step}},
        coefficients,
        terms,
    )
    low = contrast_shares(
        left,
        None,
        {**profiles, table: {**profiles[table], str(player): base - coarse_step}},
        coefficients,
        terms,
    )
    for z in range(len(profiles["zones"])):
        coarse = (high[z] - low[z]) / (2.0 * coarse_step)
        assert coarse == pytest.approx(analytic[0][z], abs=1e-4, rel=1e-2)


def test_a_swap_reports_the_terms_it_moved_with_their_pre_registered_verdicts(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """The mechanism block, and the reason it exists.

    ``spacing_x_three`` is the term whose pre-registered sign came back
    contradicted, and it is one of the three an offensive swap moves -- so the
    first swap anybody tries will show a better spacer pushing this shooter
    *away* from threes. Surfacing the verdict beside the number is the whole
    point; a UI that rendered the movement without it would look like a bug in
    the model rather than the published finding it is.
    """
    left = _request(profiles, 0)
    right = ScoreRequest(left.shooter_id, (left.shooter_id, *_five(profiles, 50)[:4]), ())
    result = compare_lineups(left, right, profiles, model, **contract)

    by_term = {term.term: term for term in result.mechanism}
    assert set(by_term) == {
        "spacing_x_three",
        "spacing_min_x_three",
        "teammate_rim_x_rim",
        "opp_rim_allowed_x_rim",
        "opp_three_allowed_x_three",
    }
    spacing = by_term["spacing_x_three"]
    assert spacing.verdict == "DISAGREES"
    assert spacing.expected_sign == 1
    assert spacing.coefficient < 0.0
    # An offensive-only swap cannot move the two opponent terms.
    assert by_term["opp_rim_allowed_x_rim"].feature_delta == 0.0
    assert by_term["opp_three_allowed_x_three"].feature_delta == 0.0


def test_every_standard_error_is_finite_and_non_negative(
    profiles: dict[str, Any], model: dict[str, Any], contract: dict[str, float]
) -> None:
    """A nan here would surface much later and look like a data problem."""
    left = _request(profiles, 0)
    for start in (5, 15, 25, 35, 45):
        right = ScoreRequest(left.shooter_id, (left.shooter_id, *_five(profiles, start)[:4]), ())
        result = compare_lineups(left, right, profiles, model, **contract)
        for zone in result.zones:
            assert math.isfinite(zone.standard_error)
            assert zone.standard_error >= 0.0
            assert zone.variance_profiles >= 0.0
            assert zone.variance_coefficients >= 0.0
            assert zone.interval[0] <= zone.points_per_100 <= zone.interval[1]


def test_the_rate_floor_is_currently_shadowed_by_the_support_floor(
    profiles: dict[str, Any], model: dict[str, Any]
) -> None:
    """The unprofiled-player refusal cannot fire on this snapshot, and that is fine.

    `min_profile_attempts` is 20 and the directional attempt floor is 30, over
    the same corpus -- so every player the support contract is willing to say
    anything about already has a fitted shooting rate, and the support gate
    always fires first. Measured, not assumed: zero players in the current
    export sit between the two.

    The guard stays, and this test is why it can be trusted to stay correct
    rather than quietly rot. It is defence against the two floors diverging --
    raise the profile floor, or export a roster from a wider frame than the one
    the profiles were fitted on, and this test fails on the same commit that
    makes the guard reachable. Deleting the guard because "it never fires" would
    then be exactly wrong.

    What is not acceptable is the guard being absent: a player with no fitted
    rate silently inherits the league rate, so the comparison returns exactly
    0.000, which reads like a finding and is a missing row.
    """
    import json
    import pathlib

    from lineupiq.models.support import load_thresholds

    thresholds = load_thresholds()
    assert thresholds.comparison_min_profile_attempts <= thresholds.directional_attempts

    players = json.loads((_ASSETS / "players.json").read_text(encoding="utf-8"))["players"]
    exposed = sorted(
        int(player_id)
        for player_id, row in players.items()
        if row["attempts"] >= thresholds.directional_attempts
        and player_id not in profiles["player_three_rate"]
    )
    assert exposed == [], (
        "These players clear the support floor but have no fitted shooting rate, so "
        f"the comparison endpoint's NO_FITTED_RATE branch is now reachable: {exposed}. "
        "That is not a bug -- it is the guard doing its job -- but the refusal is now "
        "something a user can hit, and the docs that call it unreachable are stale."
    )
    assert pathlib.Path(_ASSETS / "players.json").exists()
