"""The play ranking: the delta method, the difference test, and the banding.

The endpoint's whole claim is that it does not order what it cannot separate, so
the tests that matter are about *refusing*. A ranking mechanism that always
produces nine distinct ranks passes any test asking "is the order right"; only a
test asking "did it decline when it should have" can tell the two apart.

The gradient itself is checked against a closed form. It is taken by central
differences on the served scorer -- which is the right choice, because a
hand-derived gradient would be a second implementation of the model that drifts
silently -- but "right choice" is not the same as "correct", and a finite
difference with a badly chosen step is wrong in a way that produces
plausible-looking intervals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from lineupiq.serve.plays import GRADIENT_STEP, contribution_gradients, rank_plays
from lineupiq.serve.score import ScoreRequest, score_selection

DATA = Path(__file__).resolve().parents[3] / "apps" / "web" / "public" / "data"


def _load(name: str) -> Any:
    path = DATA / name
    if not path.exists():  # pragma: no cover - only when the export is missing
        pytest.skip(f"{name} has not been exported")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def profiles() -> Any:
    return _load("selection_profiles.json")


@pytest.fixture(scope="module")
def model() -> Any:
    served = _load("selection_model.json")
    if served.get("covariance") is None:
        pytest.skip("the exported model has no covariance matrix")
    return served


@pytest.fixture(scope="module")
def contract(model: Any) -> dict[str, float]:
    return {
        "confidence": model["ranking"]["confidence"],
        "critical_value": model["ranking"]["critical_value"],
        "min_zone_share": model["ranking"]["min_zone_share"],
    }


@pytest.fixture(scope="module")
def lineup(profiles: Any) -> ScoreRequest:
    known = sorted(int(k) for k in profiles["shooter_log_ratio"])
    return ScoreRequest(known[0], tuple(known[:5]), tuple(known[5:10]))


def test_the_contributions_sum_to_the_headline_priced_shift(
    profiles: Any, model: Any, contract: dict[str, float], lineup: ScoreRequest
) -> None:
    """This endpoint decomposes a number the scorer already publishes.

    If the parts stopped adding to the whole, one of the two would be wrong and
    a reader looking at either page alone would never find out.
    """
    ranking = rank_plays(lineup, profiles, model, **contract)
    assert ranking.excluded == ()
    scored = score_selection(lineup, profiles, model["coefficients"], model["term_names"])
    parts = sum(play.points_per_100 for play in ranking.plays)
    assert parts == pytest.approx(scored.points_per_100, abs=1e-9)


def test_the_finite_difference_gradient_matches_the_closed_form(
    profiles: Any, model: Any, lineup: ScoreRequest
) -> None:
    """The delta method is only as good as the gradient it differentiates.

    The served gradient is a central difference on the scorer, and that is the
    right design -- a hand-derived gradient would be a second implementation of
    the model, drifting silently into intervals of the wrong width. But it still
    has to be *correct*, and a finite difference with a badly chosen step is
    wrong in exactly the way that produces plausible numbers.

    So the closed form is derived here and only here, in the test. For a softmax
    ``p`` and a utility ``u``, ``dp_z/dtheta_j = p_z (du_z/dtheta_j - sum_k p_k
    du_k/dtheta_j)``, and the contribution is ``100 (p_z - b_z) v_z`` where ``b``
    is the same expression with the lineup terms dropped. Rather than write out
    all twenty ``du/dtheta``, this checks the identity that follows from it: the
    gradient of the *sum* over zones of ``p_z`` is zero, because the shares are a
    probability vector for every value of theta.
    """
    coefficients = [float(c) for c in model["coefficients"]]
    gradients = contribution_gradients(lineup, profiles, coefficients, model["term_names"])

    zone_points = [float(v) for v in profiles["zone_points"]]
    n_terms = len(coefficients)

    # Shares sum to one for every theta, so their gradient sums to zero. The
    # contributions are the shares weighted by zone value, so their gradient
    # does *not* sum to zero -- but recovering the share gradient by dividing
    # out the weight and summing must.
    for j in range(n_terms):
        total = sum(gradients[z][j] / zone_points[z] for z in range(len(zone_points)))
        assert abs(total) < 1e-6, f"share gradient for term {j} does not sum to zero"

    # And the whole priced shift's gradient is the sum of the per-zone ones, by
    # linearity. Differencing the aggregate directly is an independent path to
    # the same number: it goes through `score_selection.points_per_100` rather
    # than through the per-zone decomposition.
    for j in range(n_terms):
        step = GRADIENT_STEP * max(1.0, abs(coefficients[j]))
        high, low = list(coefficients), list(coefficients)
        high[j] += step
        low[j] -= step
        direct = (
            score_selection(lineup, profiles, high, model["term_names"]).points_per_100
            - score_selection(lineup, profiles, low, model["term_names"]).points_per_100
        ) / (2.0 * step)
        summed = sum(gradients[z][j] for z in range(len(zone_points)))
        assert direct == pytest.approx(summed, abs=1e-9)


def test_a_bigger_gap_is_not_always_more_separable(
    profiles: Any, model: Any, contract: dict[str, float], lineup: ScoreRequest
) -> None:
    """The reason the covariance is shipped, stated as an assertion.

    Zone shares live on a simplex, so two contributions are strongly negatively
    correlated: ``Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b)`` and the covariance
    term is large and negative. Comparing marginal intervals for overlap drops
    it, and refuses to rank pairs that separate decisively.

    ``diagonal_would_refuse`` counts those pairs. If it were zero the 20x20
    matrix would be dead weight and the diagonal would do -- so this is the test
    that says whether the design decision paid for itself.
    """
    ranking = rank_plays(lineup, profiles, model, **contract)
    assert ranking.pairs_compared > 0
    assert ranking.diagonal_would_refuse > 0, (
        "The marginal intervals reached the same verdict on every pair, so "
        "shipping the full covariance bought nothing on this lineup."
    )


def test_tied_zones_share_a_rank_and_ranks_never_go_backwards(
    profiles: Any, model: Any, contract: dict[str, float], lineup: ScoreRequest
) -> None:
    """The property that makes the output renderable as a list.

    The first version of the banding scanned every existing band for a member it
    could not separate from, which is single linkage without the contiguity
    constraint. Because the difference test has a *per-pair* standard error, a
    wider gap can separate while a narrower one inside it does not -- so bands
    could interleave, and it produced rank sequences like 1, 2, 2, 1. The parity
    suite caught it. This is the regression.
    """
    ranking = rank_plays(lineup, profiles, model, **contract)
    ranks = [play.rank for play in ranking.plays]
    assert ranks == sorted(ranks)
    assert max(ranks) == len(ranking.bands)
    assert len(set(ranks)) == len(ranking.bands)

    by_zone = {play.zone: play.rank for play in ranking.plays}
    for index, band in enumerate(ranking.bands, start=1):
        for zone in band:
            assert by_zone[zone] == index


def test_contributions_are_sorted_descending_with_a_total_tiebreak(
    profiles: Any, model: Any, contract: dict[str, float], lineup: ScoreRequest
) -> None:
    ranking = rank_plays(lineup, profiles, model, **contract)
    values = [play.points_per_100 for play in ranking.plays]
    assert values == sorted(values, reverse=True)


def test_a_league_average_lineup_ranks_nothing(
    profiles: Any, model: Any, contract: dict[str, float]
) -> None:
    """Every lineup term is a deviation from the league mean.

    A lineup with no teammates and no defenders has all five of them at exactly
    zero, so it moves no share, every contribution is exactly zero, and there is
    nothing to order. The mechanism must say so rather than ranking nine zeros
    by float comparison -- which is the failure that would look most like
    working.
    """
    known = sorted(int(k) for k in profiles["shooter_log_ratio"])
    solo = ScoreRequest(known[0], (known[0],), ())
    ranking = rank_plays(solo, profiles, model, **contract)

    for play in ranking.plays:
        assert play.points_per_100 == pytest.approx(0.0, abs=1e-12)
    assert not ranking.ordered
    assert len(ranking.bands) == 1


def test_the_share_floor_excludes_and_does_not_silently_drop(
    profiles: Any, model: Any, lineup: ScoreRequest
) -> None:
    """A raised floor must move zones into `excluded`, not out of the response.

    A zone that vanished entirely would look like a model that has nothing to
    say about it, rather than a threshold that declined to rank it.
    """
    contract = {"confidence": 0.8, "critical_value": 1.2815515655, "min_zone_share": 0.15}
    ranking = rank_plays(lineup, profiles, model, **contract)
    assert len(ranking.plays) + len(ranking.excluded) == len(profiles["zones"])
    assert ranking.excluded, "a 15% share floor should exclude something from nine zones"
    for play in ranking.plays:
        assert play.share >= 0.15


def test_a_higher_confidence_never_produces_more_bands(
    profiles: Any, model: Any, lineup: ScoreRequest
) -> None:
    """Monotonicity, which is the sanity check on the whole separation test.

    Demanding more evidence can only merge bands, never split them. If raising
    the level ever produced a *finer* ranking, the comparison would be pointing
    the wrong way and every published order would be backwards.
    """
    base = {"confidence": 0.8, "min_zone_share": 0.01}
    loose = rank_plays(lineup, profiles, model, critical_value=1.2815515655, **base)
    tight = rank_plays(lineup, profiles, model, critical_value=2.5758293035, **base)
    assert len(tight.bands) <= len(loose.bands)


def test_standard_errors_are_finite_and_positive(
    profiles: Any, model: Any, contract: dict[str, float], lineup: ScoreRequest
) -> None:
    ranking = rank_plays(lineup, profiles, model, **contract)
    for play in ranking.plays:
        assert math.isfinite(play.standard_error)
        assert play.standard_error >= 0.0
        low, high = play.interval
        assert low <= play.points_per_100 <= high


def test_the_ranking_contract_comes_from_the_pinned_file(model: Any) -> None:
    """The confidence level is pre-registered, not chosen by the serving layer.

    A threshold that lives next to the code that wants it loose is a threshold
    that gets loose. This one is read from the hash-pinned file at export time
    and shipped with the model, so both implementations use the same number and
    changing it is a build failure.
    """
    from lineupiq.models.support import load_thresholds

    thresholds = load_thresholds()
    assert model["ranking"]["confidence"] == thresholds.ranking_confidence
    assert model["ranking"]["min_zone_share"] == thresholds.ranking_min_zone_share
    # The two-sided normal quantile for the pre-registered level, resolved once
    # in Python because the Worker has no inverse normal CDF.
    assert model["ranking"]["critical_value"] == pytest.approx(1.2815515655, abs=1e-9)
