"""The refusal contract, and the pre-registration that makes it credible.

The support thresholds are the product's central promise: they decide when a
number may be shown at all. A promise that can be edited after seeing a result
is not a promise, so the file's hash is **pinned here**.

Changing that constant is the point. It cannot be done quietly -- a diff that
loosens a floor must also edit this test, and editing this test is editing the
claim that the floors were fixed in advance. That is a conversation, not a
commit.
"""

from __future__ import annotations

import pytest

from lineupiq.models.support import (
    LineupSupport,
    Tier,
    assess,
    load_thresholds,
    thresholds_hash,
)

#: SHA-256 of `configs/support_thresholds.json`, fixed before any lineup-level
#: result existed.
#:
#: If this test fails, one of two things happened. Either the thresholds were
#: edited -- in which case the "pre-registered" claim in the README is no longer
#: true and has to come out -- or the file was reformatted, in which case
#: reformatting a pre-registered artefact is itself worth a second look.
PRE_REGISTERED_SHA256 = "c82ee2907151a34887d640fa34cca75bc905f699738461b5e76b7cdf996d088d"


def test_thresholds_are_pre_registered_and_unchanged() -> None:
    assert thresholds_hash() == PRE_REGISTERED_SHA256, (
        "The support thresholds changed. They were fixed before any lineup-level "
        "result was computed, and the README says so. Loosening a floor to make a "
        "demo look better is a build failure by design -- if the change is "
        "legitimate, update this constant *and* the claim it backs."
    )


def test_the_floors_are_the_ones_the_arithmetic_implies() -> None:
    """200 possessions is not a taste; it comes from the standard error.

    A lineup's offensive rating has a standard error of about 115/sqrt(n) per 100
    possessions. At 200 that is +/-8.1, against a true between-lineup spread of
    roughly 6-8. Below 200 the measurement is noisier than the entire signal, so
    a point estimate there is not a weak claim -- it is a meaningless one.
    """
    thresholds = load_thresholds()
    assert thresholds.reportable_possessions == 200
    standard_error = 115.0 / (thresholds.reportable_possessions**0.5)
    assert 7.5 < standard_error < 8.5

    # The directional floor is far lower on purpose: it gates the *player*
    # terms, which have orders of magnitude more evidence than any combination.
    assert thresholds.directional_possessions < thresholds.reportable_possessions
    assert thresholds.directional_attempts < thresholds.reportable_attempts


def _table(possessions: int, attempts: int, lineup: list[int]) -> dict[str, tuple[int, int]]:
    from lineupiq.hashing import lineup_hash

    return {lineup_hash(lineup): (possessions, attempts)}


LINEUP = [201939, 1628369, 203507, 1629027, 202695]


def test_a_well_evidenced_lineup_is_reportable() -> None:
    thresholds = load_thresholds()
    result = assess(
        LINEUP,
        _table(500, 400, LINEUP),
        thresholds,
        dict.fromkeys(LINEUP, 400),
    )
    assert result.tier is Tier.REPORTABLE
    assert result.may_report_point_estimate
    assert not result.counterfactual


def test_a_counterfactual_lineup_of_known_players_is_directional() -> None:
    """The normal case for a trade lineup, and the reason it is not a refusal.

    These five have never shared the floor, so the combination has no evidence.
    Their individual terms have plenty. The honest answer is a direction with an
    interval and no centre -- not a refusal, and certainly not a point estimate.
    """
    thresholds = load_thresholds()
    result = assess(LINEUP, {}, thresholds, dict.fromkeys(LINEUP, 400))
    assert result.tier is Tier.DIRECTIONAL
    assert not result.may_report_point_estimate
    assert result.counterfactual
    assert result.possessions == 0


def test_a_lineup_with_an_unknown_player_is_refused() -> None:
    thresholds = load_thresholds()
    attempts = dict.fromkeys(LINEUP, 400)
    attempts[LINEUP[0]] = 3
    result = assess(LINEUP, _table(500, 3, LINEUP), thresholds, attempts)
    assert result.tier is Tier.REFUSED
    assert not result.may_report_point_estimate
    # The refusal names who fell short. "Not enough data" without "of what" is
    # not an answer.
    assert LINEUP[0] in result.shortfall_players


def test_the_tier_boundary_is_exactly_at_the_floor() -> None:
    """One possession either side of the floor must give different answers."""
    thresholds = load_thresholds()
    attempts = dict.fromkeys(LINEUP, thresholds.reportable_attempts)

    at_floor = assess(
        LINEUP,
        _table(thresholds.reportable_possessions, thresholds.reportable_attempts, LINEUP),
        thresholds,
        attempts,
    )
    below = assess(
        LINEUP,
        _table(thresholds.reportable_possessions - 1, thresholds.reportable_attempts, LINEUP),
        thresholds,
        attempts,
    )
    assert at_floor.tier is Tier.REPORTABLE
    assert below.tier is Tier.DIRECTIONAL


def test_attempts_gate_independently_of_possessions() -> None:
    """A lineup with the possessions but not the attempts is still not reportable.

    Both conditions have to hold. A five-man group can share the floor for 500
    possessions while one of them has barely shot, and a per-zone estimate for
    that player has nothing behind it.
    """
    thresholds = load_thresholds()
    attempts = dict.fromkeys(LINEUP, thresholds.reportable_attempts)
    attempts[LINEUP[2]] = thresholds.directional_attempts

    result = assess(
        LINEUP, _table(500, thresholds.directional_attempts, LINEUP), thresholds, attempts
    )
    assert result.tier is Tier.DIRECTIONAL


@pytest.mark.parametrize("tier", list(Tier))
def test_only_reportable_may_carry_a_point_estimate(tier: Tier) -> None:
    """The one invariant, asserted over every tier rather than the happy path."""
    support = LineupSupport(
        lineup_hash="x" * 32,
        possessions=0,
        min_player_attempts=0,
        tier=tier,
        counterfactual=False,
    )
    assert support.may_report_point_estimate == (tier is Tier.REPORTABLE)
