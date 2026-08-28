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
#:
#: **Changed twice.** At version 2 to add the `ranking` block for
#: `/lineups/optimal-plays`, and at version 3 to add the `comparison` block for
#: `/lineups/compare`. Both are additions, not loosenings, and the difference
#: between those two is the whole value of the pin -- so it is not left to a
#: commit message. Every earlier number is asserted *by value* by
#: `test_the_floors_are_the_ones_the_arithmetic_implies` (version 1) and
#: `test_the_ranking_confidence_was_fixed_before_any_ranking_existed`
#: (version 2), so the hash catches any edit and those tests catch the specific
#: edit that would matter. A future diff that lowers the possession floor and
#: updates this constant still fails.
#:
#: The digest is taken over **LF-normalised** bytes. Rewriting the file from
#: Windows put CRLF in the working copy while `.gitattributes` kept LF in the
#: repository, and the identical thresholds hashed differently on the two
#: platforms -- so the pin failed on every Linux job for a reason that had
#: nothing to do with the thresholds. A pin that a line-ending change can break
#: is a pin whose obvious repair is to edit the constant, and the constant is
#: the claim.
PRE_REGISTERED_SHA256 = "2ff4b3f2bbb0243271b1aaf7e098fa53d6cc8b76d85420bb49db083d4d938f1c"


def test_the_pin_survives_a_line_ending_change(tmp_path: object) -> None:
    """A CRLF checkout must not read as a changed pre-registration.

    This is the regression for the failure above: the raw-bytes version of the
    digest made a Windows checkout and a Linux checkout of the same commit
    disagree about whether the thresholds had been edited.
    """
    import pathlib

    from lineupiq.models.support import _thresholds_path

    source = _thresholds_path().read_bytes().replace(b"\r\n", b"\n")
    directory = pathlib.Path(str(tmp_path))
    unix = directory / "lf.json"
    windows = directory / "crlf.json"
    unix.write_bytes(source)
    windows.write_bytes(source.replace(b"\n", b"\r\n"))

    assert thresholds_hash(unix) == thresholds_hash(windows)
    assert thresholds_hash(unix) == PRE_REGISTERED_SHA256


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

    # Version-1 floors, asserted by value so an "addition" cannot smuggle one
    # through. Every number below predates any result that depends on it.
    assert thresholds.reportable_attempts == 100
    assert thresholds.directional_possessions == 25
    assert thresholds.directional_attempts == 30
    assert thresholds.min_zone_attempts == 10
    assert thresholds.conformal_bin_min_n == 50
    assert thresholds.min_reportable_minutes_share == 0.6


def test_the_ranking_confidence_was_fixed_before_any_ranking_existed() -> None:
    """80%, and deliberately not 95%.

    This level does not gate whether a number is *shown* -- it gates whether a
    list is presented as **ordered**. Those are different errors with different
    costs. Refusing to order two plays that really do differ wastes information
    the model has; ordering two that do not differ invents information it does
    not. At nine zones the second is the one that would happen constantly, so
    the level is set where the plan pre-registered it and no looser.

    Kept in the pinned file rather than as a constant in the ranking module,
    because a threshold that lives next to the code that wants it loose is a
    threshold that will get loose.
    """
    thresholds = load_thresholds()
    assert thresholds.ranking_confidence == 0.8
    assert thresholds.ranking_min_zone_share == 0.01


def test_the_comparison_thresholds_were_fixed_before_any_comparison_existed() -> None:
    """Version 3, pinned by value for the same reason version 2 is.

    The critical value is the one number here a reader cannot check by eye, so
    it is re-derived rather than asserted against itself: chi-square at 80% on
    **two** degrees of freedom.

    Two, not eight, and the difference is not a detail. Every lineup term in the
    selection model multiplies either the rim indicator or the three indicator,
    so a lineup's whole effect on the nine utilities is `a*rim + b*three` and it
    has exactly two parameters. The nine shares do live on an eight-dimensional
    simplex, but a lineup cannot move them in eight independent directions --
    the eight-by-eight covariance of the difference is structurally rank two,
    and inverting it inverts six directions of rounding error.
    `test_the_lineup_effect_really_is_two_dimensional` pins the decomposition
    itself, so a sixth term with a different zone structure fails there rather
    than silently invalidating this number.

    scipy is imported inside the test rather than at module scope: the point is
    to confirm the committed constant against an independent computation, and a
    constant that is only ever compared to itself is not pinned to anything.
    """
    from scipy import stats

    thresholds = load_thresholds()
    assert thresholds.comparison_omnibus_confidence == 0.8
    assert thresholds.comparison_min_profile_attempts == 20

    expected = float(stats.chi2.ppf(thresholds.comparison_omnibus_confidence, 2))
    assert abs(thresholds.comparison_omnibus_critical_value - expected) < 1e-12


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
