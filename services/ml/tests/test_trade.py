"""Tests for move detection, power analysis and the trade projection.

Two of these are the ones that matter.

``test_projecting_a_player_against_himself_is_exactly_zero`` is the placebo
identity. If swapping a player for himself projects any change at all, the
backtest's placebo arm is measuring a pipeline bug and every real number beside
it is worthless.

``test_power_analysis_refuses_to_be_optimistic`` pins the pre-commitment: at the
sample sizes three seasons of mid-season trades provide, the minimum detectable
effect is larger than the effects being claimed, and the verdict has to say so.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lineupiq.models.moves import (
    CLAIMED_EFFECT_PER_100,
    MIN_POSSESSIONS_EITHER_SIDE,
    detect_moves,
    player_team_by_game,
    power_analysis,
    residual_sd_of_team_rating_change,
    team_game_ratings,
)
from lineupiq.models.rapm import RapmFit, RapmReport
from lineupiq.models.trade import (
    MINUTES_RULES,
    project_swap,
    rule_by_name,
    variance_decomposition,
)


def _possessions_with_a_move(
    n_per_stint: int = 400,
    mover: int = 200_001,
    team_a: int = 10,
    team_b: int = 20,
    opponent: int = 99,
    game_prefix_a: str = "0022300",
    game_prefix_b: str = "0022380",
) -> pl.DataFrame:
    """A mover who plays for team A, then team B, with enough time on each.

    Both sides of every possession are real teams, and each game contains
    possessions in both directions -- otherwise a team never appears as both
    offence and defence and ``team_game_ratings`` has nothing to join.
    """
    rows: list[tuple[str, int, int, list[int], list[int], int]] = []
    for prefix, team, mates in (
        (game_prefix_a, team_a, [1, 2, 3, 4]),
        (game_prefix_b, team_b, [5, 6, 7, 8]),
    ):
        for i in range(n_per_stint):
            game = f"{prefix}{i // 20:03d}"
            rows.append((game, team, opponent, [mover, *mates], [90, 91, 92, 93, 94], 2022))
            # The opponent's possession in the same game, so both teams have an
            # offensive and a defensive rating for it.
            rows.append((game, opponent, team, [90, 91, 92, 93, 94], [mover, *mates], 2022))

    return pl.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "offense_team_id": [r[1] for r in rows],
            "defense_team_id": [r[2] for r in rows],
            "off_lineup": [r[3] for r in rows],
            "def_lineup": [r[4] for r in rows],
            "season": [r[5] for r in rows],
            "points": [1.1] * len(rows),
            "home_team_id": [team_a] * len(rows),
            "stint_quality": ["VALID"] * len(rows),
            "boundary_ambiguous": [False] * len(rows),
        }
    )


def test_player_team_is_derived_from_both_sides_of_the_floor() -> None:
    frame = _possessions_with_a_move()
    membership = player_team_by_game(frame)

    mover = membership.filter(pl.col("player_id") == 200_001)
    assert set(mover["team_id"].to_list()) == {10, 20}
    # Defenders resolve too, from the defensive possessions alone.
    assert 90 in membership["player_id"].to_list()


def test_detects_a_move_with_enough_evidence_on_both_sides() -> None:
    moves = detect_moves(_possessions_with_a_move())
    mover = [m for m in moves if m.player_id == 200_001]

    assert len(mover) == 1
    move = mover[0]
    assert move.from_team_id == 10
    assert move.to_team_id == 20
    assert move.mid_season
    assert move.kind == "mid-season"
    assert move.possessions_before >= MIN_POSSESSIONS_EITHER_SIDE
    assert move.possessions_after >= MIN_POSSESSIONS_EITHER_SIDE


def test_a_move_with_too_little_time_is_not_evaluable() -> None:
    """A player with ten possessions on the new team cannot be scored."""
    moves = detect_moves(_possessions_with_a_move(n_per_stint=20))
    assert [m for m in moves if m.player_id == 200_001] == []


def test_three_teams_produce_two_moves_with_disjoint_windows() -> None:
    frame = pl.concat(
        [
            _possessions_with_a_move(400, 200_001, 10, 20, 99, "0022310", "0022320"),
            _possessions_with_a_move(400, 200_001, 20, 30, 98, "0022330", "0022340"),
        ]
    )
    moves = [m for m in detect_moves(frame) if m.player_id == 200_001]
    pairs = {(m.from_team_id, m.to_team_id) for m in moves}
    assert (10, 20) in pairs
    assert (20, 30) in pairs
    assert len(moves) == 2
    # The windows must not overlap: the 10->20 move's "after" possessions stop
    # where the 20->30 move begins.
    first = next(m for m in moves if m.to_team_id == 20)
    second = next(m for m in moves if m.to_team_id == 30)
    assert first.first_game_after < second.first_game_after


def test_team_ratings_are_symmetric_by_construction() -> None:
    """A team's offensive rating is its opponent's defensive rating."""
    frame = _possessions_with_a_move()
    ratings = team_game_ratings(frame)
    assert ratings.height > 0
    assert (ratings["off_rating"] > 0).all()
    np.testing.assert_allclose(
        ratings["net_rating"].to_numpy(),
        (ratings["off_rating"] - ratings["def_rating"]).to_numpy(),
    )


def _report(off: dict[int, float], dfn: dict[int, float], se: float = 1.0) -> RapmReport:
    players = sorted(set(off) | set(dfn))
    fit = RapmFit(
        players=players,
        off_rapm=off,
        def_rapm=dfn,
        home_advantage=2.0,
        league_ppp=1.12,
        lambda_offence=1000.0,
        lambda_defence=1000.0,
        effective_df=200.0,
        condition_number=50.0,
        n_possessions=100_000,
        cv_mse=1.2,
        off_se=dict.fromkeys(players, se),
        def_se=dict.fromkeys(players, se),
    )
    return RapmReport(fit=fit, reliability={}, co_occurrence={}, lambda_trace=[])


def test_projecting_a_player_against_himself_is_exactly_zero() -> None:
    """The placebo identity. Any drift here invalidates the whole backtest."""
    report = _report({1: 3.0, 2: -1.0}, {1: 1.0, 2: 0.5})
    projection = project_swap(
        report,
        player_in=1,
        player_out=1,
        rule=MINUTES_RULES[0],
        possessions_vacated=1_500,
        team_possessions=8_000,
    )
    assert projection.delta_offence == 0.0
    assert projection.delta_defence == 0.0
    assert projection.delta_net == 0.0
    assert projection.se_from_minutes == 0.0


def test_a_better_player_projects_a_positive_delta() -> None:
    report = _report({1: 4.0, 2: -2.0}, {1: 1.0, 2: -1.0})
    better = project_swap(
        report,
        player_in=1,
        player_out=2,
        rule=MINUTES_RULES[0],
        possessions_vacated=2_000,
        team_possessions=8_000,
    )
    worse = project_swap(
        report,
        player_in=2,
        player_out=1,
        rule=MINUTES_RULES[0],
        possessions_vacated=2_000,
        team_possessions=8_000,
    )
    assert better.delta_net > 0
    assert worse.delta_net == pytest.approx(-better.delta_net)


def test_the_delta_scales_with_the_minutes_actually_affected() -> None:
    """Replacing a reserve cannot move a team as much as replacing a starter."""
    report = _report({1: 4.0, 2: -2.0}, {1: 0.0, 2: 0.0})
    starter = project_swap(
        report,
        player_in=1,
        player_out=2,
        rule=MINUTES_RULES[0],
        possessions_vacated=4_000,
        team_possessions=8_000,
    )
    reserve = project_swap(
        report,
        player_in=1,
        player_out=2,
        rule=MINUTES_RULES[0],
        possessions_vacated=400,
        team_possessions=8_000,
    )
    assert starter.delta_net > reserve.delta_net
    assert starter.delta_net == pytest.approx(10 * reserve.delta_net)


def test_the_minutes_rule_changes_the_answer_and_is_reported() -> None:
    """Toggling the rule must move the number, and the number must say which."""
    report = _report({1: 4.0, 2: -2.0}, {1: 0.0, 2: 0.0})
    results = {
        rule.name: project_swap(
            report,
            player_in=1,
            player_out=2,
            rule=rule,
            possessions_vacated=2_000,
            team_possessions=8_000,
        )
        for rule in MINUTES_RULES
    }
    assert results["inherit"].delta_net > results["conservative"].delta_net
    for name, projection in results.items():
        assert projection.minutes_rule == name


def test_variance_decomposition_names_the_dominant_term() -> None:
    report = _report({1: 4.0, 2: -2.0}, {1: 0.0, 2: 0.0}, se=0.05)
    projections = [
        project_swap(
            report,
            player_in=1,
            player_out=2,
            rule=rule,
            possessions_vacated=2_000,
            team_possessions=8_000,
        )
        for rule in MINUTES_RULES
    ]
    decomposition = variance_decomposition(projections)
    # With tiny coefficient standard errors, the minutes assumption must
    # dominate -- which is the finding that says the ML stack is not the
    # bottleneck.
    assert decomposition["mean_minutes_variance_share"] > 0.5
    assert decomposition["n"] == len(MINUTES_RULES)


def test_unidentified_players_are_flagged_not_silently_projected() -> None:
    report = _report({1: 4.0, 2: -2.0}, {1: 0.0, 2: 0.0})
    report.co_occurrence["non_identified"] = [{"player_id": 1, "max_co_occurrence": 0.97}]
    projection = project_swap(
        report,
        player_in=1,
        player_out=2,
        rule=MINUTES_RULES[0],
        possessions_vacated=2_000,
        team_possessions=8_000,
    )
    assert any("not identified" in w for w in projection.warnings)


def test_missing_player_is_flagged_as_league_average() -> None:
    report = _report({1: 4.0}, {1: 0.0})
    projection = project_swap(
        report,
        player_in=999,
        player_out=1,
        rule=MINUTES_RULES[0],
        possessions_vacated=1_000,
        team_possessions=8_000,
    )
    assert any("no RAPM estimate" in w for w in projection.warnings)


def test_power_analysis_refuses_to_be_optimistic() -> None:
    """Pre-committed: at this n and this noise, no accuracy claim follows.

    Roughly sixty evaluable mid-season moves against a team-rating noise floor
    of about five points per 100 gives an MDE near two, which is the size of the
    effects being claimed. The verdict must be UNDERPOWERED.
    """
    analysis = power_analysis(60, 5.0)
    # (1.96 + 0.8416) * 5 / sqrt(60) = 1.81 per 100, against claimed effects of
    # about 1.0 -- so the backtest cannot separate a real projection from zero.
    assert analysis.mde == pytest.approx(1.81, abs=0.02)
    assert analysis.mde > CLAIMED_EFFECT_PER_100
    assert analysis.verdict == "UNDERPOWERED"
    # A sign-accuracy estimate at n=60 spans roughly +/-13 points.
    assert 0.10 < analysis.sign_accuracy_ci_half_width < 0.15


def test_power_analysis_becomes_adequate_with_enough_data() -> None:
    """The verdict must be able to say yes, or it is not a test."""
    assert power_analysis(5_000, 5.0).verdict == "adequate"
    assert power_analysis(5_000, 5.0).mde < CLAIMED_EFFECT_PER_100


def test_power_analysis_handles_a_degenerate_sample() -> None:
    assert power_analysis(1, 5.0).verdict == "UNDERPOWERED"
    assert power_analysis(1, 5.0).mde == float("inf")


def test_residual_sd_is_positive_and_finite() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for team in range(10):
        for game in range(40):
            rows.append((f"00223{game:05d}", team, 2022, float(rng.normal(0, 8))))
    ratings = pl.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "team_id": [r[1] for r in rows],
            "season": [r[2] for r in rows],
            "net_rating": [r[3] for r in rows],
        }
    )
    sd = residual_sd_of_team_rating_change(ratings)
    assert np.isfinite(sd)
    assert sd > 0


def test_rule_by_name_rejects_an_unknown_rule() -> None:
    assert rule_by_name("inherit").share == 1.0
    with pytest.raises(KeyError, match="unknown minutes rule"):
        rule_by_name("whatever-makes-it-look-good")
