"""Tests for RAPM.

The important test plants known player effects in synthetic possessions and
checks they are recovered. Without it, a coefficient vector that correlates with
nothing is indistinguishable from a model wired up backwards -- and ridge will
happily return a smooth, plausible-looking set of numbers either way.

The second important test is that game-grouped folds actually group by game.
Splitting possessions from the same game across a fold boundary selects a far
smaller lambda than is justified, and the failure is invisible: the model just
looks better than it is.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lineupiq.models.rapm import (
    CO_OCCURRENCE_CEILING,
    LAMBDA_GRID,
    build_rapm_design,
    co_occurrence_report,
    fit_rapm,
    select_lambda,
    split_half_reliability,
    usable_possessions,
)


def _synthetic_possessions(
    n: int = 30_000,
    n_players: int = 60,
    seed: int = 3,
    effect_scale: float = 0.06,
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Possessions generated from known offensive and defensive player effects.

    Ten distinct players per possession, five a side, and points drawn around a
    league mean shifted by the sum of the offence's offensive effects minus the
    defence's defensive effects.
    """
    rng = np.random.default_rng(seed)
    players = [200_000 + i for i in range(n_players)]
    offence_effect = rng.normal(scale=effect_scale, size=n_players)
    defence_effect = rng.normal(scale=effect_scale, size=n_players)

    off_rows, def_rows, points, games, teams, home = [], [], [], [], [], []
    for i in range(n):
        chosen = rng.choice(n_players, 10, replace=False)
        off_index, def_index = chosen[:5], chosen[5:]
        if pairs:
            # Force a nearly-inseparable pair onto the floor together.
            a, b = pairs[i % len(pairs)]
            off_index = np.array([a, b, *[c for c in off_index if c not in (a, b)][:3]])
        mean = 1.10 + offence_effect[off_index].sum() - defence_effect[def_index].sum()
        points.append(float(rng.normal(mean, 1.1)))
        off_rows.append([players[j] for j in off_index])
        def_rows.append([players[j] for j in def_index])
        games.append(f"00223{i // 100:05d}")
        teams.append(1610612700 + (i % 2))
        home.append(1610612700)

    frame = pl.DataFrame(
        {
            "game_id": games,
            "off_lineup": off_rows,
            "def_lineup": def_rows,
            "points": points,
            "offense_team_id": teams,
            "home_team_id": home,
            "stint_quality": ["VALID"] * n,
            "boundary_ambiguous": [False] * n,
        }
    )
    return frame, offence_effect, defence_effect


def test_design_has_ten_players_and_a_home_indicator_per_possession() -> None:
    frame, _, _ = _synthetic_possessions(n=500)
    design = build_rapm_design(frame)

    assert design.n_possessions == 500
    assert design.matrix.shape[1] == 2 * design.n_players + 1
    # Ten player entries per row. The home column is 0 for away offence, so
    # counting nonzeros per row gives 10 or 11.
    per_row = np.asarray((design.matrix != 0).sum(axis=1)).ravel()
    assert set(np.unique(per_row)) <= {10, 11}
    # Every possession is attributed to exactly five offensive players.
    offence_block = design.matrix[:, : design.n_players]
    np.testing.assert_array_equal(np.asarray(offence_block.sum(axis=1)).ravel(), 5.0)


def test_home_advantage_column_is_never_penalised() -> None:
    frame, _, _ = _synthetic_possessions(n=200)
    design = build_rapm_design(frame)
    penalty = design.penalty_vector(1000.0, 2000.0)

    assert penalty[-1] == 0.0
    assert (penalty[: design.n_players] == 1000.0).all()
    assert (penalty[design.n_players : 2 * design.n_players] == 2000.0).all()


def test_recovers_planted_player_effects() -> None:
    """The whole point: do the coefficients track the truth?

    Ridge shrinks, so the recovered effects are attenuated rather than equal to
    the planted ones. Correlation is the right check; equality would fail for a
    correct implementation.
    """
    frame, offence, defence = _synthetic_possessions(n=40_000, n_players=60, effect_scale=0.08)
    report = fit_rapm(frame, n_folds=3, measure_boundary_sensitivity=False)
    fit = report.fit

    recovered_off = np.array([fit.off_rapm[200_000 + i] for i in range(len(offence))])
    recovered_def = np.array([fit.def_rapm[200_000 + i] for i in range(len(defence))])

    # Planted effects are per possession; coefficients are per 100.
    assert np.corrcoef(recovered_off, offence)[0, 1] > 0.75
    # def_rapm is negated so higher is better, and the generator subtracts the
    # defensive effect -- so a large planted `defence_effect` is good defence.
    assert np.corrcoef(recovered_def, defence)[0, 1] > 0.75
    assert fit.league_ppp == pytest.approx(1.10, abs=0.05)


def test_defensive_sign_convention_is_higher_is_better() -> None:
    """A defensive number where -3 means good reads wrong everywhere.

    The raw coefficient is points conceded, so it must be negated exactly once.
    This test fails if that negation is dropped or applied twice.
    """
    frame, _, defence = _synthetic_possessions(n=30_000, n_players=40, effect_scale=0.10)
    fit = fit_rapm(frame, n_folds=3, measure_boundary_sensitivity=False).fit

    best = int(np.argmax(defence))
    worst = int(np.argmin(defence))
    assert fit.def_rapm[200_000 + best] > fit.def_rapm[200_000 + worst]


def test_lambda_selection_groups_folds_by_game() -> None:
    """No game may appear in both the training and held-out side of a fold."""
    from lineupiq.models.rapm import _game_folds

    frame, _, _ = _synthetic_possessions(n=4_000)
    design = build_rapm_design(frame)
    folds = _game_folds(design.game_ids, n_folds=4, seed=0)

    assert len(folds) == 4
    for held in folds:
        held_games = set(design.game_ids[held])
        train_games = set(design.game_ids[~held])
        assert not (held_games & train_games)
    # Every possession is held out exactly once.
    np.testing.assert_array_equal(sum(folds), np.ones(design.n_possessions))


def test_selected_lambda_is_inside_the_grid() -> None:
    """An endpoint means the grid was too narrow, and that must be visible."""
    frame, _, _ = _synthetic_possessions(n=8_000, n_players=40)
    design = build_rapm_design(frame)
    lambda_offence, lambda_defence, mse, trace = select_lambda(design, n_folds=3)

    assert lambda_offence in LAMBDA_GRID
    assert lambda_defence in LAMBDA_GRID
    assert len(trace) == len(LAMBDA_GRID) ** 2
    assert mse > 0
    assert min(row["mse"] for row in trace) == pytest.approx(mse)


def test_effective_df_is_far_below_the_column_count() -> None:
    """Regularised means spending fewer parameters than you have."""
    frame, _, _ = _synthetic_possessions(n=10_000, n_players=50)
    report = fit_rapm(frame, n_folds=3, measure_boundary_sensitivity=False)

    n_columns = 2 * len(report.fit.players) + 1
    assert 0 < report.fit.effective_df < n_columns
    assert np.isfinite(report.fit.condition_number)


def test_co_occurrence_flags_an_inseparable_pair() -> None:
    """Two players who always appear together are not separately identified."""
    frame, _, _ = _synthetic_possessions(n=6_000, n_players=40, pairs=[(0, 1)])
    design = build_rapm_design(frame)
    report = co_occurrence_report(design)

    flagged = {row["player_id"] for row in report["non_identified"]}
    assert 200_000 in flagged
    assert 200_001 in flagged
    assert report["ceiling"] == CO_OCCURRENCE_CEILING


def test_split_half_reliability_is_high_when_effects_are_real() -> None:
    frame, _, _ = _synthetic_possessions(n=40_000, n_players=40, effect_scale=0.10)
    design = build_rapm_design(frame)
    result = split_half_reliability(design, 500.0, 500.0, min_possessions=50)

    assert result["n_players"] > 10
    assert result["off_split_half_r"] > 0.3
    # Spearman-Brown must lift a positive half-to-half correlation, never lower it.
    assert result["off_full_sample_reliability"] > result["off_split_half_r"]


def test_split_half_reliability_is_near_zero_when_there_is_no_signal() -> None:
    """The control: with no planted effects, reliability must collapse.

    A reliability metric that reports high agreement on pure noise is measuring
    the penalty, not the players.
    """
    frame, _, _ = _synthetic_possessions(n=30_000, n_players=40, effect_scale=0.0)
    design = build_rapm_design(frame)
    result = split_half_reliability(design, 2_000.0, 2_000.0, min_possessions=50)

    assert abs(result["off_split_half_r"]) < 0.25
    assert abs(result["def_split_half_r"]) < 0.25


def test_usable_possessions_rejects_incomplete_lineups() -> None:
    frame, _, _ = _synthetic_possessions(n=100)
    broken = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < 10)
        .then(pl.col("off_lineup").list.head(4))
        .otherwise(pl.col("off_lineup"))
        .alias("off_lineup")
    )
    assert usable_possessions(broken).height == 90


def test_boundary_ambiguous_can_be_excluded() -> None:
    frame, _, _ = _synthetic_possessions(n=100)
    flagged = frame.with_columns(
        (pl.int_range(pl.len()) < 20).alias("boundary_ambiguous"),
    )
    assert usable_possessions(flagged).height == 100
    assert usable_possessions(flagged, exclude_boundary_ambiguous=True).height == 80
