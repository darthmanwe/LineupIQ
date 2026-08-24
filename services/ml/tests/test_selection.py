"""Tests for the shot-selection model.

The gradient check is the important one. A hand-derived gradient that is subtly
wrong does not raise -- L-BFGS follows it to a nearby point, reports success,
and produces a log loss that looks entirely reasonable. Finite differences are
the only cheap way to know the optimiser is descending the function that is
actually being claimed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lineupiq.models.priors import fit_dirichlet_prior
from lineupiq.models.selection import (
    LINEUP_TERM_NAMES,
    SELECTION_TERMS,
    TERM_NAMES,
    ConditionalLogit,
    SelectionDesign,
    build_selection_design,
    fit_selection_profiles,
    lineup_wide_indices,
    score_selection,
    wide_feature_names,
    wide_features,
    zone_attribute,
)
from lineupiq.transform.zones import ZONE_IDS


def _synthetic_shots(n: int = 900, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    players = [200000 + i for i in range(24)]
    teams = [1610612700 + i for i in range(6)]
    return pl.DataFrame(
        {
            "game_id": [f"00223{i // 30:05d}" for i in range(n)],
            "event_num": list(range(n)),
            "shooter_id": [int(rng.choice(players)) for _ in range(n)],
            "team_id": [int(rng.choice(teams)) for _ in range(n)],
            "season": [2023] * n,
            "period": rng.integers(1, 5, n).tolist(),
            "seconds_remaining": rng.integers(0, 720, n).tolist(),
            "zone_id": [str(z) for z in rng.choice(list(ZONE_IDS), n)],
            "seconds_into_possession": rng.uniform(0, 24, n).tolist(),
            "live_ball_start": rng.random(n) < 0.3,
            "is_second_chance": rng.random(n) < 0.12,
            "lineup_for": [
                [int(p) for p in rng.choice(players, 5, replace=False)] for _ in range(n)
            ],
            "lineup_against": [
                [int(p) for p in rng.choice(players, 5, replace=False)] for _ in range(n)
            ],
        }
    )


def test_zone_attributes_partition_the_taxonomy() -> None:
    rim = zone_attribute("rim")
    three = zone_attribute("three")
    assert rim.shape == three.shape == (len(ZONE_IDS),)
    # A zone cannot be both at the rim and behind the arc.
    assert not np.any((rim > 0) & (three > 0))
    assert rim.sum() == 2
    assert three.sum() == 4


def test_term_names_are_unique_and_ordered() -> None:
    assert len(TERM_NAMES) == len(set(TERM_NAMES))
    assert tuple(t.name for t in SELECTION_TERMS) == TERM_NAMES
    # Eight alternative-specific constants: nine zones, one reference.
    assert sum(1 for t in SELECTION_TERMS if t.kind == "alt") == len(ZONE_IDS) - 1


def test_every_lineup_term_has_a_preregistered_sign() -> None:
    """The sign audit is worthless if a lineup term can opt out of it."""
    for term in SELECTION_TERMS:
        if term.is_lineup:
            assert term.expected_sign in (-1, 1), term.name


def _naive_utilities(design: SelectionDesign, theta: np.ndarray) -> np.ndarray:
    """The obvious implementation: one outer product per interaction term.

    Kept only as a reference for the optimised path. This is what
    ``SelectionDesign.utilities`` used to do, and at three seasons it allocated
    about 620 MB per call.
    """
    n_alt = design.alt_matrix.shape[1]
    u = np.tile(design.alt_matrix @ theta[:n_alt], (design.n, 1))
    index = n_alt
    for matrix in design.pair_matrices.values():
        u = u + theta[index] * matrix
        index += 1
    for name, shot in design.inter_shot.items():
        u = u + theta[index] * np.outer(shot, design.inter_alt[name])
        index += 1
    return u


def test_utilities_match_the_naive_reference() -> None:
    """The grouped, in-place hot path must be arithmetically identical.

    ``utilities`` collapses ten outer products into one masked add per distinct
    zone attribute and writes into a reused buffer. That is a rewrite of the
    function being optimised, which is exactly where a change stops being a
    speedup and becomes a different model.
    """
    frame = _synthetic_shots()
    design = build_selection_design(frame, fit_selection_profiles(frame))
    rng = np.random.default_rng(17)

    for _ in range(5):
        theta = rng.normal(scale=0.7, size=len(design.term_names))
        np.testing.assert_allclose(
            design.utilities(theta), _naive_utilities(design, theta), rtol=0, atol=1e-12
        )

    # A zero coefficient is skipped by the optimised path; it must still agree.
    theta = np.zeros(len(design.term_names))
    np.testing.assert_allclose(
        design.utilities(theta), _naive_utilities(design, theta), rtol=0, atol=1e-12
    )


def test_utilities_respect_a_supplied_buffer() -> None:
    frame = _synthetic_shots()
    design = build_selection_design(frame, fit_selection_profiles(frame))
    theta = np.full(len(design.term_names), 0.15)

    buffer = np.empty((design.n, design.n_zones))
    returned = design.utilities(theta, out=buffer)
    assert returned is buffer
    np.testing.assert_allclose(buffer, design.utilities(theta), atol=1e-12)


def test_attribute_grouping_covers_every_interaction_once() -> None:
    frame = _synthetic_shots()
    design = build_selection_design(frame, fit_selection_profiles(frame))
    positions = [position for _, _, members in design.attribute_groups() for position, _ in members]
    assert sorted(positions) == sorted(set(positions))
    assert len(positions) == len(design.inter_shot)
    # Two distinct zone attributes in this specification: rim and three.
    assert len(design.attribute_groups()) == 2


def test_analytic_gradient_matches_finite_differences() -> None:
    frame = _synthetic_shots()
    profiles = fit_selection_profiles(frame)
    design = build_selection_design(frame, profiles)
    model = ConditionalLogit(l2=0.03)

    rng = np.random.default_rng(3)
    theta = rng.normal(scale=0.4, size=len(design.term_names))
    _, analytic = model.objective(theta, design)

    step = 1e-6
    numeric = np.empty_like(theta)
    for k in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[k] += step
        down[k] -= step
        numeric[k] = (model.objective(up, design)[0] - model.objective(down, design)[0]) / (
            2 * step
        )

    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-8)


def test_fit_recovers_a_planted_effect() -> None:
    """Plant a spacing effect in synthetic data and check it is found.

    Without this, a coefficient of zero is indistinguishable from a model that
    cannot see the feature at all.
    """
    rng = np.random.default_rng(11)
    n = 6000
    three = zone_attribute("three")
    spacing = rng.normal(size=n)
    beta = 1.8

    base = np.log(np.array([0.22, 0.10, 0.06, 0.10, 0.08, 0.07, 0.07, 0.17, 0.13]))
    utilities = base[None, :] + beta * np.outer(spacing, three)
    probability = np.exp(utilities - utilities.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    chosen = np.array([rng.choice(len(ZONE_IDS), p=row) for row in probability])

    zeros = np.zeros(n)
    design = SelectionDesign(
        n=n,
        y=chosen,
        alt_matrix=np.eye(len(ZONE_IDS))[:, 1:],
        pair_matrices={"shooter_mix": np.zeros((n, len(ZONE_IDS)))},
        inter_shot={"spacing_x_three": spacing, "unused": zeros},
        inter_alt={"spacing_x_three": three, "unused": three},
        term_names=(
            *(f"alt_{z}" for z in ZONE_IDS[1:]),
            "shooter_mix",
            "spacing_x_three",
            "unused",
        ),
    )
    fitted = ConditionalLogit(l2=1e-6).fit(design)
    assert fitted.converged
    assert fitted.coefficient("spacing_x_three") == pytest.approx(beta, rel=0.08)
    # A column of zeros must earn a coefficient of zero.
    assert fitted.coefficient("unused") == pytest.approx(0.0, abs=1e-6)


def test_without_zeroes_only_the_named_terms() -> None:
    frame = _synthetic_shots()
    design = build_selection_design(frame, fit_selection_profiles(frame))
    stripped = design.without(LINEUP_TERM_NAMES)

    for name in LINEUP_TERM_NAMES:
        assert not stripped.inter_shot[name].any()
    for name, values in design.inter_shot.items():
        if name not in LINEUP_TERM_NAMES:
            np.testing.assert_array_equal(stripped.inter_shot[name], values)
    # The shape of the problem must not change, or the comparison is not a
    # comparison of the same model.
    assert stripped.term_names == design.term_names
    assert stripped.n == design.n


def test_alt_constants_are_not_penalised() -> None:
    frame = _synthetic_shots()
    design = build_selection_design(frame, fit_selection_profiles(frame))
    mask = design.penalty_mask
    assert mask[: len(ZONE_IDS) - 1].sum() == 0.0
    assert mask[len(ZONE_IDS) - 1 :].all()


def test_predictions_are_a_distribution() -> None:
    frame = _synthetic_shots()
    profiles = fit_selection_profiles(frame)
    design = build_selection_design(frame, profiles)
    p = ConditionalLogit().fit(design).predict_proba(design)
    assert p.shape == (frame.height, len(ZONE_IDS))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert (p > 0).all()


def test_unseen_shooter_falls_back_to_the_league_mix() -> None:
    """An unknown player must contribute exactly zero, not a random profile."""
    frame = _synthetic_shots()
    profiles = fit_selection_profiles(frame)
    stranger = frame.head(1).with_columns(pl.lit(9999999).alias("shooter_id"))
    design = build_selection_design(stranger, profiles)
    np.testing.assert_allclose(design.pair_matrices["shooter_mix"][0], 0.0, atol=1e-12)


def test_wide_features_agree_with_their_names() -> None:
    frame = _synthetic_shots()
    design = build_selection_design(frame, fit_selection_profiles(frame))
    matrix = wide_features(design)
    names = wide_feature_names(design)
    assert matrix.shape == (frame.height, len(names))
    indices = lineup_wide_indices(design)
    assert len(indices) == len(LINEUP_TERM_NAMES)
    assert {names[i] for i in indices} == set(LINEUP_TERM_NAMES)


def test_score_selection_rewards_the_truth() -> None:
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8] * 20)
    n = len(y)
    uniform = np.full((n, len(ZONE_IDS)), 1.0 / len(ZONE_IDS))
    confident = np.full((n, len(ZONE_IDS)), 0.01)
    confident[np.arange(n), y] = 1.0 - 0.01 * (len(ZONE_IDS) - 1)

    poor = score_selection(y, uniform)
    good = score_selection(y, confident)
    assert good.log_loss < poor.log_loss
    assert good.top1_accuracy == 1.0
    assert poor.log_loss == pytest.approx(np.log(len(ZONE_IDS)), rel=1e-9)
    # Uniform predictions carry no information about the three/rim split.
    assert good.three_resolution > poor.three_resolution


def test_dirichlet_prior_shrinks_toward_the_league() -> None:
    rng = np.random.default_rng(5)
    league = np.array([0.22, 0.10, 0.06, 0.10, 0.08, 0.07, 0.07, 0.17, 0.13])
    counts = np.array([rng.multinomial(400, league) for _ in range(200)], dtype=float)
    # One player with four attempts, all from a single zone.
    counts = np.vstack([counts, np.array([4.0, 0, 0, 0, 0, 0, 0, 0, 0])])

    prior = fit_dirichlet_prior(counts)
    mix, weight = prior.shrink(counts)

    np.testing.assert_allclose(mix.sum(axis=1), 1.0, atol=1e-12)
    # The four-attempt player must end up near the league mix, not at 100%.
    assert mix[-1, 0] < 0.5
    assert weight[-1] < weight[0]
    assert weight.min() >= 0.0
    assert weight.max() <= 1.0
