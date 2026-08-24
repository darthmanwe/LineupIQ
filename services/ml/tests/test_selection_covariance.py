"""The selection model's standard errors, checked against things known in advance.

A covariance matrix is the easiest thing in a model to get wrong and the hardest
to notice: it never raises, and a number that is too small by a factor of two
still looks like a standard error. So none of these tests assert a value. They
assert properties that must hold for *any* correct implementation and cannot hold
for most incorrect ones:

- the finite-difference Hessian agrees with a second-difference Hessian of the
  loss itself, which is an independent route to the same quantity,
- standard errors shrink as ``1/sqrt(n)``, which is the defining property of an
  asymptotic standard error and fails immediately if the ``1 / n`` in
  :meth:`coefficient_covariance` is missing or misplaced,
- the covariance is symmetric and positive definite,
- a term the data cannot pin down comes back as `indeterminate` rather than as a
  confirmation of whichever sign it happened to land on.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lineupiq.models.selection import (
    ConditionalLogit,
    build_selection_design,
    fit_selection_profiles,
)
from lineupiq.transform.zones import ZONE_IDS


def _corpus(n: int, *, seed: int = 3) -> pl.DataFrame:
    """A synthetic shot table with real column names and plausible values.

    Built rather than loaded so the sample size can be varied, which is what the
    ``1/sqrt(n)`` test needs. Zones are drawn with unequal probabilities so the
    alternative-specific constants are identified.
    """
    rng = np.random.default_rng(seed)
    weights = np.array([0.30, 0.22, 0.05, 0.01, 0.03, 0.05, 0.05, 0.06, 0.23])
    weights = weights / weights.sum()
    shooters = 1000 + rng.integers(0, 40, n)
    return pl.DataFrame(
        {
            "shooter_id": shooters,
            "team_id": 1610612737 + rng.integers(0, 4, n),
            "season": rng.choice([2022, 2023, 2024], n),
            "zone_id": rng.choice(list(ZONE_IDS), n, p=weights),
            "lineup_for": [
                sorted(rng.choice(np.arange(1000, 1040), 5, replace=False).tolist())
                for _ in range(n)
            ],
            "lineup_against": [
                sorted(rng.choice(np.arange(2000, 2040), 5, replace=False).tolist())
                for _ in range(n)
            ],
            "seconds_into_possession": rng.uniform(0, 24, n),
            "live_ball_start": rng.integers(0, 2, n).astype(bool),
            "is_second_chance": rng.integers(0, 2, n).astype(bool),
            "period": rng.integers(1, 5, n),
            "seconds_remaining": rng.uniform(0, 720, n),
            "is_three": [z.endswith("_three") for z in rng.choice(list(ZONE_IDS), n, p=weights)],
            "made": rng.integers(0, 2, n),
            "shot_points": rng.choice([2, 3], n),
        }
    )


def _fit(n: int) -> tuple[ConditionalLogit, object]:
    frame = _corpus(n)
    design = build_selection_design(frame, fit_selection_profiles(frame))
    model = ConditionalLogit().fit(design)
    return model.compute_standard_errors(design), design


@pytest.fixture(scope="module")
def fitted() -> tuple[ConditionalLogit, object]:
    return _fit(6_000)


def test_hessian_agrees_with_second_differences_of_the_loss(
    fitted: tuple[ConditionalLogit, object],
) -> None:
    """An independent route to the same matrix.

    `observed_information` differences the *analytic gradient*. This differences
    the *loss* twice. They share no code beyond the objective, so agreement means
    the analytic gradient and the Hessian derived from it are consistent -- which
    is the failure mode that matters, because a wrong gradient does not raise: it
    converges somewhere plausible and reports a believable log loss.
    """
    model, design = fitted
    theta = np.asarray(model.coefficients, dtype=float)
    mask = design.penalty_mask  # type: ignore[attr-defined]

    def loss(point: np.ndarray) -> float:
        value, _ = model.objective(point, design)  # type: ignore[arg-type]
        return value - 0.5 * model.l2 * float((mask * point * point).sum())

    analytic = model.observed_information(design)  # type: ignore[arg-type]

    # Only the first few parameters: this is O(p^2) loss evaluations and the
    # point is agreement, not coverage.
    step = 1e-4
    for i in range(3):
        for j in range(3):
            plus_plus, plus_minus = theta.copy(), theta.copy()
            minus_plus, minus_minus = theta.copy(), theta.copy()
            plus_plus[i] += step
            plus_plus[j] += step
            plus_minus[i] += step
            plus_minus[j] -= step
            minus_plus[i] -= step
            minus_plus[j] += step
            minus_minus[i] -= step
            minus_minus[j] -= step
            numeric = (
                loss(plus_plus) - loss(plus_minus) - loss(minus_plus) + loss(minus_minus)
            ) / (4 * step * step)
            assert numeric == pytest.approx(analytic[i, j], abs=2e-4), f"({i}, {j})"


def test_standard_errors_shrink_as_one_over_root_n() -> None:
    """The defining property, and the one that catches a misplaced 1/n.

    Quadrupling the sample must halve the standard errors. An implementation that
    forgot to divide by ``n``, or divided twice, fails this by orders of
    magnitude rather than subtly.
    """
    small, _ = _fit(4_000)
    large, _ = _fit(16_000)

    assert small.standard_errors is not None
    assert large.standard_errors is not None
    ratio = np.asarray(small.standard_errors) / np.asarray(large.standard_errors)
    finite = ratio[np.isfinite(ratio)]
    assert finite.size > 10
    # Generous: this is an asymptotic property on a synthetic corpus, so the
    # tolerance admits sampling variation while excluding any wrong power of n.
    assert np.median(finite) == pytest.approx(2.0, rel=0.35)


def test_covariance_is_symmetric_and_positive_definite(
    fitted: tuple[ConditionalLogit, object],
) -> None:
    model, design = fitted
    covariance = model.coefficient_covariance(design)  # type: ignore[arg-type]
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() > 0, f"smallest eigenvalue {eigenvalues.min():.3e}"


def test_standard_errors_are_all_finite_and_positive(
    fitted: tuple[ConditionalLogit, object],
) -> None:
    model, _ = fitted
    errors = np.asarray(model.standard_errors)
    assert np.isfinite(errors).all()
    assert (errors > 0).all()


def test_a_coefficient_the_data_cannot_pin_down_is_indeterminate() -> None:
    """The point of the whole exercise.

    On random data every lineup coefficient is truly zero, so an honest audit
    must return `indeterminate` for them rather than crediting whichever sign
    noise produced. Without standard errors the audit has no vocabulary for
    this and records a confirmation.
    """
    model, _ = _fit(3_000)
    audit = model.sign_audit()
    verdicts = {name: entry["verdict"] for name, entry in audit.items()}
    lineup_terms = {n for n, e in audit.items() if e["is_lineup"]}
    assert lineup_terms, "no lineup terms in the audit"
    assert any(verdicts[name] == "indeterminate" for name in lineup_terms), verdicts

    # And every entry with an interval must be self-consistent: the interval
    # straddling zero is exactly what `indeterminate` means.
    for name, entry in audit.items():
        if entry.get("ci95") is None:
            continue
        low, high = entry["ci95"]  # type: ignore[misc]
        straddles = low <= 0 <= high
        assert (entry["verdict"] == "indeterminate") == straddles, name


def test_sign_audit_falls_back_cleanly_without_standard_errors() -> None:
    """A fold fit has no covariance, and must still produce an audit."""
    frame = _corpus(2_000)
    design = build_selection_design(frame, fit_selection_profiles(frame))
    model = ConditionalLogit().fit(design)
    audit = model.sign_audit()
    assert audit
    for entry in audit.values():
        assert entry["standard_error"] is None
        assert entry["verdict"] in {"agrees", "DISAGREES"}
