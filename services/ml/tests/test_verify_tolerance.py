"""Which metrics get the loose tolerance, and — more importantly — which do not.

The reproducibility gate is only worth as much as this classification. Give a
smooth metric the loose bound and the gate stops catching changed models; give a
binned one the tight bound and it fails on bin-edge noise from a different BLAS.

Both mistakes were made. The first version of `BINNED_METRICS` was a set of exact
names, which held the selection model's nineteen per-zone-group variants
(`three_ece`, `rim_resolution`, `classwise_ece`) to 1e-6 and failed the gate on
nothing. Then I removed `skill_score` from it, citing `1 - brier/uncertainty` — a formula
this code does not use. It is actually `(resolution - reliability) / uncertainty`,
so two of its three inputs are binned and it inherits every bin-edge
discontinuity they have. Ten of them moved by up to 4.5e-5 on the next CI run.

So the classification is asserted here, in both directions, by name.
"""

from __future__ import annotations

import pytest

from lineupiq.models.train import BINNED_TOLERANCE, TOLERANCE, tolerance_for


@pytest.mark.parametrize(
    "metric",
    [
        # Every binning-based estimator, and every prefix the models actually use.
        "ece",
        "classwise_ece",
        "three_ece",
        "rim_ece",
        "reliability",
        "three_reliability",
        "resolution",
        "rim_resolution",
        "three_resolution",
        # Derived from two binned quantities, so binned itself. Removing this
        # broke the gate; see the module docstring.
        "skill_score",
        "three_skill_score",
    ],
)
def test_binned_estimators_get_the_loose_tolerance(metric: str) -> None:
    assert tolerance_for(metric) == BINNED_TOLERANCE


@pytest.mark.parametrize(
    "metric",
    [
        # Smooth functions of the predictions. These are the gate.
        "log_loss",
        "rim_log_loss",
        "three_log_loss",
        "brier",
        "rim_brier",
        "three_brier",
        "uncertainty",
        "calibration_slope",
        "calibration_intercept",
        "top1_accuracy",
        "n",
    ],
)
def test_smooth_metrics_keep_the_tight_tolerance(metric: str) -> None:
    assert tolerance_for(metric) == TOLERANCE


def test_the_loose_bound_is_below_the_estimator_own_sampling_error() -> None:
    """The justification, asserted rather than left in a comment.

    A 20-bin ECE on ~100k held-out shots has a sampling standard error of order
    1e-3. If the loose tolerance ever crept above that it would stop being "wide
    enough for bin-edge noise" and start being "wide enough to hide a real
    change", and the difference is the whole argument for having it.
    """
    assert BINNED_TOLERANCE <= 1e-3
    # And it must still be far tighter than any difference a changed model makes.
    assert BINNED_TOLERANCE < 0.01


def test_the_tight_bound_survives_blas_variation_but_nothing_larger() -> None:
    # Loose enough that last-place differences between two platforms' matrix
    # multiplies pass, tight enough that a genuinely different fit does not.
    assert 1e-9 < TOLERANCE <= 1e-6
