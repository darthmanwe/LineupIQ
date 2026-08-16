"""Scoring rules and calibration diagnostics.

The Brier decomposition is the reason this module exists. A model can be
perfectly calibrated and completely uninformative -- predicting the league mean
for every shot is perfectly calibrated. Only *resolution* distinguishes a model
that knows something from one that knows nothing, and reporting calibration
without it is how an uninformative model gets described as a good one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "BrierDecomposition",
    "CalibrationReport",
    "brier_decomposition",
    "calibration_report",
    "calibration_slope_intercept",
    "expected_calibration_error",
    "log_loss",
]

_EPS = 1e-15


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log likelihood, in nats."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy's three-way split of the Brier score.

    ``brier = reliability - resolution + uncertainty``

    - **reliability** -- calibration error. Lower is better; 0 is perfect.
    - **resolution** -- how far predictions move away from the base rate in a
      way that tracks the outcome. Higher is better. This is the informative
      part, and it is 0 for a model that always predicts the base rate.
    - **uncertainty** -- the base rate's own variance. A property of the data,
      not the model; it bounds what any model can achieve.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    n_bins: int

    @property
    def skill_score(self) -> float:
        """Fraction of achievable Brier improvement captured. 0 = base rate."""
        return (self.resolution - self.reliability) / self.uncertainty if self.uncertainty else 0.0


def brier_decomposition(y: np.ndarray, p: np.ndarray, *, n_bins: int = 20) -> BrierDecomposition:
    """Decompose the Brier score using equal-count bins.

    Equal-count rather than equal-width: shot probabilities cluster heavily
    around a few values, and equal-width bins leave most of the range nearly
    empty, which makes reliability estimates wildly noisy at the tails.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    base = float(np.mean(y))
    brier = float(np.mean((p - y) ** 2))

    order = np.argsort(p)
    bins = np.array_split(order, min(n_bins, max(1, n)))

    reliability = 0.0
    resolution = 0.0
    used = 0
    for idx in bins:
        if len(idx) == 0:
            continue
        used += 1
        w = len(idx) / n
        p_bar = float(np.mean(p[idx]))
        y_bar = float(np.mean(y[idx]))
        reliability += w * (p_bar - y_bar) ** 2
        resolution += w * (y_bar - base) ** 2

    return BrierDecomposition(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=base * (1 - base),
        n_bins=used,
    )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, *, n_bins: int = 20) -> float:
    """Weighted mean absolute gap between predicted and observed, by bin."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    order = np.argsort(p)
    total = 0.0
    for idx in np.array_split(order, min(n_bins, max(1, n))):
        if len(idx) == 0:
            continue
        total += (len(idx) / n) * abs(float(np.mean(p[idx])) - float(np.mean(y[idx])))
    return total


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration coefficients.

    Regress the outcome on the predicted log-odds. A perfectly calibrated model
    gives slope 1, intercept 0. Slope < 1 means predictions are too extreme --
    the classic overfitting signature, and invisible to ECE alone.
    """
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    # C=inf is the unpenalised fit. `penalty=None` was deprecated in sklearn 1.8
    # and this is the documented replacement; the recalibration must be
    # unpenalised or the slope is shrunk toward zero and the diagnostic lies in
    # the direction of "looks better calibrated than it is".
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    base_rate: float
    log_loss: float
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    skill_score: float
    ece: float
    calibration_slope: float
    calibration_intercept: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calibration_report(y: np.ndarray, p: np.ndarray, *, n_bins: int = 20) -> CalibrationReport:
    decomposition = brier_decomposition(y, p, n_bins=n_bins)
    slope, intercept = calibration_slope_intercept(y, p)
    return CalibrationReport(
        n=len(y),
        base_rate=float(np.mean(y)),
        log_loss=log_loss(y, p),
        brier=decomposition.brier,
        reliability=decomposition.reliability,
        resolution=decomposition.resolution,
        uncertainty=decomposition.uncertainty,
        skill_score=decomposition.skill_score,
        ece=expected_calibration_error(y, p, n_bins=n_bins),
        calibration_slope=slope,
        calibration_intercept=intercept,
    )
