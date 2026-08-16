"""Training driver, run log, and the reproducibility gate.

Every published number originates here and nowhere else. ``train --verify``
refits from committed gold and fails if any metric moved by more than 1e-6,
which is what makes the README's numbers checkable rather than asserted.

The sibling project learned the remaining half of this lesson the hard way: its
``--verify`` compared refits against a run log but not against its README, so
the build stayed green while the published table drifted out of date. Here the
run log is the *only* source for published numbers, and M8's report generator
reads from it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lineupiq.config import SEED
from lineupiq.eval.metrics import calibration_report
from lineupiq.eval.splits import Fold, leave_lineup_out, walk_forward_by_game
from lineupiq.models.baselines import BASELINES, predict_baseline
from lineupiq.models.shot_model import ShotModel, build_features, fit_profiles
from lineupiq.paths import DataPaths

__all__ = ["RunLog", "compare_to_committed", "latest_run", "train_and_evaluate", "write_run_log"]

#: Metrics must reproduce to this tolerance. Loose enough to survive BLAS
#: variation across platforms, tight enough that a real change is caught.
TOLERANCE = 1e-6


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


@dataclass
class RunLog:
    """One training run, fully described."""

    created_at: str
    git_sha: str
    git_dirty: bool
    seed: int
    python: str
    platform: str
    seasons: list[int]
    n_shots: int
    n_lineups: int
    #: split -> model/baseline key -> metric name -> value
    metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    controls: dict[str, float] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _evaluate_fold(fold: Fold) -> dict[str, dict[str, float]]:
    """Fit everything on this fold's train and score its test."""
    profiles = fit_profiles(fold.train)
    results: dict[str, dict[str, float]] = {}

    y_test = fold.test["made"].to_numpy().astype(float)

    for baseline in BASELINES:
        p = predict_baseline(baseline.key, fold.train, fold.test, profiles)
        results[baseline.key] = calibration_report(y_test, p).to_dict()

    X_tr, y_tr = build_features(fold.train, profiles)
    X_te, _ = build_features(fold.test, profiles)

    # The servable model: linear in precomputed per-player quantities.
    model = ShotModel().fit(X_tr, y_tr)
    results["full"] = calibration_report(y_test, model.predict_proba(X_te)).to_dict()

    # The unconstrained reference: same features, no serving constraint.
    #
    # This exists to make the ablation honest. Comparing the logistic `full`
    # against the boosted `B3` conflates two differences at once -- model class
    # and lineup information -- and would let a model-class effect be reported
    # as a lineup effect. `full_gbdt` vs `B3` differs ONLY in whether the
    # lineup columns are zeroed, so it isolates the thing being claimed. It
    # also measures what the closed-form serving constraint costs.
    from sklearn.ensemble import HistGradientBoostingClassifier

    gbdt = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_depth=6,
        l2_regularization=1.0,
        random_state=SEED,
        early_stopping=False,
    )
    gbdt.fit(X_tr, y_tr)
    results["full_gbdt"] = calibration_report(y_test, gbdt.predict_proba(X_te)[:, 1]).to_dict()

    return results


def _pool(per_fold: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Average each metric across folds, weighted by test size."""
    if not per_fold:
        return {}
    keys = per_fold[0].keys()
    pooled: dict[str, dict[str, float]] = {}
    for key in keys:
        weights = np.array([f[key]["n"] for f in per_fold], dtype=float)
        total = weights.sum() or 1.0
        metric_names = [m for m in per_fold[0][key] if m != "n"]
        pooled[key] = {"n": float(total)}
        for metric in metric_names:
            values = np.array([f[key][metric] for f in per_fold], dtype=float)
            finite = np.isfinite(values)
            if finite.any():
                pooled[key][metric] = float(np.average(values[finite], weights=weights[finite]))
            else:
                pooled[key][metric] = float("nan")
    return pooled


def _shuffled_lineup_control(shots: pl.DataFrame, *, seed: int = SEED) -> float:
    """Negative control: permute which lineup context attaches to each shot.

    If lineup features still help after this, the improvement is leakage rather
    than signal. Returns the full model's log-loss improvement over B1 on
    shuffled data -- it should be approximately zero.
    """
    rng = np.random.default_rng(seed)
    n = shots.height
    perm = rng.permutation(n)

    shuffled = shots.with_columns(
        pl.Series("lineup_for", shots["lineup_for"].to_list()).gather(perm),
        pl.Series("lineup_against", shots["lineup_against"].to_list()).gather(perm),
    )

    folds = list(walk_forward_by_game(shuffled, n_folds=2))
    if not folds:
        return 0.0

    gains = []
    for fold in folds:
        profiles = fit_profiles(fold.train)
        y = fold.test["made"].to_numpy().astype(float)
        b1 = predict_baseline("B1", fold.train, fold.test, profiles)
        X_tr, y_tr = build_features(fold.train, profiles)
        X_te, _ = build_features(fold.test, profiles)
        full = ShotModel().fit(X_tr, y_tr).predict_proba(X_te)
        from lineupiq.eval.metrics import log_loss

        gains.append(log_loss(y, b1) - log_loss(y, full))
    return float(np.mean(gains))


def train_and_evaluate(shots: pl.DataFrame, *, run_controls: bool = True) -> RunLog:
    """Fit and score across both split types, plus negative controls."""
    usable = shots.filter(
        pl.col("lineup_for_hash").is_not_null()
        & pl.col("zone_id").is_not_null()
        & pl.col("made").is_not_null()
        # Training uses only cleanly solved lineups. Imputed ones are still
        # served, with their flag, but a guess must never become a coefficient.
        & (pl.col("stint_quality") == "VALID")
    )

    log = RunLog(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        seed=SEED,
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.system(),
        seasons=sorted({int(s) for s in usable["season"].unique().to_list()}),
        n_shots=usable.height,
        n_lineups=usable["lineup_for_hash"].n_unique(),
    )

    for split_name, folds in (
        ("walk_forward", list(walk_forward_by_game(usable))),
        ("leave_lineup_out", list(leave_lineup_out(usable))),
    ):
        per_fold = [_evaluate_fold(fold) for fold in folds]
        if per_fold:
            log.metrics[split_name] = _pool(per_fold)
            log.notes.append(f"{split_name}: {len(per_fold)} folds")

    if run_controls:
        log.controls["shuffled_lineup_logloss_gain"] = _shuffled_lineup_control(usable)

    # Refit on everything for the served coefficients.
    profiles = fit_profiles(usable)
    X, y = build_features(usable, profiles)
    log.model = ShotModel().fit(X, y).to_dict()

    return log


def write_run_log(log: RunLog, paths: DataPaths) -> Path:
    directory = paths.runs / "epsa"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = log.created_at.replace(":", "").replace("-", "")
    path = directory / f"{stamp}_{log.git_sha}.json"
    path.write_text(log.to_json(), encoding="utf-8", newline="\n")
    return path


def latest_run(paths: DataPaths) -> dict[str, Any] | None:
    directory = paths.runs / "epsa"
    if not directory.exists():
        return None
    runs = sorted(directory.glob("*.json"))
    if not runs:
        return None
    return json.loads(runs[-1].read_text(encoding="utf-8"))


def compare_to_committed(fresh: RunLog, committed: dict[str, Any]) -> list[str]:
    """Report every metric that moved by more than :data:`TOLERANCE`."""
    drifts: list[str] = []

    if fresh.n_shots != committed.get("n_shots"):
        drifts.append(f"n_shots: {committed.get('n_shots')} -> {fresh.n_shots}")

    for split, models in fresh.metrics.items():
        old_split = committed.get("metrics", {}).get(split, {})
        for key, metrics in models.items():
            old = old_split.get(key, {})
            for metric, value in metrics.items():
                if metric not in old:
                    continue
                previous = old[metric]
                if not (np.isfinite(value) and np.isfinite(previous)):
                    continue
                if abs(value - previous) > TOLERANCE:
                    drifts.append(f"{split}.{key}.{metric}: {previous:.9f} -> {value:.9f}")
    return drifts
