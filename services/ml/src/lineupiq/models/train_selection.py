"""Training driver for the shot-selection model.

The ladder is built so that every verdict isolates one thing. ``S2`` and
``full`` are the same conditional logit differing only in whether the five
lineup columns carry information; ``S3`` and ``full_gbdt`` are the same boosted
model differing in exactly the same way. Comparing the logit against the boosted
model would conflate the serving constraint with the lineup claim, and that
mistake has already been made once in this repository -- on the conversion
model, where the closed form losing to a GBDT was briefly read as lineup context
adding nothing.

``S1`` is the number that matters most. It is a lookup table: the shooter's own
shrunk zone mix, ignoring the opponent, the clock and the other four players. A
model that cannot beat it has learned nothing about basketball.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from lineupiq.config import SEED
from lineupiq.eval.splits import Fold, leave_lineup_out, walk_forward_by_game
from lineupiq.models.selection import (
    LINEUP_TERM_NAMES,
    ConditionalLogit,
    SelectionDesign,
    build_selection_design,
    fit_selection_profiles,
    lineup_wide_indices,
    score_selection,
    shooter_mix_prediction,
    summarise_mix,
    usable_selection_frame,
    wide_features,
)
from lineupiq.models.train import RunLog, _git_dirty, _git_sha, _pool
from lineupiq.paths import DataPaths

__all__ = [
    "SELECTION_LADDER",
    "evaluate_selection_fold",
    "shuffled_lineup_control",
    "train_and_evaluate_selection",
    "within_shooter_robustness",
]

#: Reported every run, in this order.
SELECTION_LADDER: tuple[str, ...] = ("S0", "S1", "S2", "S3", "full", "full_gbdt")

#: Each model's no-lineup counterpart. A verdict is only ever stated against
#: the same model class with the lineup columns zeroed.
COUNTERPART: dict[str, str] = {"full": "S2", "full_gbdt": "S3"}

_GBDT_KWARGS: dict[str, Any] = {
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "l2_regularization": 1.0,
    "random_state": SEED,
    "early_stopping": False,
}


def _gbdt_probabilities(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int
) -> np.ndarray:
    """Fit a multiclass GBDT and return a full probability matrix.

    ``predict_proba`` only has columns for classes present in training. A rare
    zone missing from one fold would otherwise silently shift every column to
    the left, so the output is placed back into full zone order by class label.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(**_GBDT_KWARGS)
    model.fit(x_train, y_train)
    partial = model.predict_proba(x_test)
    full = np.zeros((x_test.shape[0], n_classes))
    for column, label in enumerate(model.classes_):
        full[:, int(label)] = partial[:, column]
    return full


def evaluate_selection_fold(fold: Fold) -> tuple[dict[str, dict[str, float]], ConditionalLogit]:
    """Fit the whole ladder on this fold's train and score its test."""
    profiles = fit_selection_profiles(fold.train)
    train_design = build_selection_design(fold.train, profiles)
    test_design = build_selection_design(fold.test, profiles)
    n_classes = train_design.n_zones
    y_test = test_design.y

    results: dict[str, dict[str, float]] = {}

    # S0 -- the league's mix, ignoring everything about the shot.
    results["S0"] = score_selection(
        y_test, np.tile(profiles.league_mix, (test_design.n, 1))
    ).to_dict()

    # S1 -- the shooter's own shrunk mix. A lookup table.
    results["S1"] = score_selection(y_test, shooter_mix_prediction(fold.test, profiles)).to_dict()

    # S2 / full -- one conditional logit, lineup columns off then on.
    stripped_train = train_design.without(LINEUP_TERM_NAMES)
    stripped_test = test_design.without(LINEUP_TERM_NAMES)
    s2 = ConditionalLogit().fit(stripped_train)
    results["S2"] = score_selection(y_test, s2.predict_proba(stripped_test)).to_dict()

    full = ConditionalLogit().fit(train_design)
    results["full"] = score_selection(y_test, full.predict_proba(test_design)).to_dict()

    # S3 / full_gbdt -- one boosted model, the same two states.
    #
    # The lineup columns are zeroed in place for S3 and then written back,
    # rather than copying the matrix. At three seasons each copy is 150 MB per
    # fold, and there are nine folds.
    x_train = wide_features(train_design)
    x_test = wide_features(test_design)
    lineup_columns = list(lineup_wide_indices(train_design))
    kept_train = x_train[:, lineup_columns].copy()
    kept_test = x_test[:, lineup_columns].copy()

    x_train[:, lineup_columns] = 0.0
    x_test[:, lineup_columns] = 0.0
    results["S3"] = score_selection(
        y_test, _gbdt_probabilities(x_train, train_design.y, x_test, n_classes)
    ).to_dict()

    x_train[:, lineup_columns] = kept_train
    x_test[:, lineup_columns] = kept_test
    results["full_gbdt"] = score_selection(
        y_test, _gbdt_probabilities(x_train, train_design.y, x_test, n_classes)
    ).to_dict()

    return results, full


def _permute_lineups(shots: pl.DataFrame, *, seed: int) -> pl.DataFrame:
    """Detach lineup context from the shot it belongs to."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(shots.height)
    return shots.with_columns(
        pl.Series("lineup_for", shots["lineup_for"].to_list()).gather(order),
        pl.Series("lineup_against", shots["lineup_against"].to_list()).gather(order),
    )


def shuffled_lineup_control(shots: pl.DataFrame, *, seed: int = SEED) -> dict[str, float]:
    """Negative control: permute which five players were on the floor.

    Two numbers come back, and the second is the sharper one. The log-loss gain
    of ``full`` over ``S2`` should collapse to zero, as usual. But this model
    also makes a *directional* claim through ``spacing_x_three``, and a
    coefficient can stay large while a pooled metric goes flat. If spacing still
    reads -0.5 once lineups are randomly reassigned, then it is not measuring
    lineups at all, and no aggregate would have said so.
    """
    shuffled = _permute_lineups(shots, seed=seed)
    folds = list(walk_forward_by_game(shuffled, n_folds=2))
    if not folds:
        return {}

    gains: list[float] = []
    spacings: list[float] = []
    for fold in folds:
        profiles = fit_selection_profiles(fold.train)
        train_design = build_selection_design(fold.train, profiles)
        test_design = build_selection_design(fold.test, profiles)
        blind = ConditionalLogit().fit(train_design.without(LINEUP_TERM_NAMES))
        full = ConditionalLogit().fit(train_design)
        blind_loss = score_selection(
            test_design.y, blind.predict_proba(test_design.without(LINEUP_TERM_NAMES))
        ).log_loss
        full_loss = score_selection(test_design.y, full.predict_proba(test_design)).log_loss
        gains.append(blind_loss - full_loss)
        spacings.append(full.coefficient("spacing_x_three"))

    return {
        "shuffled_lineup_logloss_gain": float(np.mean(gains)),
        "shuffled_lineup_spacing_coefficient": float(np.mean(spacings)),
    }


def within_shooter_robustness(shots: pl.DataFrame) -> dict[str, float]:
    """Re-estimate the lineup terms off within-player variation only.

    The lineup aggregates are anti-correlated with the shooter's own tendencies
    by roster construction: put four shooters on the floor and the fifth man is
    usually the centre, so "my teammates shoot threes" partly encodes "I am the
    big". ``shooter_mix`` absorbs a player's average mix, but the cleanest way
    to ask the question is to strip the between-player component outright and
    centre each lineup feature within shooter. What is left is: when *this*
    player gets more spacing than he usually has, what does he shoot?

    Reported rather than substituted for the headline fit, because centring
    within shooter also discards the cross-sectional information the served
    model legitimately uses.
    """
    profiles = fit_selection_profiles(shots)
    design = build_selection_design(shots, profiles)
    shooters = np.asarray([int(s) for s in shots["shooter_id"].to_list()])

    centred = {name: values.copy() for name, values in design.inter_shot.items()}
    for name in LINEUP_TERM_NAMES:
        values = centred[name]
        adjusted = np.empty_like(values)
        for shooter in np.unique(shooters):
            mask = shooters == shooter
            adjusted[mask] = values[mask] - values[mask].mean()
        centred[name] = adjusted

    within = SelectionDesign(
        n=design.n,
        y=design.y,
        alt_matrix=design.alt_matrix,
        pair_matrices=design.pair_matrices,
        inter_shot=centred,
        inter_alt=design.inter_alt,
        term_names=design.term_names,
    )
    fitted = ConditionalLogit().fit(within)
    return {name: fitted.coefficient(name) for name in LINEUP_TERM_NAMES}


def _write_partial(log: RunLog, paths: DataPaths | None, stage: str) -> None:
    """Checkpoint a partially complete run.

    A full pass is eighteen model fits over three seasons and takes tens of
    minutes. Losing all of it because the machine went down at minute thirty is
    avoidable, so each stage is written as it finishes.

    Partials go to their own directory: ``latest_run`` globs
    ``runs/selection/*.json`` and picks the last by name, and a half-finished
    log sitting in there would be served as if it were the published result.
    """
    if paths is None:
        return
    directory = paths.runs / "selection_partial"
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.loads(log.to_json())
    payload["_partial_after"] = stage
    text = json.dumps(payload, indent=2, sort_keys=True)
    (directory / "run.json").write_text(f"{text}\n", encoding="utf-8", newline="\n")


def train_and_evaluate_selection(
    shots: pl.DataFrame,
    *,
    run_controls: bool = True,
    paths: DataPaths | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunLog:
    """Fit and score the selection ladder across both split types."""
    started = time.monotonic()

    def report(message: str) -> None:
        if progress is not None:
            progress(f"[{time.monotonic() - started:6.1f}s] {message}")

    usable = usable_selection_frame(shots)

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
        per_fold = []
        for fold in folds:
            report(f"{split_name}: {fold.name} (train {fold.sizes[0]:,} / test {fold.sizes[1]:,})")
            per_fold.append(evaluate_selection_fold(fold)[0])
        if per_fold:
            log.metrics[split_name] = _pool(per_fold)
            log.notes.append(f"{split_name}: {len(per_fold)} folds")
            _write_partial(log, paths, split_name)
        report(f"{split_name}: done, {len(per_fold)} folds")

    if run_controls:
        report("negative control: shuffling lineups")
        log.controls.update(shuffled_lineup_control(usable))
        _write_partial(log, paths, "controls")

    # Refit on everything for the served coefficients and the sign audit.
    report("final refit on the full corpus")
    profiles = fit_selection_profiles(usable)
    design = build_selection_design(usable, profiles)
    fitted = ConditionalLogit().fit(design)
    log.model = {
        **fitted.to_dict(),
        "observed_mix": summarise_mix(usable),
        "shooter_prior_strength": profiles.shooter_prior_strength,
        "team_prior_strength": profiles.team_prior_strength,
        "within_shooter_coefficients": within_shooter_robustness(usable),
    }
    report("within-shooter robustness refit done")

    audit = fitted.sign_audit()
    disagreements = [name for name, row in audit.items() if row["verdict"] == "DISAGREES"]
    log.notes.append(
        f"sign audit: {len(audit) - len(disagreements)}/{len(audit)} pre-registered signs agree"
        + (f"; DISAGREES: {', '.join(disagreements)}" if disagreements else "")
    )
    return log
