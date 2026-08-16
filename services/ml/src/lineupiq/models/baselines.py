"""The baseline ladder.

Every model is reported against all of these, every run. The ladder exists so
that "the model works" is a comparison rather than an assertion, and so that a
model which adds nothing over a lookup table is visibly one that adds nothing
over a lookup table.

**B3 is the one that decides the project.** It has the shooter, the zone, the
distance and the context -- everything except who else is on the floor. The
entire lineup-interaction claim reduces to whether the full model beats it out
of sample on lineups it has never seen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from lineupiq.config import SEED
from lineupiq.models.shot_model import FEATURE_NAMES, Profiles, build_features

__all__ = ["BASELINES", "LINEUP_FEATURE_INDICES", "Baseline", "predict_baseline"]

#: Positions of the lineup-dependent columns in FEATURE_NAMES. B3 zeroes these
#: rather than dropping them, so it sees an identical matrix shape and any
#: difference is attributable to the lineup signal alone.
LINEUP_FEATURE_INDICES: tuple[int, ...] = tuple(
    FEATURE_NAMES.index(name)
    for name in ("spacing_sum", "spacing_min", "opp_rim_protection", "opp_zone_defence")
)


@dataclass(frozen=True)
class Baseline:
    key: str
    label: str
    rationale: str
    predict: Callable[[pl.DataFrame, pl.DataFrame, Profiles], np.ndarray]


def _b0_league_zone(_train: pl.DataFrame, test: pl.DataFrame, profiles: Profiles) -> np.ndarray:
    """League make rate for the zone. The scale reference."""
    return np.array(
        [profiles.zone_rate.get(z, profiles.league_fg_allowed) for z in test["zone_id"].to_list()]
    )


def _b1_shooter_zone(_train: pl.DataFrame, test: pl.DataFrame, profiles: Profiles) -> np.ndarray:
    """The shooter's own shrunk rate in that zone -- a lookup table.

    A model that cannot beat this has learned nothing beyond who is shooting
    and from where.
    """
    return np.array(
        [
            profiles.player_zone_rate.get((int(p), z), profiles.zone_rate.get(z, 0.45))
            for p, z in zip(test["shooter_id"].to_list(), test["zone_id"].to_list(), strict=True)
        ]
    )


def _b2_shooter_zone_context(
    train: pl.DataFrame, test: pl.DataFrame, profiles: Profiles
) -> np.ndarray:
    """B1 plus distance and game context, still with no lineup information."""
    from lineupiq.models.shot_model import ShotModel

    X_tr, y_tr = build_features(train, profiles)
    X_te, _ = build_features(test, profiles)
    for idx in LINEUP_FEATURE_INDICES:
        X_tr[:, idx] = 0.0
        X_te[:, idx] = 0.0
    return ShotModel().fit(X_tr, y_tr).predict_proba(X_te)


def _b3_additive_gbdt(train: pl.DataFrame, test: pl.DataFrame, profiles: Profiles) -> np.ndarray:
    """Gradient-boosted, full features, **lineup columns zeroed**.

    The strongest thing that can be built without knowing the other four
    players. If the full model does not beat this on unseen lineups, the
    project's headline claim is unsupported -- and that result gets published.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_tr, y_tr = build_features(train, profiles)
    X_te, _ = build_features(test, profiles)
    for idx in LINEUP_FEATURE_INDICES:
        X_tr[:, idx] = 0.0
        X_te[:, idx] = 0.0

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_depth=6,
        l2_regularization=1.0,
        random_state=SEED,
        early_stopping=False,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        "B0", "League zone mean", "Scale reference; the least a model may know.", _b0_league_zone
    ),
    Baseline(
        "B1",
        "Shooter x zone (shrunk)",
        "A lookup table. Beating it is the minimum bar.",
        _b1_shooter_zone,
    ),
    Baseline(
        "B2",
        "B1 + context, no lineup",
        "Isolates what context adds before lineups.",
        _b2_shooter_zone_context,
    ),
    Baseline(
        "B3",
        "Additive GBDT, no lineup",
        "The decisive comparison for the lineup claim.",
        _b3_additive_gbdt,
    ),
)


def predict_baseline(
    key: str, train: pl.DataFrame, test: pl.DataFrame, profiles: Profiles
) -> np.ndarray:
    for baseline in BASELINES:
        if baseline.key == key:
            return baseline.predict(train, test, profiles)
    raise KeyError(f"unknown baseline {key!r}; known: {[b.key for b in BASELINES]}")
