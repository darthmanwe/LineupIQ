"""The shot model: P(make | shooter, zone, lineup context).

Two things shape the design.

**Leakage is prevented structurally, not by discipline.** Every player and team
profile is fitted from a training frame into a :class:`Profiles` object, and
featurisation takes that object as an argument. There is no code path that can
see the test set's outcomes, because the function that builds features has no
access to them.

**The served model is a closed form.** Cloudflare Workers give 10 ms of CPU and
the optimizer accepts any five of ~450 players, so nothing can be precomputed.
The lineup-dependent part is therefore linear in precomputed per-player
quantities, which makes exact Python/TypeScript parity provable rather than
approximate. What that constraint costs is measured against an unconstrained
gradient-boosted reference and published.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from lineupiq.config import SEED
from lineupiq.eval.leakage import assert_no_forbidden_features
from lineupiq.models.priors import fit_beta_prior
from lineupiq.transform.zones import ZONE_IDS
from lineupiq.util import as_float

__all__ = ["FEATURE_NAMES", "Profiles", "ShotModel", "build_features", "fit_profiles"]

#: Rim zones, for the opponent interior-defence profile.
_RIM_ZONES = ("restricted_area", "paint_non_ra")

#: Order matters and is part of the serving contract: the TypeScript scorer
#: consumes coefficients positionally.
FEATURE_NAMES: tuple[str, ...] = (
    "shooter_zone_logit",
    "shooter_zone_weight",
    "spacing_sum",
    "spacing_min",
    "opp_rim_protection",
    "opp_zone_defence",
    "shot_distance_z",
    "is_three",
    "is_clutch",
    "late_clock",
)


@dataclass(frozen=True)
class Profiles:
    """Everything the model knows before it sees a shot, fitted on train only."""

    #: (player_id, zone_id) -> shrunk make rate.
    player_zone_rate: dict[tuple[int, str], float]
    #: (player_id, zone_id) -> shrinkage weight (1 = all evidence, 0 = all prior).
    player_zone_weight: dict[tuple[int, str], float]
    #: zone_id -> league make rate. The fallback for an unseen shooter.
    zone_rate: dict[str, float]
    #: player_id -> share of his attempts that are threes. The spacing signal.
    player_three_rate: dict[int, float]
    #: player_id -> opponent rim FG% allowed while he was on the floor.
    player_rim_defence: dict[int, float]
    #: player_id -> opponent overall FG% allowed while on the floor.
    player_zone_defence: dict[int, float]
    league_three_rate: float
    league_rim_allowed: float
    league_fg_allowed: float
    distance_mean: float
    distance_std: float
    seasons: tuple[int, ...] = field(default_factory=tuple)

    def base_logit(self, player_id: int, zone_id: str) -> float:
        rate = self.player_zone_rate.get((player_id, zone_id), self.zone_rate.get(zone_id, 0.45))
        return _logit(rate)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def fit_profiles(train: pl.DataFrame) -> Profiles:
    """Fit every player/team profile from a training frame.

    Called once per fold. Nothing here may touch data outside ``train``.
    """
    zone_rate = {
        z: float(r)
        for z, r in train.group_by("zone_id").agg(pl.col("made").mean().alias("r")).iter_rows()
    }

    # --- shooter x zone, empirical-Bayes shrunk within each zone ----------
    per_zone = train.group_by(["shooter_id", "zone_id"]).agg(
        pl.col("made").sum().alias("makes"), pl.len().alias("attempts")
    )

    player_zone_rate: dict[tuple[int, str], float] = {}
    player_zone_weight: dict[tuple[int, str], float] = {}
    for zone in ZONE_IDS:
        block = per_zone.filter(pl.col("zone_id") == zone)
        if block.is_empty():
            continue
        makes = block["makes"].to_numpy().astype(float)
        attempts = block["attempts"].to_numpy().astype(float)
        prior = fit_beta_prior(makes, attempts)
        denom = attempts + prior.strength
        shrunk = (makes + prior.alpha) / denom
        weight = attempts / denom
        for pid, s, w in zip(block["shooter_id"].to_list(), shrunk, weight, strict=True):
            player_zone_rate[(int(pid), zone)] = float(s)
            player_zone_weight[(int(pid), zone)] = float(w)

    # --- spacing: how often a player shoots threes -----------------------
    three = (
        train.group_by("shooter_id")
        .agg(pl.col("is_three").mean().alias("r"), pl.len().alias("n"))
        .filter(pl.col("n") >= 20)
    )
    player_three_rate = {int(p): float(r) for p, r, _ in three.iter_rows()}
    league_three_rate = as_float(train["is_three"].mean(), 0.35)

    # --- defence: what opponents shot with this player on the floor ------
    defenders = (
        train.select(pl.col("lineup_against").alias("defender_id"), "made", "zone_id")
        .explode("defender_id", empty_as_null=True)
        .drop_nulls("defender_id")
    )

    rim = (
        defenders.filter(pl.col("zone_id").is_in(list(_RIM_ZONES)))
        .group_by("defender_id")
        .agg(pl.col("made").mean().alias("r"), pl.len().alias("n"))
        .filter(pl.col("n") >= 50)
    )
    overall = (
        defenders.group_by("defender_id")
        .agg(pl.col("made").mean().alias("r"), pl.len().alias("n"))
        .filter(pl.col("n") >= 100)
    )

    league_rim = as_float(
        train.filter(pl.col("zone_id").is_in(list(_RIM_ZONES)))["made"].mean(), 0.60
    )
    league_fg = as_float(train["made"].mean(), 0.46)

    return Profiles(
        player_zone_rate=player_zone_rate,
        player_zone_weight=player_zone_weight,
        zone_rate=zone_rate,
        player_three_rate=player_three_rate,
        player_rim_defence={int(p): float(r) for p, r, _ in rim.iter_rows()},
        player_zone_defence={int(p): float(r) for p, r, _ in overall.iter_rows()},
        league_three_rate=league_three_rate,
        league_rim_allowed=league_rim,
        league_fg_allowed=league_fg,
        distance_mean=as_float(train["shot_distance_ft"].mean(), 13.0),
        distance_std=as_float(train["shot_distance_ft"].std(), 9.0) or 1.0,
        seasons=tuple(sorted({int(s) for s in train["season"].unique().to_list()})),
    )


def build_features(frame: pl.DataFrame, profiles: Profiles) -> tuple[np.ndarray, np.ndarray]:
    """Build the design matrix and label vector.

    Returns ``(X, y)`` with columns in :data:`FEATURE_NAMES` order.
    """
    assert_no_forbidden_features([c for c in frame.columns if c in {"is_assisted"}])

    shooters = frame["shooter_id"].to_list()
    zones = frame["zone_id"].to_list()
    for_lineups = frame["lineup_for"].to_list()
    against_lineups = frame["lineup_against"].to_list()

    n = frame.height
    base_logit = np.empty(n)
    base_weight = np.empty(n)
    spacing_sum = np.empty(n)
    spacing_min = np.empty(n)
    opp_rim = np.empty(n)
    opp_zone = np.empty(n)

    for i in range(n):
        pid, zone = int(shooters[i]), zones[i]
        base_logit[i] = profiles.base_logit(pid, zone)
        base_weight[i] = profiles.player_zone_weight.get((pid, zone), 0.0)

        teammates = [int(p) for p in (for_lineups[i] or []) if int(p) != pid]
        rates = [profiles.player_three_rate.get(t, profiles.league_three_rate) for t in teammates]
        spacing_sum[i] = sum(rates) - len(rates) * profiles.league_three_rate if rates else 0.0
        # The worst spacer on the floor. One non-shooter collapses spacing in a
        # way a sum cannot express, because a good spacer offsets him in the sum.
        spacing_min[i] = (min(rates) - profiles.league_three_rate) if rates else 0.0

        defenders = [int(p) for p in (against_lineups[i] or [])]
        rim_vals = [
            profiles.player_rim_defence.get(d, profiles.league_rim_allowed) for d in defenders
        ]
        fg_vals = [
            profiles.player_zone_defence.get(d, profiles.league_fg_allowed) for d in defenders
        ]
        opp_rim[i] = (
            sum(rim_vals) / len(rim_vals) - profiles.league_rim_allowed if rim_vals else 0.0
        )
        opp_zone[i] = sum(fg_vals) / len(fg_vals) - profiles.league_fg_allowed if fg_vals else 0.0

    distance = frame["shot_distance_ft"].to_numpy().astype(float)
    distance_z = (distance - profiles.distance_mean) / profiles.distance_std
    is_three = frame["is_three"].to_numpy().astype(float)
    period = frame["period"].to_numpy().astype(float)
    secs = frame["seconds_remaining"].to_numpy().astype(float)

    is_clutch = ((period >= 4) & (secs <= 300)).astype(float)
    late_clock = (secs <= 24).astype(float)

    X = np.column_stack(
        [
            base_logit,
            base_weight,
            spacing_sum,
            spacing_min,
            opp_rim,
            opp_zone,
            distance_z,
            is_three,
            is_clutch,
            late_clock,
        ]
    )
    y = frame["made"].to_numpy().astype(float)
    return X, y


@dataclass
class ShotModel:
    """Regularised logistic model over :data:`FEATURE_NAMES`.

    Logistic rather than boosted **on purpose**: this is the form that gets
    served, and a linear model in precomputed per-player quantities is one that
    a Worker can evaluate exactly. The gradient-boosted comparison exists to
    measure what that choice costs, not to be deployed.
    """

    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def fit(self, X: np.ndarray, y: np.ndarray, *, C: float = 1.0) -> ShotModel:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED,
        )
        model.fit(X, y)
        self.coefficients = model.coef_[0].copy()
        self.intercept = float(model.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")
        z = X @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict[str, object]:
        # `self.coefficients or []` is a trap: numpy arrays raise on truthiness.
        coefficients = [] if self.coefficients is None else [float(c) for c in self.coefficients]
        return {
            "feature_names": list(self.feature_names),
            "coefficients": coefficients,
            "intercept": self.intercept,
        }
