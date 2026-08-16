"""The refusal contract.

The product's central promise is that a number never appears without the
evidence behind it. This module is where that promise is made operational: it
computes how much support a lineup actually has and decides which of three
things the API may do -- report a point estimate, report only a direction, or
refuse outright.

Thresholds are loaded from a pre-registered, hash-pinned file. They were fixed
before any lineup-level result existed, so they cannot be tuned after the fact
to make an answer look better.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import polars as pl

from lineupiq.hashing import lineup_hash
from lineupiq.paths import DataPaths

__all__ = [
    "LineupSupport",
    "SupportThresholds",
    "Tier",
    "assess",
    "build_lineup_support",
    "load_thresholds",
    "thresholds_hash",
]


class Tier(StrEnum):
    REPORTABLE = "reportable"
    DIRECTIONAL = "directional"
    REFUSED = "refused"


@dataclass(frozen=True)
class SupportThresholds:
    reportable_possessions: int
    reportable_attempts: int
    directional_possessions: int
    directional_attempts: int
    min_zone_attempts: int
    conformal_bin_min_n: int
    min_reportable_minutes_share: float


@lru_cache(maxsize=1)
def _thresholds_path() -> Path:
    return DataPaths.discover().configs / "support_thresholds.json"


def load_thresholds(path: Path | None = None) -> SupportThresholds:
    raw = json.loads((path or _thresholds_path()).read_text(encoding="utf-8"))
    return SupportThresholds(
        reportable_possessions=raw["reportable"]["lineup_possessions"],
        reportable_attempts=raw["reportable"]["min_player_attempts"],
        directional_possessions=raw["directional"]["lineup_possessions"],
        directional_attempts=raw["directional"]["min_player_attempts"],
        min_zone_attempts=raw["shot_model"]["min_zone_attempts_for_point_estimate"],
        conformal_bin_min_n=raw["shot_model"]["conformal_bin_min_n"],
        min_reportable_minutes_share=raw["trade"]["min_reportable_minutes_share"],
    )


def thresholds_hash(path: Path | None = None) -> str:
    """SHA-256 of the thresholds file, asserted unchanged by CI."""
    return hashlib.sha256((path or _thresholds_path()).read_bytes()).hexdigest()


@dataclass(frozen=True)
class LineupSupport:
    """What is known about one five-man group."""

    lineup_hash: str
    possessions: int
    min_player_attempts: int
    tier: Tier
    counterfactual: bool
    shortfall_players: tuple[int, ...] = ()

    @property
    def may_report_point_estimate(self) -> bool:
        return self.tier is Tier.REPORTABLE


def build_lineup_support(stints: pl.DataFrame, shot_facts: pl.DataFrame) -> pl.DataFrame:
    """Aggregate observed evidence per five-man offensive lineup.

    Possessions are approximated by stint duration: a possession averages
    roughly 24 seconds of game clock, so seconds/24 is a reasonable proxy and
    is stated as a proxy rather than dressed up as a count. The alternative --
    the possessions oracle -- is only available for some seasons, and a support
    figure that silently changes definition between seasons would be worse than
    an approximate one that does not.
    """
    per_side = []
    for col in ("home_lineup", "away_lineup"):
        per_side.append(
            stints.filter(pl.col(col).is_not_null() & (pl.col(col).list.len() == 5)).select(
                pl.col(col).list.sort().cast(pl.List(pl.Utf8)).list.join(",").alias("_key"),
                "duration_seconds",
            )
        )
    durations = (
        pl.concat(per_side)
        .group_by("_key")
        .agg(pl.col("duration_seconds").sum().alias("seconds"))
        .with_columns((pl.col("seconds") / 24.0).round().cast(pl.Int64).alias("possessions"))
    )

    # Map the joined-id key back to the md5 hash used everywhere else.
    durations = durations.with_columns(
        pl.col("_key")
        .map_elements(lambda k: lineup_hash([int(x) for x in k.split(",")]), return_dtype=pl.Utf8)
        .alias("lineup_hash")
    )

    attempts = (
        shot_facts.filter(pl.col("lineup_for_hash").is_not_null())
        .group_by("lineup_for_hash")
        .agg(pl.len().alias("shots"))
        .rename({"lineup_for_hash": "lineup_hash"})
    )

    player_attempts = shot_facts.group_by("shooter_id").agg(pl.len().alias("player_attempts"))
    attempt_lookup = dict(player_attempts.iter_rows())

    def _min_attempts(key: str) -> int:
        return min((attempt_lookup.get(int(p), 0) for p in key.split(",")), default=0)

    return (
        durations.join(attempts, on="lineup_hash", how="left")
        .with_columns(
            pl.col("shots").fill_null(0),
            pl.col("_key")
            .map_elements(_min_attempts, return_dtype=pl.Int64)
            .alias("min_player_attempts"),
        )
        .drop("_key")
        .sort("possessions", descending=True)
    )


def assess(
    lineup_ids: list[int],
    support_table: dict[str, tuple[int, int]],
    thresholds: SupportThresholds,
    player_attempts: dict[int, int] | None = None,
) -> LineupSupport:
    """Decide what may be said about this lineup.

    Three outcomes, and the boundary between them is the product:

    - **reportable** -- enough evidence for a point estimate.
    - **directional** -- sign and rough magnitude only; the API returns 200 with
      a null point estimate and a populated interval.
    - **refused** -- no basis at all; the API returns 422.
    """
    key = lineup_hash(lineup_ids)
    possessions, min_attempts = support_table.get(key, (0, 0))
    counterfactual = key not in support_table

    if player_attempts:
        min_attempts = min((player_attempts.get(p, 0) for p in lineup_ids), default=0)

    shortfall = tuple(
        p
        for p in lineup_ids
        if (player_attempts or {}).get(p, min_attempts) < thresholds.directional_attempts
    )

    if (
        possessions >= thresholds.reportable_possessions
        and min_attempts >= thresholds.reportable_attempts
    ):
        tier = Tier.REPORTABLE
    elif min_attempts >= thresholds.directional_attempts:
        # Player-level terms have support even when the combination does not.
        # This is the normal case for a post-trade lineup, and it is why the
        # answer is a null centre with a real interval rather than a refusal.
        tier = Tier.DIRECTIONAL
    else:
        tier = Tier.REFUSED

    return LineupSupport(
        lineup_hash=key,
        possessions=possessions,
        min_player_attempts=min_attempts,
        tier=tier,
        counterfactual=counterfactual,
        shortfall_players=shortfall,
    )
