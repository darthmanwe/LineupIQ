"""The bronze -> silver -> gold pipeline, one season at a time.

One season per invocation is the unit of work. It holds peak memory to roughly
one season of events (~570k rows) instead of the whole corpus, and it makes a
partial failure recoverable: seasons already built stay built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from lineupiq.ingest.sources import Source, sdv_sources, shufinskiy_sources
from lineupiq.io.bronze import BronzeCache
from lineupiq.paths import DataPaths
from lineupiq.seasons import Season
from lineupiq.transform.events import canonical_order, type_events
from lineupiq.transform.gold import build_dim_player, build_shot_facts
from lineupiq.transform.segments import minutes_agreement, segment_stints
from lineupiq.transform.stints import reconstruct_stints
from lineupiq.util import as_float, as_int

__all__ = ["MinutesCheck", "SeasonBuild", "build_season", "ingest_season"]


@dataclass(frozen=True)
class MinutesCheck:
    """Agreement between derived stint minutes and the box score.

    The gate is stated on two numbers because they fail differently. A large
    ``mean_abs_delta`` means the reconstruction is systematically wrong; a small
    mean with a large ``p95`` means it is right except in specific games, which
    is a much more tractable bug.
    """

    n_player_games: int
    mean_abs_delta: float
    p95_abs_delta: float
    max_abs_delta: int
    pct_total_discrepancy: float
    #: A player the box score says did not play, who derived non-zero minutes.
    #: This is a hard failure, never a tolerance miss.
    n_dnp_with_minutes: int


@dataclass
class SeasonBuild:
    """What one season's build produced, for reporting and gating."""

    season: Season
    n_events: int = 0
    n_games: int = 0
    n_shots: int = 0
    n_stint_units: int = 0
    n_stints: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    quality_counts: dict[str, int] = field(default_factory=dict)
    missing_sources: list[str] = field(default_factory=list)
    minutes: MinutesCheck | None = None
    #: Share of shots that resolved to a complete five-man lineup.
    shot_lineup_coverage: float = 0.0
    #: Agreement between derived and feed-reported three-point classification.
    zone_agreement: float = 0.0

    @property
    def exact_rate(self) -> float:
        total = sum(self.status_counts.values())
        return self.status_counts.get("EXACT", 0) / total if total else 0.0

    @property
    def resolved_rate(self) -> float:
        """Share of period-team units that produced any lineup at all."""
        total = sum(self.status_counts.values())
        if not total:
            return 0.0
        return 1.0 - (self.status_counts.get("UNDERDETERMINED", 0) / total)


def ingest_season(
    season: Season, paths: DataPaths, *, oracles: bool = True
) -> dict[str, pl.DataFrame]:
    """Populate the bronze cache and return the raw frames.

    Optional artifacts that upstream does not have come back absent rather than
    raising -- a playoff file for a season still in progress, for example.
    """
    cache = BronzeCache(paths.bronze)
    sources: list[Source] = list(shufinskiy_sources(season))
    if oracles:
        sources.extend(sdv_sources(season))

    frames: dict[str, pl.DataFrame] = {}
    for source in sources:
        frame = cache.fetch(source)
        if frame is not None:
            frames[source.namespace] = frame
    return frames


def build_season(season: Season, paths: DataPaths, *, oracles: bool = True) -> SeasonBuild:
    """bronze -> silver for one season. Gold is assembled separately."""
    report = SeasonBuild(season=season)
    frames = ingest_season(season, paths, oracles=oracles)

    pbp_parts = [frames[k] for k in ("shufinskiy/pbp", "shufinskiy/pbp_po") if k in frames]
    if not pbp_parts:
        raise RuntimeError(f"no play-by-play available for {season.label}")
    for key in ("shufinskiy/pbp_po", "shufinskiy/shots_po", "sdv/lineup_oracle", "sdv/possessions"):
        if key not in frames:
            report.missing_sources.append(key)

    # Playoff and regular-season archives share a schema; concatenate before
    # typing so the season assertion sees every game at once.
    raw_pbp = pl.concat(pbp_parts, how="vertical_relaxed")
    events = canonical_order(type_events(raw_pbp, season))

    report.n_events = events.height
    report.n_games = events["game_id"].n_unique()

    shot_parts = [frames[k] for k in ("shufinskiy/shots", "shufinskiy/shots_po") if k in frames]
    if shot_parts:
        report.n_shots = sum(part.height for part in shot_parts)

    enriched, solutions = reconstruct_stints(events)
    report.n_stint_units = solutions.height
    if solutions.height:
        report.status_counts = {
            str(k): int(v) for k, v in solutions.group_by("status").len().iter_rows()
        }
        report.quality_counts = {
            str(k): int(v) for k, v in solutions.group_by("quality").len().iter_rows()
        }

    stints = segment_stints(enriched)
    report.n_stints = stints.height

    # The independent check: derived minutes against the box score.
    if "sdv/boxscore" in frames:
        agreement = minutes_agreement(stints, frames["sdv/boxscore"])
        report.minutes = _summarise_minutes(agreement)
    else:
        report.missing_sources.append("sdv/boxscore (minutes check skipped)")

    silver = paths.silver / f"season={season.start_year}"
    silver.mkdir(parents=True, exist_ok=True)
    events.write_parquet(silver / "events_typed.parquet")
    enriched.write_parquet(silver / "events_enriched.parquet")
    solutions.write_parquet(silver / "period_solutions.parquet")
    stints.write_parquet(silver / "stints.parquet")

    # --- gold: committed, partitioned by season -------------------------
    part = f"season={season.start_year}"
    if shot_parts:
        shots_raw = pl.concat(shot_parts, how="vertical_relaxed")
        shots_raw.write_parquet(silver / "shots_raw.parquet")
        shot_facts = build_shot_facts(shots_raw, enriched, season)
        report.n_shots = shot_facts.height
        report.shot_lineup_coverage = (
            shot_facts.filter(pl.col("lineup_for_hash").is_not_null()).height / shot_facts.height
            if shot_facts.height
            else 0.0
        )
        report.zone_agreement = _zone_agreement(shot_facts)
        _write_gold(paths, "shot_facts", part, shot_facts)

    _write_gold(paths, "stints", part, stints)
    _write_gold(paths, "dim_player", part, build_dim_player(events))

    return report


def _write_gold(paths: DataPaths, table: str, partition: str, frame: pl.DataFrame) -> None:
    target = paths.gold / table / partition
    target.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target / "part.parquet")


def _zone_agreement(shot_facts: pl.DataFrame) -> float:
    """How often our geometric zone matches the feed's own coarse label.

    Not an equality check -- the taxonomies differ in granularity. It compares
    only the three/two-point split, which both sources do encode, so a
    disagreement means the arc geometry is wrong rather than that the buckets
    are named differently.
    """
    if shot_facts.is_empty() or "shot_type_raw" not in shot_facts.columns:
        return 0.0
    feed_three = pl.col("shot_type_raw").str.contains("3PT")
    matched = shot_facts.filter(pl.col("shot_type_raw").is_not_null()).with_columns(
        (feed_three == pl.col("is_three")).alias("_agree")
    )
    return as_float(matched["_agree"].mean())


def _summarise_minutes(agreement: pl.DataFrame) -> MinutesCheck:
    """Reduce the per-player-game join to the numbers the gate is stated in."""
    played = agreement.filter(pl.col("box_seconds") > 0)
    dnp = agreement.filter(pl.col("box_seconds") == 0)

    total_box = as_float(played["box_seconds"].sum())
    total_derived = as_float(played["derived_seconds"].sum())
    return MinutesCheck(
        n_player_games=played.height,
        mean_abs_delta=as_float(played["abs_delta"].mean()),
        p95_abs_delta=as_float(played["abs_delta"].quantile(0.95)),
        max_abs_delta=as_int(played["abs_delta"].max()),
        pct_total_discrepancy=(abs(total_derived - total_box) / total_box if total_box else 0.0),
        n_dnp_with_minutes=int(dnp.filter(pl.col("derived_seconds") > 0).height),
    )


def silver_path(paths: DataPaths, season: Season, name: str) -> Path:
    return paths.silver / f"season={season.start_year}" / f"{name}.parquet"
