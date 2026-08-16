"""Gold tables: model-ready facts with full lineup context.

Gold is **committed to the repository**. That is what lets a clean clone
re-derive every published number offline, with no network and no account, and
it is why the contract checksums in ``data/gold/_contracts/`` matter.

Table names and grains match the original Snowflake design exactly, so the
optional warehouse adapter is a thin swap rather than a translation.
"""

from __future__ import annotations

import hashlib

import polars as pl

from lineupiq.hashing import LINEUP_SIZE
from lineupiq.seasons import Season
from lineupiq.transform.zones import derive_zone, is_three_expr

__all__ = ["build_dim_player", "build_shot_facts"]


def build_shot_facts(
    shots_raw: pl.DataFrame, enriched: pl.DataFrame, season: Season
) -> pl.DataFrame:
    """One row per shot attempt, carrying the five-man context it happened in.

    The join key is ``GAME_EVENT_ID`` on the shot side against ``EVENTNUM`` on
    the play-by-play side. Shots that do not match an enriched event -- and
    therefore have no lineup -- are kept with null lineups and a quarantine
    flag rather than dropped, so coverage is measurable instead of invisible.
    """
    shots = shots_raw.select(
        pl.col("GAME_ID").cast(pl.Utf8).str.strip_chars().str.zfill(10).alias("game_id"),
        pl.col("GAME_EVENT_ID").cast(pl.Int64).alias("event_num"),
        pl.col("PLAYER_ID").cast(pl.Int64).alias("shooter_id"),
        pl.col("TEAM_ID").cast(pl.Int64).alias("team_id"),
        pl.col("PERIOD").cast(pl.Int64).alias("period"),
        (
            pl.col("MINUTES_REMAINING").cast(pl.Int64) * 60
            + pl.col("SECONDS_REMAINING").cast(pl.Int64)
        ).alias("seconds_remaining"),
        pl.col("LOC_X").cast(pl.Float64).alias("loc_x"),
        pl.col("LOC_Y").cast(pl.Float64).alias("loc_y"),
        pl.col("SHOT_DISTANCE").cast(pl.Float64).alias("shot_distance_ft"),
        pl.col("ACTION_TYPE").cast(pl.Utf8).alias("action_type"),
        pl.col("SHOT_TYPE").cast(pl.Utf8).alias("shot_type_raw"),
        pl.col("SHOT_ZONE_BASIC").cast(pl.Utf8).alias("feed_zone_basic"),
        pl.col("SHOT_MADE_FLAG").cast(pl.Int64).alias("made"),
    ).with_columns(
        derive_zone(),
        is_three_expr().alias("is_three"),
        pl.lit(season.start_year).cast(pl.Int64).alias("season"),
    )

    shots = shots.with_columns(
        pl.when(pl.col("is_three")).then(3).otherwise(2).cast(pl.Int64).alias("shot_points")
    )

    lineups = enriched.select(
        "game_id",
        "event_num",
        "home_lineup",
        "away_lineup",
        "lineup_quality",
        "lineup_method",
    )

    joined = shots.join(lineups, on=["game_id", "event_num"], how="left")

    # Which side shot decides which lineup is "for" and which is "against".
    # Membership is the reliable test: team ids in the shot feed and the pbp
    # feed do not always agree in form.
    joined = joined.with_columns(
        pl.col("home_lineup").list.contains(pl.col("shooter_id")).alias("_shooter_is_home")
    )

    joined = joined.with_columns(
        pl.when(pl.col("_shooter_is_home"))
        .then(pl.col("home_lineup"))
        .otherwise(pl.col("away_lineup"))
        .alias("lineup_for"),
        pl.when(pl.col("_shooter_is_home"))
        .then(pl.col("away_lineup"))
        .otherwise(pl.col("home_lineup"))
        .alias("lineup_against"),
    )

    has_lineups = (
        pl.col("lineup_for").is_not_null()
        & pl.col("lineup_against").is_not_null()
        & (pl.col("lineup_for").list.len() == LINEUP_SIZE)
        & (pl.col("lineup_against").list.len() == LINEUP_SIZE)
        & pl.col("lineup_for").list.contains(pl.col("shooter_id"))
    )

    return (
        joined.with_columns(
            pl.when(has_lineups)
            .then(pl.col("lineup_quality").fill_null("QUARANTINED"))
            .otherwise(pl.lit("QUARANTINED"))
            .alias("stint_quality"),
            _hash_list("lineup_for").alias("lineup_for_hash"),
            _hash_list("lineup_against").alias("lineup_against_hash"),
        )
        .drop("_shooter_is_home")
        .sort(["game_id", "event_num"])
    )


def _hash_list(column: str) -> pl.Expr:
    """MD5 of the numerically-sorted, comma-joined ids. Null when incomplete.

    ``list.sort()`` on an integer list sorts numerically, then the cast to Utf8
    happens afterwards -- that order is the whole correctness condition, and it
    is asserted against both DuckDB and pure Python in ``test_hashing.py``.
    """
    joined = pl.col(column).list.sort().cast(pl.List(pl.Utf8)).list.join(",")
    return (
        pl.when(pl.col(column).is_null() | (pl.col(column).list.len() != LINEUP_SIZE))
        .then(None)
        .otherwise(joined.map_elements(_md5, return_dtype=pl.Utf8))
    )


def _md5(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.md5(value.encode("ascii")).hexdigest()


def build_dim_player(events: pl.DataFrame) -> pl.DataFrame:
    """Player id -> name, from whichever slot mentioned them.

    Names come from the play-by-play feed rather than a separate roster pull, so
    the dimension is always consistent with the facts and needs no network.
    """
    parts = []
    for slot in (1, 2, 3):
        idc, namec = f"PLAYER{slot}_ID", f"PLAYER{slot}_NAME"
        typec = f"PERSON{slot}TYPE"
        if idc not in events.columns or namec not in events.columns:
            continue
        parts.append(
            events.filter(pl.col(typec).is_in([4, 5]) & (pl.col(idc) > 0)).select(
                pl.col(idc).cast(pl.Int64).alias("player_id"),
                pl.col(namec).cast(pl.Utf8).alias("player_name"),
            )
        )
    if not parts:
        return pl.DataFrame(schema={"player_id": pl.Int64, "player_name": pl.Utf8})

    return (
        pl.concat(parts)
        .drop_nulls()
        .group_by("player_id")
        .agg(pl.col("player_name").mode().first().alias("player_name"))
        .sort("player_id")
    )
