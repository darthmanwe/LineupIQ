"""Stint segmentation and the minutes invariant.

A stint is a maximal run of play over which *both* lineups are constant. Once
events carry lineups, segmenting is bookkeeping -- but the durations it produces
are what make the whole reconstruction falsifiable.

Summed per player, stint durations must equal the minutes the box score
records. That check is the one genuinely independent validation available here:
the box score is produced by a different system from a different source, and
minutes played is a physical quantity. A lineup reconstruction can agree with
another *derived* lineup file and still be wrong in the same way; it cannot
disagree with the clock and be right.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "OVERTIME_PERIOD_SECONDS",
    "REGULATION_PERIOD_SECONDS",
    "minutes_agreement",
    "period_length_seconds",
    "player_seconds",
    "segment_stints",
]

REGULATION_PERIOD_SECONDS: Final = 720
OVERTIME_PERIOD_SECONDS: Final = 300


def period_length_seconds(period: int) -> int:
    return REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS


def segment_stints(enriched: pl.DataFrame) -> pl.DataFrame:
    """Collapse enriched events into stint intervals.

    Expects the output of :func:`lineupiq.transform.stints.reconstruct_stints`,
    already in canonical order.
    """
    if enriched.is_empty():
        return _empty_stints()

    rows: list[dict[str, object]] = []

    for (game_id, period), group in enriched.group_by(["game_id", "period"], maintain_order=True):
        period_start = period_length_seconds(int(period))
        current: tuple[list[int], list[int]] | None = None
        start_sec = period_start
        quality = "VALID"
        stint_index = 0

        records = group.to_dicts()
        for row in records:
            home, away = row["home_lineup"], row["away_lineup"]
            secs = row["seconds_remaining"]
            if home is None or away is None or secs is None:
                continue
            key = (list(home), list(away))

            if current is None:
                current = key
                start_sec = period_start
                quality = row["lineup_quality"]
                continue

            if key != current:
                end_sec = int(secs)
                if start_sec > end_sec:
                    rows.append(
                        _stint_row(
                            game_id, period, stint_index, start_sec, end_sec, current, quality
                        )
                    )
                    stint_index += 1
                current = key
                start_sec = int(secs)
                quality = row["lineup_quality"]
            elif row["lineup_quality"] == "QUARANTINED":
                quality = "QUARANTINED"

        if current is not None and start_sec > 0:
            rows.append(_stint_row(game_id, period, stint_index, start_sec, 0, current, quality))

    return pl.DataFrame(rows) if rows else _empty_stints()


def _stint_row(
    game_id: str,
    period: int,
    index: int,
    start_sec: int,
    end_sec: int,
    lineups: tuple[list[int], list[int]],
    quality: str,
) -> dict[str, object]:
    home, away = lineups
    return {
        "game_id": game_id,
        "period": int(period),
        "stint_index": index,
        "start_seconds_remaining": start_sec,
        "end_seconds_remaining": end_sec,
        "duration_seconds": start_sec - end_sec,
        "home_lineup": home,
        "away_lineup": away,
        "stint_quality": quality,
    }


def player_seconds(stints: pl.DataFrame) -> pl.DataFrame:
    """Seconds on court per ``(game_id, player_id)``, from stint durations."""
    if stints.is_empty():
        return pl.DataFrame(schema={"game_id": pl.Utf8, "player_id": pl.Int64, "seconds": pl.Int64})

    sides = [
        stints.select(
            "game_id",
            pl.col(col).alias("player_id"),
            "duration_seconds",
        ).explode("player_id")
        for col in ("home_lineup", "away_lineup")
    ]
    return (
        pl.concat(sides)
        .drop_nulls("player_id")
        .group_by(["game_id", "player_id"])
        .agg(pl.col("duration_seconds").sum().alias("seconds"))
        .sort(["game_id", "player_id"])
    )


def _parse_box_minutes(column: str = "minutes") -> pl.Expr:
    """Box-score minutes arrive as ``"MM:SS"`` (sometimes ``"M:SS"``).

    An empty string means the player did not appear. That must become 0, not
    null -- a DNP deriving non-zero minutes is a hard failure, and nulls would
    silently drop those rows out of the comparison instead of failing it.
    """
    raw = pl.col(column).cast(pl.Utf8).str.strip_chars()
    parts = raw.str.split(":")
    # null_on_oob is required: polars evaluates every branch of a when/then, so
    # a bare `.list.get(1)` raises on the DNP rows this expression exists to
    # handle -- the guard never gets a chance to short-circuit it.
    mins = parts.list.get(0, null_on_oob=True).cast(pl.Int64, strict=False)
    secs = parts.list.get(1, null_on_oob=True).cast(pl.Int64, strict=False)
    return (
        pl.when(raw.is_null() | (raw.str.len_chars() == 0))
        .then(pl.lit(0, dtype=pl.Int64))
        .when(parts.list.len() == 2)
        .then(mins * 60 + secs)
        .otherwise(pl.lit(0, dtype=pl.Int64))
        .alias("box_seconds")
    )


def minutes_agreement(stints: pl.DataFrame, boxscore: pl.DataFrame) -> pl.DataFrame:
    """Join derived seconds to box-score seconds per player-game."""
    derived = player_seconds(stints)

    cols = set(boxscore.columns)
    game_col = next((c for c in ("game_id", "gameId", "GAME_ID") if c in cols), None)
    if game_col is None or "person_id" not in cols or "minutes" not in cols:
        raise ValueError(
            f"boxscore must carry game id, person_id and minutes; got {sorted(cols)[:15]}..."
        )

    box = boxscore.select(
        pl.col(game_col).cast(pl.Utf8).str.strip_chars().str.zfill(10).alias("game_id"),
        pl.col("person_id").cast(pl.Int64).alias("player_id"),
        _parse_box_minutes(),
    )

    return (
        box.join(derived, on=["game_id", "player_id"], how="left")
        .with_columns(pl.col("seconds").fill_null(0).alias("derived_seconds"))
        .drop("seconds")
        .with_columns((pl.col("derived_seconds") - pl.col("box_seconds")).abs().alias("abs_delta"))
    )


def _empty_stints() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "game_id": pl.Utf8,
            "period": pl.Int64,
            "stint_index": pl.Int64,
            "start_seconds_remaining": pl.Int64,
            "end_seconds_remaining": pl.Int64,
            "duration_seconds": pl.Int64,
            "home_lineup": pl.List(pl.Int64),
            "away_lineup": pl.List(pl.Int64),
            "stint_quality": pl.Utf8,
        }
    )
