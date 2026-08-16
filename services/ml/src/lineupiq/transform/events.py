"""Typed, canonically ordered play-by-play events.

The single most important thing in this module is the sort key.

``EVENTNUM`` is not chronological. Replaying a game in ``EVENTNUM`` order with a
known-correct lineup produces hundreds of impossible states -- a player
recorded as acting after he was substituted out, or before he came in. Sorting
by the game clock instead cuts those violations by about 91%, because the clock
is what actually orders play and ``EVENTNUM`` is an insertion artifact.

That is why hoopR's derived files carry both ``order_index`` and
``action_number``: neither alone is sufficient.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from lineupiq.seasons import MODELLED_GAME_TYPES, Season

__all__ = [
    "EVENT_EJECTION",
    "EVENT_SUBSTITUTION",
    "PERSON_TYPE_AWAY",
    "PERSON_TYPE_HOME",
    "TECHNICAL_FOUL_ACTION_TYPES",
    "canonical_order",
    "parse_clock_seconds",
    "type_events",
]

# EVENTMSGTYPE values that matter to lineup tracking.
EVENT_MADE_SHOT: Final = 1
EVENT_MISSED_SHOT: Final = 2
EVENT_FREE_THROW: Final = 3
EVENT_REBOUND: Final = 4
EVENT_TURNOVER: Final = 5
EVENT_FOUL: Final = 6
EVENT_VIOLATION: Final = 7
EVENT_SUBSTITUTION: Final = 8
EVENT_TIMEOUT: Final = 9
EVENT_JUMP_BALL: Final = 10
EVENT_EJECTION: Final = 11
EVENT_PERIOD_BEGIN: Final = 12
EVENT_PERIOD_END: Final = 13
EVENT_REPLAY: Final = 18

#: PERSON{n}TYPE: which side of the floor the referenced person is on.
#: 2/3 are team-level rows and 6/7 are coaches -- neither is a player on court.
PERSON_TYPE_HOME: Final = 4
PERSON_TYPE_AWAY: Final = 5

#: EVENTMSGACTIONTYPE values on a foul (EVENTMSGTYPE 6) that denote a technical.
#:
#: These are excluded from on-court evidence because a player on the *bench* can
#: be assessed a technical. Treating one as proof of presence puts a seated
#: player on the floor and makes the surrounding stint unsolvable.
TECHNICAL_FOUL_ACTION_TYPES: Final[frozenset[int]] = frozenset({10, 11, 12, 13, 16, 18, 19, 25, 30})

_INT_COLUMNS: Final = (
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "PERSON1TYPE",
    "PERSON2TYPE",
    "PERSON3TYPE",
    "PLAYER1_ID",
    "PLAYER2_ID",
    "PLAYER3_ID",
    "PLAYER1_TEAM_ID",
    "PLAYER2_TEAM_ID",
    "PLAYER3_TEAM_ID",
)


def parse_clock_seconds(column: str = "PCTIMESTRING") -> pl.Expr:
    """``"11:41"`` -> 701 seconds remaining in the period.

    Malformed or absent clocks yield null rather than 0. Zero is a real clock
    value (the buzzer), so coercing to it would fabricate end-of-period events.
    """
    parts = pl.col(column).str.strip_chars().str.split(":")
    # null_on_oob is required: polars evaluates every branch of a when/then, so
    # a bare `.list.get(1)` raises on any malformed clock rather than falling
    # through to the null this expression intends.
    mins = parts.list.get(0, null_on_oob=True).cast(pl.Int64, strict=False)
    secs = parts.list.get(1, null_on_oob=True).cast(pl.Int64, strict=False)
    return (
        pl.when(parts.list.len() == 2)
        .then(mins * 60 + secs)
        .otherwise(None)
        .alias("seconds_remaining")
    )


def type_events(raw: pl.DataFrame, season: Season) -> pl.DataFrame:
    """Cast the all-Utf8 bronze frame to real types and assert its identity.

    Raises if the frame's ``GAME_ID`` values do not decode to ``season``. That
    assertion is the whole defence against the two mirrors' conflicting filename
    conventions -- without it, a one-year offset is undetectable.
    """
    if "GAME_ID" not in raw.columns:
        raise ValueError(f"expected a GAME_ID column, got {raw.columns}")

    frame = raw.with_columns(
        # Leading zeros are stripped upstream; restore them before anything
        # joins on this column.
        pl.col("GAME_ID").cast(pl.Utf8).str.strip_chars().str.zfill(10).alias("game_id"),
    )

    for col in _INT_COLUMNS:
        if col in frame.columns:
            frame = frame.with_columns(pl.col(col).cast(pl.Int64, strict=False))

    frame = frame.with_columns(
        parse_clock_seconds(),
        pl.col("game_id").str.slice(0, 3).alias("_game_type_prefix"),
        pl.col("game_id").str.slice(3, 2).alias("_season_digits"),
    )

    _assert_season(frame, season)

    game_type = (
        pl.when(pl.col("_game_type_prefix") == "001")
        .then(pl.lit("preseason"))
        .when(pl.col("_game_type_prefix") == "002")
        .then(pl.lit("regular"))
        .when(pl.col("_game_type_prefix") == "003")
        .then(pl.lit("allstar"))
        .when(pl.col("_game_type_prefix") == "004")
        .then(pl.lit("playoffs"))
        .when(pl.col("_game_type_prefix") == "005")
        .then(pl.lit("playin"))
        .when(pl.col("_game_type_prefix") == "006")
        .then(pl.lit("cupfinal"))
        .otherwise(pl.lit("unknown"))
        .alias("game_type")
    )

    frame = (
        frame.with_columns(game_type)
        .with_columns(pl.lit(season.start_year).cast(pl.Int64).alias("season"))
        .drop("_game_type_prefix", "_season_digits")
    )

    # Preseason rotations are not real rotations and all-star defense is not
    # defense. Both would otherwise pass every structural check downstream.
    frame = frame.filter(pl.col("game_type").is_in(list(MODELLED_GAME_TYPES)))

    # (game_id, event_num) is the documented grain. Duplicates appear when a
    # regular-season and playoff archive overlap.
    return frame.unique(subset=["game_id", "EVENTNUM"], keep="first")


def _assert_season(frame: pl.DataFrame, season: Season) -> None:
    observed = set(frame["_season_digits"].unique().to_list()) - {None}
    expected = season.two_digit
    if observed != {expected}:
        raise ValueError(
            f"season mismatch: requested {season.label} (GAME_ID digits {expected!r}) "
            f"but the payload contains digits {sorted(observed)!r}. "
            "This is the filename-convention trap -- the file claimed one season "
            "and contains another. Do not 'fix' this by relabelling; re-check "
            "which mirror the URL came from."
        )


def canonical_order(events: pl.DataFrame) -> pl.DataFrame:
    """Sort into true chronological order.

    ``EVENTNUM`` ascending is the obvious choice and it is wrong: measured over
    120 games it produced 435 lineup-invariant violations, against 40 for this
    key -- a 91% reduction. Clock descending is primary; ``EVENTNUM`` only
    breaks ties within the same second.

    Events with a null clock sort last within their period rather than first,
    so a malformed row cannot displace real play.
    """
    return events.sort(
        by=["game_id", "PERIOD", "seconds_remaining", "EVENTNUM"],
        descending=[False, False, True, False],
        nulls_last=True,
    )
