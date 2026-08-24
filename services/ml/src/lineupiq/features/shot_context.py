"""Where in a possession each shot was taken.

``shot_facts`` knows the game clock; it does not know how long the offence had
been holding the ball. That distinction is most of what drives shot *selection*:
a shot two seconds after a steal and a shot eighteen seconds into a half-court
set are different decisions taken by the same player from the same spot.

The feed carries no shot clock, so it is derived here by joining each shot to
the possession that contains it. The join keys on the offensive team as well as
the clock, which is what disambiguates the common case of a possession's last
event and the next possession's first event landing on the same second.
"""

from __future__ import annotations

import polars as pl

from lineupiq.transform.possessions import (
    OVERTIME_PERIOD_SECONDS,
    REGULATION_PERIOD_SECONDS,
    possession_windows,
)

__all__ = [
    "BOUNDARY_TOLERANCE_SECONDS",
    "POSSESSION_CONTEXT_COLUMNS",
    "attach_possession_context",
    "context_coverage",
]

#: Slack allowed at a possession's edges, in seconds.
#:
#: The two feeds carry the clock at different resolutions: play-by-play parses a
#: ``MM:SS`` string to a whole second, while the possession feed keeps tenths.
#: A shot recorded at 501 against a possession ending at 501.4 is the same
#: event, and with no tolerance it matched nothing.
#:
#: One second is the coarser feed's own resolution, and the coverage curve says
#: that is all this is: 95.84% at zero tolerance, 99.74% at one second, then
#: essentially flat -- 99.749% at two, 99.759% at three. A boundary problem that
#: resolves at exactly one unit of quantisation and then stops improving is a
#: rounding artefact, not a tuning parameter.
BOUNDARY_TOLERANCE_SECONDS = 1.0

#: Columns added by :func:`attach_possession_context`.
POSSESSION_CONTEXT_COLUMNS: tuple[str, ...] = (
    "seconds_into_possession",
    "possession_seconds",
    "possession_start_type",
    "live_ball_start",
    "transition",
    "is_second_chance",
    "possession_points",
    "has_possession_context",
)


def attach_possession_context(shots: pl.DataFrame, possessions: pl.DataFrame) -> pl.DataFrame:
    """Add possession-relative context to every shot.

    Shots that cannot be placed in a possession keep the columns with null
    values and ``has_possession_context = False``. They are not dropped: a shot
    silently disappearing from a model's training set because a join missed is
    exactly the failure the coverage gate exists to catch.

    The window used is ``[end_seconds_remaining, possession_start_clock]``,
    where the start is the change of hands rather than the possession's first
    recorded event. With the feed's own start the windows left 4.4% of shots in
    gaps between possessions; deriving the start closes them.

    **Three of the columns this adds are outcome-contaminated and must never be
    used as features.** ``possession_seconds`` is measured to the possession's
    last event, and a possession ends on a make at the shot but on a miss at the
    rebound a beat later -- so a short possession is evidence the shot went in.
    The effect is not subtle: shots that end their possession convert at 93.3%,
    shots that do not at 1.3%. ``transition`` inherits it through the duration
    cut, and ``possession_points`` contains the shot's own points outright. They
    are attached for reporting and are listed in
    :data:`lineupiq.eval.leakage.FORBIDDEN_FEATURES`.

    ``seconds_into_possession``, ``possession_start_type``, ``live_ball_start``
    and ``is_second_chance`` are all fixed before the shot resolves and are safe.
    """
    # An as-of join, not an equi-join plus a filter.
    #
    # The obvious form -- join on (game, period, team) and then keep the row
    # whose window contains the shot -- matches every possession that team had
    # in that period, roughly twelve per shot, and materialises about 8 million
    # rows before the filter runs. Measured peak committed memory for that was
    # 3.7 GB, which is more than the whole model costs and was the actual reason
    # a capped run died.
    #
    # ``join_asof`` finds the most recent possession start at or before each
    # shot in one linear merge. It needs a monotone key, so the countdown clock
    # is converted to elapsed time within the period.
    period_open = (
        pl.when(pl.col("period") <= 4)
        .then(REGULATION_PERIOD_SECONDS)
        .otherwise(OVERTIME_PERIOD_SECONDS)
    )

    windows = (
        possessions.select(
            "game_id",
            "period",
            "possession_number",
            "offense_team_id",
            "possession_start_clock",
            "end_seconds_remaining",
            "possession_seconds",
            "possession_start_type",
            "live_ball_start",
            "transition",
            "is_second_chance",
            pl.col("points").alias("possession_points"),
        )
        .with_columns(
            (period_open - pl.col("possession_start_clock")).alias("_start_elapsed"),
            (period_open - pl.col("end_seconds_remaining")).alias("_end_elapsed"),
        )
        .sort("_start_elapsed")
    )

    attempts = (
        shots.select("game_id", "event_num", "period", "seconds_remaining", "team_id")
        .with_columns((period_open - pl.col("seconds_remaining")).alias("_elapsed"))
        .sort("_elapsed")
    )

    matched = (
        attempts.join_asof(
            windows,
            left_on="_elapsed",
            right_on="_start_elapsed",
            by_left=["game_id", "period", "team_id"],
            by_right=["game_id", "period", "offense_team_id"],
            strategy="backward",
            # Both sides are sorted on the as-of key immediately above. Polars
            # cannot verify that itself once `by` groups are involved, and its
            # warning to that effect is noise here rather than information.
            check_sortedness=False,
        )
        # The as-of join guarantees the possession started at or before the
        # shot; it does not guarantee the shot happened before the possession
        # ended. A shot after the last possession of a period, or in a gap the
        # feed does not cover, has to fall out here rather than be attributed to
        # whatever came before it.
        .filter(
            pl.col("_start_elapsed").is_not_null()
            & (pl.col("_elapsed") <= pl.col("_end_elapsed") + BOUNDARY_TOLERANCE_SECONDS)
        )
        .with_columns(
            # Clipped at zero: the tolerance admits shots a second before the
            # derived start, and a negative elapsed time is not a thing.
            (pl.col("_elapsed") - pl.col("_start_elapsed"))
            .clip(lower_bound=0.0)
            .alias("seconds_into_possession")
        )
        .select(
            "game_id",
            "event_num",
            "seconds_into_possession",
            "possession_seconds",
            "possession_start_type",
            "live_ball_start",
            "transition",
            "is_second_chance",
            "possession_points",
        )
    )

    return shots.join(matched, on=["game_id", "event_num"], how="left").with_columns(
        pl.col("seconds_into_possession").is_not_null().alias("has_possession_context")
    )


def context_coverage(shots: pl.DataFrame) -> float:
    """Share of shots that were placed inside a possession."""
    if shots.is_empty() or "has_possession_context" not in shots.columns:
        return 0.0
    return shots.filter(pl.col("has_possession_context")).height / shots.height


def rebuild_windows(raw_possessions: pl.DataFrame) -> pl.DataFrame:
    """Re-derive possession windows from a raw upstream frame.

    Only needed when the committed ``possession_facts`` table is unavailable;
    the gold table already carries ``possession_start_clock``.
    """
    return possession_windows(raw_possessions)
