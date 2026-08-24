"""Tests for possession-window derivation and the shot-to-possession join.

Two bugs motivated every test here, and both were silent.

The feed's ``start_seconds_remaining`` is the clock at a possession's first
*recorded event*, not at the change of hands, so ``start - end`` was zero for
45% of possessions. Nothing raised; a published transition/half-court split was
simply computed on a duration that was not a duration.

The second is the reason the join carries a tolerance: the two feeds keep the
clock at different resolutions, so a shot recorded at 501 against a possession
ending at 501.4 is the same event, and matching exactly dropped 4% of shots
with no error anywhere.
"""

from __future__ import annotations

import polars as pl

from lineupiq.features.shot_context import (
    BOUNDARY_TOLERANCE_SECONDS,
    attach_possession_context,
    context_coverage,
)
from lineupiq.transform.possessions import LIVE_BALL_STARTS, possession_windows


def _raw_possessions() -> pl.DataFrame:
    """Two teams trading possessions through one period.

    The clock values reproduce the shape that caused the original bug: the first
    possession's recorded window spans 720 -> 695, and the second team's first
    recorded event is not until 675, leaving twenty seconds unaccounted for.
    """
    return pl.DataFrame(
        {
            "game_id": ["0022300001"] * 4,
            "period": [1, 1, 1, 1],
            "number_in_period": [1, 2, 3, 4],
            "possession_number": [1, 2, 3, 4],
            "offense_team_id": [10, 20, 10, 20],
            "start_seconds_remaining": [720.0, 675.0, 665.0, 646.0],
            "end_seconds_remaining": [695.0, 675.0, 663.0, 640.0],
            "count_as_possession": [True, True, True, True],
        }
    )


def test_window_start_is_the_change_of_hands_not_the_first_event() -> None:
    windows = possession_windows(_raw_possessions())
    starts = windows["possession_start_clock"].to_list()

    # First possession of the period opens at the period clock.
    assert starts[0] == 720.0
    # The second team gained the ball when the first team lost it, at 695 --
    # not at 675 where its own first event happens to be recorded.
    assert starts[1] == 695.0
    assert starts[2] == 675.0
    assert starts[3] == 663.0


def test_derived_durations_are_positive_and_plausible() -> None:
    windows = possession_windows(_raw_possessions())
    durations = windows["possession_seconds"].to_list()
    assert durations == [25.0, 20.0, 12.0, 23.0]
    # The feed's own arithmetic would have called the second possession zero
    # seconds long, which is the bug this replaces.
    feed = [
        s - e
        for s, e in zip(
            windows["start_seconds_remaining"].to_list(),
            windows["end_seconds_remaining"].to_list(),
            strict=True,
        )
    ]
    assert feed[1] == 0.0
    assert durations[1] > 0.0


def test_overtime_periods_open_at_five_minutes() -> None:
    raw = _raw_possessions().with_columns(pl.lit(5).alias("period"))
    windows = possession_windows(raw)
    assert windows["possession_start_clock"].to_list()[0] == 300.0


def test_a_dropped_possession_does_not_splice_its_time_onto_a_neighbour() -> None:
    """The window has to be derived before ``count_as_possession`` filtering.

    If the third possession is dropped first, the fourth would inherit the
    second's end and silently absorb the dropped possession's time.
    """
    raw = _raw_possessions().with_columns(
        pl.Series("count_as_possession", [True, True, False, True])
    )
    kept = possession_windows(raw).filter(pl.col("count_as_possession"))
    fourth = kept.filter(pl.col("possession_number") == 4)
    # 663 is the third possession's end, even though the third row is gone.
    assert fourth["possession_start_clock"].item() == 663.0


def _shots() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["0022300001"] * 4,
            "event_num": [1, 2, 3, 4],
            "period": [1, 1, 1, 1],
            # 700 is inside possession 1; 680 inside possession 2 (only because
            # its window starts at the change of hands); 664 inside possession
            # 3; 641 inside possession 4.
            "seconds_remaining": [700, 680, 664, 641],
            "team_id": [10, 20, 10, 20],
        }
    )


def _possession_facts() -> pl.DataFrame:
    """Windows plus the outcome columns the join carries through."""
    return possession_windows(_raw_possessions()).with_columns(
        pl.lit("OffMissedShot").alias("possession_start_type"),
        pl.lit(True).alias("live_ball_start"),
        pl.lit(False).alias("transition"),
        pl.lit(False).alias("is_second_chance"),
        pl.lit(2).alias("points"),
    )


def test_shots_are_placed_in_the_right_possession() -> None:
    out = attach_possession_context(_shots(), _possession_facts())

    assert context_coverage(out) == 1.0
    # Seconds into the possession is measured from the change of hands.
    assert out["seconds_into_possession"].to_list() == [20.0, 15.0, 11.0, 22.0]


def test_a_shot_in_the_gap_would_be_lost_without_the_derived_start() -> None:
    """The 20-second hole between the feed's own windows.

    A shot at 680 sits inside no feed-reported window at all: possession 1 ends
    at 695 and possession 2 reports starting at 675. It is only placeable
    because the start is derived.
    """
    raw = _raw_possessions()
    second = raw.filter(pl.col("possession_number") == 2)
    assert second["start_seconds_remaining"].item() < 680
    assert raw.filter(pl.col("possession_number") == 1)["end_seconds_remaining"].item() > 680


def test_boundary_tolerance_is_one_second() -> None:
    """A clock-resolution allowance, not a tuning knob.

    Coverage goes 95.84% -> 99.74% at one second and then flattens (99.749% at
    two, 99.759% at three). If this ever grows, the justification has to change
    with it.
    """
    assert BOUNDARY_TOLERANCE_SECONDS == 1.0


def test_unmatched_shots_are_kept_and_flagged() -> None:
    """A shot that cannot be placed must not vanish from the frame."""
    shots = _shots().with_columns(pl.Series("seconds_remaining", [700, 680, 664, 5]))
    out = attach_possession_context(shots, _possession_facts())

    assert out.height == shots.height
    assert out["has_possession_context"].to_list() == [True, True, True, False]
    assert (
        out.filter(~pl.col("has_possession_context"))["seconds_into_possession"].null_count() == 1
    )


def test_live_ball_starts_exclude_a_made_basket() -> None:
    """After the opponent scores the clock stops and the ball is inbounded.

    Median possession length says so: 17s after a made shot against 18s after a
    timeout, versus 10s after a defensive rebound and 7s after a steal.
    """
    assert "OffMadeShot" not in LIVE_BALL_STARTS
    assert set(LIVE_BALL_STARTS) == {"OffMissedShot", "OffLiveBallTurnover"}
