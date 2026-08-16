"""Runtime leakage assertions.

These are called *inside* the cross-validation loop, not from a test. A leakage
bug does not raise; it produces a suspiciously good number, and by the time the
number is in a README nobody remembers which split produced it. Asserting at
fit time is the only place the check is cheap and unambiguous.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "FORBIDDEN_FEATURES",
    "LeakageError",
    "assert_no_game_overlap",
    "assert_no_lineup_overlap",
    "assert_players_seen",
    "assert_temporal_disjoint",
]


class LeakageError(AssertionError):
    """Train and test share information they must not."""


#: Columns that must never enter a feature matrix, with the reason.
FORBIDDEN_FEATURES: dict[str, str] = {
    # The play-by-play feed records an assister only on MADE shots. Using it
    # per-shot leaks the label directly and produces a spectacular fake AUC.
    "is_assisted": "assist is recorded only on makes -- direct label leak",
    "made": "the label",
    "shot_points": "derived from the label's zone at scoring time",
}


def assert_no_forbidden_features(columns: list[str]) -> None:
    found = [c for c in columns if c in FORBIDDEN_FEATURES]
    if found:
        detail = "; ".join(f"{c}: {FORBIDDEN_FEATURES[c]}" for c in found)
        raise LeakageError(f"forbidden feature(s) in the matrix -- {detail}")


def assert_no_game_overlap(train: pl.DataFrame, test: pl.DataFrame, col: str = "game_id") -> None:
    """No game may appear on both sides.

    Splitting within a game leaks: the same lineups, the same opponent, the
    same night's shooting variance sit on both sides of the boundary.
    """
    shared = set(train[col].unique().to_list()) & set(test[col].unique().to_list())
    if shared:
        raise LeakageError(
            f"{len(shared)} game(s) appear in both train and test, e.g. {sorted(shared)[:3]}"
        )


def assert_temporal_disjoint(train: pl.DataFrame, test: pl.DataFrame, col: str = "game_id") -> None:
    """Every training game must precede every test game.

    Game ids are sequential within a season, so ordering them is equivalent to
    ordering by date without needing a date column.
    """
    if train.is_empty() or test.is_empty():
        return
    # Game ids are zero-padded fixed-width strings, so lexicographic order is
    # chronological order. Comparing as text keeps that true and avoids the
    # broad union polars returns from an aggregate.
    latest_train = train[col].max()
    earliest_test = test[col].min()
    if latest_train is None or earliest_test is None:
        return
    if str(latest_train) >= str(earliest_test):
        raise LeakageError(
            f"train reaches {latest_train!s} but test starts at {earliest_test!s} -- "
            "the fold is not temporally disjoint"
        )


def assert_no_lineup_overlap(
    train: pl.DataFrame, test: pl.DataFrame, col: str = "lineup_for_hash"
) -> None:
    """For leave-lineup-out: no five-man combination may appear in both."""
    shared = set(train[col].drop_nulls().unique().to_list()) & set(
        test[col].drop_nulls().unique().to_list()
    )
    if shared:
        raise LeakageError(
            f"{len(shared)} lineup(s) appear in both train and test; this is not a "
            "leave-lineup-out split"
        )


def assert_players_seen(train: pl.DataFrame, test: pl.DataFrame, col: str = "shooter_id") -> None:
    """The inverse assertion, and the one usually forgotten.

    In a leave-lineup-out split every held-out *player* must still appear in
    training. Without this check, a split that happens to hold out a rookie's
    only lineup is silently measuring cold-start performance while being
    reported as a lineup-combination result -- a much stronger claim than the
    experiment supports.
    """
    unseen = set(test[col].unique().to_list()) - set(train[col].unique().to_list())
    if unseen:
        raise LeakageError(
            f"{len(unseen)} player(s) in test never appear in train, e.g. "
            f"{sorted(unseen)[:3]}. This is a cold-start split, not a "
            "leave-lineup-out split."
        )
