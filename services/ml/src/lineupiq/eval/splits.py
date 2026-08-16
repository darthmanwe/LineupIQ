"""Cross-validation splits.

Two kinds, testing different claims.

**Walk-forward** answers "does this generalise to later games?" -- the ordinary
temporal question.

**Leave-lineup-out** answers the question this project actually makes a claim
about: does it generalise to a five-man combination it has never seen, when it
*has* seen all five players individually? That is the counterfactual the trade
simulator depends on, and it is the split most models quietly avoid.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from lineupiq.config import SEED
from lineupiq.eval.leakage import (
    assert_no_game_overlap,
    assert_no_lineup_overlap,
    assert_players_seen,
    assert_temporal_disjoint,
)

__all__ = ["Fold", "leave_lineup_out", "walk_forward_by_game"]


@dataclass(frozen=True)
class Fold:
    name: str
    train: pl.DataFrame
    test: pl.DataFrame

    @property
    def sizes(self) -> tuple[int, int]:
        return self.train.height, self.test.height


def walk_forward_by_game(
    shots: pl.DataFrame, *, n_folds: int = 4, min_train_fraction: float = 0.4
) -> Iterator[Fold]:
    """Expanding-window temporal folds, split on game boundaries.

    Games are ordered by id, which is chronological within a season. Splitting
    inside a game would leak: both sides would share the same lineups, the same
    opponent, and the same night's shooting variance.
    """
    games = sorted(shots["game_id"].unique().to_list())
    if len(games) < n_folds + 1:
        return

    start = int(len(games) * min_train_fraction)
    step = max(1, (len(games) - start) // n_folds)

    for k in range(n_folds):
        cut = start + k * step
        end = min(cut + step, len(games))
        if cut >= len(games) or end <= cut:
            break
        train_games = set(games[:cut])
        test_games = set(games[cut:end])

        train = shots.filter(pl.col("game_id").is_in(list(train_games)))
        test = shots.filter(pl.col("game_id").is_in(list(test_games)))
        if train.is_empty() or test.is_empty():
            continue

        assert_no_game_overlap(train, test)
        assert_temporal_disjoint(train, test)
        yield Fold(f"walk_forward_{k + 1}", train, test)


def leave_lineup_out(
    shots: pl.DataFrame,
    *,
    n_folds: int = 5,
    min_shots_per_lineup: int = 25,
    seed: int = SEED,
) -> Iterator[Fold]:
    """Hold out whole five-man combinations, keeping every player in training.

    The two assertions at the end are opposites and both are required. Without
    ``assert_no_lineup_overlap`` this is not a lineup-out split at all; without
    ``assert_players_seen`` a fold that happens to hold out a player's only
    lineup is measuring cold start while being reported as a combination
    result, which is a much stronger claim than the experiment supports.
    """
    eligible = (
        shots.filter(pl.col("lineup_for_hash").is_not_null())
        .group_by("lineup_for_hash")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_shots_per_lineup)
    )
    hashes = eligible["lineup_for_hash"].to_list()
    if len(hashes) < n_folds:
        return

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(hashes))
    buckets = np.array_split(order, n_folds)

    for k, bucket in enumerate(buckets):
        held = {hashes[i] for i in bucket}
        test = shots.filter(pl.col("lineup_for_hash").is_in(list(held)))
        train = shots.filter(
            pl.col("lineup_for_hash").is_null() | ~pl.col("lineup_for_hash").is_in(list(held))
        )
        if train.is_empty() or test.is_empty():
            continue

        # Keep only held-out shots whose shooter is still represented in train,
        # so the fold measures the combination and not cold start.
        seen = set(train["shooter_id"].unique().to_list())
        test = test.filter(pl.col("shooter_id").is_in(list(seen)))
        if test.is_empty():
            continue

        assert_no_lineup_overlap(train, test)
        assert_players_seen(train, test)
        yield Fold(f"lolo_{k + 1}", train, test)
