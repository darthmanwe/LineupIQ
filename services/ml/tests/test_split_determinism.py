"""Folds must not depend on how many cores the machine has.

`leave_lineup_out` permutes a list of lineup hashes with a fixed seed. The
permutation is over *positions*, so the identity of fold k depends on the order
of that list -- and the list came out of a parallel `group_by`, which makes no
ordering promise. The result reproduced perfectly on one machine and moved 60
metrics on another.

These tests pin the property that was missing rather than the numbers that
happened to come out.
"""

from __future__ import annotations

import polars as pl

from lineupiq.eval.splits import leave_lineup_out


def _corpus(n_lineups: int = 12, per_lineup: int = 30) -> pl.DataFrame:
    rows = []
    for lineup in range(n_lineups):
        for shot in range(per_lineup):
            rows.append(
                {
                    "game_id": f"002230{lineup:04d}",
                    "lineup_for_hash": f"{lineup:032x}",
                    "shooter_id": 1000 + (shot % 7),
                    "made": shot % 2,
                }
            )
    return pl.DataFrame(rows)


def _fold_membership(frame: pl.DataFrame) -> list[list[str]]:
    return [
        sorted(fold.test["lineup_for_hash"].unique().to_list())
        for fold in leave_lineup_out(frame, n_folds=3, min_shots_per_lineup=25)
    ]


def test_folds_are_stable_under_row_order() -> None:
    """A shuffled corpus must produce identical folds.

    Row order is the observable proxy for the real hazard: `group_by` output
    order varies with the thread pool, and re-ordering the input is how that is
    reproduced deterministically in a test.
    """
    frame = _corpus()
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=7)
    assert _fold_membership(frame) == _fold_membership(shuffled)


def test_folds_are_stable_under_reversal() -> None:
    frame = _corpus()
    assert _fold_membership(frame) == _fold_membership(frame.reverse())


def test_every_eligible_lineup_lands_in_exactly_one_fold() -> None:
    frame = _corpus()
    folds = _fold_membership(frame)
    flat = [h for fold in folds for h in fold]
    assert len(flat) == len(set(flat)), "a lineup appeared in two test folds"
    assert set(flat) == set(frame["lineup_for_hash"].unique().to_list())
