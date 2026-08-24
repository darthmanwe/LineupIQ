"""`lineup_slots` must reproduce `to_list` semantics exactly.

Three models had their lineup reads changed from `Series.to_list()` to
`lineup_slots`. The arithmetic downstream sums floats, and floating-point
addition is not associative, so "same members" is not enough -- the order has
to match too, or the fitted metrics move and `--verify` starts failing for a
reason that has nothing to do with reproducibility.
"""

from __future__ import annotations

import polars as pl

from lineupiq.util import ABSENT_PLAYER, lineup_slots


def _via_slots(series: pl.Series) -> list[list[int]]:
    slots = lineup_slots(series)
    return [
        [player for player in (int(slot[i]) for slot in slots) if player != ABSENT_PLAYER]
        for i in range(series.len())
    ]


def test_matches_to_list_on_well_formed_lineups() -> None:
    rows = [[201143, 1630552, 2544, 203507, 1629029], [101108, 202681, 203954, 1628369, 1627759]]
    series = pl.Series("lineup", rows, dtype=pl.List(pl.Int64))
    assert _via_slots(series) == [[int(p) for p in row] for row in rows]


def test_order_is_preserved_not_sorted() -> None:
    # The callers sum floats keyed on these ids. A sorted read would silently
    # change the summation order and move the fitted numbers.
    rows = [[5, 3, 1, 4, 2]]
    series = pl.Series("lineup", rows, dtype=pl.List(pl.Int64))
    assert _via_slots(series) == [[5, 3, 1, 4, 2]]


def test_null_lineup_reads_as_empty() -> None:
    # `to_list()` returned None for these rows and every call site wrote
    # `(lineup or [])`. The sentinel has to reproduce that, not a row of -1s.
    series = pl.Series("lineup", [None, [1, 2, 3, 4, 5]], dtype=pl.List(pl.Int64))
    assert _via_slots(series) == [[], [1, 2, 3, 4, 5]]


def test_short_lineup_yields_only_its_real_members() -> None:
    series = pl.Series("lineup", [[7, 8, 9]], dtype=pl.List(pl.Int64))
    assert _via_slots(series) == [[7, 8, 9]]


def test_agrees_with_to_list_on_committed_gold() -> None:
    """The equivalence that actually matters, on real data rather than fixtures."""
    from lineupiq.io.gold import load_all_gold
    from lineupiq.paths import DataPaths

    frame = load_all_gold(DataPaths.discover(), "shot_facts").head(5_000)
    for column in ("lineup_for", "lineup_against"):
        expected = [[int(p) for p in (row or [])] for row in frame[column].to_list()]
        assert _via_slots(frame[column]) == expected, column
