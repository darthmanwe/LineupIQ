"""Derived artefacts must not depend on the order rows arrive in.

Three separate bugs of this shape shipped before any of them was noticed, and
they were only found by running the same code on a machine with a different core
count:

1. `leave_lineup_out` permuted a list that came out of a parallel `group_by`, so
   fold membership varied -- see `test_split_determinism.py`.
2. The parity fixture's real-lineup sample sorted on a tied integer duration, so
   30 of 2,604 cases swapped.
3. The retrieval corpus sorted on a tied possession count, so the groundedness
   harness scored a different 200 documents and published a different rate.

The common cause is a sort whose key has ties, over a `group_by` that makes no
ordering promise. Nothing raises; the numbers are all individually plausible.

These tests use row order as the observable proxy. Permuting the input is a
deterministic way to reproduce what a different thread count does to a parallel
aggregation, and any function whose output survives permutation is safe from it.
"""

from __future__ import annotations

import polars as pl
import pytest

from lineupiq.io.gold import load_all_gold
from lineupiq.paths import DataPaths


@pytest.fixture(scope="module")
def paths() -> DataPaths:
    return DataPaths.discover()


def _permutations(frame: pl.DataFrame) -> list[pl.DataFrame]:
    """The same rows, three ways. Reversal is the harshest of the three."""
    return [
        frame,
        frame.reverse(),
        frame.sample(fraction=1.0, shuffle=True, seed=11),
    ]


def test_lineup_support_order_is_stable(paths: DataPaths) -> None:
    from lineupiq.models.support import build_lineup_support

    stints = load_all_gold(paths, "stints")
    shots = load_all_gold(paths, "shot_facts")

    orders = [
        build_lineup_support(variant, shots)["lineup_hash"].to_list()[:200]
        for variant in _permutations(stints)
    ]
    assert orders[0] == orders[1] == orders[2]


def test_retrieval_corpus_order_is_stable(paths: DataPaths) -> None:
    """The one that actually moved a published number."""
    from lineupiq.retrieval.docs import build_documents

    possessions = load_all_gold(paths, "possession_facts")
    shots = load_all_gold(paths, "shot_facts")
    players = load_all_gold(paths, "dim_player")

    orders = []
    for variant in _permutations(possessions):
        docs = build_documents(shots, variant, players)
        orders.append([doc.lineup_hash for doc in docs[:200]])
    assert orders[0] == orders[1] == orders[2]


def test_dim_player_survives_deduplication_identically(paths: DataPaths) -> None:
    """Which of a player's rows wins must not be a race.

    A player who changed teams mid-season has rows that differ, so an
    order-dependent `unique(keep="first")` can export a different name for the
    same id from one run to the next.
    """
    frame = load_all_gold(paths, "dim_player")
    assert frame["player_id"].n_unique() == frame.height, "dim_player has duplicate ids"

    names = [
        variant.unique(subset=["player_id"], keep="first", maintain_order=True)
        .sort("player_id")["player_name"]
        .to_list()
        for variant in _permutations(frame)
    ]
    assert names[0] == names[1] == names[2]
