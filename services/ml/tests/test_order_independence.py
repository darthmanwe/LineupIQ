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


def test_parity_fixtures_reproduce(paths: DataPaths) -> None:
    """The gate CI runs, run here too.

    `lineupiq parity --check` regenerates both fixtures in memory and compares
    them: exactly for the hash-and-tier fixture, and to 1e-9 for the scorer's,
    because float64 results of logs and exponentials are not bit-portable
    between platforms. Requiring byte-identity of a float artefact fails on a
    platform change and passes on a rounding coincidence.
    """
    from lineupiq.serve.parity import check_fixtures

    drifts = check_fixtures(paths)
    assert not drifts, "\n".join(str(d) for d in drifts[:10])


def test_rank_order_breaks_ties_by_index() -> None:
    """Ties must resolve the same way on every machine.

    `np.argsort` defaults to an unstable quicksort, and reciprocal-rank fusion
    produces exact ties constantly -- any two documents holding the same pair of
    ranks across the two legs sum to the same fused score. An unstable sort left
    their order to the partitioning, which moved a published MRR from 0.950 to
    0.967 between two machines with Recall@10 identical to sixteen digits.
    """
    import numpy as np

    from lineupiq.retrieval.index import rank_order

    scores = np.array([0.5, 0.9, 0.5, 0.9, 0.1])
    order = rank_order(scores)
    # Best first, and among equals the lower index first.
    assert order.tolist() == [1, 3, 0, 2, 4]


def test_rank_order_is_immune_to_last_place_noise() -> None:
    """A one-bit difference must not decide an ordering.

    BM25 sums and cosine similarities are not bit-portable, so two documents
    whose real scores are equal can arrive differing in the last place. Rounding
    before comparing turns that back into the exact tie it should have been.
    """
    import numpy as np

    from lineupiq.retrieval.index import rank_order

    base = np.array([0.4, 0.4, 0.4])
    jittered = base + np.array([0.0, np.spacing(0.4), -np.spacing(0.4)])
    assert rank_order(base).tolist() == rank_order(jittered).tolist()


def test_fusion_is_immune_to_last_place_noise() -> None:
    import numpy as np

    from lineupiq.retrieval.index import reciprocal_rank_fusion

    lexical = np.array([3.0, 1.0, 2.0, 1.0])
    dense = np.array([0.2, 0.8, 0.2, 0.8])
    jitter = np.spacing(1.0)
    fused = reciprocal_rank_fusion([lexical, dense])
    nudged = reciprocal_rank_fusion([lexical + jitter, dense - jitter])
    assert np.allclose(fused, nudged, rtol=0, atol=1e-12)


def test_co_occurrence_report_is_order_independent(paths: DataPaths) -> None:
    """The RAPM diagnostic that was publishing an arbitrary fifty rows.

    `max_co_occurrence` is exactly 1.0 for every player who never took the floor
    without a particular teammate, so the top of the non-identified list is a
    solid block of ties. An unstable sort over those ties, plus an `argmax` over
    ratios that differ in the last place between platforms, meant the published
    list depended on the machine.
    """
    from lineupiq.models.rapm import build_rapm_design, co_occurrence_report, usable_possessions

    possessions = usable_possessions(load_all_gold(paths, "possession_facts"))

    reports = [
        co_occurrence_report(build_rapm_design(variant)) for variant in _permutations(possessions)
    ]
    keys = [
        [(row["player_id"], row["partner_id"]) for row in report["non_identified"]]
        for report in reports
    ]
    assert keys[0] == keys[1] == keys[2]
    # And the flagged count itself, which feeds the README.
    assert len({report["n_flagged"] for report in reports}) == 1


def test_cluster_robust_rate_errors_are_order_independent(paths: DataPaths) -> None:
    """The newest artefact of this shape, pinned before it can go wrong.

    These errors are a sum of squared per-game influences, taken over rows that
    come out of a `group_by`. Float addition is not associative, so an unordered
    aggregation is reproducible only on the machine that produced it -- which is
    the property every other entry in this file was written after discovering.

    The values are compared exactly rather than approximately. A tolerance here
    would pass on precisely the last-bit drift the test exists to catch, and the
    sums are over identical terms, so exact equality is what correct ordering
    actually produces.
    """
    from lineupiq.models.selection import (
        _THREE_ZONES,
        _cluster_robust_rate_errors,
    )

    shots = load_all_gold(paths, "shot_facts")
    results = [
        _cluster_robust_rate_errors(variant, _THREE_ZONES, 20) for variant in _permutations(shots)
    ]
    assert results[0] == results[1] == results[2]
    assert len(results[0]) > 0


def test_the_cluster_error_reduces_to_the_binomial_one_on_singleton_clusters() -> None:
    """The algebra, checked rather than asserted in a docstring.

    With one shot per game every cluster is a single Bernoulli draw, and the
    linearised sandwich collapses to `p(1-p)/(n-1)`. If it did not, the
    estimator would be measuring something other than what its name says, and no
    other test in the suite would notice -- a standard error that is wrong by a
    constant factor still looks exactly like a standard error.
    """
    import math

    import polars as pl

    from lineupiq.models.selection import _THREE_ZONES, _cluster_robust_rate_errors

    n, threes = 200, 60
    zones = ["top_three"] * threes + ["restricted_area"] * (n - threes)
    frame = pl.DataFrame(
        {
            "shooter_id": [7] * n,
            "game_id": [f"g{i:04d}" for i in range(n)],
            "zone_id": zones,
        }
    )
    observed = _cluster_robust_rate_errors(frame, _THREE_ZONES, 20)[7]
    rate = threes / n
    expected = math.sqrt(rate * (1.0 - rate) / (n - 1))
    assert observed == pytest.approx(expected, rel=1e-12)
