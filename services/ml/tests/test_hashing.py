"""Lineup hashing, including cross-engine agreement with DuckDB.

The failure this guards against is not an exception. If Python and the query
engine order player ids differently, every join on ``lineup_hash`` returns zero
rows and the app reports "these five have never played together" for lineups
that played hundreds of possessions.
"""

from __future__ import annotations

import hashlib

import duckdb
import pytest

from lineupiq.hashing import canonical_lineup, lineup_hash

# A real 2023-24 Miami lineup. Deliberately mixes 6-digit (older) and 7-digit
# (newer) player ids -- that mix is what makes the sort order observable.
MIAMI = [1628389, 202710, 1629639, 1628974, 101108]

# Every id here is 7 digits, so numeric and lexicographic order coincide. Used
# to show the parity test would pass vacuously on a badly chosen fixture.
ALL_SEVEN_DIGIT = [1628389, 1629639, 1628974, 1630170, 1631094]


def _duckdb_hash(player_ids: list[int]) -> str:
    """The warehouse-side expression, evaluated for real.

    Mirrors Snowflake's ``MD5(ARRAY_TO_STRING(ARRAY_SORT(ids), ','))``. The ids
    are bound as BIGINT so ``list_sort`` orders them numerically -- which is the
    entire contract being tested.
    """
    con = duckdb.connect()
    try:
        (result,) = con.execute(
            "SELECT md5(array_to_string(list_sort(?::BIGINT[]), ','))", [player_ids]
        ).fetchone()
        return str(result)
    finally:
        con.close()


class TestCanonicalLineup:
    def test_sorts_numerically(self) -> None:
        assert canonical_lineup(MIAMI) == (101108, 202710, 1628389, 1628974, 1629639)

    def test_accepts_numeric_strings(self) -> None:
        assert canonical_lineup([str(p) for p in MIAMI]) == canonical_lineup(MIAMI)

    @pytest.mark.parametrize(
        ("ids", "match"),
        [
            ([1, 2, 3, 4], "exactly 5"),
            ([1, 2, 3, 4, 5, 6], "exactly 5"),
            ([1, 1, 2, 3, 4], "duplicate"),
            ([0, 1, 2, 3, 4], "positive"),
            ([-1, 1, 2, 3, 4], "positive"),
        ],
    )
    def test_refuses_malformed_lineups(self, ids: list[int], match: str) -> None:
        # Refused, never repaired. A four-man lineup is a bug upstream, and
        # padding it would bury that bug inside a meaningless hash.
        with pytest.raises(ValueError, match=match):
            canonical_lineup(ids)

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError, match="integer-valued"):
            canonical_lineup(["bam", 202710, 1629639, 1628974, 101108])


class TestOrderInvariance:
    def test_permutations_agree(self) -> None:
        import itertools

        hashes = {lineup_hash(p) for p in itertools.permutations(MIAMI)}
        assert len(hashes) == 1, "hash must not depend on input order"

    def test_reversal_agrees(self) -> None:
        assert lineup_hash(MIAMI) == lineup_hash(list(reversed(MIAMI)))

    def test_different_lineups_differ(self) -> None:
        other = [*MIAMI[:4], 1630170]
        assert lineup_hash(MIAMI) != lineup_hash(other)


class TestNumericVersusLexicographicSort:
    """The specific bug the docstring in `hashing.py` warns about."""

    def test_lexicographic_ordering_would_produce_a_different_hash(self) -> None:
        numeric = ",".join(str(p) for p in sorted(MIAMI))
        lexicographic = ",".join(sorted(str(p) for p in MIAMI))

        assert numeric != lexicographic, (
            "fixture is useless if both orderings coincide -- pick ids with differing digit counts"
        )
        assert lineup_hash(MIAMI) != hashlib.md5(lexicographic.encode()).hexdigest()

    def test_fixture_actually_exercises_the_difference(self) -> None:
        # 101108 sorts first numerically; '1011...' also sorts first as a string
        # here, but 202710 vs 1628389 is where they diverge.
        assert sorted(MIAMI) != [int(s) for s in sorted(str(p) for p in MIAMI)]


class TestDuckDbParity:
    """Python and the query engine must agree byte for byte."""

    def test_agrees_on_mixed_width_ids(self) -> None:
        assert lineup_hash(MIAMI) == _duckdb_hash(MIAMI)

    def test_agrees_on_uniform_width_ids(self) -> None:
        assert lineup_hash(ALL_SEVEN_DIGIT) == _duckdb_hash(ALL_SEVEN_DIGIT)

    def test_agrees_under_permutation(self) -> None:
        shuffled = [202710, 101108, 1629639, 1628389, 1628974]
        assert lineup_hash(shuffled) == _duckdb_hash(shuffled)

    def test_agrees_across_many_synthetic_lineups(self) -> None:
        # Deterministic pseudo-random ids spanning the 6-to-7-digit boundary,
        # which is exactly where the two sort orders disagree.
        import random

        rng = random.Random(20260815)
        for _ in range(200):
            ids = rng.sample(range(100_000, 1_700_000), 5)
            assert lineup_hash(ids) == _duckdb_hash(ids), f"disagreement on {ids}"


class TestStability:
    def test_hash_is_pinned(self) -> None:
        # A golden value. If canonicalisation ever changes, every committed
        # lineup_hash in gold is invalidated -- this test is the tripwire.
        assert (
            lineup_hash(MIAMI) == hashlib.md5(b"101108,202710,1628389,1628974,1629639").hexdigest()
        )

    def test_is_32_hex_characters(self) -> None:
        h = lineup_hash(MIAMI)
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)
