"""Python/TypeScript parity fixtures.

The Worker re-implements two things: the lineup hash and the support tier. Both
are decisions, not approximations, and a disagreement between the two languages
does not raise anywhere -- it silently returns the wrong answer, or zero rows.

So the Python side computes both for a fixed sample, writes the answers to a
committed fixture, and a vitest suite running inside ``workerd`` asserts the
TypeScript produces the same ones. The fixture is the contract; neither
implementation is allowed to be the reference by convention.

The sample deliberately includes the cases that break things:

- lineups whose ids sort differently numerically and lexicographically,
- lineups at each side of every threshold boundary,
- a counterfactual lineup that has never played,
- a lineup containing a player with almost no attempts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lineupiq.config import SEED
from lineupiq.hashing import lineup_hash
from lineupiq.models.support import Tier, assess, build_lineup_support, load_thresholds
from lineupiq.paths import DataPaths

__all__ = ["build_parity_fixture", "write_parity_fixture"]

#: How many random lineups to include beyond the hand-picked edge cases.
N_RANDOM = 2_000


def _lexicographic_disagreement_cases() -> list[list[int]]:
    """Lineups where a lexicographic sort would produce a different hash.

    This is the failure mode most likely to survive review: every engine agrees
    on the digest, and they disagree on what to feed it. A seven-digit id and a
    six-digit id starting with a larger first character is all it takes.
    """
    return [
        [1630552, 201143, 2544, 203999, 1629029],
        [201939, 1628369, 203507, 1629027, 202695],
        [2544, 3, 22, 111, 1000000],
        [999999, 1000000, 1000001, 100000, 10000],
    ]


def build_parity_fixture(paths: DataPaths) -> dict[str, Any]:
    """Compute every fixture case with the Python implementation."""
    from lineupiq.io.gold import load_all_gold

    thresholds = load_thresholds()
    support_table = build_lineup_support(
        load_all_gold(paths, "stints"), load_all_gold(paths, "shot_facts")
    )
    shots = load_all_gold(paths, "shot_facts")
    attempts = {
        int(pid): int(n)
        for pid, n in shots.group_by("shooter_id").agg(pl.len().alias("n")).iter_rows()
    }

    lookup: dict[str, tuple[int, int]] = {
        row["lineup_hash"]: (int(row["possessions"]), int(row["min_player_attempts"]))
        for row in support_table.iter_rows(named=True)
    }

    rng = np.random.default_rng(SEED)
    players = sorted(attempts)
    cases: list[list[int]] = list(_lexicographic_disagreement_cases())

    # Real lineups first, taken from the stints themselves and ordered by time
    # played. Random five-player draws are the natural way to build a fixture
    # and they produce **zero** reportable cases -- out of 49,827 observed
    # groups only 485 clear the 200-possession floor, so a random combination
    # essentially never is one. A parity fixture that never exercises the
    # reportable branch cannot prove that branch agrees, which is why the
    # generator warns when a tier comes back empty.
    stints = load_all_gold(paths, "stints")
    observed = (
        pl.concat(
            [
                stints.filter(pl.col(col).is_not_null() & (pl.col(col).list.len() == 5)).select(
                    pl.col(col).alias("lineup"), "duration_seconds"
                )
                for col in ("home_lineup", "away_lineup")
            ]
        )
        .with_columns(pl.col("lineup").list.sort().alias("lineup"))
        .group_by("lineup")
        .agg(pl.col("duration_seconds").sum().alias("seconds"))
        .sort("seconds", descending=True)
        .head(600)
    )
    cases.extend([int(p) for p in row] for row in observed["lineup"].to_list())

    for _ in range(N_RANDOM):
        cases.append([int(p) for p in rng.choice(players, 5, replace=False)])

    entries: list[dict[str, Any]] = []
    tier_counts: dict[str, int] = {tier.value: 0 for tier in Tier}
    for ids in cases:
        if len(set(ids)) != 5:
            continue
        result = assess(ids, lookup, thresholds, attempts)
        tier_counts[result.tier.value] += 1
        entries.append(
            {
                "players": ids,
                "canonical": ",".join(str(i) for i in sorted(ids)),
                "lineup_hash": lineup_hash(ids),
                "possessions": result.possessions,
                "min_player_attempts": result.min_player_attempts,
                "tier": result.tier.value,
                "counterfactual": result.counterfactual,
            }
        )

    return {
        "generated_by": "lineupiq parity",
        "seed": SEED,
        "thresholds": {
            "reportable_possessions": thresholds.reportable_possessions,
            "reportable_attempts": thresholds.reportable_attempts,
            "directional_possessions": thresholds.directional_possessions,
            "directional_attempts": thresholds.directional_attempts,
        },
        "tier_counts": tier_counts,
        "n_cases": len(entries),
        "cases": entries,
    }


def write_parity_fixture(paths: DataPaths) -> Path:
    fixture = build_parity_fixture(paths)
    paths.parity.mkdir(parents=True, exist_ok=True)
    path = paths.parity / "lineups.json"
    path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path
