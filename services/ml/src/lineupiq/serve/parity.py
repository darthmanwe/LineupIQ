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
from lineupiq.validate.reproduce import Drift, compare_artefacts

__all__ = [
    "FLOAT_TOLERANCE",
    "build_parity_fixture",
    "build_selection_parity_fixture",
    "check_fixtures",
    "write_parity_fixture",
    "write_selection_parity_fixture",
]

#: How far a regenerated float may move before it counts as a real change.
#:
#: The two fixtures are compared differently on purpose.
#:
#: `lineups.json` holds integers, canonical id strings and MD5 digests. Those are
#: exact quantities: a single differing bit is a bug, and it is compared byte for
#: byte.
#:
#: `selection.json` holds utilities and softmax outputs -- float64 results of
#: logs, means and exponentials. Those are *not* bit-portable. The same source
#: and the same library versions on Linux and on Windows differ in the last
#: place, because the underlying BLAS and libm do. Requiring byte-identity of a
#: float artefact means the gate fails on a platform change and passes on a
#: rounding coincidence, which is the wrong way round.
#:
#: 1e-9 is three orders of magnitude above the noise and three orders below any
#: difference that could come from a changed model. It is also the tolerance the
#: TypeScript parity suite asserts, so the two agree about what "the same" means.
FLOAT_TOLERANCE = 1e-9

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
        # **The tiebreaker is load-bearing.** `seconds` ties constantly -- stint
        # durations are whole seconds and there are 49,827 groups -- and
        # `group_by` makes no ordering promise, so ties were resolved by whatever
        # order the parallel aggregation happened to produce. Regenerating the
        # fixture on a machine with a different core count swapped adjacent
        # equal-duration lineups, which changed 30 of 2,604 cases and failed the
        # parity gate for a reason that had nothing to do with parity.
        #
        # Sorting on the canonical id list after `seconds` makes the order a
        # function of the data alone. The list is already sorted numerically, so
        # comparing lists compares them elementwise and is total.
        .sort(["seconds", "lineup"], descending=[True, False])
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


#: How many random (shooter, offence, defence, context) draws to score.
N_SELECTION_CASES = 500


def build_selection_parity_fixture(paths: DataPaths) -> dict[str, Any]:
    """Score a sample of counterfactuals with the Python served scorer.

    The fixture stores **utilities**, not just the softmax output. A softmax is a
    contraction: two implementations that disagree in the fourth decimal of a
    utility can agree to 1e-9 on the resulting share, especially for the small
    zones. Comparing before the normalisation is the stronger check, so both are
    stored and both are asserted.

    The sample is built to exercise the branches that a random draw would miss:

    - a shooter with no profile at all, whose log ratio must be exactly zero
      rather than some arbitrary player's,
    - a lineup with only one real teammate, so the ``min`` in ``spacing_min``
      has nothing to hide behind,
    - the shooter listed among his own five, which must be excluded from his own
      spacing,
    - a team/season key that does not exist, which must fall back to the league,
    - every context flag on and off, since three of them multiply zone
      indicators and a sign error in one is invisible when the flag is false.
    """
    from lineupiq.serve.export import export_selection_model, export_selection_profiles
    from lineupiq.serve.score import ScoreRequest, score_selection

    profiles = export_selection_profiles(paths)
    model = export_selection_model(paths)
    if not model.get("available"):
        raise RuntimeError("no selection run log committed; run `lineupiq selection` first")

    known = sorted(int(k) for k in profiles["shooter_log_ratio"])
    team_keys = sorted(profiles["team_log_ratio"])
    rng = np.random.default_rng(SEED)

    requests: list[ScoreRequest] = [
        # An unseen shooter. `0` is not a valid NBA player id, which is the point.
        ScoreRequest(0, tuple(known[:5]), tuple(known[5:10])),
        # One teammate. `spacing` and `spacing_min` must coincide here, and a
        # test that only ever sees five-man lineups cannot check that.
        ScoreRequest(known[0], (known[0], known[1]), tuple(known[5:10])),
        # The shooter appears in his own lineup and must not space himself.
        ScoreRequest(known[0], tuple(known[:5]), tuple(known[5:10])),
        # A team/season that never existed.
        ScoreRequest(known[0], tuple(known[:5]), tuple(known[5:10]), team_id=1, season=1999),
        # No defenders at all: every opponent term must be exactly zero.
        ScoreRequest(known[0], tuple(known[:5]), ()),
        # Every context flag on, at a fast-break clock.
        ScoreRequest(
            known[1],
            tuple(known[:5]),
            tuple(known[5:10]),
            seconds_into_possession=0.0,
            live_ball=True,
            second_chance=True,
            clutch=True,
        ),
        # A late-clock possession, which drives `seconds_z` strongly positive.
        ScoreRequest(known[1], tuple(known[:5]), tuple(known[5:10]), seconds_into_possession=24.0),
    ]

    for _ in range(N_SELECTION_CASES):
        offence = [int(known[i]) for i in rng.choice(len(known), size=5, replace=False)]
        defence = [int(known[i]) for i in rng.choice(len(known), size=5, replace=False)]
        team_key = team_keys[int(rng.integers(len(team_keys)))]
        team_id, season = team_key.split(":")
        requests.append(
            ScoreRequest(
                shooter_id=offence[0],
                offense=tuple(offence),
                defense=tuple(defence),
                team_id=int(team_id),
                season=2000 + int(season),
                seconds_into_possession=float(rng.uniform(0.0, 24.0)),
                live_ball=bool(rng.integers(2)),
                second_chance=bool(rng.integers(2)),
                clutch=bool(rng.integers(2)),
            )
        )

    cases: list[dict[str, Any]] = []
    for request in requests:
        result = score_selection(request, profiles, model["coefficients"], model["term_names"])
        cases.append(
            {
                "request": {
                    "shooter_id": request.shooter_id,
                    "offense": list(request.offense),
                    "defense": list(request.defense),
                    "team_id": request.team_id,
                    "season": request.season,
                    "seconds_into_possession": request.seconds_into_possession,
                    "live_ball": request.live_ball,
                    "second_chance": request.second_chance,
                    "clutch": request.clutch,
                },
                "utilities": list(result.utilities),
                "mix": list(result.mix),
                "baseline_mix": list(result.baseline_mix),
                "shooter_known": result.shooter_known,
                "shooter_weight": result.shooter_weight,
                "points_per_100": result.points_per_100,
            }
        )

    return {
        "seed": SEED,
        "zones": list(profiles["zones"]),
        "term_names": list(model["term_names"]),
        "n_cases": len(cases),
        "n_unknown_shooters": sum(1 for c in cases if not c["shooter_known"]),
        "cases": cases,
    }


def write_selection_parity_fixture(paths: DataPaths) -> Path:
    fixture = build_selection_parity_fixture(paths)
    paths.parity.mkdir(parents=True, exist_ok=True)
    path = paths.parity / "selection.json"
    path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def check_fixtures(paths: DataPaths) -> list[Drift]:
    """Regenerate both fixtures and report every real disagreement.

    This replaces a `git diff` on the committed files, and the reason is in
    :data:`FLOAT_TOLERANCE`: one fixture is exact and the other is floating
    point, so one comparison cannot serve both. Structure is still compared
    exactly in both -- a missing key, a changed tier, a different player id or a
    list that grew is a difference at any tolerance.
    """
    drifts: list[Drift] = []

    for name, builder, tolerance in (
        ("lineups.json", build_parity_fixture, 0.0),
        ("selection.json", build_selection_parity_fixture, FLOAT_TOLERANCE),
    ):
        path = paths.parity / name
        if not path.exists():
            drifts.append(Drift(name, "", "committed", "missing"))
            continue
        committed = json.loads(path.read_text(encoding="utf-8"))
        fresh = json.loads(json.dumps(builder(paths)))
        compare_artefacts(name, committed, fresh, tolerance=tolerance, drifts=drifts)

    return drifts
