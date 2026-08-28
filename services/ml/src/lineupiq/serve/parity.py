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
    "build_plays_parity_fixture",
    "build_selection_parity_fixture",
    "check_fixtures",
    "write_parity_fixture",
    "write_plays_parity_fixture",
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

#: Ranking cases. Far fewer than the scoring cases, and deliberately so.
#:
#: Every ranking costs forty scorer calls for the gradient plus thirty-six
#: quadratic forms, and it exercises the *same* scorer the 507 scoring cases
#: already cover. What these add is the delta method, the difference test and
#: the banding, and a hundred-odd cases saturate those: the branch that has to
#: be hit is a ranking the data cannot order at all, and that one is included
#: by construction below rather than left to the sample.
N_PLAYS_CASES = 120


def _selection_requests(profiles: dict[str, Any]) -> list[Any]:
    """The counterfactuals both selection fixtures score.

    Shared rather than duplicated, so the ranking fixture cannot drift onto a
    different corpus from the scoring fixture and quietly stop covering the edge
    cases. The hand-built ones come first and in a fixed order, which is what lets
    the ranking fixture take a prefix and still get all of them.

    The sample is built to exercise the branches a random draw would miss:

    - a shooter with no profile at all, whose log ratio must be exactly zero
      rather than some arbitrary player's,
    - a lineup with only one real teammate, so the ``min`` in ``spacing_min``
      has nothing to hide behind,
    - the shooter listed among his own five, which must be excluded from his own
      spacing,
    - a team/season key that does not exist, which must fall back to the league,
    - no defenders at all, so every opponent term is exactly zero,
    - every context flag on and off, since three of them multiply zone
      indicators and a sign error in one is invisible when the flag is false.
    """
    from lineupiq.serve.score import ScoreRequest

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

    return requests


def build_selection_parity_fixture(paths: DataPaths) -> dict[str, Any]:
    """Score a sample of counterfactuals with the Python served scorer.

    The fixture stores **utilities**, not just the softmax output. A softmax is a
    contraction: two implementations that disagree in the fourth decimal of a
    utility can agree to 1e-9 on the resulting share, especially for the small
    zones. Comparing before the normalisation is the stronger check, so both are
    stored and both are asserted.

    The corpus is :func:`_selection_requests`, shared with the ranking fixture.
    """
    from lineupiq.serve.export import export_selection_model, export_selection_profiles
    from lineupiq.serve.score import score_selection

    profiles = export_selection_profiles(paths)
    model = export_selection_model(paths)
    if not model.get("available"):
        raise RuntimeError("no selection run log committed; run `lineupiq selection` first")

    requests = _selection_requests(profiles)
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


def build_plays_parity_fixture(paths: DataPaths) -> dict[str, Any]:
    """Rank a sample of counterfactuals, and store the intervals as well as the order.

    The order alone would be a weak fixture. Two implementations can produce the
    same ranked list from intervals that differ by a factor of two -- the ranking
    is a sequence of comparisons, and comparisons are robust to exactly the kind
    of error a variance calculation makes. So the standard errors are stored and
    asserted at 1e-9 alongside the ranks.

    The requests are the **same** ones the scoring fixture uses, taken from the
    front of that list so the hand-built edge cases are all included: the unseen
    shooter, the one-teammate lineup, the empty defence, the flags-all-on case.
    A ranking of a lineup whose scorer is untested would be testing two things at
    once.

    One case is chosen rather than sampled: the first request in the corpus whose
    zones the model cannot order at all. That branch is the point of the whole
    mechanism, it fires on about three per cent of requests, and a fixture that
    covers it only by luck is a fixture that stops covering it when the sample
    changes.
    """
    from lineupiq.serve.export import export_selection_model, export_selection_profiles
    from lineupiq.serve.plays import rank_plays

    profiles = export_selection_profiles(paths)
    model = export_selection_model(paths)
    if not model.get("available"):
        raise RuntimeError("no selection run log committed; run `lineupiq selection` first")
    if model.get("covariance") is None:
        raise RuntimeError(
            "the committed selection run log has no covariance matrix; "
            "re-run `lineupiq selection` and `lineupiq export`"
        )

    contract = model["ranking"]
    requests = _selection_requests(profiles)

    def rank(request: Any) -> Any:
        return rank_plays(
            request,
            profiles,
            model,
            confidence=contract["confidence"],
            critical_value=contract["critical_value"],
            min_zone_share=contract["min_zone_share"],
        )

    chosen = list(requests[:N_PLAYS_CASES])
    # Guarantee the unordered branch. Searching forward from the end of the
    # sample keeps the first N cases stable when this constant changes.
    if all(rank(request).ordered for request in chosen):
        for request in requests[N_PLAYS_CASES:]:
            if not rank(request).ordered:
                chosen.append(request)
                break

    cases: list[dict[str, Any]] = []
    for request in chosen:
        ranking = rank(request)
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
                "plays": [
                    {
                        "zone": play.zone,
                        "points_per_100": play.points_per_100,
                        "standard_error": play.standard_error,
                        "rank": play.rank,
                    }
                    for play in ranking.plays
                ],
                "bands": [list(band) for band in ranking.bands],
                "ordered": ranking.ordered,
                "excluded": list(ranking.excluded),
                "diagonal_would_refuse": ranking.diagonal_would_refuse,
                "pairs_compared": ranking.pairs_compared,
                "ties_spanning_bands": ranking.ties_spanning_bands,
            }
        )

    refused = sum(c["diagonal_would_refuse"] for c in cases)
    compared = sum(c["pairs_compared"] for c in cases)
    return {
        "seed": SEED,
        "zones": list(profiles["zones"]),
        "term_names": list(model["term_names"]),
        "ranking": contract,
        "n_cases": len(cases),
        # Summary counters, stored so a change in behaviour is one line of a diff
        # rather than a hundred. Each is a sum over the cases in this file, so
        # nothing here is a number the fixture cannot itself justify.
        "n_unordered": sum(1 for c in cases if not c["ordered"]),
        "n_bands_total": sum(len(c["bands"]) for c in cases),
        "diagonal_would_refuse": refused,
        "pairs_compared": compared,
        # What contiguity costs, summed. If this is zero the constraint is free
        # on this corpus; if it is not, the number is the honest size of what the
        # ranked-list rendering gives up.
        "ties_spanning_bands": sum(c["ties_spanning_bands"] for c in cases),
        "cases": cases,
    }


def write_plays_parity_fixture(paths: DataPaths) -> Path:
    fixture = build_plays_parity_fixture(paths)
    paths.parity.mkdir(parents=True, exist_ok=True)
    path = paths.parity / "plays.json"
    path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path


#: How many random lineup pairs the comparison fixture scores.
#:
#: Smaller than the scoring corpus because each case runs a gradient sweep over
#: twenty coefficients *and* one per distinct player rate -- roughly two hundred
#: scorer calls against a scoring case's one. Ninety is enough to cover the
#: branches; the hand-built pairs below cover the ones that matter.
N_COMPARE_CASES = 90


def _compare_pairs(profiles: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """The lineup pairs the comparison fixture scores, as ``(label, left, right)``.

    ``right=None`` is the league-average arm, and it is not a special case bolted
    on: it is the quantity ``/lineups/score`` already publishes per zone, routed
    through the same function so the two cannot drift.

    The hand-built pairs come first and in a fixed order, each one a branch a
    random draw would not reliably hit:

    - the league-average arm itself,
    - a lineup against itself, which must return exactly zero with exactly zero
      variance and a degenerate omnibus -- the placebo identity,
    - a one-player swap, which is the shape the product actually serves,
    - a swap that changes *which* teammate is the worst spacer, so the ``min`` in
      ``spacing_min`` moves its argmin rather than only its value,
    - a change of defence only, so both offensive terms are exactly zero and the
      two opponent terms carry the whole difference,
    - two entirely different fives, where every lineup term moves at once,
    - a swap bringing in a player with no fitted rate, which must refuse rather
      than return the zero the league fallback would produce.

    Only players carrying both a rate and a standard error are drawn, because
    those are the only ones the endpoint will score at all.
    """
    from lineupiq.serve.score import ScoreRequest

    rated = sorted(
        int(k)
        for k in profiles["player_three_rate"]
        if k in profiles["player_three_rate_se"] and k in profiles["player_rim_rate_se"]
    )
    defenders = sorted(
        int(k) for k in profiles["opp_three_allowed"] if k in profiles["opp_three_allowed_se"]
    )
    unrated = sorted(
        int(k) for k in profiles["shooter_log_ratio"] if k not in profiles["player_three_rate"]
    )
    rng = np.random.default_rng(SEED)

    shooter = rated[0]
    base = (shooter, *rated[1:5])
    defence = tuple(defenders[:5])

    # The teammate in `base` with the lowest three-point rate, and a replacement
    # below him. Swapping this one moves the argmin of `spacing_min` rather than
    # only its value, which is the branch where the finite difference is taken
    # across the kink of a `min`.
    rates = {p: float(profiles["player_three_rate"][str(p)]) for p in base[1:]}
    worst = min(rates, key=lambda p: (rates[p], p))
    lower = [p for p in rated if float(profiles["player_three_rate"][str(p)]) < rates[worst]]
    argmin_mover = lower[0] if lower else rated[6]

    pairs: list[tuple[str, Any, Any]] = [
        ("league_average", ScoreRequest(shooter, base, defence), None),
        ("identity", ScoreRequest(shooter, base, defence), ScoreRequest(shooter, base, defence)),
        (
            "one_player_swap",
            ScoreRequest(shooter, base, defence),
            ScoreRequest(shooter, (*base[:4], rated[7]), defence),
        ),
        (
            "argmin_moves",
            ScoreRequest(shooter, base, defence),
            ScoreRequest(shooter, (*(p for p in base if p != worst), argmin_mover), defence),
        ),
        (
            "defence_only",
            ScoreRequest(shooter, base, defence),
            ScoreRequest(shooter, base, tuple(defenders[5:10])),
        ),
        (
            "disjoint_fives",
            ScoreRequest(shooter, base, defence),
            ScoreRequest(shooter, (shooter, *rated[20:24]), defence),
        ),
        (
            "no_defence",
            ScoreRequest(shooter, base, ()),
            ScoreRequest(shooter, (*base[:4], rated[9]), ()),
        ),
        (
            "unprofiled_swap_in",
            ScoreRequest(shooter, base, defence),
            ScoreRequest(shooter, (*base[:4], unrated[0]), defence),
        ),
    ]

    for _ in range(N_COMPARE_CASES):
        picked = [int(rated[i]) for i in rng.choice(len(rated), size=9, replace=False)]
        common = tuple(picked[:4])
        opponents = tuple(
            int(defenders[i]) for i in rng.choice(len(defenders), size=5, replace=False)
        )
        pairs.append(
            (
                "random",
                ScoreRequest(common[0], (*common, picked[4]), opponents),
                ScoreRequest(common[0], (*common, picked[5]), opponents),
            )
        )
    return pairs


def build_compare_parity_fixture(paths: DataPaths) -> dict[str, Any]:
    """Compare a sample of lineup pairs, storing both variance components.

    The two components are stored **separately** rather than only their sum, and
    that is the point of the fixture rather than a detail of it. A TypeScript
    mirror that dropped the profile term entirely would still agree on every
    delta, every share and every rank; it would disagree only on the interval
    widths, and only by the third or so that the profile term is worth. Storing
    the split makes such a mirror fail on the component instead of passing on a
    plausible-looking total.

    Refusals are stored too, with the players they named. A mirror that silently
    scored an unprofiled player would otherwise look correct, because the number
    it produced would be exactly zero -- which is also what a real "this swap
    changes nothing" answer looks like.
    """
    from lineupiq.serve.compare import UnprofiledPlayerError, compare_lineups
    from lineupiq.serve.export import export_selection_model, export_selection_profiles

    profiles = export_selection_profiles(paths)
    model = export_selection_model(paths)
    if not model.get("available"):
        raise RuntimeError("no selection run log committed; run `lineupiq selection` first")
    if model.get("covariance") is None:
        raise RuntimeError(
            "the committed selection run log has no covariance matrix; "
            "re-run `lineupiq selection` and `lineupiq export`"
        )

    # Read from the exported model rather than rebuilt from the thresholds, so
    # the fixture pins what the Worker will actually be handed.
    exported = model["comparison"]
    comparison_contract = {
        "confidence": exported["confidence"],
        "critical_value": exported["critical_value"],
        "omnibus_critical_value": exported["omnibus_critical_value"],
    }

    def _serialise(request: Any) -> dict[str, Any] | None:
        if request is None:
            return None
        return {
            "shooter_id": request.shooter_id,
            "offense": list(request.offense),
            "defense": list(request.defense),
            "team_id": request.team_id,
            "season": request.season,
            "seconds_into_possession": request.seconds_into_possession,
            "live_ball": request.live_ball,
            "second_chance": request.second_chance,
            "clutch": request.clutch,
        }

    cases: list[dict[str, Any]] = []
    for label, left, right in _compare_pairs(profiles):
        case: dict[str, Any] = {
            "label": label,
            "left": _serialise(left),
            "right": _serialise(right),
        }
        try:
            result = compare_lineups(left, right, profiles, model, **comparison_contract)
        except UnprofiledPlayerError as refused:
            case["unprofiled"] = list(refused.players)
            cases.append(case)
            continue
        case["unprofiled"] = None
        case["omnibus"] = {
            "statistic": result.omnibus.statistic,
            "degrees_of_freedom": result.omnibus.degrees_of_freedom,
            "distinguishable": result.omnibus.distinguishable,
            "degenerate": result.omnibus.degenerate,
            "rim_shift": result.omnibus.rim_shift,
            "three_shift": result.omnibus.three_shift,
            "rim_shift_error": result.omnibus.rim_shift_error,
            "three_shift_error": result.omnibus.three_shift_error,
        }
        case["profile_variance_share"] = result.profile_variance_share
        case["argmin_unstable"] = result.argmin_unstable
        case["zones"] = [
            {
                "zone": zone.zone,
                "delta_share": zone.delta_share,
                "points_per_100": zone.points_per_100,
                "standard_error": zone.standard_error,
                "variance_coefficients": zone.variance_coefficients,
                "variance_profiles": zone.variance_profiles,
            }
            for zone in result.zones
        ]
        case["mechanism"] = [
            {
                "term": term.term,
                "feature_delta": term.feature_delta,
                "coefficient": term.coefficient,
                "verdict": term.verdict,
            }
            for term in result.mechanism
        ]
        cases.append(case)

    scored = [c for c in cases if c["unprofiled"] is None]
    shares = [c["profile_variance_share"] for c in scored]
    return {
        "seed": SEED,
        "zones": list(profiles["zones"]),
        "term_names": list(model["term_names"]),
        "comparison": comparison_contract,
        "n_cases": len(cases),
        "n_refused": len(cases) - len(scored),
        "n_degenerate": sum(1 for c in scored if c["omnibus"]["degenerate"]),
        "n_distinguishable": sum(1 for c in scored if c["omnibus"]["distinguishable"]),
        # The headline of the whole feature, summarised so a change in it shows
        # up as one line of a diff: how much of a comparison's uncertainty comes
        # from the two players' own shooting rates rather than from the model.
        "mean_profile_variance_share": (sum(shares) / len(shares)) if shares else 0.0,
        "cases": cases,
    }


def write_compare_parity_fixture(paths: DataPaths) -> Path:
    fixture = build_compare_parity_fixture(paths)
    paths.parity.mkdir(parents=True, exist_ok=True)
    path = paths.parity / "compare.json"
    path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def check_fixtures(paths: DataPaths) -> list[Drift]:
    """Regenerate every fixture and report each real disagreement.

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
        ("plays.json", build_plays_parity_fixture, FLOAT_TOLERANCE),
        ("compare.json", build_compare_parity_fixture, FLOAT_TOLERANCE),
    ):
        path = paths.parity / name
        if not path.exists():
            drifts.append(Drift(name, "", "committed", "missing"))
            continue
        committed = json.loads(path.read_text(encoding="utf-8"))
        fresh = json.loads(json.dumps(builder(paths)))
        compare_artefacts(name, committed, fresh, tolerance=tolerance, drifts=drifts)

    return drifts
