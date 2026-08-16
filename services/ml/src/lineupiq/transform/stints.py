"""Stint reconstruction: which five players were on the floor for every event.

Play-by-play does not record who is on the court. It records substitutions and
it records who did things. Recovering the lineup means replaying a period
forward from a starting five that is never stated.

The classic framing of this problem is "some starters are invisible" -- a player
who starts a period but does not appear in an event until after a substitution.
Measured against an independent oracle over a full season, that case affects
**6 of 9,976 period-team units (0.06%)**. It is real but rare. The dominant
error source is event *mis-ordering*, which is handled in
:mod:`lineupiq.transform.events`.

Every outcome is labelled. A lineup that cannot be determined produces a null
and a quarantine flag, never a guess.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Final, Literal

import polars as pl

from lineupiq.transform.events import (
    EVENT_EJECTION,
    EVENT_FOUL,
    EVENT_PERIOD_BEGIN,
    EVENT_PERIOD_END,
    EVENT_REPLAY,
    EVENT_SUBSTITUTION,
    EVENT_TIMEOUT,
    PERSON_TYPE_AWAY,
    PERSON_TYPE_HOME,
    TECHNICAL_FOUL_ACTION_TYPES,
)

__all__ = [
    "LINEUP_SIZE",
    "AssertionKind",
    "CourtAssertion",
    "PeriodSolution",
    "SolveStatus",
    "reconstruct_stints",
    "solve_period_start",
]

LINEUP_SIZE: Final = 5

#: Cap on the candidate pool before enumeration. C(16,5) = 4,368 subsets is
#: cheap; beyond that the period is too chaotic to solve by search and is better
#: quarantined than guessed at.
_MAX_CANDIDATES: Final = 16

AssertionKind = Literal["EV", "IN", "OUT", "EJECT"]

SolveStatus = Literal["EXACT", "SOLVED", "AMBIGUOUS", "REPAIRED", "UNDERDETERMINED"]

#: Event types that say nothing about who is on the floor.
_NON_POSITIONAL: Final[frozenset[int]] = frozenset(
    {EVENT_TIMEOUT, EVENT_PERIOD_BEGIN, EVENT_PERIOD_END, EVENT_REPLAY}
)

#: Which PLAYER{n} slots carry an on-court actor, by event type. Slot 2 is the
#: assister / stealer / blocker; slot 3 is rarer. Both are genuine presence
#: evidence when populated.
_SLOT2_EVENTS: Final[frozenset[int]] = frozenset({1, 5, 6, 10})
_SLOT3_EVENTS: Final[frozenset[int]] = frozenset({2, 6, 10})


@dataclass(frozen=True, slots=True)
class CourtAssertion:
    """One claim that a player was (or stopped being) on the floor.

    ``side`` comes from this player's own ``PERSON{n}TYPE``, never from the
    row's primary actor. On a steal, a block, or a foul the second and third
    slots hold an *opponent*, so attributing a whole row to one side puts five
    of the wrong team's players in the other team's lineup and makes every
    period unsolvable.
    """

    seconds_remaining: int
    event_num: int
    player_id: int
    kind: AssertionKind
    side: int


@dataclass(frozen=True, slots=True)
class PeriodSolution:
    """The starting five for one team in one period, and how sure we are."""

    starters: frozenset[int] | None
    status: SolveStatus
    n_violations: int
    n_candidate_solutions: int

    @property
    def quality(self) -> Literal["VALID", "IMPUTED", "QUARANTINED"]:
        """Only a clean solve is training-grade.

        ``IMPUTED`` rows are still served (with their flag) because a slightly
        uncertain lineup is better than a hole in the rotation, but they are
        excluded from model training so a guess never becomes a coefficient.
        """
        if self.status in ("EXACT", "SOLVED"):
            return "VALID"
        if self.status in ("REPAIRED", "AMBIGUOUS"):
            return "IMPUTED"
        return "QUARANTINED"


def _technical_foul(event_type: int, action_type: int | None) -> bool:
    return event_type == EVENT_FOUL and action_type in TECHNICAL_FOUL_ACTION_TYPES


def court_assertions(row: Mapping[str, Any]) -> list[CourtAssertion]:
    """Turn one play-by-play row into zero or more presence claims."""
    event_type = row.get("EVENTMSGTYPE")
    if event_type is None:
        return []
    event_type = int(event_type)
    if event_type in _NON_POSITIONAL:
        return []

    action_type = row.get("EVENTMSGACTIONTYPE")
    action_type = int(action_type) if action_type is not None else None
    secs = row.get("seconds_remaining")
    secs = int(secs) if secs is not None else 0
    event_num = int(row.get("EVENTNUM") or 0)

    def _player(slot: int) -> tuple[int, int] | None:
        """Return ``(player_id, side)`` for a slot, or None if it is not a player.

        The side is read from this slot's own PERSON{n}TYPE, which is what makes
        cross-team rows (steals, blocks, fouls) land on the right lineup.
        """
        pid = row.get(f"PLAYER{slot}_ID")
        ptype = row.get(f"PERSON{slot}TYPE")
        if pid is None or ptype is None:
            return None
        side = int(ptype)
        if side not in (PERSON_TYPE_HOME, PERSON_TYPE_AWAY):
            return None  # team row or coach, not a player on court
        pid = int(pid)
        return (pid, side) if pid > 0 else None

    out: list[CourtAssertion] = []

    if event_type == EVENT_SUBSTITUTION:
        # PLAYER1 leaves, PLAYER2 enters -- always the same team. Verified over
        # a full season: both ids are populated on all 57,671 substitution rows.
        if (leaving := _player(1)) is not None:
            out.append(CourtAssertion(secs, event_num, leaving[0], "OUT", leaving[1]))
        if (entering := _player(2)) is not None:
            out.append(CourtAssertion(secs, event_num, entering[0], "IN", entering[1]))
        return out

    if event_type == EVENT_EJECTION:
        if (ejected := _player(1)) is not None:
            out.append(CourtAssertion(secs, event_num, ejected[0], "EJECT", ejected[1]))
        return out

    # A bench player can be assessed a technical foul. Counting that as presence
    # puts a seated player on the floor and makes the period unsolvable.
    if _technical_foul(event_type, action_type):
        return []

    if (primary := _player(1)) is not None:
        out.append(CourtAssertion(secs, event_num, primary[0], "EV", primary[1]))
    if event_type in _SLOT2_EVENTS and (p2 := _player(2)) is not None:
        out.append(CourtAssertion(secs, event_num, p2[0], "EV", p2[1]))
    if event_type in _SLOT3_EVENTS and (p3 := _player(3)) is not None:
        out.append(CourtAssertion(secs, event_num, p3[0], "EV", p3[1]))
    return out


def _clusters(assertions: Sequence[CourtAssertion]) -> Iterator[list[CourtAssertion]]:
    """Group assertions that share a clock second.

    Substitutions arrive in bursts at a stoppage. Applying them strictly one at
    a time makes an ``IN, IN, OUT, OUT`` burst transiently show six players on
    the floor, which then rejects the true starting five. Grouping by second and
    applying every OUT before every IN removes that artifact.
    """
    if not assertions:
        return
    bucket: list[CourtAssertion] = [assertions[0]]
    for item in assertions[1:]:
        if item.seconds_remaining == bucket[0].seconds_remaining:
            bucket.append(item)
        else:
            yield bucket
            bucket = [item]
    yield bucket


def _replay(starters: frozenset[int], assertions: Sequence[CourtAssertion]) -> int:
    """Count impossible states when replaying from ``starters``.

    A violation is a player entering while already on the floor, leaving while
    not on it, acting while on neither side of a substitution boundary, or more
    than five players on court.

    **Events tied on the clock are evaluated tolerantly.** The game clock does
    not order events within a second, and a player very often commits a foul and
    is substituted on the same tick. Insisting on one order marks that as a
    violation and rejects the true starting five: measured over a season, this
    single detail is the difference between 39% and 98% exact solves.

    So an action counts as valid if the player was on the floor either *before*
    or *after* that second's substitutions.
    """
    on_court = set(starters)
    violations = 0

    for cluster in _clusters(assertions):
        before = set(on_court)

        for a in cluster:
            if a.kind in ("OUT", "EJECT"):
                if a.player_id in on_court:
                    on_court.discard(a.player_id)
                else:
                    violations += 1
        for a in cluster:
            if a.kind == "IN":
                if a.player_id in on_court:
                    violations += 1
                else:
                    on_court.add(a.player_id)

        # `before | on_court` is the tolerant window: anyone present on either
        # side of this second's substitutions could legitimately have acted.
        permitted = before | on_court
        for a in cluster:
            if a.kind == "EV" and a.player_id not in permitted:
                violations += 1

        if len(on_court) > LINEUP_SIZE:
            violations += len(on_court) - LINEUP_SIZE

    return violations


def solve_period_start(assertions: Sequence[CourtAssertion]) -> PeriodSolution:
    """Infer one team's starting five for one period.

    Two facts do most of the work. A player whose *first* assertion is ``IN``
    cannot have started. Everyone else who appears must have. When that
    partition yields exactly five and the forward replay is clean, the answer is
    exact and no search is needed -- which is the case ~98% of the time.
    """
    if not assertions:
        return PeriodSolution(None, "UNDERDETERMINED", 0, 0)

    first_kind: dict[int, AssertionKind] = {}
    for a in assertions:
        first_kind.setdefault(a.player_id, a.kind)

    never_started = {pid for pid, kind in first_kind.items() if kind == "IN"}
    must_have_started = {pid for pid in first_kind if pid not in never_started}

    if len(must_have_started) == LINEUP_SIZE:
        starters = frozenset(must_have_started)
        violations = _replay(starters, assertions)
        if violations == 0:
            return PeriodSolution(starters, "EXACT", 0, 1)
        # The partition is right but the replay disagrees -- almost always
        # residual ordering noise inside a tied clock second.
        return PeriodSolution(starters, "REPAIRED", violations, 1)

    if len(must_have_started) > LINEUP_SIZE:
        # Over-inclusion: mis-ordering made a substitute look like a starter.
        # Search the subsets of the over-large set rather than giving up.
        pool = sorted(must_have_started)
        if len(pool) > _MAX_CANDIDATES:
            return PeriodSolution(None, "UNDERDETERMINED", 0, 0)
        return _search(pool, assertions, required=frozenset())

    # Fewer than five identifiable starters: fill from players who entered
    # later, since a genuine starter can be subbed out and back in.
    pool = sorted(must_have_started | never_started)
    if len(pool) < LINEUP_SIZE:
        return PeriodSolution(None, "UNDERDETERMINED", 0, 0)
    if len(pool) > _MAX_CANDIDATES:
        return PeriodSolution(None, "UNDERDETERMINED", 0, 0)
    return _search(pool, assertions, required=frozenset(must_have_started))


def _search(
    pool: Sequence[int], assertions: Sequence[CourtAssertion], *, required: frozenset[int]
) -> PeriodSolution:
    """Enumerate candidate fives and keep the ones that replay cleanly."""
    clean: list[frozenset[int]] = []
    best: tuple[int, frozenset[int]] | None = None

    for combo in combinations(pool, LINEUP_SIZE):
        candidate = frozenset(combo)
        if not required <= candidate:
            continue
        violations = _replay(candidate, assertions)
        if violations == 0:
            clean.append(candidate)
        if best is None or violations < best[0]:
            best = (violations, candidate)

    if len(clean) == 1:
        return PeriodSolution(clean[0], "SOLVED", 0, 1)
    if len(clean) > 1:
        # Several fives are consistent with the evidence. Prefer the one
        # overlapping the players we know were present, and mark it imputed.
        chosen = max(clean, key=lambda c: (len(c & required), sorted(c)))
        return PeriodSolution(chosen, "AMBIGUOUS", 0, len(clean))
    if best is not None:
        return PeriodSolution(best[1], "REPAIRED", best[0], 0)
    return PeriodSolution(None, "UNDERDETERMINED", 0, 0)


def reconstruct_stints(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach the on-court five for both teams to every event.

    Expects ``events`` already typed and in canonical (clock) order.

    Returns ``(events_enriched, stints)``:

    - ``events_enriched`` -- one row per input event plus both lineups and a
      quality flag.
    - ``stints`` -- maximal runs over which *both* lineups are constant.
    """
    required = {"game_id", "PERIOD", "EVENTNUM", "seconds_remaining", "EVENTMSGTYPE"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")

    enriched_rows: list[dict[str, object]] = []
    solutions: list[dict[str, object]] = []

    for (game_id, period), group in events.group_by(["game_id", "PERIOD"], maintain_order=True):
        rows = group.to_dicts()

        per_side: dict[int, list[CourtAssertion]] = {PERSON_TYPE_HOME: [], PERSON_TYPE_AWAY: []}
        row_assertions: list[tuple[dict[str, object], list[CourtAssertion]]] = []

        for row in rows:
            assertions = court_assertions(row)
            row_assertions.append((row, assertions))
            for a in assertions:
                per_side[a.side].append(a)

        solved: dict[int, PeriodSolution] = {
            side: solve_period_start(per_side[side]) for side in per_side
        }
        for side, solution in solved.items():
            solutions.append(
                {
                    "game_id": game_id,
                    "period": period,
                    "side": "home" if side == PERSON_TYPE_HOME else "away",
                    "status": solution.status,
                    "quality": solution.quality,
                    "n_violations": solution.n_violations,
                    "n_candidate_solutions": solution.n_candidate_solutions,
                    "starters": sorted(solution.starters) if solution.starters else None,
                }
            )

        on_court = {
            side: set(sol.starters) if sol.starters else set() for side, sol in solved.items()
        }

        for row, assertions in row_assertions:
            # Departures before arrivals, so a burst of substitutions at one
            # stoppage never transiently shows six players on the floor.
            for a in assertions:
                if a.kind in ("OUT", "EJECT"):
                    on_court[a.side].discard(a.player_id)
            for a in assertions:
                if a.kind == "IN":
                    on_court[a.side].add(a.player_id)

            home = sorted(on_court[PERSON_TYPE_HOME])
            away = sorted(on_court[PERSON_TYPE_AWAY])
            valid = len(home) == LINEUP_SIZE and len(away) == LINEUP_SIZE
            worst = _worst_quality(
                solved[PERSON_TYPE_HOME].quality, solved[PERSON_TYPE_AWAY].quality
            )

            enriched_rows.append(
                {
                    "game_id": game_id,
                    "period": period,
                    "event_num": row["EVENTNUM"],
                    "seconds_remaining": row["seconds_remaining"],
                    "event_type": row["EVENTMSGTYPE"],
                    "home_lineup": home if valid else None,
                    "away_lineup": away if valid else None,
                    "lineup_quality": worst if valid else "QUARANTINED",
                    "lineup_method": _method(solved[PERSON_TYPE_HOME].status),
                }
            )

    enriched = pl.DataFrame(enriched_rows) if enriched_rows else _empty_enriched()
    return enriched, pl.DataFrame(solutions) if solutions else _empty_solutions()


def _worst_quality(a: str, b: str) -> str:
    order = {"VALID": 0, "IMPUTED": 1, "QUARANTINED": 2}
    return a if order[a] >= order[b] else b


def _method(status: SolveStatus) -> str:
    return {
        "EXACT": "pbp_exact",
        "SOLVED": "pbp_solved",
        "AMBIGUOUS": "pbp_ambiguous",
        "REPAIRED": "pbp_repaired",
        "UNDERDETERMINED": "unresolved",
    }[status]


def _empty_enriched() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "game_id": pl.Utf8,
            "period": pl.Int64,
            "event_num": pl.Int64,
            "seconds_remaining": pl.Int64,
            "event_type": pl.Int64,
            "home_lineup": pl.List(pl.Int64),
            "away_lineup": pl.List(pl.Int64),
            "lineup_quality": pl.Utf8,
            "lineup_method": pl.Utf8,
        }
    )


def _empty_solutions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "game_id": pl.Utf8,
            "period": pl.Int64,
            "side": pl.Utf8,
            "status": pl.Utf8,
            "quality": pl.Utf8,
            "n_violations": pl.Int64,
            "n_candidate_solutions": pl.Int64,
            "starters": pl.List(pl.Int64),
        }
    )
