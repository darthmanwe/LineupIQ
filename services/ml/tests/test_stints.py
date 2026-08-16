"""The stint solver, on hand-built games with known answers.

These fixtures encode the specific failure modes that cost real accuracy during
development: cross-team slot attribution, tied-clock ordering, and substitution
bursts. Each test names the rate it protects.
"""

from __future__ import annotations

import polars as pl
import pytest

from lineupiq.transform.events import PERSON_TYPE_AWAY, PERSON_TYPE_HOME
from lineupiq.transform.segments import segment_stints
from lineupiq.transform.stints import (
    CourtAssertion,
    court_assertions,
    reconstruct_stints,
    solve_period_start,
)

HOME = [101, 102, 103, 104, 105]
AWAY = [201, 202, 203, 204, 205]


def ev(secs: int, num: int, pid: int, side: int, kind: str = "EV") -> CourtAssertion:
    return CourtAssertion(secs, num, pid, kind, side)  # type: ignore[arg-type]


def _row(
    *,
    event_num: int,
    event_type: int,
    secs: int,
    p1: int | None = None,
    t1: int | None = None,
    p2: int | None = None,
    t2: int | None = None,
    p3: int | None = None,
    t3: int | None = None,
    action: int = 0,
    game_id: str = "0022300001",
    period: int = 1,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "PERIOD": period,
        "EVENTNUM": event_num,
        "EVENTMSGTYPE": event_type,
        "EVENTMSGACTIONTYPE": action,
        "seconds_remaining": secs,
        "PLAYER1_ID": p1,
        "PERSON1TYPE": t1,
        "PLAYER2_ID": p2,
        "PERSON2TYPE": t2,
        "PLAYER3_ID": p3,
        "PERSON3TYPE": t3,
    }


class TestCourtAssertions:
    def test_substitution_yields_out_then_in_on_one_side(self) -> None:
        row = _row(
            event_num=10,
            event_type=8,
            secs=500,
            p1=101,
            t1=PERSON_TYPE_HOME,
            p2=106,
            t2=PERSON_TYPE_HOME,
        )
        got = court_assertions(row)
        assert [(a.player_id, a.kind, a.side) for a in got] == [
            (101, "OUT", PERSON_TYPE_HOME),
            (106, "IN", PERSON_TYPE_HOME),
        ]

    def test_steal_attributes_each_player_to_his_own_team(self) -> None:
        # A turnover: PLAYER1 lost it (home), PLAYER2 stole it (away). Reading
        # the row's side from PLAYER1 alone puts an away player in the home
        # lineup -- the bug that held exact solves at 0.04%.
        row = _row(
            event_num=11,
            event_type=5,
            secs=480,
            p1=101,
            t1=PERSON_TYPE_HOME,
            p2=201,
            t2=PERSON_TYPE_AWAY,
        )
        sides = {a.player_id: a.side for a in court_assertions(row)}
        assert sides == {101: PERSON_TYPE_HOME, 201: PERSON_TYPE_AWAY}

    def test_block_attributes_blocker_to_the_defending_team(self) -> None:
        row = _row(
            event_num=12,
            event_type=2,
            secs=470,
            p1=101,
            t1=PERSON_TYPE_HOME,
            p3=201,
            t3=PERSON_TYPE_AWAY,
        )
        sides = {a.player_id: a.side for a in court_assertions(row)}
        assert sides == {101: PERSON_TYPE_HOME, 201: PERSON_TYPE_AWAY}

    def test_technical_foul_is_not_presence_evidence(self) -> None:
        # A bench player can be assessed a technical. Counting it puts a seated
        # player on the floor and makes the period unsolvable.
        row = _row(event_num=13, event_type=6, secs=460, p1=110, t1=PERSON_TYPE_HOME, action=11)
        assert court_assertions(row) == []

    def test_ordinary_foul_is_presence_evidence(self) -> None:
        row = _row(event_num=14, event_type=6, secs=450, p1=101, t1=PERSON_TYPE_HOME, action=1)
        assert [a.kind for a in court_assertions(row)] == ["EV"]

    def test_team_rebound_is_ignored(self) -> None:
        # PERSON1TYPE 2 is a team row, not a player.
        row = _row(event_num=15, event_type=4, secs=440, p1=1610612748, t1=2)
        assert court_assertions(row) == []


class TestSolvePeriodStart:
    def test_exact_when_five_players_act_before_any_substitution(self) -> None:
        asserts = [ev(720 - i * 10, i, HOME[i % 5], PERSON_TYPE_HOME) for i in range(10)]
        solution = solve_period_start(asserts)
        assert solution.status == "EXACT"
        assert solution.starters == frozenset(HOME)
        assert solution.quality == "VALID"

    def test_player_whose_first_appearance_is_a_sub_in_did_not_start(self) -> None:
        asserts = [ev(720 - i * 10, i, HOME[i % 5], PERSON_TYPE_HOME) for i in range(10)]
        asserts.append(ev(600, 20, HOME[0], PERSON_TYPE_HOME, "OUT"))
        asserts.append(ev(600, 21, 106, PERSON_TYPE_HOME, "IN"))
        asserts.append(ev(590, 22, 106, PERSON_TYPE_HOME))
        solution = solve_period_start(asserts)
        assert solution.status == "EXACT"
        assert 106 not in (solution.starters or frozenset())

    def test_action_tied_on_the_clock_with_a_substitution_is_not_a_violation(self) -> None:
        # A player fouls and is subbed out on the same tick. Insisting on one
        # order marks this as impossible and rejects the true five -- worth
        # ~59 percentage points of exact solves.
        asserts = [ev(720 - i * 10, i, HOME[i % 5], PERSON_TYPE_HOME) for i in range(10)]
        asserts.append(ev(600, 20, HOME[0], PERSON_TYPE_HOME))  # acts at t=600
        asserts.append(ev(600, 21, HOME[0], PERSON_TYPE_HOME, "OUT"))  # leaves at t=600
        asserts.append(ev(600, 22, 106, PERSON_TYPE_HOME, "IN"))
        solution = solve_period_start(asserts)
        assert solution.status == "EXACT", "tied-clock action must be tolerated"
        assert solution.n_violations == 0

    def test_substitution_burst_does_not_transiently_exceed_five(self) -> None:
        asserts = [ev(720 - i * 10, i, HOME[i % 5], PERSON_TYPE_HOME) for i in range(10)]
        # Two in, two out, all at one stoppage, INs listed first.
        asserts += [
            ev(500, 30, 106, PERSON_TYPE_HOME, "IN"),
            ev(500, 31, 107, PERSON_TYPE_HOME, "IN"),
            ev(500, 32, HOME[0], PERSON_TYPE_HOME, "OUT"),
            ev(500, 33, HOME[1], PERSON_TYPE_HOME, "OUT"),
        ]
        solution = solve_period_start(asserts)
        assert solution.status == "EXACT"
        assert solution.n_violations == 0

    def test_empty_period_is_underdetermined_not_guessed(self) -> None:
        solution = solve_period_start([])
        assert solution.status == "UNDERDETERMINED"
        assert solution.starters is None
        assert solution.quality == "QUARANTINED"

    def test_too_few_identifiable_players_refuses(self) -> None:
        asserts = [ev(700, 1, 101, PERSON_TYPE_HOME), ev(690, 2, 102, PERSON_TYPE_HOME)]
        solution = solve_period_start(asserts)
        assert solution.starters is None
        assert solution.quality == "QUARANTINED"


class TestReconstruct:
    def _game(self) -> pl.DataFrame:
        rows = []
        num = 0
        for i in range(10):
            num += 1
            rows.append(
                _row(
                    event_num=num,
                    event_type=1,
                    secs=720 - i * 20,
                    p1=HOME[i % 5],
                    t1=PERSON_TYPE_HOME,
                )
            )
            num += 1
            rows.append(
                _row(
                    event_num=num,
                    event_type=1,
                    secs=715 - i * 20,
                    p1=AWAY[i % 5],
                    t1=PERSON_TYPE_AWAY,
                )
            )
        # One home substitution at 400.
        num += 1
        rows.append(
            _row(
                event_num=num,
                event_type=8,
                secs=400,
                p1=HOME[0],
                t1=PERSON_TYPE_HOME,
                p2=106,
                t2=PERSON_TYPE_HOME,
            )
        )
        num += 1
        rows.append(_row(event_num=num, event_type=1, secs=380, p1=106, t1=PERSON_TYPE_HOME))
        return pl.DataFrame(rows)

    def test_attaches_both_lineups_to_every_event(self) -> None:
        enriched, solutions = reconstruct_stints(self._game())
        assert enriched.height == 22
        assert solutions.height == 2  # one per side
        assert set(solutions["status"].to_list()) == {"EXACT"}

        first = enriched.row(0, named=True)
        assert sorted(first["home_lineup"]) == HOME
        assert sorted(first["away_lineup"]) == AWAY

        last = enriched.row(enriched.height - 1, named=True)
        assert 106 in last["home_lineup"]
        assert HOME[0] not in last["home_lineup"]

    def test_segments_into_stints_that_tile_the_period(self) -> None:
        enriched, _ = reconstruct_stints(self._game())
        stints = segment_stints(enriched)
        assert stints.height == 2, "one substitution splits the period in two"
        assert stints["duration_seconds"].sum() == 720, "stints must tile the full period"
        assert (stints["duration_seconds"] > 0).all()


class TestZones:
    @pytest.mark.parametrize(
        ("x", "y", "expected"),
        [
            (0, 0, "restricted_area"),
            (0, 30, "restricted_area"),
            (0, 120, "paint_non_ra"),
            (-230, 50, "corner_three_left"),
            (230, 50, "corner_three_right"),
            (0, 260, "top_three"),
            # 20 ft straight on: past the 19 ft lane, inside the 23.75 ft arc.
            (0, 200, "mid_top"),
            # 15 ft on the baseline side, outside the lane.
            (150, 40, "mid_baseline"),
        ],
    )
    def test_zone_geometry(self, x: int, y: int, expected: str) -> None:
        from lineupiq.transform.zones import derive_zone

        got = pl.DataFrame({"loc_x": [float(x)], "loc_y": [float(y)]}).select(derive_zone())
        assert got["zone_id"][0] == expected

    def test_corner_three_is_shorter_than_the_arc(self) -> None:
        # 22.5 ft in the corner is a three; the same distance up top is not.
        from lineupiq.transform.zones import is_three_expr

        frame = pl.DataFrame({"loc_x": [225.0, 0.0], "loc_y": [50.0, 225.0]})
        got = frame.select(is_three_expr().alias("three"))["three"].to_list()
        assert got == [True, False]
