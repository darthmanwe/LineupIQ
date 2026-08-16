"""Season decoding.

These tests exist because of two specific, silent failure modes: a CSV reader
that strips the leading zeros off ``GAME_ID``, and a mirror that labels the same
season with two different years one character apart in the filename.
"""

from __future__ import annotations

import itertools

import pytest

from lineupiq.seasons import (
    MODELLED_GAME_TYPES,
    SEASON_COVERAGE,
    Season,
    game_type_from_game_id,
    season_from_game_id,
)


class TestSeason:
    def test_label_spans_two_calendar_years(self) -> None:
        assert Season(2023).label == "2023-24"
        assert Season(2023).end_year == 2024

    def test_label_pads_single_digit_end_year(self) -> None:
        # 2009-10, not 2009-1. The obvious formatting bug.
        assert Season(2009).label == "2009-10"

    def test_two_digit_pads(self) -> None:
        assert Season(2005).two_digit == "05"

    def test_round_trips_through_label(self) -> None:
        for season in SEASON_COVERAGE:
            assert Season.from_label(season.label) == season

    def test_rejects_internally_inconsistent_label(self) -> None:
        # 2023-25 is not a season; catching it beats silently taking the 2023.
        with pytest.raises(ValueError, match="inconsistent"):
            Season.from_label("2023-25")

    def test_orders_chronologically(self) -> None:
        assert sorted([Season(2024), Season(2022), Season(2023)]) == [
            Season(2022),
            Season(2023),
            Season(2024),
        ]

    def test_is_hashable_and_usable_as_a_key(self) -> None:
        assert len({Season(2023), Season(2023), Season(2024)}) == 2


class TestMirrorConventions:
    """The landmine: one season, two filename years, both in active use."""

    def test_shufinskiy_uses_start_year(self) -> None:
        assert Season(2023).shufinskiy_year() == 2023

    def test_sportsdataverse_uses_end_year(self) -> None:
        assert Season(2023).sportsdataverse_year() == 2024

    def test_the_two_conventions_disagree_by_one(self) -> None:
        # Stated as an assertion so that if anyone "simplifies" these to the
        # same accessor, the suite says exactly what broke and why.
        season = Season(2023)
        assert season.sportsdataverse_year() - season.shufinskiy_year() == 1


class TestSeasonFromGameId:
    @pytest.mark.parametrize(
        ("game_id", "expected"),
        [
            ("0022300001", Season(2023)),  # regular season, 2023-24
            ("0022400001", Season(2024)),  # regular season, 2024-25
            ("0042200301", Season(2022)),  # playoffs, 2022-23
        ],
    )
    def test_decodes_season(self, game_id: str, expected: Season) -> None:
        assert season_from_game_id(game_id) == expected

    def test_recovers_stripped_leading_zeros(self) -> None:
        # What a CSV reader hands back after inferring int: 0022300001 -> 22300001.
        assert season_from_game_id("22300001") == season_from_game_id("0022300001")

    def test_rejects_an_integer_outright(self) -> None:
        # Not coerced. An int GAME_ID means the leading zeros are already gone
        # upstream, and silently zero-filling would hide that.
        with pytest.raises(TypeError, match="leading zeros"):
            season_from_game_id(22300001)  # type: ignore[arg-type]

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="valid 10-digit"):
            season_from_game_id("not-a-game-id")


class TestGameType:
    @pytest.mark.parametrize(
        ("game_id", "expected"),
        [
            ("0012300001", "preseason"),
            ("0022300001", "regular"),
            ("0032300001", "allstar"),
            ("0042300101", "playoffs"),
            ("0052300001", "playin"),
        ],
    )
    def test_decodes_type(self, game_id: str, expected: str) -> None:
        assert game_type_from_game_id(game_id) == expected

    def test_preseason_and_allstar_are_excluded_from_modelling(self) -> None:
        # Preseason rotations are not real rotations, and all-star defense is
        # not defense. Both would otherwise pass every structural check.
        assert "preseason" not in MODELLED_GAME_TYPES
        assert "allstar" not in MODELLED_GAME_TYPES
        assert "regular" in MODELLED_GAME_TYPES

    def test_rejects_unknown_prefix(self) -> None:
        with pytest.raises(ValueError, match="unknown GAME_ID type prefix"):
            game_type_from_game_id("0092300001")


class TestDeclaredCoverage:
    def test_coverage_is_three_consecutive_seasons(self) -> None:
        years = [s.start_year for s in SEASON_COVERAGE]
        assert years == sorted(years), "coverage must be in chronological order"
        assert all(b - a == 1 for a, b in itertools.pairwise(years)), (
            "coverage must be consecutive; a gap would silently change what "
            "'three seasons' means in every published number"
        )

    def test_coverage_is_non_empty(self) -> None:
        assert len(SEASON_COVERAGE) >= 1
