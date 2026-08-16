"""Season identity, declared exactly once.

Two upstream sources label the same season with different years, and one of them
is internally inconsistent. Getting this wrong shifts the entire project by a
year, silently, with no error anywhere -- every join still succeeds, every row
count still looks plausible, and every number is about the wrong season.

So the rule is: **never trust a filename.** The season is decoded from
``GAME_ID``, which carries it in positions 3-4, and asserted against whatever
the file claimed to be.

Conventions observed in the wild
--------------------------------

``shufinskiy/nba_data`` uses the **start** year::

    nbastats_2023.csv  ->  GAME_ID 0022300001  ->  2023-24

The sportsdataverse mirror mostly uses the **end** year -- but not consistently,
and the inconsistency lives inside a single release tag::

    nba_stats_pbp/play_by_play_2024.parquet      -> 0022400001 -> 2024-25  (START)
    nba_stats_pbp/nba_play_by_play_2024.parquet  -> 0012300001 -> 2023-24  (END)
    nba_stats_game_lineups/nba_lineups_2024      -> 0012300001 -> 2023-24  (END)
    nba_stats_player_boxscores/..._2024          -> 0022300001 -> 2023-24  (END)

Two files in the same repository, one character apart in the name, a year apart
in the content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "GAME_ID_RE",
    "SEASON_COVERAGE",
    "GameType",
    "Season",
    "game_type_from_game_id",
    "season_from_game_id",
]

GAME_ID_RE: Final = re.compile(r"^\d{10}$")

#: ``GAME_ID`` prefix -> what kind of game it is. Position 2 of the 10-character
#: id. Verified present in 2023-24.
_GAME_TYPE_BY_PREFIX: Final[dict[str, GameType]] = {
    "001": "preseason",
    "002": "regular",
    "003": "allstar",
    "004": "playoffs",
    "005": "playin",
    "006": "cupfinal",
}

GameType = Literal["preseason", "regular", "allstar", "playoffs", "playin", "cupfinal"]

#: The NBA's own two-digit season encoding rolled over at 1999-00. Every season
#: this project covers is post-2000, so a two-digit year maps unambiguously into
#: the 2000s. Asserted rather than assumed -- see ``Season.from_two_digit``.
_CENTURY_PIVOT: Final = 2000


@dataclass(frozen=True, order=True)
class Season:
    """A single NBA season, identified by the calendar year it starts in.

    ``Season(2023)`` is the 2023-24 season. That is the only representation
    stored; every other convention is a derived accessor, so a caller has to say
    which one it means and cannot silently pick up the wrong one.
    """

    start_year: int

    def __post_init__(self) -> None:
        if not 1946 <= self.start_year <= 2100:
            raise ValueError(f"implausible season start year: {self.start_year}")

    @property
    def end_year(self) -> int:
        """The calendar year the season finishes in. 2023 -> 2024."""
        return self.start_year + 1

    @property
    def label(self) -> str:
        """Human and NBA-stats form: ``"2023-24"``."""
        return f"{self.start_year}-{self.end_year % 100:02d}"

    @property
    def two_digit(self) -> str:
        """The two characters that appear in ``GAME_ID``: 2023 -> ``"23"``."""
        return f"{self.start_year % 100:02d}"

    def shufinskiy_year(self) -> int:
        """Filename year used by ``shufinskiy/nba_data`` (start year)."""
        return self.start_year

    def sportsdataverse_year(self) -> int:
        """Filename year used by most sportsdataverse assets (end year).

        Not universal -- see the module docstring. Always confirm with
        :func:`season_from_game_id` after loading.
        """
        return self.end_year

    @classmethod
    def from_two_digit(cls, digits: str) -> Season:
        if len(digits) != 2 or not digits.isdigit():
            raise ValueError(f"expected two digits, got {digits!r}")
        return cls(_CENTURY_PIVOT + int(digits))

    @classmethod
    def from_label(cls, label: str) -> Season:
        """Parse ``"2023-24"``. Rejects a mismatched pair like ``"2023-25"``."""
        m = re.fullmatch(r"(\d{4})-(\d{2})", label)
        if m is None:
            raise ValueError(f"expected a season label like '2023-24', got {label!r}")
        start = int(m.group(1))
        season = cls(start)
        if season.label != label:
            raise ValueError(f"inconsistent season label {label!r}; did you mean {season.label!r}?")
        return season

    def __str__(self) -> str:
        return self.label


def _validate_game_id(game_id: str) -> str:
    """Normalise and check a ``GAME_ID``.

    The leading zeros are load-bearing and are the first thing lost to type
    inference: a CSV reader that infers int turns ``0022300001`` into
    ``22300001``, and every downstream join against a correctly-typed source
    then misses. Read ``GAME_ID`` as a string and zero-fill to 10.
    """
    if not isinstance(game_id, str):
        raise TypeError(
            f"GAME_ID must be a string, got {type(game_id).__name__}. "
            "Reading it as an integer drops the leading zeros."
        )
    padded = game_id.strip().zfill(10)
    if not GAME_ID_RE.fullmatch(padded):
        raise ValueError(f"not a valid 10-digit GAME_ID: {game_id!r}")
    return padded


def season_from_game_id(game_id: str) -> Season:
    """Decode the season from a ``GAME_ID``. The only trustworthy source.

    ``0022300001`` -> ``Season(2023)`` (the 2023-24 season).

    Layout: ``00`` + game-type digit + two-digit season start year + 5-digit
    sequence.
    """
    return Season.from_two_digit(_validate_game_id(game_id)[3:5])


def game_type_from_game_id(game_id: str) -> GameType:
    """Decode preseason / regular / playoffs / ... from a ``GAME_ID``.

    Preseason and all-star games must be excluded before anything is modelled:
    the rotations are not real, and all-star defense is not defense.
    """
    prefix = _validate_game_id(game_id)[:3]
    try:
        return _GAME_TYPE_BY_PREFIX[prefix]
    except KeyError:
        raise ValueError(
            f"unknown GAME_ID type prefix {prefix!r} (from {game_id!r}); "
            f"known prefixes: {sorted(_GAME_TYPE_BY_PREFIX)}"
        ) from None


#: The declared scope of this project, stated once.
#:
#: Every document that names a season range -- README, model cards, the design
#: docs -- must agree with this tuple, and CI checks that they do. A stale
#: coverage claim is a data-integrity bug, not a typo.
#:
#: These three are the most recent seasons confirmed complete in the upstream
#: mirror. M2 asserts availability at build time and fails loudly rather than
#: silently building a short season.
SEASON_COVERAGE: Final[tuple[Season, ...]] = (
    Season(2022),
    Season(2023),
    Season(2024),
)

#: Game types that reach the model. Preseason, all-star and (for now) the Cup
#: final are excluded; the Cup final is a regular-season game by record but is
#: duplicated under prefix 006, so including it would double-count.
MODELLED_GAME_TYPES: Final[frozenset[GameType]] = frozenset({"regular", "playoffs", "playin"})
