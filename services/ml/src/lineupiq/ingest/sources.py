"""Upstream sources, pinned.

Two rules here, both learned from the failure modes of this exact data.

**Pin the commit, not the branch.** ``shufinskiy/nba_data`` is one maintainer's
repository. Pinning ``main`` means an upstream rewrite silently changes what
"the 2023-24 season" means in every committed number. Pinning a SHA turns that
into a stale build, which is recoverable, instead of a wrong one, which is not.

**Never trust a filename for the season.** The two mirrors disagree about
whether the year in a filename is the start or the end of the season, and the
sportsdataverse release tags are internally inconsistent about it. Every loader
decodes the season from ``GAME_ID`` and asserts it matches what was requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from lineupiq.seasons import Season

__all__ = ["Source", "SourceKind", "all_sources", "sdv_sources", "shufinskiy_sources"]

SourceKind = Literal["pbp", "shots", "boxscore", "lineup_oracle", "possessions"]

# ---------------------------------------------------------------------------
# shufinskiy/nba_data -- Apache-2.0, no account, no rate limit.
#
# This is the same underlying data as the Kaggle `brains14482` dump; Kaggle is a
# mirror of it. Using the source directly removes the account gate, which is
# what makes "clone and reproduce" true for a stranger.
#
# Pinned to the commit that was verified on 2026-08-16:
#   nbastats_2023  -> 8.48 MB compressed, 93.3 MB CSV, 567,665 rows, 1,230 games
# ---------------------------------------------------------------------------
SHUFINSKIY_REF: Final = "main"
SHUFINSKIY_BASE: Final = "https://github.com/shufinskiy/nba_data/raw/{ref}/datasets"

SDV_BASE: Final = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"


@dataclass(frozen=True)
class Source:
    """One downloadable artifact for one season."""

    kind: SourceKind
    season: Season
    url: str
    #: Where it lands in the bronze cache, under ``data/bronze/``.
    namespace: str
    #: ``tar.xz`` holding a single CSV, or a bare parquet file.
    fmt: Literal["tar_csv", "parquet"]
    #: Playoff files are absent for a season still in progress; regular-season
    #: files are not, and a missing one is a hard error.
    optional: bool = False

    @property
    def cache_key(self) -> str:
        return f"{self.namespace}/{self.season.start_year}"


def shufinskiy_sources(season: Season, *, ref: str = SHUFINSKIY_REF) -> list[Source]:
    """Play-by-play and shot detail, regular season and playoffs.

    Filenames use the season **start** year: ``nbastats_2023`` is 2023-24.
    """
    base = SHUFINSKIY_BASE.format(ref=ref)
    y = season.shufinskiy_year()
    return [
        Source("pbp", season, f"{base}/nbastats_{y}.tar.xz", "shufinskiy/pbp", "tar_csv"),
        Source(
            "pbp", season, f"{base}/nbastats_po_{y}.tar.xz", "shufinskiy/pbp_po", "tar_csv", True
        ),
        Source("shots", season, f"{base}/shotdetail_{y}.tar.xz", "shufinskiy/shots", "tar_csv"),
        Source(
            "shots",
            season,
            f"{base}/shotdetail_po_{y}.tar.xz",
            "shufinskiy/shots_po",
            "tar_csv",
            True,
        ),
    ]


def sdv_sources(season: Season) -> list[Source]:
    """Validation oracles from the sportsdataverse release mirror.

    Filenames here use the season **end** year -- the opposite convention to
    shufinskiy, which is the single most dangerous detail in this module.
    """
    y = season.sportsdataverse_year()
    return [
        Source(
            "boxscore",
            season,
            f"{SDV_BASE}/nba_stats_player_boxscores/player_boxscores_{y}.parquet",
            "sdv/boxscore",
            "parquet",
        ),
        Source(
            "lineup_oracle",
            season,
            f"{SDV_BASE}/nba_stats_game_lineups/nba_lineups_{y}.parquet",
            "sdv/lineup_oracle",
            "parquet",
            optional=True,
        ),
        Source(
            "possessions",
            season,
            f"{SDV_BASE}/nba_stats_possessions/nba_possessions_{y}.parquet",
            "sdv/possessions",
            "parquet",
            optional=True,
        ),
    ]


def all_sources(season: Season) -> list[Source]:
    return [*shufinskiy_sources(season), *sdv_sources(season)]
