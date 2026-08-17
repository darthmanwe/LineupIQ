"""Reading and writing the committed gold layer."""

from __future__ import annotations

import polars as pl

from lineupiq.paths import DataPaths
from lineupiq.seasons import SEASON_COVERAGE, Season
from lineupiq.validate.contracts import derive_contract, write_contract

__all__ = ["GOLD_TABLES", "available_seasons", "load_all_gold", "load_gold", "refresh_contracts"]

#: Tables that are committed and contract-checked.
GOLD_TABLES: tuple[str, ...] = ("shot_facts", "stints", "dim_player", "possession_facts")


def available_seasons(paths: DataPaths, table: str = "shot_facts") -> list[Season]:
    """Seasons actually present on disk, not merely declared."""
    root = paths.gold / table
    if not root.exists():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("season="):
            found.append(Season(int(child.name.split("=", 1)[1])))
    return found


def load_gold(paths: DataPaths, table: str, season: Season) -> pl.DataFrame:
    path = paths.gold / table / f"season={season.start_year}" / "part.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{table} for {season.label} is not built. Run `lineupiq build --season "
            f"{season.start_year}` first."
        )
    return pl.read_parquet(path)


def load_all_gold(
    paths: DataPaths, table: str, seasons: list[Season] | None = None
) -> pl.DataFrame:
    """Concatenate one gold table across seasons.

    ``dim_player`` is deduplicated on load: the same player appears in every
    season he played, and downstream joins assume one row per id.
    """
    targets = seasons or available_seasons(paths, table) or list(SEASON_COVERAGE)
    parts = [load_gold(paths, table, s) for s in targets]
    if not parts:
        raise FileNotFoundError(f"no partitions found for {table}")
    frame = pl.concat(parts, how="vertical_relaxed")
    if table == "dim_player":
        frame = frame.unique(subset=["player_id"], keep="first").sort("player_id")
    return frame


def refresh_contracts(paths: DataPaths) -> list[str]:
    """Re-derive and write a contract for every committed gold partition."""
    written: list[str] = []
    for table in GOLD_TABLES:
        for season in available_seasons(paths, table):
            frame = load_gold(paths, table, season)
            contract = derive_contract(frame, table=table, partition=f"season={season.start_year}")
            write_contract(contract, paths.contracts)
            written.append(f"{table}__season={season.start_year}")
    return written
