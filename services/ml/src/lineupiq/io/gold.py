"""Reading and writing the committed gold layer."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from lineupiq.paths import DataPaths
from lineupiq.seasons import SEASON_COVERAGE, Season
from lineupiq.validate.contracts import derive_contract, write_contract

__all__ = [
    "GOLD_TABLES",
    "POOLED_GOLD_TABLES",
    "POOLED_PARTITION",
    "available_seasons",
    "load_all_gold",
    "load_gold",
    "load_pooled_gold",
    "refresh_contracts",
    "write_pooled_gold",
]

#: Season-partitioned tables that are committed and contract-checked.
GOLD_TABLES: tuple[str, ...] = ("shot_facts", "stints", "dim_player", "possession_facts")

#: Committed tables that are *not* partitioned by season, because the model
#: behind them pools seasons by construction. RAPM is fitted across the whole
#: corpus at once -- a per-season partition would imply three independent fits
#: that do not exist.
POOLED_GOLD_TABLES: tuple[str, ...] = ("player_rapm",)

#: Partition name used for pooled tables.
POOLED_PARTITION = "pooled"


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
        # `maintain_order=True` is not decoration. `keep="first"` only means
        # anything if "first" is defined, and polars does not preserve row order
        # through `unique` unless asked -- so which of a player's rows survived
        # was whichever the parallel pass reached first. A player who changed
        # teams mid-season has rows that differ, so the surviving *name* could
        # differ between machines, and it would show up as a changed export with
        # no code change behind it.
        #
        # Partitions are concatenated in season order, so "first" is the earliest
        # season the player appears in. That is a choice, and this is where it is
        # made.
        frame = frame.unique(subset=["player_id"], keep="first", maintain_order=True).sort(
            "player_id"
        )
    return frame


def pooled_path(paths: DataPaths, table: str) -> Path:
    return paths.gold / table / POOLED_PARTITION / "part.parquet"


def load_pooled_gold(paths: DataPaths, table: str) -> pl.DataFrame:
    path = pooled_path(paths, table)
    if not path.exists():
        raise FileNotFoundError(f"{table} is not built. Run the command that produces it first.")
    return pl.read_parquet(path)


def write_pooled_gold(paths: DataPaths, table: str, frame: pl.DataFrame) -> Path:
    path = pooled_path(paths, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def refresh_contracts(paths: DataPaths) -> list[str]:
    """Re-derive and write a contract for every committed gold partition."""
    written: list[str] = []
    for table in GOLD_TABLES:
        for season in available_seasons(paths, table):
            frame = load_gold(paths, table, season)
            contract = derive_contract(frame, table=table, partition=f"season={season.start_year}")
            write_contract(contract, paths.contracts)
            written.append(f"{table}__season={season.start_year}")
    for table in POOLED_GOLD_TABLES:
        if not pooled_path(paths, table).exists():
            continue
        frame = load_pooled_gold(paths, table)
        write_contract(
            derive_contract(frame, table=table, partition=POOLED_PARTITION), paths.contracts
        )
        written.append(f"{table}__{POOLED_PARTITION}")
    return written
