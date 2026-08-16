"""Filesystem layout, discovered from a marker rather than guessed.

Deriving the repository root from ``__file__`` with a fixed number of
``.parent`` hops breaks the moment a module moves between packages, and it
breaks quietly -- the path still resolves, just to the wrong directory. Walking
up to a marker survives refactors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ["DataPaths", "RepoRootNotFound", "find_repo_root"]

#: Paths that must *all* exist for a directory to be the repository root.
#: Deliberately more than one: `pyproject.toml` alone would match
#: `services/ml/`, and `.git` alone would match any checkout of any project.
_ROOT_MARKERS: Final[tuple[str, ...]] = ("services/ml/pyproject.toml", "docs")


class RepoRootNotFound(RuntimeError):
    """Raised when no ancestor directory carries every root marker."""


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until every marker in ``_ROOT_MARKERS`` is present."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and all((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate
    raise RepoRootNotFound(
        f"no ancestor of {here} contains all of {_ROOT_MARKERS}. Is this a partial checkout?"
    )


@dataclass(frozen=True)
class DataPaths:
    """Every directory the pipeline reads or writes.

    Bronze and silver are regenerable and gitignored. **Gold is committed** --
    that is what makes `lineupiq verify` meaningful on a clean clone with no
    network. See the note in `.gitignore` before adding an ignore rule here.
    """

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> DataPaths:
        return cls(root=find_repo_root(start))

    # -- data layers ------------------------------------------------------
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def bronze(self) -> Path:
        """Raw upstream payloads, content-addressed. Gitignored, regenerable."""
        return self.data / "bronze"

    @property
    def silver(self) -> Path:
        """Typed events, substitutions, stints. Gitignored, regenerable."""
        return self.data / "silver"

    @property
    def gold(self) -> Path:
        """Model-ready facts. COMMITTED, partitioned by season."""
        return self.data / "gold"

    @property
    def contracts(self) -> Path:
        """Checksum sidecars for every gold table. COMMITTED."""
        return self.gold / "_contracts"

    # -- committed evidence ----------------------------------------------
    @property
    def parity(self) -> Path:
        """Python-scored fixtures the TypeScript scorer must reproduce to 1e-9."""
        return self.data / "parity"

    @property
    def llm_cache(self) -> Path:
        """Content-addressed model responses, readable JSON. Keeps the demo free."""
        return self.data / "llm_cache"

    @property
    def llm_labels(self) -> Path:
        """Hand-graded narratives. Real human labels, not model-produced ones."""
        return self.data / "llm_labels"

    @property
    def retrieval_labels(self) -> Path:
        return self.data / "retrieval_labels"

    # -- outputs ----------------------------------------------------------
    @property
    def runs(self) -> Path:
        """Run logs. The single source of every number published anywhere."""
        return self.root / "services" / "ml" / "runs"

    @property
    def configs(self) -> Path:
        return self.root / "services" / "ml" / "src" / "lineupiq" / "configs"

    @property
    def sql_snowflake(self) -> Path:
        return self.root / "sql" / "snowflake"

    def season_partition(self, layer: Path, season_start_year: int) -> Path:
        """``data/gold/shot_facts/season=2023/`` -- Hive-style so DuckDB globs it."""
        return layer / f"season={season_start_year}"
