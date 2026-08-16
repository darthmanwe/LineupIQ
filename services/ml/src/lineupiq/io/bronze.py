"""Content-addressed, resumable bronze cache.

Downloads land here once and are never re-fetched unless the bytes upstream
change. Three properties matter:

**Resumable.** Ingesting three seasons is ~88 MB across 21 artifacts. A failure
on the last one must not re-download the first twenty.

**Auditable.** The manifest records the resolved URL, byte count, and SHA-256 of
every payload. If upstream silently rewrites a file, the digest changes and the
next run says so instead of quietly producing different numbers.

**Offline-after-first-run.** Once bronze is populated, the whole build runs with
no network, which is what makes the ingest step reproducible rather than a live
dependency.
"""

from __future__ import annotations

import hashlib
import io
import json

# Imported for its side effect: `tarfile` can only open .xz when the lzma
# module is present, and some minimal Python builds ship without it. Importing
# it here fails at module load with a real traceback instead of deep inside a
# download with an opaque "unknown compression" error.
import lzma  # noqa: F401
import tarfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from lineupiq.ingest.sources import Source

__all__ = ["BronzeCache", "CacheEntry", "FetchError"]

_USER_AGENT = "lineupiq/0.1 (+https://github.com/darthmanwe/LineupIQ)"
_CHUNK = 1 << 16


class FetchError(RuntimeError):
    """A required artifact could not be retrieved."""


@dataclass(frozen=True)
class CacheEntry:
    """One line of the manifest."""

    cache_key: str
    url: str
    sha256: str
    bytes_downloaded: int
    rows: int
    fetched_at: str
    parquet: str


class BronzeCache:
    """Append-only cache of upstream payloads, normalised to parquet."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "manifest.jsonl"

    # -- manifest ---------------------------------------------------------
    def _entries(self) -> dict[str, CacheEntry]:
        if not self.manifest_path.exists():
            return {}
        out: dict[str, CacheEntry] = {}
        with self.manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entry = CacheEntry(**json.loads(line))
                    out[entry.cache_key] = entry
        return out

    def _append(self, entry: CacheEntry) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

    def entry_for(self, source: Source) -> CacheEntry | None:
        return self._entries().get(source.cache_key)

    def path_for(self, source: Source) -> Path:
        return self.root / f"{source.namespace}" / f"{source.season.start_year}.parquet"

    # -- fetch ------------------------------------------------------------
    def fetch(self, source: Source, *, force: bool = False) -> pl.DataFrame | None:
        """Return the artifact as a DataFrame, downloading only if needed.

        Returns ``None`` for an optional artifact that upstream does not have --
        a playoff file for a season still in progress, for example. A missing
        *required* artifact raises.
        """
        target = self.path_for(source)
        if target.exists() and not force:
            return pl.read_parquet(target)

        try:
            payload = self._download(source.url)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if source.optional:
                return None
            raise FetchError(f"required source {source.cache_key} unavailable: {exc}") from exc

        frame = self._decode(payload, source)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)

        self._append(
            CacheEntry(
                cache_key=source.cache_key,
                url=source.url,
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes_downloaded=len(payload),
                rows=frame.height,
                fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                parquet=str(target.relative_to(self.root)).replace("\\", "/"),
            )
        )
        return frame

    @staticmethod
    def _download(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        buf = io.BytesIO()
        with urllib.request.urlopen(request, timeout=300) as response:
            while chunk := response.read(_CHUNK):
                buf.write(chunk)
        return buf.getvalue()

    @staticmethod
    def _decode(payload: bytes, source: Source) -> pl.DataFrame:
        if source.fmt == "parquet":
            return pl.read_parquet(io.BytesIO(payload))

        # tar.xz holding exactly one CSV.
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            if len(members) != 1:
                raise FetchError(
                    f"{source.cache_key}: expected one file in the archive, found "
                    f"{[m.name for m in members]}"
                )
            handle = tf.extractfile(members[0])
            if handle is None:
                raise FetchError(f"{source.cache_key}: archive member was not readable")
            csv_bytes = handle.read()

        # infer_schema_length=0 reads every column as Utf8. This is not laziness:
        # GAME_ID arrives 8 characters wide with its leading zeros already
        # stripped upstream, and letting a reader infer int would make that
        # unrecoverable. Typing happens once, explicitly, in transform/events.py.
        return pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=0)

    # -- reporting --------------------------------------------------------
    def summary(self) -> pl.DataFrame:
        entries = list(self._entries().values())
        if not entries:
            return pl.DataFrame(
                schema={"cache_key": pl.Utf8, "rows": pl.Int64, "mb": pl.Float64, "sha256": pl.Utf8}
            )
        return pl.DataFrame(
            [
                {
                    "cache_key": e.cache_key,
                    "rows": e.rows,
                    "mb": round(e.bytes_downloaded / 1e6, 2),
                    "sha256": e.sha256[:12],
                }
                for e in entries
            ]
        ).sort("cache_key")
