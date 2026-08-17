"""Data contracts: a committed fingerprint for every gold table.

A contract records what a table contained when its numbers were published --
row count, schema, null rates, ranges, and a content hash. ``lineupiq verify``
re-derives all of it offline and fails on any drift.

The content hash is order-independent (rows are hashed individually, then the
digests are sorted) so a change in row order is not reported as a change in
data. It is also platform-stable: floats are formatted at fixed precision
rather than repr'd, because ``repr(0.1)`` has differed across builds and would
make Windows and Linux CI disagree about identical data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from lineupiq.util import as_float

__all__ = ["TableContract", "content_hash", "derive_contract", "verify_table", "write_contract"]

#: Fixed precision for float formatting inside the hash. Enough to catch a real
#: change, coarse enough to survive last-bit variation between BLAS builds.
_FLOAT_PRECISION = 9

#: Rows per streaming chunk when hashing. Bounds peak memory on large tables.
_HASH_CHUNK = 50_000


def _format(value: Any) -> str:
    if value is None:
        return "\x00"
    if isinstance(value, float):
        return f"{value:.{_FLOAT_PRECISION}f}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format(v) for v in value) + "]"
    return str(value)


def content_hash(frame: pl.DataFrame) -> str:
    """Order-independent SHA-256 over the frame's contents.

    Canonicalisation happens inside polars rather than by iterating rows in
    Python. That is not only faster: materialising 1.5M rows of nested list
    columns as Python objects exhausted memory and crashed the interpreter with
    an access violation, which is a spectacularly unhelpful way to learn that a
    checksum is expensive.

    Floats are rounded to a fixed precision before stringification so that
    last-bit differences between BLAS builds do not read as changed data, and
    SHA-256 is used rather than polars' internal hash so the digest is stable
    across polars versions and platforms.
    """
    columns = sorted(frame.columns)
    if not columns:
        return hashlib.sha256(b"").hexdigest()

    parts: list[pl.Expr] = []
    for name in columns:
        dtype = frame.schema[name]
        col = pl.col(name)
        if isinstance(dtype, pl.List):
            # A List cannot be cast straight to Utf8; it has to be rendered
            # element-wise and joined. Float elements carry the same precision
            # problem as scalar floats, so they are rounded on the way through.
            inner = dtype.inner
            if inner is not None and inner.is_float():
                col = col.list.eval(pl.element().round(_FLOAT_PRECISION))
            col = col.cast(pl.List(pl.Utf8)).list.join(",")
        elif dtype.is_float():
            col = col.round(_FLOAT_PRECISION).cast(pl.Utf8)
        else:
            col = col.cast(pl.Utf8)
        parts.append(col.fill_null("\x00"))

    canonical = (
        frame.select(pl.concat_str(parts, separator="\x1f").alias("_row")).get_column("_row").sort()
    )

    outer = hashlib.sha256()
    outer.update("\x1e".join(columns).encode("utf-8"))
    # Stream in chunks so peak memory stays bounded regardless of table size.
    for offset in range(0, canonical.len(), _HASH_CHUNK):
        chunk = canonical.slice(offset, _HASH_CHUNK).to_list()
        outer.update("\x1d".join("" if v is None else v for v in chunk).encode("utf-8"))
        outer.update(b"\x1c")
    return outer.hexdigest()


@dataclass(frozen=True)
class TableContract:
    table: str
    partition: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    null_rates: dict[str, float]
    numeric_ranges: dict[str, list[float]]
    content_sha256: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def derive_contract(frame: pl.DataFrame, *, table: str, partition: str) -> TableContract:
    null_rates: dict[str, float] = {}
    ranges: dict[str, list[float]] = {}

    for name, dtype in zip(frame.columns, frame.dtypes, strict=True):
        series = frame[name]
        null_rates[name] = round(series.null_count() / frame.height, 6) if frame.height else 0.0
        if dtype.is_numeric():
            lo, hi = series.min(), series.max()
            if lo is not None and hi is not None:
                ranges[name] = [as_float(lo), as_float(hi)]

    return TableContract(
        table=table,
        partition=partition,
        rows=frame.height,
        columns=list(frame.columns),
        dtypes={n: str(d) for n, d in zip(frame.columns, frame.dtypes, strict=True)},
        null_rates=null_rates,
        numeric_ranges=ranges,
        content_sha256=content_hash(frame),
    )


def write_contract(contract: TableContract, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{contract.table}__{contract.partition}.json"
    path.write_text(contract.to_json(), encoding="utf-8", newline="\n")
    return path


def verify_table(frame: pl.DataFrame, contract_path: Path) -> list[str]:
    """Return a list of human-readable drifts. Empty means the table matches."""
    stored = json.loads(contract_path.read_text(encoding="utf-8"))
    fresh = derive_contract(frame, table=stored["table"], partition=stored["partition"])
    problems: list[str] = []

    if fresh.rows != stored["rows"]:
        problems.append(f"rows: {stored['rows']} -> {fresh.rows}")
    if fresh.columns != stored["columns"]:
        added = sorted(set(fresh.columns) - set(stored["columns"]))
        removed = sorted(set(stored["columns"]) - set(fresh.columns))
        if added:
            problems.append(f"columns added: {added}")
        if removed:
            problems.append(f"columns removed: {removed}")
    for name, dtype in stored["dtypes"].items():
        if name in fresh.dtypes and fresh.dtypes[name] != dtype:
            problems.append(f"dtype {name}: {dtype} -> {fresh.dtypes[name]}")
    if fresh.content_sha256 != stored["content_sha256"]:
        problems.append(
            f"content hash: {stored['content_sha256'][:12]} -> {fresh.content_sha256[:12]}"
        )
    return problems
