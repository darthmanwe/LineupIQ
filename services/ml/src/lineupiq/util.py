"""Small shared helpers."""

from __future__ import annotations

import numpy as np
import polars as pl

from lineupiq.hashing import LINEUP_SIZE

__all__ = ["ABSENT_PLAYER", "as_float", "as_int", "lineup_slots"]

# Sentinel for an empty lineup slot. Player ids are positive, so it cannot collide.
ABSENT_PLAYER = -1


def as_float(value: object, default: float = 0.0) -> float:
    """Coerce a polars aggregate to a float.

    Polars types every aggregate as a broad union -- it may in principle return
    a date, a Decimal, or a list. Narrowing once here keeps call sites readable
    and turns a null aggregate on an empty frame into the default rather than a
    TypeError three layers up.
    """
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    return int(as_float(value, float(default)))


def lineup_slots(series: pl.Series, *, absent: int = ABSENT_PLAYER) -> list[np.ndarray]:
    """Read a ``List(Int64)`` lineup column as five flat integer arrays.

    Returns one ``(n,)`` array per lineup position, in the column's own order,
    with :data:`ABSENT_PLAYER` where a slot is null or the row's lineup is
    shorter than five.

    **Why this exists rather than ``series.to_list()``.** ``to_list()`` on 600k
    rows allocates 600k Python lists of five ints -- roughly 120 MB of small
    objects. Called once per column per cross-validation fold, the churn
    fragments the allocator badly enough that resident memory does not come
    back: the conversion model's peak climbed 1,097 MB to 2,604 MB across nine
    folds. ``list.get`` stays inside polars and returns a fixed-width primitive
    column, so nothing variable-length is ever handed to Python.

    Filtering out ``absent`` at the call site reproduces the old
    ``(lineup or [])`` semantics exactly: a null lineup yields five sentinels
    and therefore an empty list, and a short lineup yields exactly its real
    members, in order. Order is preserved because slot ``k`` is read from
    position ``k``, which matters -- the callers sum floats, and floating-point
    addition is not associative.
    """
    return [
        series.list.get(slot, null_on_oob=True).fill_null(absent).to_numpy()
        for slot in range(LINEUP_SIZE)
    ]
