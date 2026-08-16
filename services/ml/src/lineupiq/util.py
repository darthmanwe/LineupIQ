"""Small shared helpers."""

from __future__ import annotations

__all__ = ["as_float", "as_int"]


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
