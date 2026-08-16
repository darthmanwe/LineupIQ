"""Order-invariant lineup identity.

Three engines have to agree on this byte for byte: Python (which builds gold),
DuckDB (which serves local queries), and Snowflake (the optional adapter). If
any of them disagrees, joins on lineup identity silently return nothing --
no error, just an empty result that looks like "these five never played
together".

The original design specifies::

    MD5(ARRAY_TO_STRING(ARRAY_SORT(player_id_array), ','))

which is right, with one correction it does not make: **the sort has to be
numeric.** Sorting the string forms puts ``1630552`` before ``201143``, because
``'1' < '2'``. Modern player ids are 7 digits and older ones are 6, so a mixed
lineup -- which is most of them -- hashes differently under the two orderings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

__all__ = ["LINEUP_SIZE", "canonical_lineup", "lineup_hash"]

LINEUP_SIZE = 5


def canonical_lineup(player_ids: Iterable[int | str]) -> tuple[int, ...]:
    """Validate and numerically sort five distinct player ids.

    Raises rather than coercing. A four-man or duplicated lineup is a bug
    upstream, and quietly padding or de-duplicating it would bury that bug in a
    hash that no longer means anything.
    """
    try:
        ordered = sorted(int(p) for p in player_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"player ids must be integer-valued: {exc}") from exc

    if len(ordered) != LINEUP_SIZE:
        raise ValueError(f"lineup must have exactly {LINEUP_SIZE} players, got {len(ordered)}")
    if len(set(ordered)) != LINEUP_SIZE:
        raise ValueError(f"lineup has duplicate players: {ordered}")
    if any(p <= 0 for p in ordered):
        raise ValueError(f"player ids must be positive: {ordered}")
    return tuple(ordered)


def lineup_hash(player_ids: Iterable[int | str]) -> str:
    """Order-invariant 32-character hex identity for a five-man lineup.

    Byte-identical to DuckDB's
    ``md5(array_to_string(list_sort(ids), ','))`` and Snowflake's
    ``MD5(ARRAY_TO_STRING(ARRAY_SORT(ids), ','))`` **when the ids are sorted as
    numbers on both sides**.

    Not a security boundary -- MD5 is chosen because it is what the warehouses
    expose, and cross-engine agreement is the whole point.
    """
    ordered = canonical_lineup(player_ids)
    joined = ",".join(str(p) for p in ordered)
    return hashlib.md5(joined.encode("ascii")).hexdigest()
