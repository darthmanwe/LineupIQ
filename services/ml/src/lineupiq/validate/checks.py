"""Data-quality gates.

Every check here exists because of a specific way this data can be wrong while
still looking fine. A gate that cannot fail is decoration, so each one states
the threshold it is measured against and what the failure would have meant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import polars as pl

from lineupiq.hashing import LINEUP_SIZE, lineup_hash
from lineupiq.seasons import Season, season_from_game_id
from lineupiq.util import as_float

__all__ = ["Gate", "GateResult", "run_gates"]

Severity = Literal["blocking", "reported"]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    measured: float
    threshold: float
    comparison: str
    detail: str
    severity: Severity

    @property
    def verdict(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.severity == "blocking" else "WARN"


@dataclass(frozen=True)
class Gate:
    name: str
    threshold: float
    #: "min" -- measured must be >= threshold. "max" -- measured must be <=.
    comparison: Literal["min", "max"]
    detail: str
    severity: Severity
    measure: Callable[[dict[str, pl.DataFrame]], float]

    def run(self, tables: dict[str, pl.DataFrame]) -> GateResult:
        measured = self.measure(tables)
        passed = (
            measured >= self.threshold if self.comparison == "min" else measured <= self.threshold
        )
        return GateResult(
            name=self.name,
            passed=passed,
            measured=measured,
            threshold=self.threshold,
            comparison=self.comparison,
            detail=self.detail,
            severity=self.severity,
        )


# --- measures --------------------------------------------------------------


def _valid_stints_have_five(t: dict[str, pl.DataFrame]) -> float:
    stints = t.get("stints")
    if stints is None or stints.is_empty():
        return 0.0
    valid = stints.filter(pl.col("stint_quality") == "VALID")
    if valid.is_empty():
        return 0.0
    ok = valid.filter(
        (pl.col("home_lineup").list.len() == LINEUP_SIZE)
        & (pl.col("away_lineup").list.len() == LINEUP_SIZE)
    )
    return ok.height / valid.height


def _shot_lineup_coverage(t: dict[str, pl.DataFrame]) -> float:
    shots = t.get("shot_facts")
    if shots is None or shots.is_empty():
        return 0.0
    return shots.filter(pl.col("lineup_for_hash").is_not_null()).height / shots.height


def _shots_have_shooter(t: dict[str, pl.DataFrame]) -> float:
    shots = t.get("shot_facts")
    if shots is None or shots.is_empty():
        return 0.0
    return (
        shots.filter(pl.col("shooter_id").is_not_null() & (pl.col("shooter_id") > 0)).height
        / shots.height
    )


def _three_point_agreement(t: dict[str, pl.DataFrame]) -> float:
    shots = t.get("shot_facts")
    if shots is None or shots.is_empty() or "shot_type_raw" not in shots.columns:
        return 0.0
    scored = shots.filter(pl.col("shot_type_raw").is_not_null())
    if scored.is_empty():
        return 0.0
    agree = scored.with_columns(
        (pl.col("shot_type_raw").str.contains("3PT") == pl.col("is_three")).alias("_a")
    )
    return as_float(agree["_a"].mean())


def _season_prefix_consistent(t: dict[str, pl.DataFrame]) -> float:
    """Every GAME_ID must decode to the season the partition claims.

    This is the guard against the two mirrors' conflicting filename
    conventions. A failure here means the whole partition is off by a year, and
    every number computed from it is about the wrong season.
    """
    shots = t.get("shot_facts")
    if shots is None or shots.is_empty() or "season" not in shots.columns:
        return 0.0
    # Compare each game against the season *its own row* claims. Checking every
    # row against one declared value would read as a 67% failure the moment
    # more than one season is loaded, which is a bug in the check rather than
    # in the data.
    pairs = shots.select("game_id", "season").unique()
    if pairs.is_empty():
        return 0.0
    ok = sum(
        1
        for gid, declared in pairs.iter_rows()
        if season_from_game_id(gid) == Season(int(declared))
    )
    return ok / pairs.height


def _no_nan_or_inf(t: dict[str, pl.DataFrame]) -> float:
    """Share of numeric gold cells that are finite. Must be exactly 1.0."""
    shots = t.get("shot_facts")
    if shots is None or shots.is_empty():
        return 0.0
    bad = 0
    total = 0
    for name, dtype in zip(shots.columns, shots.dtypes, strict=True):
        if not dtype.is_float():
            continue
        series = shots[name]
        total += series.len()
        bad += int(series.is_infinite().sum() or 0) + int(series.is_nan().sum() or 0)
    return 1.0 if total == 0 else (total - bad) / total


def _hash_order_invariant(t: dict[str, pl.DataFrame]) -> float:
    """Recompute a sample of lineup hashes from the raw ids, shuffled.

    Guards the one property every join on lineup identity depends on. A
    regression here returns zero rows everywhere rather than an error.
    """
    shots = t.get("shot_facts")
    if shots is None or shots.is_empty():
        return 0.0
    sample = (
        shots.filter(pl.col("lineup_for_hash").is_not_null())
        .select("lineup_for", "lineup_for_hash")
        .head(2000)
    )
    if sample.is_empty():
        return 0.0
    ok = 0
    for ids, stored in sample.iter_rows():
        if lineup_hash(reversed(list(ids))) == stored:
            ok += 1
    return ok / sample.height


def _stint_durations_positive(t: dict[str, pl.DataFrame]) -> float:
    stints = t.get("stints")
    if stints is None or stints.is_empty():
        return 0.0
    return stints.filter(pl.col("duration_seconds") > 0).height / stints.height


GATES: tuple[Gate, ...] = (
    Gate(
        "valid_stints_have_five_per_team",
        1.0,
        "min",
        "A stint flagged VALID must have exactly five players on each side.",
        "blocking",
        _valid_stints_have_five,
    ),
    Gate(
        "shot_lineup_coverage",
        0.99,
        "min",
        "Share of shots that resolved to a complete five-man lineup.",
        "blocking",
        _shot_lineup_coverage,
    ),
    Gate(
        "every_shot_has_a_shooter",
        1.0,
        "min",
        "A shot with no shooter cannot be attributed and must not exist.",
        "blocking",
        _shots_have_shooter,
    ),
    Gate(
        "derived_three_matches_feed",
        0.99,
        "min",
        "Our arc geometry against the feed's own 2PT/3PT label -- an independent check.",
        "blocking",
        _three_point_agreement,
    ),
    Gate(
        "season_matches_game_id_prefix",
        1.0,
        "min",
        "Every GAME_ID decodes to the season the partition claims.",
        "blocking",
        _season_prefix_consistent,
    ),
    Gate(
        "no_nan_or_inf_in_gold",
        1.0,
        "min",
        "Non-finite floats poison every downstream aggregate silently.",
        "blocking",
        _no_nan_or_inf,
    ),
    Gate(
        "lineup_hash_order_invariant",
        1.0,
        "min",
        "Recomputed hashes from shuffled ids must match what was stored.",
        "blocking",
        _hash_order_invariant,
    ),
    Gate(
        "stint_durations_positive",
        1.0,
        "min",
        "A zero or negative stint duration means the clock ran backwards.",
        "blocking",
        _stint_durations_positive,
    ),
)


def run_gates(tables: dict[str, pl.DataFrame]) -> list[GateResult]:
    return [gate.run(tables) for gate in GATES]
