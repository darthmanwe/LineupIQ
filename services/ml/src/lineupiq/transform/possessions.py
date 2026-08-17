"""Possession-grain facts, and why they matter more than shot-grain ones.

The shot model answers "will this shot go in?". That turned out to be almost
independent of who else is on the floor -- which is the correct answer to the
wrong question. Spacing does not make a player shoot better from the corner; it
gets him *more open corner threes instead of contested mid-range*. Lineup
effects live in shot **selection** and in what a possession is worth, not in
conversion once a shot is taken.

This module builds the layer where those effects can actually be measured: one
row per possession, with points scored, how the possession started, and the ten
players on the floor.

The lineups attached here are **ours**, reconstructed from play-by-play. The
upstream file ships its own lineup columns, and using those would delete the
hardest engineering in the project and replace a measurable claim with an
unmeasurable one. They are used instead as a second independent oracle: two
different implementations, from two different languages, disagreeing about the
same possession is a finding worth having.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from lineupiq.hashing import LINEUP_SIZE
from lineupiq.seasons import MODELLED_GAME_TYPES, Season, game_type_from_game_id
from lineupiq.util import as_float

__all__ = ["PossessionBuild", "build_possession_facts", "home_team_by_game"]

#: Possessions that began from a live-ball change of hands. Everything else
#: (deadball, timeout) starts against a set defence, which is a different game.
_LIVE_BALL_STARTS = ("OffMadeShot", "OffMissedShot", "OffLiveBallTurnover")

#: Seconds from possession start within which a shot counts as transition.
TRANSITION_SECONDS = 7


@dataclass(frozen=True)
class PossessionBuild:
    n_possessions: int
    n_with_lineup: int
    lineup_coverage: float
    #: Share of possessions where our five-man offensive lineup matches theirs.
    oracle_agreement: float
    #: The same, restricted to possessions that do NOT start on a substitution.
    #: This is the number that actually measures the reconstruction; the overall
    #: figure is dragged down by an ambiguity neither implementation can resolve.
    oracle_agreement_unambiguous: float
    n_oracle_compared: int
    #: Share of possessions beginning on the exact second of a substitution.
    boundary_ambiguous_rate: float
    transition_ppp: float
    halfcourt_ppp: float


def _lineup_hash_expr(column: str) -> pl.Expr:
    """Numeric sort, then stringify -- matching lineupiq.hashing.lineup_hash."""
    import hashlib

    def _md5(value: str | None) -> str | None:
        if value is None:
            return None
        return hashlib.md5(value.encode("ascii")).hexdigest()

    joined = pl.col(column).list.sort().cast(pl.List(pl.Utf8)).list.join(",")
    return (
        pl.when(pl.col(column).is_null() | (pl.col(column).list.len() != LINEUP_SIZE))
        .then(None)
        .otherwise(joined.map_elements(_md5, return_dtype=pl.Utf8))
    )


def home_team_by_game(events: pl.DataFrame) -> pl.DataFrame:
    """``game_id -> home_team_id``, derived from our own play-by-play.

    ``PERSON1TYPE == 4`` marks the home side, so the modal team id across that
    side's events identifies the home team without consulting any other feed.
    """
    return (
        events.filter((pl.col("PERSON1TYPE") == 4) & (pl.col("PLAYER1_TEAM_ID") > 0))
        .group_by("game_id")
        .agg(pl.col("PLAYER1_TEAM_ID").mode().first().alias("home_team_id"))
    )


def build_possession_facts(
    raw: pl.DataFrame, stints: pl.DataFrame, events: pl.DataFrame, season: Season
) -> tuple[pl.DataFrame, PossessionBuild]:
    """Join upstream possession outcomes to our reconstructed lineups.

    A possession is attributed to the stint that contains its start. The clock
    counts down, so a stint spans ``[end_seconds_remaining, start_seconds_remaining]``
    and a possession belongs to it when its start falls inside that window.
    """
    home_team = home_team_by_game(events)
    poss = raw.with_columns(
        pl.col("game_id").cast(pl.Utf8).str.strip_chars().str.zfill(10).alias("game_id"),
        pl.col("period").cast(pl.Int64),
        pl.col("start_seconds_remaining").cast(pl.Float64),
        pl.col("points").cast(pl.Int64),
    ).filter(pl.col("count_as_possession"))

    # Preseason rotations are not real rotations; all-star defence is not defence.
    keep = [
        gid
        for gid in poss["game_id"].unique().to_list()
        if game_type_from_game_id(gid) in MODELLED_GAME_TYPES
    ]
    poss = poss.filter(pl.col("game_id").is_in(keep))

    # Their lineups, kept only for the oracle comparison.
    oracle = poss.select(
        "game_id",
        "period",
        "possession_number",
        pl.concat_list([f"off_player_{i}" for i in range(1, 6)])
        .cast(pl.List(pl.Int64))
        .alias("oracle_off_lineup"),
    )

    slim = poss.select(
        "game_id",
        "period",
        "possession_number",
        "offense_team_id",
        "defense_team_id",
        "start_seconds_remaining",
        "end_seconds_remaining",
        "points",
        "possession_start_type",
        "fg2a",
        "fg3a",
        "fta",
        "tov",
        "oreb",
    )

    windows = stints.select(
        "game_id",
        "period",
        "stint_index",
        "start_seconds_remaining",
        "end_seconds_remaining",
        "home_lineup",
        "away_lineup",
        "stint_quality",
    ).rename(
        {
            "start_seconds_remaining": "stint_start",
            "end_seconds_remaining": "stint_end",
        }
    )

    # Attribute a possession to the stint containing its start.
    #
    # About 15% of possessions begin on the exact second a substitution happens,
    # and for those the attribution is genuinely ambiguous rather than merely
    # difficult. Measured against the independent oracle:
    #
    #     >30s into a stint   97.8% agreement
    #     6-30s               95.4%
    #     at a boundary       43.2%
    #
    # Three quarters of the disagreements differ by exactly one player -- the
    # signature of one substitution applied on different sides of one instant.
    # Midpoint attribution was tried and measured *worse* (89.0% vs 89.7%
    # overall), so this is not a rule that can be tuned into agreement: the two
    # implementations simply answer a genuinely ambiguous question differently.
    #
    # The honest response is to keep the simple rule and mark the ambiguity, so
    # a downstream model can exclude or downweight those possessions instead of
    # inheriting a coin flip it cannot see.
    joined = (
        slim.join(windows, on=["game_id", "period"], how="left")
        .filter(
            (pl.col("start_seconds_remaining") <= pl.col("stint_start"))
            & (pl.col("start_seconds_remaining") > pl.col("stint_end"))
        )
        .unique(subset=["game_id", "period", "possession_number"], keep="first")
    )

    # Which side is on offence is decided by TEAM ID, from our own event feed --
    # not by matching against the upstream lineups.
    #
    # An earlier version compared their offensive five to our home five and fell
    # through to "away" whenever it did not match. That silently mis-attributed
    # every possession where either side was wrong, and it made our facts depend
    # on the very file we are trying to check ourselves against. Deciding by team
    # id keeps the oracle genuinely independent.
    joined = joined.join(home_team, on="game_id", how="left").join(
        oracle, on=["game_id", "period", "possession_number"], how="left"
    )

    joined = joined.with_columns(
        (pl.col("offense_team_id") == pl.col("home_team_id")).alias("_off_is_home")
    )

    facts = (
        joined.with_columns(
            # A possession whose offence we cannot place is nulled and flagged,
            # never assigned to a default side.
            pl.when(pl.col("_off_is_home").is_null())
            .then(None)
            .when(pl.col("_off_is_home"))
            .then(pl.col("home_lineup"))
            .otherwise(pl.col("away_lineup"))
            .alias("off_lineup"),
            pl.when(pl.col("_off_is_home").is_null())
            .then(None)
            .when(pl.col("_off_is_home"))
            .then(pl.col("away_lineup"))
            .otherwise(pl.col("home_lineup"))
            .alias("def_lineup"),
        )
        .with_columns(
            _lineup_hash_expr("off_lineup").alias("off_lineup_hash"),
            _lineup_hash_expr("def_lineup").alias("def_lineup_hash"),
            pl.col("possession_start_type").is_in(_LIVE_BALL_STARTS).alias("live_ball_start"),
            # True when the possession begins at the exact instant the stint
            # does -- i.e. on a substitution. Attribution there is a coin flip
            # between two defensible answers, so it is flagged rather than
            # silently trusted.
            (pl.col("start_seconds_remaining") >= pl.col("stint_start")).alias(
                "boundary_ambiguous"
            ),
            pl.lit(season.start_year).cast(pl.Int64).alias("season"),
            (pl.col("start_seconds_remaining") - pl.col("end_seconds_remaining")).alias(
                "possession_seconds"
            ),
        )
        .drop("_off_is_home", "stint_start", "stint_end")
        .sort(["game_id", "period", "possession_number"])
    )

    report = _summarise(facts)
    return facts, report


def _summarise(facts: pl.DataFrame) -> PossessionBuild:
    n = facts.height
    with_lineup = facts.filter(pl.col("off_lineup_hash").is_not_null()).height

    # The real comparison: our *offensive* five against theirs, on the same
    # possession. Asking only whether their five matched either of our two would
    # pass even when we had the sides backwards.
    compared = facts.filter(
        pl.col("oracle_off_lineup").is_not_null() & pl.col("off_lineup").is_not_null()
    )
    if compared.height:
        agree = compared.with_columns(
            pl.col("oracle_off_lineup")
            .list.sort()
            .eq(pl.col("off_lineup").list.sort())
            .alias("_match")
        )
        oracle_agreement = as_float(agree["_match"].mean())
        clean = agree.filter(~pl.col("boundary_ambiguous"))
        unambiguous = as_float(clean["_match"].mean()) if clean.height else 0.0
    else:
        oracle_agreement = 0.0
        unambiguous = 0.0

    # Transition is worth materially more than half-court. If our possession
    # classification is right, this gap should reproduce the well-known one --
    # computed, not quoted.
    live = facts.filter(
        pl.col("live_ball_start") & (pl.col("possession_seconds") <= TRANSITION_SECONDS)
    )
    half = facts.filter(
        ~pl.col("live_ball_start") | (pl.col("possession_seconds") > TRANSITION_SECONDS)
    )

    return PossessionBuild(
        n_possessions=n,
        n_with_lineup=with_lineup,
        lineup_coverage=with_lineup / n if n else 0.0,
        oracle_agreement=oracle_agreement,
        oracle_agreement_unambiguous=unambiguous,
        n_oracle_compared=compared.height,
        boundary_ambiguous_rate=as_float(facts["boundary_ambiguous"].mean()),
        transition_ppp=as_float(live["points"].mean()),
        halfcourt_ppp=as_float(half["points"].mean()),
    )
