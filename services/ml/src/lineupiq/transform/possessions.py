"""Possession-grain facts, and why they matter more than shot-grain ones.

The shot model answers "will this shot go in?". That turned out to be almost
independent of who else is on the floor -- which is the correct answer to the
wrong question. Spacing does not make a player shoot better from the corner; it
gets him *more open corner threes instead of contested mid-range*. Lineup
effects live in shot **selection** and in what a possession is worth, not in
conversion once a shot is taken.

This module builds the layer where those effects can actually be measured: one
row per possession, with points scored, how the possession started, how long it
lasted, and the ten players on the floor.

The lineups attached here are **ours**, reconstructed from play-by-play. The
upstream file ships its own lineup columns, and using those would delete the
hardest engineering in the project and replace a measurable claim with an
unmeasurable one. They are used instead as a second independent oracle: two
different implementations, from two different languages, disagreeing about the
same possession is a finding worth having.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from lineupiq.hashing import LINEUP_SIZE
from lineupiq.seasons import MODELLED_GAME_TYPES, Season, game_type_from_game_id
from lineupiq.util import as_float

__all__ = [
    "LIVE_BALL_STARTS",
    "OVERTIME_PERIOD_SECONDS",
    "REGULATION_PERIOD_SECONDS",
    "TRANSITION_SECONDS",
    "PossessionBuild",
    "build_possession_facts",
    "home_team_by_game",
    "possession_windows",
]

#: Period lengths, used to open the clock on a period's first possession.
REGULATION_PERIOD_SECONDS = 720.0
OVERTIME_PERIOD_SECONDS = 300.0

#: Possession starts where the ball was already live: a defensive rebound or a
#: steal. Everything else is inbounded against a defence that had time to set.
#:
#: ``OffMadeShot`` was originally in this tuple and the data rejected it. Median
#: possession length by start type:
#:
#:     OffLiveBallTurnover   7s     <- live
#:     OffMissedShot        10s     <- live
#:     OffDeadball          16s
#:     OffMadeShot          17s
#:     OffTimeout           18s
#:
#: A possession beginning after the opponent scores is inbounded from the
#: baseline with the clock stopped, and it behaves exactly like a timeout. The
#: split above is not a judgement call; it is where the durations separate.
LIVE_BALL_STARTS: tuple[str, ...] = ("OffMissedShot", "OffLiveBallTurnover")

#: Seconds from the change of hands within which a possession counts as
#: transition.
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
    #: Derived possession length. Published because it is checkable against a
    #: number everyone already knows: NBA possessions average about 14 seconds.
    mean_possession_seconds: float
    median_possession_seconds: float
    transition_ppp: float
    halfcourt_ppp: float
    #: Points per possession by how the possession began. Start type is fixed
    #: before the offence does anything, so unlike duration it cannot be
    #: contaminated by the outcome.
    ppp_by_start_type: dict[str, float] = field(default_factory=dict)


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


def possession_windows(raw: pl.DataFrame) -> pl.DataFrame:
    """Derive when each possession actually began, and how long it lasted.

    The feed's ``start_seconds_remaining`` is the clock at the possession's
    first *recorded event*, not at the change of hands. A team that secures a
    defensive rebound at 9:35 and takes its first shot at 9:15 is recorded as
    starting at 9:15, so the twenty seconds it spent holding the ball belong to
    no possession at all. Concretely, from the first period of the first game in
    the corpus:

        possession 1   720 -> 695   (ends on a turnover)
        possession 2   675 -> 675   (the other team, one recorded event)

    Twenty seconds are missing between them, and the second possession's
    recorded window is a single instant. Across the corpus **45% of possessions
    had start == end**, which made ``start - end`` useless as a duration and
    made 4.4% of shots fall into a gap between windows.

    The change of hands is the previous possession's last recorded event, so
    that is what the true start is taken from -- and the period's opening clock
    for the first possession of each period. This is not free of assumptions,
    but it is checkable, and it checks out: the resulting durations have a
    median of 14s and a mean of 14.7s, which is the possession length the NBA
    has been playing at for years. The feed's own arithmetic gave a median of 2s.

    Ordering uses ``number_in_period`` over **every** row, including the handful
    that ``count_as_possession`` excludes. Filtering first would splice the
    dropped possession's time onto its neighbour.
    """
    ordered = raw.sort(["game_id", "period", "number_in_period"])
    return ordered.with_columns(
        pl.coalesce(
            pl.col("end_seconds_remaining").shift(1).over(["game_id", "period"]),
            pl.when(pl.col("period") <= 4)
            .then(REGULATION_PERIOD_SECONDS)
            .otherwise(OVERTIME_PERIOD_SECONDS),
        ).alias("possession_start_clock")
    ).with_columns(
        (pl.col("possession_start_clock") - pl.col("end_seconds_remaining")).alias(
            "possession_seconds"
        )
    )


def build_possession_facts(
    raw: pl.DataFrame, stints: pl.DataFrame, events: pl.DataFrame, season: Season
) -> tuple[pl.DataFrame, PossessionBuild]:
    """Join upstream possession outcomes to our reconstructed lineups.

    A possession is attributed to the stint that contains the change of hands.
    The clock counts down, so a stint spans
    ``[end_seconds_remaining, start_seconds_remaining]`` and a possession
    belongs to it when its start falls inside that window.
    """
    home_team = home_team_by_game(events)
    typed = raw.with_columns(
        pl.col("game_id").cast(pl.Utf8).str.strip_chars().str.zfill(10).alias("game_id"),
        pl.col("period").cast(pl.Int64),
        pl.col("number_in_period").cast(pl.Int64),
        pl.col("start_seconds_remaining").cast(pl.Float64),
        pl.col("end_seconds_remaining").cast(pl.Float64),
        pl.col("points").cast(pl.Int64),
    )

    # The window has to be derived before any row is dropped, then the drops
    # applied. Preseason rotations are not real rotations; all-star defence is
    # not defence.
    keep = [
        gid
        for gid in typed["game_id"].unique().to_list()
        if game_type_from_game_id(gid) in MODELLED_GAME_TYPES
    ]
    poss = (
        possession_windows(typed.filter(pl.col("game_id").is_in(keep)))
        .filter(pl.col("count_as_possession"))
        .rename({"start_seconds_remaining": "first_event_seconds_remaining"})
    )

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
        "number_in_period",
        "offense_team_id",
        "defense_team_id",
        "possession_start_clock",
        "first_event_seconds_remaining",
        "end_seconds_remaining",
        "possession_seconds",
        "points",
        "possession_start_type",
        "is_second_chance",
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

    # Attribute a possession to the stint containing the change of hands.
    #
    # Roughly 9% of possessions begin on the exact second a substitution
    # happens, and for those the attribution is genuinely ambiguous rather than
    # merely difficult. Measured against the independent oracle, agreement is
    # 97.4% away from a substitution boundary and far lower on it, and three
    # quarters of the disagreements differ by exactly one player -- the
    # signature of one substitution applied on different sides of one instant.
    #
    # Midpoint attribution was tried and measured *worse*, so this is not a rule
    # that can be tuned into agreement. The honest response is to keep the
    # simple rule and mark the ambiguity, so a downstream model can exclude or
    # downweight those possessions instead of inheriting a coin flip it cannot
    # see.
    joined = (
        slim.join(windows, on=["game_id", "period"], how="left")
        .filter(
            (pl.col("possession_start_clock") <= pl.col("stint_start"))
            & (pl.col("possession_start_clock") > pl.col("stint_end"))
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
            pl.col("possession_start_type").is_in(LIVE_BALL_STARTS).alias("live_ball_start"),
            # True when the possession begins at the exact instant the stint
            # does -- i.e. on a substitution. Attribution there is a coin flip
            # between two defensible answers, so it is flagged rather than
            # silently trusted.
            (pl.col("possession_start_clock") >= pl.col("stint_start")).alias("boundary_ambiguous"),
            pl.lit(season.start_year).cast(pl.Int64).alias("season"),
        )
        .with_columns(
            (
                pl.col("live_ball_start") & (pl.col("possession_seconds") <= TRANSITION_SECONDS)
            ).alias("transition")
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

    # Transition is worth materially more than half-court. If the possession
    # classification is right, this gap should reproduce the well-known one --
    # computed, not quoted.
    #
    # Duration is partly determined by the outcome: a made shot ends the
    # possession at the shot, a miss ends it at the rebound a second or two
    # later, so short possessions over-represent makes. That bias inflates
    # transition PPP, which is why ``ppp_by_start_type`` is published beside it
    # -- start type is fixed before the offence does anything and cannot be
    # contaminated the same way.
    live = facts.filter(pl.col("transition"))
    half = facts.filter(~pl.col("transition"))

    by_start = {
        str(row[0]): as_float(row[1])
        for row in facts.group_by("possession_start_type")
        .agg(pl.col("points").mean())
        .sort("possession_start_type")
        .iter_rows()
    }

    return PossessionBuild(
        n_possessions=n,
        n_with_lineup=with_lineup,
        lineup_coverage=with_lineup / n if n else 0.0,
        oracle_agreement=oracle_agreement,
        oracle_agreement_unambiguous=unambiguous,
        n_oracle_compared=compared.height,
        boundary_ambiguous_rate=as_float(facts["boundary_ambiguous"].mean()),
        mean_possession_seconds=as_float(facts["possession_seconds"].mean()),
        median_possession_seconds=as_float(facts["possession_seconds"].median()),
        transition_ppp=as_float(live["points"].mean()),
        halfcourt_ppp=as_float(half["points"].mean()),
        ppp_by_start_type=by_start,
    )
