"""Backtesting the trade projection against moves that actually happened.

This is the milestone the original design document had no plan for. It makes one
falsifiable claim -- *this swap changes the receiving team by that much* -- and
checks it three ways, each weaker than the last and each reported.

**Training is strictly pre-move.** For every evaluated move, the player effects
come from possessions played before the mover's first game for his new team.
Nothing after the cutoff enters the fit, for any team.

**Difference-in-differences, not a raw before/after.** Teams that trade are
usually underperforming, so their rating tends to improve afterwards whether the
trade helped or not -- regression to the mean plus the schedule. The comparison
is against teams that made no move over the same stretch of games.

**Placebo moves calibrate the noise.** The identical machinery is run on players
who did *not* move, pretending they "arrived" at their own team on a matched
date. The projected and observed deltas there must be indistinguishable from
zero. Whatever magnitude the placebos produce is the floor below which a real
result means nothing, and it is published beside the real one.

The power analysis is computed **before** any of this and printed first. At
these sample sizes the expected honest answer is that no accuracy claim is
supported, and that has to be a pre-commitment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from lineupiq.models.moves import (
    Move,
    detect_moves,
    power_analysis,
    residual_sd_of_team_rating_change,
    team_game_ratings,
)
from lineupiq.models.rapm import (
    LAMBDA_GRID,
    build_rapm_design,
    select_lambda,
    usable_possessions,
)
from lineupiq.models.trade import (
    LEAGUE_AVERAGE_REPLACEMENT,
    MINUTES_RULES,
    MinutesRule,
    project_swap,
    variance_decomposition,
)
from lineupiq.util import as_float

__all__ = [
    "MIN_GAMES_EITHER_SIDE",
    "BacktestResult",
    "MoveOutcome",
    "run_trade_backtest",
]

#: Games the receiving team needs on each side of the move for its rating change
#: to be worth differencing at all.
MIN_GAMES_EITHER_SIDE = 15

#: Number of cutoff buckets. One RAPM fit per bucket rather than per move; each
#: bucket trains on possessions before the *earliest* cutoff it contains, so no
#: move in a bucket ever sees data from after its own arrival. Later moves in a
#: bucket therefore train on slightly less data than they could -- conservative
#: in the right direction.
N_CUTOFF_BUCKETS = 6


@dataclass(frozen=True)
class MoveOutcome:
    """One move, projected and observed."""

    player_id: int
    to_team_id: int
    season: int
    kind: str
    cutoff_game: str
    games_before: int
    games_after: int
    projected_delta: float
    projected_se: float
    observed_delta: float
    #: Observed delta minus the mean delta of teams that made no move.
    did_delta: float
    minutes_variance_share: float
    is_placebo: bool = False


@dataclass
class BacktestResult:
    """Everything the backtest produces, including the reasons to doubt it."""

    minutes_rule: str
    n_moves: int
    n_placebo: int
    residual_sd: float
    power: dict[str, Any] = field(default_factory=dict)
    real: dict[str, Any] = field(default_factory=dict)
    placebo: dict[str, Any] = field(default_factory=dict)
    variance: dict[str, float] = field(default_factory=dict)
    outcomes: list[MoveOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _team_window(
    ratings: pl.DataFrame, team_id: int, season: int, cutoff_game: str
) -> tuple[float, float, int, int]:
    """Mean net rating before and after a cutoff, for one team-season."""
    block = ratings.filter((pl.col("team_id") == team_id) & (pl.col("season") == season))
    before = block.filter(pl.col("game_id") < cutoff_game)
    after = block.filter(pl.col("game_id") >= cutoff_game)
    return (
        as_float(before["net_rating"].mean()),
        as_float(after["net_rating"].mean()),
        before.height,
        after.height,
    )


def _control_delta(
    ratings: pl.DataFrame, season: int, cutoff_game: str, exclude: set[int]
) -> float:
    """Mean before/after change for teams that made no move at this cutoff.

    This is the counterfactual the real delta is measured against. Without it,
    a trading team's improvement is indistinguishable from the regression to the
    mean that made it trade in the first place.
    """
    deltas: list[float] = []
    # Sorted, because the mean below is a left-to-right float sum and `unique()`
    # does not promise an order. Not a structural bug like the placebo pool -- the
    # same teams are always included -- but it made the control delta differ in
    # the last places between machines, and every real move's DiD is measured
    # against it.
    for team in sorted(ratings.filter(pl.col("season") == season)["team_id"].unique().to_list()):
        if int(team) in exclude:
            continue
        before, after, n_before, n_after = _team_window(ratings, int(team), season, cutoff_game)
        if n_before >= MIN_GAMES_EITHER_SIDE and n_after >= MIN_GAMES_EITHER_SIDE:
            deltas.append(after - before)
    return float(np.mean(deltas)) if deltas else 0.0


def _bucket_cutoffs(moves: list[Move], n_buckets: int) -> list[tuple[str, list[Move]]]:
    """Group moves into buckets, each labelled by its earliest cutoff game."""
    ordered = sorted(moves, key=lambda m: m.first_game_after)
    if not ordered:
        return []
    buckets = np.array_split(np.arange(len(ordered)), min(n_buckets, len(ordered)))
    grouped: list[tuple[str, list[Move]]] = []
    for bucket in buckets:
        if not len(bucket):
            continue
        members = [ordered[i] for i in bucket]
        grouped.append((members[0].first_game_after, members))
    return grouped


def _fit_pre_cutoff(
    possessions: pl.DataFrame,
    cutoff_game: str,
    lambda_offence: float,
    lambda_defence: float,
) -> object:
    """Fit RAPM on possessions strictly before ``cutoff_game``.

    One filter does all the work required, for every team and every player:
    nothing at or after the mover's first game for his new side enters the fit.
    The mover's *pre*-move possessions must stay in -- they are the only evidence
    his coefficient has, and dropping them would leave the projection with
    nothing to project from.
    """
    from lineupiq.models.rapm import RapmReport, _fit_from_design, co_occurrence_report

    pre = usable_possessions(possessions).filter(pl.col("game_id") < cutoff_game)
    design = build_rapm_design(pre)
    solution, intercept = _fit_from_design(
        design, None, lambda_offence, lambda_defence, with_covariance=True
    )

    n = design.n_players
    scale = 100.0
    errors = solution.standard_errors
    from lineupiq.models.rapm import RapmFit

    fit = RapmFit(
        players=design.players,
        off_rapm={p: float(solution.coefficients[i] * scale) for i, p in enumerate(design.players)},
        def_rapm={
            p: float(-solution.coefficients[n + i] * scale) for i, p in enumerate(design.players)
        },
        home_advantage=float(solution.coefficients[-1] * scale),
        league_ppp=intercept,
        lambda_offence=lambda_offence,
        lambda_defence=lambda_defence,
        effective_df=solution.effective_df,
        condition_number=solution.condition_number,
        n_possessions=design.n_possessions,
        cv_mse=float("nan"),
        off_se=(
            {p: float(errors[i] * scale) for i, p in enumerate(design.players)}
            if errors is not None
            else {}
        ),
        def_se=(
            {p: float(errors[n + i] * scale) for i, p in enumerate(design.players)}
            if errors is not None
            else {}
        ),
    )
    return RapmReport(
        fit=fit,
        appearances=design.appearances,
        reliability={},
        co_occurrence=co_occurrence_report(design),
        lambda_trace=[],
        covariance=solution.covariance,
        column_index={p: i for i, p in enumerate(design.players)},
    )


def _player_possessions(possessions: pl.DataFrame, player_id: int, cutoff_game: str) -> int:
    """Offensive possessions the player was on the floor for, before the cutoff."""
    return (
        possessions.filter(pl.col("game_id") < cutoff_game)
        .select(pl.col("off_lineup").list.contains(player_id).sum())
        .item()
        or 0
    )


def _team_possessions(possessions: pl.DataFrame, team_id: int, season: int) -> int:
    return possessions.filter(
        (pl.col("offense_team_id") == team_id) & (pl.col("season") == season)
    ).height


def run_trade_backtest(
    possessions: pl.DataFrame,
    *,
    rule: MinutesRule | None = None,
    n_buckets: int = N_CUTOFF_BUCKETS,
    seed: int = 0,
    progress: Any = None,
) -> BacktestResult:
    """Project and score every evaluable move, plus a matched placebo set."""
    rule = rule or MINUTES_RULES[0]

    def report(message: str) -> None:
        if callable(progress):
            progress(message)

    ratings = team_game_ratings(possessions)
    residual_sd = residual_sd_of_team_rating_change(ratings)

    moves = [m for m in detect_moves(possessions) if m.mid_season]
    evaluable: list[Move] = []
    for move in moves:
        _, _, n_before, n_after = _team_window(
            ratings, move.to_team_id, move.season_after, move.first_game_after
        )
        if n_before >= MIN_GAMES_EITHER_SIDE and n_after >= MIN_GAMES_EITHER_SIDE:
            evaluable.append(move)

    result = BacktestResult(
        minutes_rule=rule.name,
        n_moves=len(evaluable),
        n_placebo=0,
        residual_sd=residual_sd,
    )

    # --- power, computed and recorded BEFORE anything is projected ---------
    analysis = power_analysis(len(evaluable), residual_sd)
    result.power = {
        "n": analysis.n,
        "residual_sd": analysis.residual_sd,
        "mde": analysis.mde,
        "alpha": analysis.alpha,
        "target_power": analysis.power,
        "sign_accuracy_ci_half_width": analysis.sign_accuracy_ci_half_width,
        "verdict": analysis.verdict,
        "description": analysis.describe(),
    }
    report(f"power: {analysis.describe()}")
    if not evaluable:
        result.notes.append("no evaluable mid-season moves at this sample")
        return result

    # Lambda is selected once, on the earliest season only, so that no move's
    # projection is tuned using data from after its own cutoff.
    first_season = int(min(m.season_after for m in evaluable))
    early = usable_possessions(possessions).filter(pl.col("season") < first_season)
    if early.height < 50_000:
        early = usable_possessions(possessions).filter(pl.col("season") == first_season)
        result.notes.append(
            "lambda selected on the first modelled season rather than a strictly earlier "
            "one; there is no earlier season in the corpus"
        )
    lambda_offence, lambda_defence, _, _ = select_lambda(
        build_rapm_design(early), n_folds=3, seed=seed, grid=LAMBDA_GRID
    )
    report(f"lambda selected on pre-period: off={lambda_offence:.0f} def={lambda_defence:.0f}")

    rng = np.random.default_rng(seed)
    outcomes: list[MoveOutcome] = []
    projections = []

    for cutoff, members in _bucket_cutoffs(evaluable, n_buckets):
        movers = {m.player_id for m in members}
        report(f"bucket cutoff {cutoff}: {len(members)} moves")
        fitted = _fit_pre_cutoff(possessions, cutoff, lambda_offence, lambda_defence)

        for move in members:
            before, after, n_before, n_after = _team_window(
                ratings, move.to_team_id, move.season_after, move.first_game_after
            )
            control = _control_delta(
                ratings, move.season_after, move.first_game_after, {move.to_team_id}
            )

            # The departing player is unknown without a transactions feed, so
            # the counterfactual is against a league-average replacement: the
            # arriving player displaces the marginal rotation player, whose
            # effect is zero by the ridge's own centring. Stating that is more
            # honest than inventing a specific departure.
            vacated = max(move.possessions_after, 1)
            team_total = _team_possessions(possessions, move.to_team_id, move.season_after)
            projection = project_swap(
                fitted,  # type: ignore[arg-type]
                player_in=move.player_id,
                player_out=LEAGUE_AVERAGE_REPLACEMENT,
                rule=rule,
                possessions_vacated=vacated,
                team_possessions=max(team_total, 1),
            )
            projections.append(projection)
            outcomes.append(
                MoveOutcome(
                    player_id=move.player_id,
                    to_team_id=move.to_team_id,
                    season=move.season_after,
                    kind=move.kind,
                    cutoff_game=move.first_game_after,
                    games_before=n_before,
                    games_after=n_after,
                    projected_delta=projection.delta_net,
                    projected_se=projection.se_total,
                    observed_delta=after - before,
                    did_delta=(after - before) - control,
                    minutes_variance_share=projection.minutes_variance_share,
                )
            )

        # --- placebo: the same machinery on players who did not move --------
        incumbents = (
            usable_possessions(possessions)
            .filter(pl.col("game_id") < cutoff)
            .select(pl.col("off_lineup").alias("player_id"))
            .explode("player_id", empty_as_null=True)
            .drop_nulls("player_id")
            .group_by("player_id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= 500)
            # **The sort is load-bearing, and its absence was a real bug.**
            #
            # `group_by` makes no ordering promise, so this list arrived in
            # whatever order the parallel aggregation produced -- and `rng.choice`
            # below then drew a *different set of players* on a machine with a
            # different core count. The seed made the draw look reproducible while
            # the population it drew from was not. It surfaced as `n_placebo`
            # moving from 64 to 66 between two runs on the same machine, which is
            # not a rounding difference: it is a different experiment.
            #
            # `player_id` is unique after the aggregation, so this is a total order.
            .sort("player_id")
        )
        candidates = [p for p in incumbents["player_id"].to_list() if int(p) not in movers]
        if candidates:
            chosen = rng.choice(candidates, size=min(len(members), len(candidates)), replace=False)
            for player in chosen:
                # The team-season this player belongs to, defined rather than
                # picked. The previous version took `.head(1)` of an unordered
                # filter, which is an arbitrary row -- and since a player appears
                # across seasons, that could attribute him to a season he barely
                # played in. "Where most of his possessions were" is both
                # deterministic and the answer the question actually wants.
                team = (
                    usable_possessions(possessions)
                    .filter(
                        (pl.col("game_id") < cutoff)
                        & pl.col("off_lineup").list.contains(int(player))
                    )
                    .group_by("offense_team_id", "season")
                    .agg(pl.len().alias("n"))
                    .sort(["n", "season", "offense_team_id"], descending=[True, True, False])
                    .head(1)
                )
                if team.is_empty():
                    continue
                team_id = int(team["offense_team_id"].item())
                season = int(team["season"].item())
                before, after, n_before, n_after = _team_window(ratings, team_id, season, cutoff)
                if n_before < MIN_GAMES_EITHER_SIDE or n_after < MIN_GAMES_EITHER_SIDE:
                    continue
                control = _control_delta(ratings, season, cutoff, {team_id})
                projection = project_swap(
                    fitted,  # type: ignore[arg-type]
                    player_in=int(player),
                    player_out=int(player),
                    rule=rule,
                    possessions_vacated=_player_possessions(possessions, int(player), cutoff),
                    team_possessions=max(_team_possessions(possessions, team_id, season), 1),
                )
                outcomes.append(
                    MoveOutcome(
                        player_id=int(player),
                        to_team_id=team_id,
                        season=season,
                        kind="placebo",
                        cutoff_game=cutoff,
                        games_before=n_before,
                        games_after=n_after,
                        projected_delta=projection.delta_net,
                        projected_se=projection.se_total,
                        observed_delta=after - before,
                        did_delta=(after - before) - control,
                        minutes_variance_share=projection.minutes_variance_share,
                        is_placebo=True,
                    )
                )

    result.outcomes = outcomes
    result.variance = variance_decomposition(projections)

    real = [o for o in outcomes if not o.is_placebo]
    placebo = [o for o in outcomes if o.is_placebo]
    result.n_placebo = len(placebo)
    result.real = _score(real)
    result.placebo = _score(placebo)

    # A placebo whose projection is not centred on zero is a broken pipeline,
    # not a finding, and it invalidates the real numbers beside it.
    placebo_projection = result.placebo.get("mean_projected")
    if isinstance(placebo_projection, float) and abs(placebo_projection) > 1e-9:
        result.notes.append(
            f"placebo mean projected delta is {placebo_projection:+.4f}, not zero -- "
            "swapping a player for himself must project exactly no change"
        )
    return result


def _score(outcomes: list[MoveOutcome]) -> dict[str, Any]:
    """Correlation, sign agreement and calibration for a set of outcomes."""
    if len(outcomes) < 3:
        return {"n": len(outcomes)}

    projected = np.array([o.projected_delta for o in outcomes])
    observed = np.array([o.observed_delta for o in outcomes])
    did = np.array([o.did_delta for o in outcomes])

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # Sign agreement is undefined where the projection is exactly zero:
    # ``sign(0)`` matches neither +1 nor -1, so counting those rows as
    # disagreements drives the statistic to zero and reads as catastrophic
    # failure. The placebo arm projects exactly zero *by construction*, so this
    # is not an edge case -- it is the entire arm.
    signed = projected != 0.0
    n = len(outcomes)
    n_signed = int(signed.sum())
    if n_signed:
        agreement = float(np.mean(np.sign(projected[signed]) == np.sign(did[signed])))
        half = 1.96 * np.sqrt(0.25 / n_signed)
        interval = [float(max(0.0, agreement - half)), float(min(1.0, agreement + half))]
    else:
        agreement = float("nan")
        interval = [float("nan"), float("nan")]

    return {
        "n": n,
        "n_with_signed_projection": n_signed,
        "mean_projected": float(projected.mean()),
        "mean_observed": float(observed.mean()),
        "mean_did": float(did.mean()),
        "sd_observed": float(observed.std(ddof=1)),
        "sd_did": float(did.std(ddof=1)),
        "correlation_projected_observed": correlation(projected, observed),
        "correlation_projected_did": correlation(projected, did),
        "sign_agreement_vs_did": agreement,
        "sign_agreement_ci": interval,
        "mean_abs_error_vs_did": float(np.mean(np.abs(projected - did))),
        # For the placebo arm this is the number that matters: how far a team's
        # rating moves across an arbitrary mid-season cutoff with no roster
        # change at all. A real effect smaller than this is not measurable here,
        # whatever the projection says.
        "mean_abs_did": float(np.mean(np.abs(did))),
    }
