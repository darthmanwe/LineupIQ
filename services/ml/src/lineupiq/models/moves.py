"""Detecting roster moves, and asking honestly whether they can be evaluated.

A trade simulator makes a counterfactual claim: *this* swap changes the team by
*that* much. The only way to check such a claim is to find moves that already
happened, project them from data available beforehand, and compare.

**Moves are derived from the data, not from a transactions feed.** Every
possession carries the offensive team and the five players on the floor, so a
player's team is observable per game. A player whose team changes between
consecutive appearances moved. That keeps the pipeline dependency-free and makes
the move list reproducible from committed gold, at the cost of missing moves that
never resulted in a minute played -- which is the right trade for this purpose,
because a player who never took the floor cannot be evaluated anyway.

**The power analysis comes first.** It is computed and published before any
backtest result, because at these sample sizes the honest output may be "no
accuracy claim is supported", and that has to be a pre-commitment rather than a
retreat after seeing an unflattering number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from lineupiq.hashing import LINEUP_SIZE
from lineupiq.util import as_float

__all__ = [
    "MIN_POSSESSIONS_EITHER_SIDE",
    "Move",
    "PowerAnalysis",
    "detect_moves",
    "player_team_by_game",
    "power_analysis",
    "team_game_ratings",
]

#: A move is only evaluable if the player logged real time on both sides of it.
#: Below this the "before" or "after" rating is noise, and a projection cannot
#: be scored against noise.
MIN_POSSESSIONS_EITHER_SIDE = 200

#: Possessions a team-season needs before its rating enters a difference.
MIN_TEAM_POSSESSIONS = 500


def player_team_by_game(possessions: pl.DataFrame) -> pl.DataFrame:
    """``(player_id, game_id) -> team_id``, from the possessions themselves.

    A player on the floor for an offensive possession is on the offensive team;
    on the floor for a defensive possession, the defensive team. Both directions
    are used so a player who never appeared on offence still resolves.
    """
    offence = (
        possessions.select(
            pl.col("off_lineup").alias("player_id"),
            "game_id",
            pl.col("offense_team_id").alias("team_id"),
            "season",
        )
        .explode("player_id", empty_as_null=True)
        .drop_nulls("player_id")
    )
    defence = (
        possessions.select(
            pl.col("def_lineup").alias("player_id"),
            "game_id",
            pl.col("defense_team_id").alias("team_id"),
            "season",
        )
        .explode("player_id", empty_as_null=True)
        .drop_nulls("player_id")
    )
    return (
        pl.concat([offence, defence])
        .group_by(["player_id", "game_id"])
        .agg(
            # A player cannot be on two teams in one game; the mode guards
            # against a stray mis-attributed possession deciding it.
            pl.col("team_id").mode().first().alias("team_id"),
            pl.col("season").first().alias("season"),
        )
    )


@dataclass(frozen=True)
class Move:
    """One observed change of team by one player."""

    player_id: int
    from_team_id: int
    to_team_id: int
    #: Last game played for the old team, first for the new one. Game ids are
    #: chronological within a season, so these bound the move.
    last_game_before: str
    first_game_after: str
    season_before: int
    season_after: int
    possessions_before: int
    possessions_after: int

    @property
    def mid_season(self) -> bool:
        return self.season_before == self.season_after

    @property
    def kind(self) -> str:
        return "mid-season" if self.mid_season else "offseason"


def detect_moves(
    possessions: pl.DataFrame, *, min_possessions: int = MIN_POSSESSIONS_EITHER_SIDE
) -> list[Move]:
    """Find every player-team change with enough evidence on both sides.

    Consecutive appearances are compared in game order. A player with three
    teams in one season produces two moves, which is correct: each is a separate
    counterfactual.
    """
    membership = player_team_by_game(possessions)
    counts = (
        possessions.select(pl.col("off_lineup").alias("player_id"), "game_id")
        .explode("player_id", empty_as_null=True)
        .drop_nulls("player_id")
        .group_by(["player_id", "game_id"])
        .agg(pl.len().alias("off_possessions"))
    )
    joined = (
        membership.join(counts, on=["player_id", "game_id"], how="left")
        .with_columns(pl.col("off_possessions").fill_null(0))
        .sort(["player_id", "game_id"])
    )

    moves: list[Move] = []
    for (player,), block in joined.group_by(["player_id"], maintain_order=True):
        teams = block["team_id"].to_list()
        games = block["game_id"].to_list()
        seasons = block["season"].to_list()
        possession_counts = block["off_possessions"].to_list()

        # Index of each change of team.
        changes = [i for i in range(1, len(teams)) if teams[i] != teams[i - 1]]
        for change in changes:
            # Only the contiguous stretch with the *same* team on each side of
            # the boundary counts. A player who goes A -> B -> C produces two
            # moves, and the possessions credited to the A->B move must stop at
            # the B->C boundary rather than running to the end of his season.
            start = next(
                (i for i in range(change - 1, -1, -1) if teams[i] != teams[change - 1]), -1
            )
            end = next(
                (i for i in range(change, len(teams)) if teams[i] != teams[change]), len(teams)
            )
            before = possession_counts[start + 1 : change]
            after = possession_counts[change:end]
            if sum(before) < min_possessions or sum(after) < min_possessions:
                continue
            moves.append(
                Move(
                    player_id=int(player),
                    from_team_id=int(teams[change - 1]),
                    to_team_id=int(teams[change]),
                    last_game_before=str(games[change - 1]),
                    first_game_after=str(games[change]),
                    season_before=int(seasons[change - 1]),
                    season_after=int(seasons[change]),
                    possessions_before=int(sum(before)),
                    possessions_after=int(sum(after)),
                )
            )
    return moves


def team_game_ratings(possessions: pl.DataFrame) -> pl.DataFrame:
    """Offensive and defensive points per 100 for each team in each game.

    The unit the trade backtest is scored on. Both sides come from the same
    possession table, so a team's offensive rating and its opponent's defensive
    rating are the same number by construction -- which is a property worth
    having when the two are differenced.
    """
    offence = possessions.group_by(["game_id", "offense_team_id", "season"]).agg(
        pl.len().alias("off_possessions"), pl.col("points").sum().alias("points_for")
    )
    defence = possessions.group_by(["game_id", "defense_team_id", "season"]).agg(
        pl.len().alias("def_possessions"), pl.col("points").sum().alias("points_against")
    )
    return (
        offence.join(
            defence,
            left_on=["game_id", "offense_team_id", "season"],
            right_on=["game_id", "defense_team_id", "season"],
            how="inner",
        )
        .rename({"offense_team_id": "team_id"})
        .with_columns(
            (100.0 * pl.col("points_for") / pl.col("off_possessions")).alias("off_rating"),
            (100.0 * pl.col("points_against") / pl.col("def_possessions")).alias("def_rating"),
        )
        .with_columns((pl.col("off_rating") - pl.col("def_rating")).alias("net_rating"))
        .sort(["team_id", "game_id"])
    )


@dataclass(frozen=True)
class PowerAnalysis:
    """What effect size this sample could detect, computed before the result.

    ``mde`` is the smallest true effect a two-sided test at ``alpha`` would
    detect with probability ``power``. If the effects a trade simulator claims
    are smaller than this, the backtest cannot distinguish them from zero, and
    the correct output is a statement about power rather than a point estimate
    dressed up as a finding.
    """

    n: int
    residual_sd: float
    alpha: float
    power: float
    mde: float
    #: 95% CI half-width for a sign-accuracy estimate at this n.
    sign_accuracy_ci_half_width: float
    verdict: str

    def describe(self) -> str:
        return (
            f"n = {self.n}, residual sd = {self.residual_sd:.2f} per 100. "
            f"Minimum detectable effect = {self.mde:.2f} per 100 "
            f"at alpha {self.alpha:g} and power {self.power:g}. "
            f"Sign accuracy carries a +/-{self.sign_accuracy_ci_half_width:.1%} "
            f"95% interval. Verdict: {self.verdict}."
        )


#: The effect size the MDE is judged against, in points per 100 possessions.
#:
#: This is not "the smallest effect anyone would care about" -- it is **the size
#: of the effects this project actually projects**, which is the only bar that
#: makes an MDE meaningful. A RAPM-based projection for a realistic mid-season
#: move, scaled by the share of team possessions it touches, lands well under a
#: point per 100. Setting the bar at two would let a study declare itself
#: adequately powered to detect effects twice as large as any it claims, which
#: is a way of passing a power analysis rather than doing one.
CLAIMED_EFFECT_PER_100 = 1.0


def power_analysis(
    n: int,
    residual_sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    interesting_effect: float = CLAIMED_EFFECT_PER_100,
) -> PowerAnalysis:
    """Minimum detectable effect for a paired comparison of ``n`` moves.

    Uses the normal approximation, which is adequate at these sample sizes and
    keeps the calculation legible: ``mde = (z_alpha/2 + z_power) * sd / sqrt(n)``.

    If the MDE exceeds ``interesting_effect`` the verdict is UNDERPOWERED and no
    accuracy claim follows from the backtest -- whatever the backtest goes on to
    report.
    """
    from scipy import stats

    if n < 2 or residual_sd <= 0:
        return PowerAnalysis(
            n=n,
            residual_sd=residual_sd,
            alpha=alpha,
            power=power,
            mde=float("inf"),
            sign_accuracy_ci_half_width=1.0,
            verdict="UNDERPOWERED",
        )

    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(power))
    mde = (z_alpha + z_power) * residual_sd / np.sqrt(n)

    # Worst-case (p = 0.5) half-width for a proportion at this n.
    half_width = z_alpha * np.sqrt(0.25 / n)

    return PowerAnalysis(
        n=n,
        residual_sd=residual_sd,
        alpha=alpha,
        power=power,
        mde=float(mde),
        sign_accuracy_ci_half_width=float(half_width),
        verdict="UNDERPOWERED" if mde > interesting_effect else "adequate",
    )


def residual_sd_of_team_rating_change(ratings: pl.DataFrame, *, min_games: int = 20) -> float:
    """Spread of team net-rating changes between adjacent stretches of games.

    The natural noise floor for a trade backtest: how much a team's net rating
    moves across a mid-season boundary *without* any roster change, in the same
    units the projection is made in. Estimated from every team-season by
    splitting its games in half and taking the standard deviation of the
    difference, so it includes schedule, health and shooting variance -- all the
    things a trade projection is competing against.
    """
    changes: list[float] = []
    for _, block in ratings.group_by(["team_id", "season"], maintain_order=True):
        if block.height < min_games:
            continue
        half = block.height // 2
        first = as_float(block.head(half)["net_rating"].mean())
        second = as_float(block.tail(block.height - half)["net_rating"].mean())
        changes.append(second - first)
    if len(changes) < 2:
        return float("nan")
    return float(np.std(np.asarray(changes), ddof=1))


def summarise_moves(moves: list[Move]) -> dict[str, Any]:
    """Counts and totals, for the run log and the power analysis."""
    mid = [m for m in moves if m.mid_season]
    off = [m for m in moves if not m.mid_season]
    return {
        "n_moves": len(moves),
        "n_mid_season": len(mid),
        "n_offseason": len(off),
        "min_possessions_either_side": MIN_POSSESSIONS_EITHER_SIDE,
        "median_possessions_after": (
            float(np.median([m.possessions_after for m in moves])) if moves else 0.0
        ),
        "lineup_size": LINEUP_SIZE,
    }
