"""Trade projection: what a swap is worth, and how much of that is guesswork.

The projection itself is arithmetic. Two things around it are the actual work.

**The minutes rule is a visible input, never a silent assumption.** A trade's
projected value depends entirely on how much the arriving player plays, and that
is a coaching decision nobody in this repository can observe. So it is a
parameter with a name, printed next to every number it produced, and the API
returns it in the response. Toggling it and watching the answer move is the most
honest thing this product does.

**The variance is decomposed.** If most of a projected delta's uncertainty comes
from the minutes assumption rather than from the player estimates, then the
entire modelling stack is not the bottleneck and saying so is more useful than
another decimal place. That decomposition is computed and published.

The estimand is deliberately narrow: **the change in the receiving team's points
per 100 possessions, holding everything except these two players fixed.** It is
not a win projection, it does not model fit, role conflict, locker rooms or
coaching. Every one of those is real and none of them is in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lineupiq.models.rapm import RapmReport

__all__ = [
    "LEAGUE_AVERAGE_REPLACEMENT",
    "MINUTES_RULES",
    "MinutesRule",
    "TradeProjection",
    "project_swap",
    "variance_decomposition",
]

#: Stand-in for "whoever the arriving player displaces", used when the departing
#: player is genuinely unknown -- which is the case in the backtest, because the
#: corpus records who arrived and not who left.
#:
#: Its effect is **exactly zero with exactly zero variance**, and that is not a
#: convenience: ridge centres the coefficients on the league mean, so the
#: marginal rotation player *is* the zero point. Treating it as an ordinary
#: player id would ask the covariance matrix for a row that does not exist.
LEAGUE_AVERAGE_REPLACEMENT = None

PlayerRef = int | None


@dataclass(frozen=True)
class MinutesRule:
    """How much the arriving player is assumed to play.

    ``share`` is the fraction of the departing player's possessions that the
    arriving player inherits. ``spread`` is the standard deviation of that
    assumption -- the honest admission that a coach's rotation is not observable
    from play-by-play, and the term that usually dominates the projection's
    uncertainty.
    """

    name: str
    share: float
    spread: float
    detail: str


#: The rules a caller may choose between. Presented as a set rather than a
#: default so that picking one is a visible act.
MINUTES_RULES: tuple[MinutesRule, ...] = (
    MinutesRule(
        "inherit",
        1.0,
        0.15,
        "The arriving player takes exactly the departing player's minutes. The "
        "cleanest counterfactual and the least realistic: it assumes the coach "
        "slots one for the other with no rotation change at all.",
    ),
    MinutesRule(
        "historical",
        0.85,
        0.25,
        "The arriving player plays at his own recent rate, scaled into the "
        "vacated minutes. Wider spread because two rotations rarely match.",
    ),
    MinutesRule(
        "conservative",
        0.60,
        0.20,
        "The arriving player absorbs only part of the vacated minutes and the "
        "rest is redistributed to incumbents. Appropriate when the departing "
        "player was a starter and the arrival is not.",
    ),
)


@dataclass(frozen=True)
class TradeProjection:
    """A projected change in the receiving team's rating, with its uncertainty."""

    player_in: PlayerRef
    player_out: PlayerRef
    minutes_rule: str
    minutes_share: float
    delta_offence: float
    delta_defence: float
    delta_net: float
    #: Standard error from the player estimates alone, holding minutes fixed.
    se_from_players: float
    #: Standard error contributed by the minutes assumption alone.
    se_from_minutes: float
    se_total: float
    #: Share of total variance attributable to the minutes rule.
    minutes_variance_share: float
    #: True when the 80% interval contains zero, which is the normal case.
    interval_includes_zero: bool
    #: Populated when either player's coefficients are not separately identified.
    warnings: list[str] = field(default_factory=list)

    def interval(self, z: float = 1.2816) -> tuple[float, float]:
        """Default is an 80% interval, matching the API's conformal bins."""
        return self.delta_net - z * self.se_total, self.delta_net + z * self.se_total

    def describe(self) -> str:
        low, high = self.interval()
        return (
            f"net {self.delta_net:+.2f} per 100 "
            f"(80% interval {low:+.2f} to {high:+.2f}); "
            f"{self.minutes_variance_share:.0%} of the variance is the minutes rule"
        )


def _coefficient(fit: object, player: PlayerRef, side: str) -> float:
    if player is LEAGUE_AVERAGE_REPLACEMENT:
        return 0.0
    table = fit.off_rapm if side == "off" else fit.def_rapm  # type: ignore[attr-defined]
    return float(table.get(player, 0.0))


def _contrast_se(
    report: RapmReport, player_in: PlayerRef, player_out: PlayerRef, side: str
) -> float:
    """Exact standard error of one side's difference between two players.

    ``Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b)``. The covariance term matters:
    two players who share floor time compete for the same credit and their
    estimates are negatively correlated, so dropping it *understates* the
    uncertainty of exactly the quantity being projected.

    A league-average replacement contributes no variance, so the contrast
    collapses to the other player's own standard error.

    Returns ``nan`` when the variance genuinely cannot be computed, and callers
    must treat that as unknown rather than as zero. An earlier version let a nan
    fall through a ``> 0`` test into a literal ``0.0``, which reported "the
    minutes rule carries 0% of the variance" when the truth was "no variance was
    computed at all" -- a far worse failure than a missing number.
    """
    fit = report.fit
    errors = fit.off_se if side == "off" else fit.def_se

    if player_in is LEAGUE_AVERAGE_REPLACEMENT and player_out is LEAGUE_AVERAGE_REPLACEMENT:
        return 0.0
    if player_in == player_out:
        # Swapping a player for himself is exactly no change, with no
        # uncertainty. The placebo arm depends on this being exact.
        return 0.0

    known = [p for p in (player_in, player_out) if p is not LEAGUE_AVERAGE_REPLACEMENT]
    if report.covariance is None or not report.column_index:
        return float(np.sqrt(sum(errors.get(p, float("nan")) ** 2 for p in known)))

    offset = 0 if side == "off" else len(fit.players)
    indices: list[int] = []
    for player in known:
        column = report.column_index.get(player)
        if column is None:
            return float("nan")
        indices.append(offset + column)

    covariance = report.covariance
    value = float(sum(covariance[i, i] for i in indices))
    if len(indices) == 2:
        first, second = indices
        value -= 2.0 * float(covariance[first, second])
    # Coefficients are reported per 100 possessions; the covariance is in
    # per-possession units, so the standard error scales by 100.
    return float(100.0 * np.sqrt(max(value, 0.0)))


def project_swap(
    report: RapmReport,
    *,
    player_in: PlayerRef,
    player_out: PlayerRef,
    rule: MinutesRule,
    possessions_vacated: int,
    team_possessions: int,
) -> TradeProjection:
    """Project the receiving team's rating change from swapping two players.

    ``possessions_vacated`` is what the departing player was on the floor for;
    ``team_possessions`` is the team's total. The delta is scaled by the share of
    team possessions actually affected, because replacing a 200-possession
    reserve cannot move a team's rating by the same amount as replacing a
    2,000-possession starter, and an unscaled difference of coefficients quietly
    claims that it can.
    """
    fit = report.fit
    warnings: list[str] = []

    for player in (player_in, player_out):
        if player is LEAGUE_AVERAGE_REPLACEMENT:
            warnings.append(
                "one side of this swap is a league-average replacement, because the "
                "departing player is not recorded. The projection is therefore what "
                "adding the arriving player is worth against the marginal rotation "
                "player, not against a specific one."
            )
        elif player not in fit.off_rapm:
            warnings.append(f"player {player} has no RAPM estimate; treated as league average")

    flagged = {row["player_id"] for row in report.co_occurrence.get("non_identified", [])}
    for player in (player_in, player_out):
        if player is not LEAGUE_AVERAGE_REPLACEMENT and player in flagged:
            warnings.append(
                f"player {player} shares most of his floor time with one teammate, so his "
                "individual coefficient is not identified -- only the pair's sum is"
            )

    exposure = (possessions_vacated * rule.share) / max(team_possessions, 1)

    off_difference = _coefficient(fit, player_in, "off") - _coefficient(fit, player_out, "off")
    def_difference = _coefficient(fit, player_in, "def") - _coefficient(fit, player_out, "def")

    delta_offence = off_difference * exposure
    delta_defence = def_difference * exposure
    delta_net = delta_offence + delta_defence

    off_se = _contrast_se(report, player_in, player_out, "off")
    def_se = _contrast_se(report, player_in, player_out, "def")
    if report.covariance is None and player_in != player_out:
        warnings.append(
            "covariance unavailable; interval uses diagonal variances only and is "
            "narrower than the exact contrast"
        )

    # Player term: offence and defence occupy disjoint coefficient blocks, so
    # adding their variances is exact rather than an approximation.
    player_sd = float(np.sqrt(off_se**2 + def_se**2)) * exposure

    # Minutes term: the delta is linear in the share, so a relative spread on
    # the share becomes the same relative spread on the delta.
    minutes_sd = abs(off_difference + def_difference) * (
        possessions_vacated * rule.spread / max(team_possessions, 1)
    )

    total_variance = player_sd**2 + minutes_sd**2
    total_sd = float(np.sqrt(total_variance))

    # An unknown variance stays unknown. Collapsing nan to zero here is how a
    # report ends up claiming a decomposition it never computed.
    if np.isnan(total_variance) or total_variance <= 0.0:
        minutes_share = float("nan") if np.isnan(total_variance) else 0.0
    else:
        minutes_share = minutes_sd**2 / total_variance
    if np.isnan(total_sd):
        warnings.append(
            "the projection interval could not be computed: at least one player has no "
            "variance estimate in this fit"
        )

    low, high = delta_net - 1.2816 * total_sd, delta_net + 1.2816 * total_sd

    return TradeProjection(
        player_in=player_in,
        player_out=player_out,
        minutes_rule=rule.name,
        minutes_share=rule.share,
        delta_offence=delta_offence,
        delta_defence=delta_defence,
        delta_net=delta_net,
        se_from_players=player_sd,
        se_from_minutes=minutes_sd,
        se_total=total_sd,
        minutes_variance_share=minutes_share,
        interval_includes_zero=bool(low <= 0.0 <= high),
        warnings=warnings,
    )


def variance_decomposition(projections: list[TradeProjection]) -> dict[str, float]:
    """Where a typical projection's uncertainty comes from.

    Published because it decides what is worth improving. If the minutes rule
    carries most of the variance, then a better shot model changes nothing that
    matters, and the useful next step is a minutes model or a wider interval --
    not another feature.

    Projections whose variance could not be computed are **counted and
    excluded**, never averaged in as zeros. Reporting "the minutes rule carries
    0% of the variance" when the truth is "no variance was computed" is the
    worse of the two failures, because it looks like a finding.
    """
    if not projections:
        return {}
    shares = np.array([p.minutes_variance_share for p in projections])
    usable = shares[~np.isnan(shares)]
    missing = float(len(projections) - usable.size)
    if not usable.size:
        return {"n": float(len(projections)), "n_without_variance": missing}
    return {
        "n": float(len(projections)),
        "n_without_variance": missing,
        "mean_minutes_variance_share": float(usable.mean()),
        "median_minutes_variance_share": float(np.median(usable)),
        "share_where_minutes_dominates": float((usable > 0.5).mean()),
        "mean_se_total": float(np.nanmean([p.se_total for p in projections])),
        "share_interval_includes_zero": float(
            np.mean([p.interval_includes_zero for p in projections])
        ),
    }


def rule_by_name(name: str) -> MinutesRule:
    for rule in MINUTES_RULES:
        if rule.name == name:
            return rule
    known = ", ".join(r.name for r in MINUTES_RULES)
    raise KeyError(f"unknown minutes rule {name!r}; known: {known}")
