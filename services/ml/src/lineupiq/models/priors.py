"""Empirical-Bayes shrinkage for rate estimates.

A player who has taken 6 corner threes and made 4 has not shot 66.7% from the
corner. Reporting that number, or feeding it to a model as a feature, is the
single most common way a basketball model learns noise.

Every shrunk value here ships with the weight that produced it and the effective
sample behind it, so a consumer can always see how much of an estimate is the
player and how much is the prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = [
    "BetaPrior",
    "DirichletPrior",
    "fit_beta_prior",
    "fit_dirichlet_prior",
    "shrink_rates",
]


@dataclass(frozen=True)
class BetaPrior:
    """A Beta(alpha, beta) prior fitted by moment matching."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        """Prior weight in units of attempts -- how many shots it is worth."""
        return self.alpha + self.beta


def fit_beta_prior(makes: np.ndarray, attempts: np.ndarray, *, min_attempts: int = 20) -> BetaPrior:
    """Moment-match a Beta prior to the observed spread of player rates.

    Only players above ``min_attempts`` inform the prior. Including everyone
    would let the noisiest observations set the variance, which inflates the
    spread and then *under*-shrinks exactly the players who need it most.
    """
    mask = attempts >= min_attempts
    if mask.sum() < 2:
        # Not enough signal to estimate a spread. Fall back to a weak prior
        # centred on the pooled rate rather than inventing a variance.
        pooled = float(makes.sum() / attempts.sum()) if attempts.sum() else 0.5
        return BetaPrior(alpha=pooled * 10.0, beta=(1.0 - pooled) * 10.0)

    rates = makes[mask] / attempts[mask]
    mean = float(np.mean(rates))
    var = float(np.var(rates, ddof=1))

    # Binomial noise inflates the observed spread; subtract the part explained
    # by sampling so the prior reflects true between-player variation.
    mean_n = float(np.mean(attempts[mask]))
    sampling_var = mean * (1.0 - mean) / mean_n if mean_n > 0 else 0.0
    true_var = max(var - sampling_var, 1e-6)

    max_var = mean * (1.0 - mean)
    if true_var >= max_var:
        true_var = max_var * 0.99

    strength = mean * (1.0 - mean) / true_var - 1.0
    strength = float(np.clip(strength, 1.0, 5000.0))
    return BetaPrior(alpha=mean * strength, beta=(1.0 - mean) * strength)


def shrink_rates(
    frame: pl.DataFrame,
    *,
    makes_col: str,
    attempts_col: str,
    group_cols: tuple[str, ...],
    prefix: str = "",
) -> pl.DataFrame:
    """Shrink a rate toward a prior fitted within each group.

    Adds ``{prefix}rate_raw``, ``{prefix}rate_shrunk``, ``{prefix}shrink_weight``
    and ``{prefix}n_eff``. Shipping the weight alongside the value is the point:
    a consumer can see immediately whether a number is evidence or prior.
    """
    out: list[pl.DataFrame] = []

    for _, group in frame.group_by(list(group_cols), maintain_order=True):
        makes = group[makes_col].to_numpy().astype(float)
        attempts = group[attempts_col].to_numpy().astype(float)
        prior = fit_beta_prior(makes, attempts)

        denom = attempts + prior.strength
        shrunk = (makes + prior.alpha) / denom
        weight = attempts / denom  # 1 = all evidence, 0 = all prior

        out.append(
            group.with_columns(
                pl.Series(
                    f"{prefix}rate_raw",
                    np.divide(makes, attempts, out=np.full_like(makes, np.nan), where=attempts > 0),
                ),
                pl.Series(f"{prefix}rate_shrunk", shrunk),
                pl.Series(f"{prefix}shrink_weight", weight),
                pl.Series(f"{prefix}n_eff", attempts),
                pl.lit(prior.mean).alias(f"{prefix}prior_mean"),
            )
        )

    return pl.concat(out) if out else frame


@dataclass(frozen=True)
class DirichletPrior:
    """A Dirichlet prior over a categorical distribution, moment-matched.

    The multinomial analogue of :class:`BetaPrior`. Where that shrinks one rate
    toward a league mean, this shrinks a whole *mix* -- a player's distribution
    of attempts across shot zones -- toward the league's mix.
    """

    #: League mean mix. Sums to one.
    mean: np.ndarray
    #: Prior weight in units of attempts.
    strength: float

    def shrink(self, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shrink observed count rows toward the prior.

        Returns ``(mix, weight)`` where ``mix`` rows sum to one and ``weight``
        is the share of each row's estimate carried by evidence rather than by
        the prior -- 1 for a player with thousands of attempts, near 0 for a
        player with four.
        """
        totals = counts.sum(axis=1, keepdims=True)
        mix = (counts + self.strength * self.mean) / (totals + self.strength)
        weight = totals / (totals + self.strength)
        return mix, weight.ravel()


def fit_dirichlet_prior(
    counts: np.ndarray, *, min_total: int = 50, max_strength: float = 5000.0
) -> DirichletPrior:
    """Moment-match a Dirichlet prior to the observed spread of category mixes.

    ``counts`` is ``(n_units, n_categories)`` -- one row per player, one column
    per zone.

    The concentration is estimated from how much more the observed mixes vary
    than multinomial sampling alone would explain. Per category, the observed
    variance of the rates is

        Var(p_j) ~ m_j (1 - m_j) / n_bar * (1 + (n_bar - 1) * rho)

    so each category implies its own ``rho``, an intra-unit correlation, and
    ``strength = (1 - rho) / rho``. The estimates are pooled across categories
    by a weighted median rather than a mean: a rare zone with a handful of
    attempts league-wide produces a wild ``rho``, and a mean lets that one
    category set the shrinkage for all nine.

    Units below ``min_total`` attempts are excluded from the fit for the same
    reason ``fit_beta_prior`` excludes them -- letting the noisiest rows set the
    variance inflates the apparent spread and then under-shrinks precisely the
    rows that need it most.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1)
    league = counts.sum(axis=0)
    mean = (
        league / league.sum()
        if league.sum() > 0
        else np.full(counts.shape[1], 1.0 / counts.shape[1])
    )

    mask = totals >= min_total
    if mask.sum() < 2:
        # Not enough units to estimate a spread. A weak prior is the honest
        # fallback; inventing a variance is not.
        return DirichletPrior(mean=mean, strength=25.0)

    rates = counts[mask] / totals[mask][:, None]
    n_bar = float(totals[mask].mean())
    if n_bar <= 1.0:
        return DirichletPrior(mean=mean, strength=25.0)

    observed = rates.var(axis=0, ddof=1)
    binomial = mean * (1.0 - mean) / n_bar

    rhos: list[float] = []
    weights: list[float] = []
    for j in range(counts.shape[1]):
        if binomial[j] <= 0 or mean[j] <= 0:
            continue
        rho = (observed[j] / binomial[j] - 1.0) / (n_bar - 1.0)
        if np.isfinite(rho) and rho > 0:
            rhos.append(float(rho))
            weights.append(float(mean[j]))
    if not rhos:
        return DirichletPrior(mean=mean, strength=float(max_strength))

    order = np.argsort(rhos)
    r = np.asarray(rhos)[order]
    w = np.asarray(weights)[order]
    cumulative = np.cumsum(w) / w.sum()
    rho_pooled = float(r[int(np.searchsorted(cumulative, 0.5))])
    strength = (1.0 - rho_pooled) / rho_pooled
    return DirichletPrior(mean=mean, strength=float(np.clip(strength, 1.0, max_strength)))
