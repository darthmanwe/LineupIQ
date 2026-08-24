"""RAPM: regularised adjusted plus-minus, fitted on possessions.

One row per possession, ten indicator columns set -- five for the offence, five
for the defence -- and points scored as the target. Ridge regression on that
design is RAPM, and it is the additive player-effect model everything downstream
needs: a trade delta is a difference of player effects, and without them a trade
simulator is just a lookup table with opinions.

Three things about this implementation are load-bearing.

**The penalty is not one number.** Offence and defence are estimated with
separate ridge parameters, because they are not equally identified: offensive
production is concentrated in a few players per possession while defensive
credit is diffuse, so the same shrinkage applied to both over-shrinks one and
under-shrinks the other. Both are selected by cross-validation.

**Folds are grouped by game, never by possession.** Two possessions from the
same game share lineups, opponent, rest, altitude and that night's shooting
variance. Splitting between them lets the model see its own answer, and the
resulting lambda is far too small -- the classic way a ridge model is reported
as better than it is.

**Reliability is measured by split-half, not by fit quality.** Ridge always
improves in-sample fit as lambda falls, and cross-validated error on possession
outcomes is dominated by shot noise, so neither tells you whether the *player
coefficients* mean anything. Fitting odd and even games separately and
correlating the two coefficient vectors does: if the same player gets +3 from
one half and -1 from the other, the number is noise however good the log
likelihood looked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from lineupiq.hashing import LINEUP_SIZE
from lineupiq.util import as_float, lineup_slots

__all__ = [
    "LAMBDA_GRID",
    "RapmDesign",
    "RapmFit",
    "RapmReport",
    "build_rapm_design",
    "co_occurrence_report",
    "fit_rapm",
    "solve_ridge",
    "split_half_reliability",
    "usable_possessions",
]

#: Candidate ridge parameters, in possession units. The useful range for NBA
#: RAPM at three seasons sits in the low thousands; the grid is wide enough that
#: a selected endpoint is visible as a warning rather than hidden as a choice.
LAMBDA_GRID: tuple[float, ...] = (
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_000.0,
    4_000.0,
    8_000.0,
    16_000.0,
    32_000.0,
)

#: Above this share of shared floor time, two players' coefficients are not
#: separately identified and neither is served as a point estimate.
CO_OCCURRENCE_CEILING = 0.85

#: Possessions a player needs in each half before split-half reliability
#: includes him. Below it the correlation measures sampling noise.
RELIABILITY_MIN_POSSESSIONS = 200


def usable_possessions(
    possessions: pl.DataFrame, *, exclude_boundary_ambiguous: bool = False
) -> pl.DataFrame:
    """Filter to possessions this model may be fitted on.

    ``exclude_boundary_ambiguous`` drops the ~9% of possessions that begin on
    the exact second of a substitution, where two defensible lineup attributions
    exist. Whether that matters is measured rather than assumed -- see
    :func:`fit_rapm`, which reports both.
    """
    frame = possessions.filter(
        pl.col("off_lineup").is_not_null()
        & pl.col("def_lineup").is_not_null()
        & (pl.col("off_lineup").list.len() == LINEUP_SIZE)
        & (pl.col("def_lineup").list.len() == LINEUP_SIZE)
        & (pl.col("stint_quality") == "VALID")
        & pl.col("points").is_not_null()
    )
    if exclude_boundary_ambiguous:
        frame = frame.filter(~pl.col("boundary_ambiguous"))
    return frame


@dataclass(frozen=True)
class RapmDesign:
    """A sparse possession design, plus the labels needed to read it back."""

    #: (n_possessions, 2 * n_players + 1). Offence block, defence block, then
    #: the home-offence indicator. The intercept is handled separately so the
    #: penalty can exclude it.
    #: ``scipy.sparse.csr_matrix``. Typed loosely because scipy ships no stubs,
    #: so a precise annotation here would be a fiction mypy cannot check.
    matrix: Any
    y: np.ndarray
    players: list[int]
    game_ids: np.ndarray
    #: Possessions each player appears in, either side.
    appearances: dict[int, int]

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def n_possessions(self) -> int:
        return len(self.y)

    def penalty_vector(self, lambda_offence: float, lambda_defence: float) -> np.ndarray:
        """Diagonal ridge penalty. Home advantage is never penalised.

        Shrinking the home-court term toward zero would push its effect into the
        player coefficients of whichever teams happened to play at home more.
        """
        penalty = np.empty(2 * self.n_players + 1)
        penalty[: self.n_players] = lambda_offence
        penalty[self.n_players : 2 * self.n_players] = lambda_defence
        penalty[-1] = 0.0
        return penalty


def build_rapm_design(possessions: pl.DataFrame) -> RapmDesign:
    """Assemble the sparse design.

    Dense would be ``774k x 1141`` float64, about 7 GB. Sparse it is 7.7M
    nonzeros, roughly 90 MB, and the normal equations are only ``1141 x 1141``.
    """
    from scipy import sparse

    # Five flat arrays per column rather than 500k Python lists. The caller
    # has already filtered to lineups of exactly five, so no slot is absent.
    off = lineup_slots(possessions["off_lineup"])
    dfn = lineup_slots(possessions["def_lineup"])
    points = possessions["points"].to_numpy().astype(float)
    games = possessions["game_id"].to_numpy()

    everyone = sorted({int(p) for slot in off + dfn for p in slot})
    index = {player: i for i, player in enumerate(everyone)}
    n_players = len(everyone)

    n = len(points)
    # Ten player entries per possession, plus the home-offence indicator.
    rows = np.empty(n * (2 * LINEUP_SIZE + 1), dtype=np.int32)
    cols = np.empty_like(rows)
    data = np.ones(rows.shape[0], dtype=np.float64)

    offence_is_home = (
        (possessions["offense_team_id"] == possessions["home_team_id"]).to_numpy().astype(float)
    )

    appearances: dict[int, int] = dict.fromkeys(everyone, 0)
    cursor = 0
    for i in range(n):
        for player in (slot[i] for slot in off):
            rows[cursor] = i
            cols[cursor] = index[int(player)]
            appearances[int(player)] += 1
            cursor += 1
        for player in (slot[i] for slot in dfn):
            rows[cursor] = i
            cols[cursor] = n_players + index[int(player)]
            appearances[int(player)] += 1
            cursor += 1
        rows[cursor] = i
        cols[cursor] = 2 * n_players
        data[cursor] = offence_is_home[i]
        cursor += 1

    matrix = sparse.csr_matrix(
        (data[:cursor], (rows[:cursor], cols[:cursor])), shape=(n, 2 * n_players + 1)
    )
    return RapmDesign(
        matrix=matrix, y=points, players=everyone, game_ids=games, appearances=appearances
    )


@dataclass(frozen=True)
class RidgeSolution:
    coefficients: np.ndarray
    intercept: float
    effective_df: float
    condition_number: float
    #: Full covariance of the coefficients, or ``None`` when not requested.
    #: 1,141 x 1,141 is 10 MB -- cheap to hold in a fitting process, far too
    #: large to ship to a Worker, which is why the serving path uses the
    #: diagonal and publishes what that approximation costs.
    covariance: np.ndarray | None = None

    @property
    def standard_errors(self) -> np.ndarray | None:
        if self.covariance is None:
            return None
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    def contrast_se(self, i: int, j: int) -> float:
        """Exact standard error of ``beta_i - beta_j``.

        ``Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b)``. For two players who
        share floor time the covariance is materially negative -- they compete
        for the same credit -- so dropping it understates the uncertainty of
        their difference, which is precisely the quantity a trade projection is.
        """
        if self.covariance is None:
            raise RuntimeError("covariance was not computed")
        value = self.covariance[i, i] + self.covariance[j, j] - 2.0 * self.covariance[i, j]
        return float(np.sqrt(max(value, 0.0)))


def solve_ridge(
    gram: np.ndarray,
    moment: np.ndarray,
    penalty: np.ndarray,
    *,
    intercept: float,
    diagnostics: bool = True,
    residual_variance: float | None = None,
) -> RidgeSolution:
    """Solve ``(X'X + diag(penalty)) beta = X'y`` for centred ``y``.

    The Gram matrix is formed once per fold and reused across the whole lambda
    grid, which turns a two-dimensional search into one Cholesky solve per
    candidate.

    ``effective_df`` is ``trace((X'X + D)^-1 X'X)`` -- how many parameters the
    fit is really spending. With ~1,140 columns and a lambda in the thousands it
    lands in the low hundreds, and publishing it is the difference between
    "regularised" and "regularised by an amount nobody stated".

    When ``residual_variance`` is supplied the coefficient covariance is
    computed as the ridge sandwich ``sigma^2 A^-1 G A^-1`` with ``A = G + D``.
    Using the ordinary least-squares form ``sigma^2 A^-1`` instead would be the
    easy mistake: it ignores the penalty's effect on the sampling distribution
    and reports intervals that are too narrow for exactly the low-minute players
    whose estimates are mostly prior.
    """
    from scipy import linalg

    regularised = gram + np.diag(penalty)
    try:
        factor = linalg.cho_factor(regularised, lower=True, check_finite=False)
        beta = linalg.cho_solve(factor, moment, check_finite=False)
    except linalg.LinAlgError:
        # A singular system means the penalty is too weak to identify the
        # design. Falling back to least squares would hide that; reporting it
        # via an infinite condition number does not.
        beta = linalg.lstsq(regularised, moment)[0]
        return RidgeSolution(beta, intercept, float("nan"), float("inf"))

    if not diagnostics:
        return RidgeSolution(beta, intercept, float("nan"), float("nan"))

    hat = linalg.cho_solve(factor, gram, check_finite=False)
    covariance = None
    if residual_variance is not None:
        inverse = linalg.cho_solve(factor, np.eye(gram.shape[0]), check_finite=False)
        covariance = residual_variance * (inverse @ gram @ inverse)

    return RidgeSolution(
        coefficients=beta,
        intercept=intercept,
        effective_df=float(np.trace(hat)),
        condition_number=float(np.linalg.cond(regularised)),
        covariance=covariance,
    )


def _game_folds(game_ids: np.ndarray, *, n_folds: int, seed: int) -> list[np.ndarray]:
    """Assign whole games to folds.

    Grouping by possession would let two rows from the same game land on
    opposite sides of the split, and they share far too much: lineups, opponent,
    rest, and that night's shooting variance.
    """
    unique = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    buckets = np.array_split(shuffled, n_folds)
    return [np.isin(game_ids, bucket) for bucket in buckets]


#: Decimal places co-occurrence ratios are rounded to before they are compared.
#:
#: These are ratios of possession counts read out of a sparse matrix product, so
#: they are not bit-portable, and every use of them below is a *comparison* --
#: which means one bit in the last place decides an ordering as firmly as a whole
#: possession would. Six places is far finer than any distinction that matters
#: between two teammates and far coarser than the noise.
_CO_OCCURRENCE_PRECISION = 6


def co_occurrence_report(
    design: RapmDesign, *, ceiling: float = CO_OCCURRENCE_CEILING
) -> dict[str, Any]:
    """Find players whose minutes are nearly inseparable from a teammate's.

    If two players are on the floor together for almost all of both their
    possessions, ridge can trade their coefficients off against each other
    freely and the split between them is an artefact of the penalty. Their
    *sum* is identified; neither of them individually is.

    Reported, and excluded from point estimates by the serving layer, rather
    than quietly published as if it were measured.
    """
    from scipy import sparse

    n = design.n_players
    presence = design.matrix[:, :n] + design.matrix[:, n : 2 * n]
    presence = (presence > 0).astype(np.float64)
    shared = (
        (presence.T @ presence).toarray() if sparse.issparse(presence) else presence.T @ presence
    )
    totals = np.diag(shared).copy()
    np.fill_diagonal(shared, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = np.where(totals[:, None] > 0, shared / totals[:, None], 0.0)

    # Rounded before any comparison, and the report is only sorted afterwards.
    # Both steps below are decided by comparisons on these values, and both were
    # non-deterministic without this.
    #
    # `argmax` picks the *first* maximal index, so a player whose two most
    # frequent teammates tie gets whichever one came out microscopically larger
    # -- and these are ratios of possession counts computed through a sparse
    # matrix product, so the last place differs between platforms. Rounding makes
    # a genuine tie an exact tie, and `argmax` then resolves it by index, which
    # is the same everywhere.
    #
    # The sort has the same problem more visibly: `worst` is exactly 1.0 for
    # every player who never took the floor without a particular teammate, so the
    # top of this list is a solid block of ties, and `flagged[:50]` was
    # publishing an arbitrary fifty of them.
    conditional = np.round(conditional, _CO_OCCURRENCE_PRECISION)
    worst = conditional.max(axis=1)
    partner = conditional.argmax(axis=1)

    flagged = sorted(
        (
            {
                "player_id": design.players[i],
                "possessions": int(totals[i]),
                "max_co_occurrence": float(worst[i]),
                "partner_id": design.players[int(partner[i])],
            }
            for i in range(n)
            if worst[i] > ceiling
        ),
        # Player id breaks the tie. `design.players` is sorted, so this is a
        # total order and it is the same on every machine.
        key=lambda row: (-float(row["max_co_occurrence"]), int(row["player_id"])),
    )
    return {
        "ceiling": ceiling,
        "n_flagged": len(flagged),
        "n_players": n,
        "median_max_co_occurrence": float(np.median(worst)),
        "non_identified": flagged[:50],
    }


@dataclass
class RapmFit:
    """Fitted coefficients and every diagnostic needed to judge them."""

    players: list[int]
    off_rapm: dict[int, float]
    def_rapm: dict[int, float]
    home_advantage: float
    league_ppp: float
    lambda_offence: float
    lambda_defence: float
    effective_df: float
    condition_number: float
    n_possessions: int
    cv_mse: float
    #: Standard errors, per 100 possessions, same sign convention as the point
    #: estimates. Shipped beside every coefficient because a RAPM number without
    #: one is indistinguishable from a number with no evidence behind it.
    off_se: dict[int, float] = field(default_factory=dict)
    def_se: dict[int, float] = field(default_factory=dict)

    def interval(self, player_id: int, side: str, *, z: float = 1.96) -> tuple[float, float]:
        """Two-sided interval for one side of one player's effect."""
        point = (self.off_rapm if side == "off" else self.def_rapm).get(player_id, 0.0)
        error = (self.off_se if side == "off" else self.def_se).get(player_id, float("nan"))
        return point - z * error, point + z * error

    def includes_zero(self, player_id: int, side: str, *, z: float = 1.96) -> bool:
        """True when the interval straddles zero.

        The refusal contract uses this: a defensive coefficient whose interval
        contains zero is not evidence of defence, and must not be served as a
        point estimate however confident the ordering looks.
        """
        low, high = self.interval(player_id, side, z=z)
        return low <= 0.0 <= high

    def total(self, player_id: int) -> float:
        """Offence plus defence, both signed so higher is better."""
        return self.off_rapm.get(player_id, 0.0) + self.def_rapm.get(player_id, 0.0)

    def to_frame(self, appearances: dict[int, int]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_id": self.players,
                "off_rapm": [self.off_rapm[p] for p in self.players],
                "def_rapm": [self.def_rapm[p] for p in self.players],
                "total_rapm": [self.total(p) for p in self.players],
                "off_se": [self.off_se.get(p, float("nan")) for p in self.players],
                "def_se": [self.def_se.get(p, float("nan")) for p in self.players],
                "off_includes_zero": [self.includes_zero(p, "off") for p in self.players],
                "def_includes_zero": [self.includes_zero(p, "def") for p in self.players],
                "possessions": [appearances.get(p, 0) for p in self.players],
            }
        ).sort(["total_rapm", "player_id"], descending=[True, False])


def _fit_from_design(
    design: RapmDesign,
    mask: np.ndarray | None,
    lambda_offence: float,
    lambda_defence: float,
    *,
    diagnostics: bool = True,
    with_covariance: bool = False,
) -> tuple[RidgeSolution, float]:
    matrix = design.matrix if mask is None else design.matrix[mask]
    y = design.y if mask is None else design.y[mask]
    intercept = float(y.mean())
    centred = y - intercept

    gram = (matrix.T @ matrix).toarray()
    moment = np.asarray(matrix.T @ centred).ravel()
    penalty = design.penalty_vector(lambda_offence, lambda_defence)

    variance = None
    if with_covariance:
        # Two passes: solve, then use the residuals to estimate sigma^2 and
        # solve again for the covariance. The alternative -- estimating sigma^2
        # from the target's own variance -- ignores everything the model
        # explains and inflates every interval uniformly.
        first = solve_ridge(gram, moment, penalty, intercept=intercept, diagnostics=False)
        fitted = np.asarray(matrix @ first.coefficients).ravel()
        residuals = centred - fitted
        # Denominator uses the effective, not nominal, degrees of freedom.
        provisional = solve_ridge(gram, moment, penalty, intercept=intercept)
        denominator = max(len(residuals) - provisional.effective_df, 1.0)
        variance = float((residuals @ residuals) / denominator)

    return solve_ridge(
        gram,
        moment,
        penalty,
        intercept=intercept,
        diagnostics=diagnostics,
        residual_variance=variance,
    ), intercept


def _predict(design: RapmDesign, mask: np.ndarray, solution: RidgeSolution) -> np.ndarray:
    return np.asarray(design.matrix[mask] @ solution.coefficients).ravel() + solution.intercept


def select_lambda(
    design: RapmDesign,
    *,
    n_folds: int = 5,
    seed: int = 0,
    grid: tuple[float, ...] = LAMBDA_GRID,
) -> tuple[float, float, float, list[dict[str, float]]]:
    """Game-grouped cross-validation over the offence/defence lambda grid.

    Returns ``(lambda_offence, lambda_defence, mse, trace)``. The full trace is
    kept so a selected grid endpoint is visible in the run log -- that means the
    grid was too narrow, and it is not something to discover later.
    """
    folds = _game_folds(design.game_ids, n_folds=n_folds, seed=seed)
    trace: list[dict[str, float]] = []

    # The Gram matrix per training fold is reused across the whole grid.
    prepared = []
    for held in folds:
        train = ~held
        matrix = design.matrix[train]
        y = design.y[train]
        intercept = float(y.mean())
        prepared.append(
            (
                held,
                (matrix.T @ matrix).toarray(),
                np.asarray(matrix.T @ (y - intercept)).ravel(),
                intercept,
            )
        )

    best = (grid[len(grid) // 2], grid[len(grid) // 2], float("inf"))
    for lambda_offence in grid:
        for lambda_defence in grid:
            penalty = design.penalty_vector(lambda_offence, lambda_defence)
            errors: list[float] = []
            weights: list[int] = []
            for held, gram, moment, intercept in prepared:
                solution = solve_ridge(
                    gram, moment, penalty, intercept=intercept, diagnostics=False
                )
                predicted = _predict(design, held, solution)
                errors.append(float(np.mean((design.y[held] - predicted) ** 2)))
                weights.append(int(held.sum()))
            mse = float(np.average(errors, weights=weights))
            trace.append(
                {"lambda_offence": lambda_offence, "lambda_defence": lambda_defence, "mse": mse}
            )
            if mse < best[2]:
                best = (lambda_offence, lambda_defence, mse)

    return best[0], best[1], best[2], trace


def split_half_reliability(
    design: RapmDesign,
    lambda_offence: float,
    lambda_defence: float,
    *,
    min_possessions: int = RELIABILITY_MIN_POSSESSIONS,
) -> dict[str, Any]:
    """Fit odd and even games separately and correlate the coefficients.

    This is the honest test of whether RAPM measures anything. Cross-validated
    error on possession outcomes is dominated by shot noise -- a model can cut it
    while its player coefficients are close to arbitrary. Two disjoint halves of
    the same league agreeing about who is good cannot happen by accident.

    The Spearman-Brown correction converts the half-to-half correlation into the
    reliability implied for the full-sample fit, which is the number a consumer
    of ``off_rapm`` actually cares about.
    """
    from scipy import stats

    unique = np.unique(design.game_ids)
    order = {game: i for i, game in enumerate(np.sort(unique))}
    parity = np.array([order[game] % 2 for game in design.game_ids])

    first = parity == 0
    second = parity == 1
    if not first.any() or not second.any():
        return {"n_players": 0}

    left, _ = _fit_from_design(design, first, lambda_offence, lambda_defence, diagnostics=False)
    right, _ = _fit_from_design(design, second, lambda_offence, lambda_defence, diagnostics=False)

    n = design.n_players
    counts_left = np.asarray((design.matrix[first] > 0).sum(axis=0)).ravel()
    counts_right = np.asarray((design.matrix[second] > 0).sum(axis=0)).ravel()

    keep = [
        i
        for i in range(n)
        if min(counts_left[i] + counts_left[n + i], counts_right[i] + counts_right[n + i])
        >= min_possessions
    ]
    if len(keep) < 10:
        return {"n_players": len(keep)}

    out: dict[str, Any] = {"n_players": len(keep), "min_possessions": min_possessions}
    for label, offset in (("off", 0), ("def", n)):
        a = np.array([left.coefficients[offset + i] for i in keep])
        b = np.array([right.coefficients[offset + i] for i in keep])
        r = float(np.corrcoef(a, b)[0, 1])
        out[f"{label}_split_half_r"] = r
        out[f"{label}_spearman_rho"] = float(stats.spearmanr(a, b).statistic)
        # Spearman-Brown: reliability of the doubled-length test.
        out[f"{label}_full_sample_reliability"] = float(2 * r / (1 + r)) if r > -1 else float("nan")
    return out


@dataclass
class RapmReport:
    """Everything published about a RAPM fit."""

    fit: RapmFit
    reliability: dict[str, Any]
    co_occurrence: dict[str, Any]
    lambda_trace: list[dict[str, float]]
    #: Possessions each player appears in, either side. Shipped beside the
    #: coefficients so a consumer can always see how much evidence is behind one.
    appearances: dict[int, int] = field(default_factory=dict)
    #: Coefficient covariance and the column index of each player, kept so a
    #: trade contrast can use the exact ``Var(a) + Var(b) - 2 Cov(a, b)`` rather
    #: than the diagonal-only approximation.
    covariance: np.ndarray | None = None
    column_index: dict[int, int] = field(default_factory=dict)
    boundary_sensitivity: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)


def fit_rapm(
    possessions: pl.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 0,
    measure_boundary_sensitivity: bool = True,
) -> RapmReport:
    """Fit RAPM end to end, with every diagnostic the numbers need.

    Coefficients are returned in **points per 100 possessions**, and both are
    signed so that higher is better: ``def_rapm`` is negated, because the raw
    coefficient is points *conceded* and a defensive number where -3 is good
    reads wrong on every surface that displays it.
    """
    frame = usable_possessions(possessions)
    design = build_rapm_design(frame)

    lambda_offence, lambda_defence, cv_mse, trace = select_lambda(
        design, n_folds=n_folds, seed=seed
    )
    solution, intercept = _fit_from_design(
        design, None, lambda_offence, lambda_defence, with_covariance=True
    )

    n = design.n_players
    scale = 100.0
    errors = solution.standard_errors
    fit = RapmFit(
        players=design.players,
        off_rapm={
            player: float(solution.coefficients[i] * scale)
            for i, player in enumerate(design.players)
        },
        # Negated: the raw coefficient is points conceded per possession.
        def_rapm={
            player: float(-solution.coefficients[n + i] * scale)
            for i, player in enumerate(design.players)
        },
        home_advantage=float(solution.coefficients[-1] * scale),
        league_ppp=intercept,
        lambda_offence=lambda_offence,
        lambda_defence=lambda_defence,
        effective_df=solution.effective_df,
        condition_number=solution.condition_number,
        n_possessions=design.n_possessions,
        cv_mse=cv_mse,
        off_se=(
            {player: float(errors[i] * scale) for i, player in enumerate(design.players)}
            if errors is not None
            else {}
        ),
        def_se=(
            {player: float(errors[n + i] * scale) for i, player in enumerate(design.players)}
            if errors is not None
            else {}
        ),
    )

    reliability = split_half_reliability(design, lambda_offence, lambda_defence)
    co_occurrence = co_occurrence_report(design)

    offence = np.array(list(fit.off_rapm.values()))
    defence = np.array(list(fit.def_rapm.values()))
    spread = {
        "off_sd": float(offence.std()),
        "def_sd": float(defence.std()),
        "off_p95_minus_p05": float(np.quantile(offence, 0.95) - np.quantile(offence, 0.05)),
        "def_p95_minus_p05": float(np.quantile(defence, 0.95) - np.quantile(defence, 0.05)),
    }

    # Does dropping the ambiguously-attributed possessions change anything? The
    # answer belongs in the run log either way: "we excluded 9% of the data and
    # it moved nothing" and "we excluded 9% and it moved everything" call for
    # very different amounts of caution downstream.
    boundary: dict[str, float] = {}
    if measure_boundary_sensitivity:
        clean = usable_possessions(possessions, exclude_boundary_ambiguous=True)
        clean_design = build_rapm_design(clean)
        clean_solution, _ = _fit_from_design(
            clean_design, None, lambda_offence, lambda_defence, diagnostics=False
        )
        shared = [p for p in clean_design.players if p in fit.off_rapm]
        clean_index = {p: i for i, p in enumerate(clean_design.players)}
        m = clean_design.n_players
        a = np.array([fit.off_rapm[p] for p in shared])
        b = np.array([clean_solution.coefficients[clean_index[p]] * scale for p in shared])
        c = np.array([fit.def_rapm[p] for p in shared])
        d = np.array([-clean_solution.coefficients[m + clean_index[p]] * scale for p in shared])
        boundary = {
            "n_possessions_excluded": design.n_possessions - clean_design.n_possessions,
            "share_excluded": 1.0 - clean_design.n_possessions / design.n_possessions,
            "off_correlation": float(np.corrcoef(a, b)[0, 1]),
            "def_correlation": float(np.corrcoef(c, d)[0, 1]),
            "off_mean_abs_change": float(np.abs(a - b).mean()),
            "def_mean_abs_change": float(np.abs(c - d).mean()),
        }

    return RapmReport(
        fit=fit,
        appearances=design.appearances,
        covariance=solution.covariance,
        column_index={player: i for i, player in enumerate(design.players)},
        reliability=reliability,
        co_occurrence=co_occurrence,
        lambda_trace=trace,
        boundary_sensitivity=boundary,
        spread=spread,
    )


def summarise_rapm(report: RapmReport, names: dict[int, str] | None = None) -> pl.DataFrame:
    """Top and bottom of the total-RAPM ordering, for eyeballing."""
    fit = report.fit
    frame = pl.DataFrame(
        {
            "player_id": fit.players,
            "off_rapm": [fit.off_rapm[p] for p in fit.players],
            "def_rapm": [fit.def_rapm[p] for p in fit.players],
            "total_rapm": [fit.total(p) for p in fit.players],
        }
    )
    if names:
        frame = frame.with_columns(
            pl.col("player_id")
            .map_elements(lambda p: names.get(int(p), str(p)), return_dtype=pl.Utf8)
            .alias("player_name")
        )
    return frame.sort(["total_rapm", "player_id"], descending=[True, False])


def league_ppp(possessions: pl.DataFrame) -> float:
    return as_float(usable_possessions(possessions)["points"].mean())
