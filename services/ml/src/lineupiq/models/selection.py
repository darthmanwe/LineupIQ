"""The shot-selection model: P(zone | shooter, lineup, context).

This is the model the project should have started with.

The conversion model asks "will this shot go in?" and finds that who else is on
the floor barely matters -- a correct answer to a question nobody was asking.
Spacing does not make a player a better corner shooter. It gets him *a corner
three instead of a contested pull-up*. Lineup effects live in which shot gets
taken, so that is what this predicts: given that an attempt happened, which of
the nine zones did it come from.

**Form: conditional logit, not multinomial logistic.** A multinomial fit gives
every zone its own coefficient vector, so "spacing shifts attempts toward
threes" arrives as a pattern across 45 numbers that has to be eyeballed. Here
each hypothesis is one shared coefficient on a shot-level driver interacted with
a zone attribute, so ``spacing_x_three`` *is* the hypothesis, with a sign and a
magnitude. Twenty-odd parameters instead of two hundred also matters when the
effect being measured may be near zero: there is far less room for a model to
manufacture one.

**Every lineup coefficient has a pre-registered expected sign** (see
:data:`SELECTION_TERMS`). A model that improves log loss while pushing spacing
*away* from threes has found something, but not the thing being claimed, and the
sign audit is reported next to the metrics so that case cannot be quietly
presented as a win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from lineupiq.eval.leakage import assert_no_forbidden_features
from lineupiq.models.priors import fit_dirichlet_prior
from lineupiq.transform.zones import ZONE_IDS
from lineupiq.util import ABSENT_PLAYER, as_float, lineup_slots

__all__ = [
    "DESIGN_COLUMNS",
    "LINEUP_TERMS",
    "SELECTION_TERMS",
    "ConditionalLogit",
    "SelectionDesign",
    "SelectionMetrics",
    "SelectionProfiles",
    "SelectionTerm",
    "build_selection_design",
    "fit_selection_profiles",
    "score_selection",
    "usable_selection_frame",
    "zone_attribute",
]

#: Zones counted as at-the-rim and as threes. Derived from the taxonomy rather
#: than restated, so a change to the zone list cannot silently desynchronise.
_RIM_ZONES = ("restricted_area", "paint_non_ra")
_THREE_ZONES = ("corner_three_left", "corner_three_right", "wing_three", "top_three")

#: Alternative-specific constants are taken relative to this zone.
_REFERENCE_ZONE = "restricted_area"

_EPS = 1e-9

#: Exactly the columns :func:`build_selection_design` is allowed to read.
#:
#: The frame is narrowed to these before anything is computed, so the model
#: cannot read a leaky column by accident -- an unlisted name raises a missing
#: column error at the first touch rather than quietly becoming a feature. The
#: whitelist itself is checked against
#: :data:`lineupiq.eval.leakage.FORBIDDEN_FEATURES`, which closes the loop: a
#: forbidden column can neither be read nor added to the list without failing.
#:
#: ``possession_seconds`` and ``transition`` are deliberately absent. They are
#: measured to the possession's last event, and a possession ends on a make at
#: the shot but on a miss at the rebound a beat later, so a short possession is
#: evidence the shot went in -- 93.3% of shots that end their possession are
#: makes against 1.3% of those that do not.
DESIGN_COLUMNS: tuple[str, ...] = (
    "zone_id",
    "shooter_id",
    "team_id",
    "season",
    "period",
    "seconds_remaining",
    "seconds_into_possession",
    "live_ball_start",
    "is_second_chance",
    "lineup_for",
    "lineup_against",
)


def zone_attribute(name: str) -> np.ndarray:
    """A 0/1 indicator over :data:`ZONE_IDS` in canonical order."""
    members = {"rim": _RIM_ZONES, "three": _THREE_ZONES}[name]
    return np.array([1.0 if z in members else 0.0 for z in ZONE_IDS])


@dataclass(frozen=True)
class SelectionTerm:
    """One column of the utility, and what it is claimed to do.

    ``expected_sign`` is pre-registered: it says which way the coefficient must
    come out for the term to mean what its name says. ``None`` marks a term
    whose direction is genuinely not predicted in advance -- claiming otherwise
    after seeing the fit would make the sign audit worthless.
    """

    name: str
    kind: str  # "alt" | "pair" | "inter"
    detail: str
    expected_sign: int | None = None
    is_lineup: bool = False


#: The five terms that carry the lineup claim, each with the sign it must take.
LINEUP_TERMS: tuple[SelectionTerm, ...] = (
    SelectionTerm(
        "spacing_x_three",
        "inter",
        "Teammates' three-point attempt rate, times whether the zone is a three. "
        "Shooters surrounded by shooters should take more threes.",
        expected_sign=1,
        is_lineup=True,
    ),
    SelectionTerm(
        "spacing_min_x_three",
        "inter",
        "The *worst* spacer on the floor. One non-shooter collapses the geometry "
        "in a way a mean cannot express, because a good spacer offsets him.",
        expected_sign=1,
        is_lineup=True,
    ),
    SelectionTerm(
        "teammate_rim_x_rim",
        "inter",
        "Teammates' rim attempt rate, times whether the zone is at the rim. "
        "Teammates who live in the paint crowd it.",
        expected_sign=-1,
        is_lineup=True,
    ),
    SelectionTerm(
        "opp_rim_allowed_x_rim",
        "inter",
        "Share of opponents' attempts that came at the rim against these five "
        "defenders. A defence that concedes the rim concedes more rim attempts.",
        expected_sign=1,
        is_lineup=True,
    ),
    SelectionTerm(
        "opp_three_allowed_x_three",
        "inter",
        "The same for threes conceded.",
        expected_sign=1,
        is_lineup=True,
    ),
)

#: The full utility specification, in coefficient order. This order is part of
#: the serving contract: the TypeScript scorer consumes coefficients positionally.
SELECTION_TERMS: tuple[SelectionTerm, ...] = (
    *(
        SelectionTerm(f"alt_{zone}", "alt", f"Baseline preference for {zone}.")
        for zone in ZONE_IDS
        if zone != _REFERENCE_ZONE
    ),
    SelectionTerm(
        "shooter_mix",
        "pair",
        "Log ratio of the shooter's own shrunk zone mix to the league's. The "
        "single strongest signal in the model, and the bar the lineup terms have "
        "to clear.",
        expected_sign=1,
    ),
    SelectionTerm(
        "team_mix",
        "pair",
        "The same for his team's mix that season -- the offensive system. Included "
        "so that a lineup term cannot be credited for what is really a team "
        "effect: five-man units share personnel, so without this control "
        "'who is on the floor' and 'which team this is' are not separable.",
        expected_sign=1,
    ),
    SelectionTerm(
        "into_possession_x_rim",
        "inter",
        "Seconds elapsed in the possession, times whether the zone is at the rim.",
        expected_sign=-1,
    ),
    SelectionTerm(
        "into_possession_x_three",
        "inter",
        "The same for threes. Not pre-registered: the observed three-rate rises "
        "over the first seconds of a possession and falls after, so a single "
        "linear coefficient has no predicted direction.",
    ),
    SelectionTerm(
        "live_ball_x_rim",
        "inter",
        "Possession began with a defensive rebound or a steal, times rim.",
        expected_sign=1,
    ),
    SelectionTerm(
        "second_chance_x_rim",
        "inter",
        "Possession followed an offensive rebound, times rim. Putbacks happen "
        "where the rebound happened.",
        expected_sign=1,
    ),
    SelectionTerm(
        "clutch_x_three",
        "inter",
        "Final five minutes of the fourth or later, times three. Not "
        "pre-registered: trailing teams hunt threes, leading teams protect the "
        "ball, and the flag does not know which is which.",
    ),
    *LINEUP_TERMS,
)

TERM_NAMES: tuple[str, ...] = tuple(t.name for t in SELECTION_TERMS)
LINEUP_TERM_NAMES: tuple[str, ...] = tuple(t.name for t in LINEUP_TERMS)


@dataclass(frozen=True)
class SelectionProfiles:
    """Everything fitted from a training frame before any shot is scored.

    Featurisation takes this object as an argument and has no access to
    outcomes, so a test-set leak is not something a caller has to remember to
    avoid -- there is no code path that permits it.
    """

    #: player_id -> shrunk mix over ZONE_IDS (sums to 1).
    shooter_mix: dict[int, np.ndarray]
    #: player_id -> share of the estimate carried by evidence rather than prior.
    shooter_weight: dict[int, float]
    #: (team_id, two-digit season) -> shrunk mix over ZONE_IDS.
    team_mix: dict[tuple[int, int], np.ndarray]
    #: League mix over ZONE_IDS.
    league_mix: np.ndarray
    #: player_id -> share of his own attempts that were threes / at the rim.
    player_three_rate: dict[int, float]
    player_rim_rate: dict[int, float]
    #: player_id -> share of opponents' attempts that were threes / at the rim
    #: while he was on the floor.
    opp_three_allowed: dict[int, float]
    opp_rim_allowed: dict[int, float]
    league_three_rate: float
    league_rim_rate: float
    seconds_mean: float
    seconds_std: float
    #: Dirichlet concentration behind the mixes, published because it is the
    #: number that decides how much a low-volume player is trusted.
    shooter_prior_strength: float
    team_prior_strength: float


def _mix_matrix(frame: pl.DataFrame, group: str) -> tuple[list[int], np.ndarray]:
    """Counts per group per zone, in canonical zone order."""
    counted = (
        frame.group_by([group, "zone_id"])
        .agg(pl.len().alias("n"))
        .pivot(on="zone_id", index=group, values="n")
        .fill_null(0)
    )
    missing = [zone for zone in ZONE_IDS if zone not in counted.columns]
    if missing:
        counted = counted.with_columns([pl.lit(0).cast(pl.Int64).alias(z) for z in missing])
    keys = [int(k) for k in counted[group].to_list()]
    matrix = counted.select(list(ZONE_IDS)).to_numpy().astype(float)
    return keys, matrix


def _attribute_rate(counts: np.ndarray, attribute: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1)
    return np.divide(counts @ attribute, totals, out=np.zeros_like(totals), where=totals > 0)


def fit_selection_profiles(train: pl.DataFrame) -> SelectionProfiles:
    """Fit every profile the design matrix needs, from ``train`` alone."""
    league_counts = np.array(
        [train.filter(pl.col("zone_id") == zone).height for zone in ZONE_IDS], dtype=float
    )
    league_mix = league_counts / max(league_counts.sum(), 1.0)

    shooters, shooter_counts = _mix_matrix(train, "shooter_id")
    shooter_prior = fit_dirichlet_prior(shooter_counts, min_total=50)
    shooter_shrunk, shooter_w = shooter_prior.shrink(shooter_counts)

    # Team mix is fitted per team-season: a roster turning over between seasons
    # is a different offence, and pooling them would smear a rebuild into a
    # contender. The key is packed so one pivot handles both dimensions.
    teamed = train.with_columns(
        (pl.col("team_id") * 100 + (pl.col("season") % 100)).alias("_team_season")
    )
    team_keys, team_counts = _mix_matrix(teamed, "_team_season")
    team_prior = fit_dirichlet_prior(team_counts, min_total=200)
    team_shrunk, _ = team_prior.shrink(team_counts)

    three = zone_attribute("three")
    rim = zone_attribute("rim")

    shooter_three = _attribute_rate(shooter_counts, three)
    shooter_rim = _attribute_rate(shooter_counts, rim)
    totals = shooter_counts.sum(axis=1)

    # Defence: what the opposition's shot mix looked like with each player on
    # the floor. Exploding the defensive lineup makes this a per-defender
    # average rather than anything causal, and it is used as a control, not as a
    # claim about individual defenders.
    defenders = (
        train.select(pl.col("lineup_against").alias("defender_id"), "zone_id")
        .explode("defender_id", empty_as_null=True)
        .drop_nulls("defender_id")
    )
    def_keys, def_counts = _mix_matrix(defenders, "defender_id")
    def_totals = def_counts.sum(axis=1)
    def_three = _attribute_rate(def_counts, three)
    def_rim = _attribute_rate(def_counts, rim)

    seconds = train["seconds_into_possession"].drop_nulls().to_numpy().astype(float)

    return SelectionProfiles(
        shooter_mix=dict(zip(shooters, shooter_shrunk, strict=True)),
        shooter_weight={p: float(w) for p, w in zip(shooters, shooter_w, strict=True)},
        team_mix={
            (key // 100, key % 100): row for key, row in zip(team_keys, team_shrunk, strict=True)
        },
        league_mix=league_mix,
        # A 20-attempt floor on the offensive rates: below it the "rate" is a
        # restatement of the prior and only adds noise to a lineup aggregate.
        player_three_rate={
            p: float(v) for p, v, n in zip(shooters, shooter_three, totals, strict=True) if n >= 20
        },
        player_rim_rate={
            p: float(v) for p, v, n in zip(shooters, shooter_rim, totals, strict=True) if n >= 20
        },
        opp_three_allowed={
            p: float(v) for p, v, n in zip(def_keys, def_three, def_totals, strict=True) if n >= 100
        },
        opp_rim_allowed={
            p: float(v) for p, v, n in zip(def_keys, def_rim, def_totals, strict=True) if n >= 100
        },
        league_three_rate=float(league_mix @ three),
        league_rim_rate=float(league_mix @ rim),
        seconds_mean=float(seconds.mean()) if seconds.size else 14.0,
        seconds_std=(float(seconds.std()) if seconds.size else 1.0) or 1.0,
        shooter_prior_strength=shooter_prior.strength,
        team_prior_strength=team_prior.strength,
    )


@dataclass
class SelectionDesign:
    """The utility matrix, held in factored form so it fits in memory.

    Materialising a conditional logit's design as ``(n_shots, n_zones,
    n_terms)`` would be 6.3M rows by 20 columns for three seasons. Almost none
    of that is needed: an alternative-specific constant varies only by zone, and
    an interaction is an outer product of a shot-level driver and a zone
    attribute. Only the two mix terms genuinely vary along both axes. Stored
    this way the whole design is a handful of ``(n, 9)`` matrices plus some
    vectors, and every gradient stays a matrix product.
    """

    n: int
    y: np.ndarray
    alt_matrix: np.ndarray  # (n_zones, n_alt)
    pair_matrices: dict[str, np.ndarray]  # name -> (n, n_zones)
    inter_shot: dict[str, np.ndarray]  # name -> (n,)
    inter_alt: dict[str, np.ndarray]  # name -> (n_zones,)
    term_names: tuple[str, ...] = TERM_NAMES
    #: Cached interaction grouping. Derived, never passed in.
    _groups: list[tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def n_zones(self) -> int:
        return len(ZONE_IDS)

    def without(self, names: tuple[str, ...]) -> SelectionDesign:
        """A copy with the named terms zeroed rather than removed.

        Zeroing keeps the coefficient vector the same length and the design the
        same shape, so a comparison between two rungs of the ladder differs in
        exactly one thing: whether those columns carry information.
        """
        return SelectionDesign(
            n=self.n,
            y=self.y,
            alt_matrix=self.alt_matrix,
            pair_matrices={
                k: (np.zeros_like(v) if k in names else v) for k, v in self.pair_matrices.items()
            },
            inter_shot={
                k: (np.zeros_like(v) if k in names else v) for k, v in self.inter_shot.items()
            },
            inter_alt=self.inter_alt,
            term_names=self.term_names,
        )

    def attribute_groups(self) -> list[tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]]:
        """Interaction terms grouped by the zone attribute they multiply.

        Every interaction shares one of a very small number of zone attributes --
        here just ``rim`` and ``three``. Since

            sum_k theta_k b_k[i] a_k[z]
                = sum_groups (sum_{k in group} theta_k b_k[i]) a_group[z]

        the ten outer products collapse to one per distinct attribute, and each
        of those touches only the columns its attribute selects. The grouping is
        derived from ``inter_alt``, not hardcoded, so a term with a new attribute
        needs no change here.

        Returns ``(columns, weights, [(theta_index, shot_vector)])`` per group.
        """
        if self._groups is None:
            offset = self.alt_matrix.shape[1] + len(self.pair_matrices)
            buckets: dict[bytes, list[tuple[int, np.ndarray]]] = {}
            attributes: dict[bytes, np.ndarray] = {}
            for position, name in enumerate(self.inter_shot):
                alt = self.inter_alt[name]
                key = alt.tobytes()
                attributes.setdefault(key, alt)
                buckets.setdefault(key, []).append((offset + position, self.inter_shot[name]))
            self._groups = [
                (
                    np.flatnonzero(attributes[key]),
                    attributes[key][np.flatnonzero(attributes[key])],
                    members,
                )
                for key, members in buckets.items()
            ]
        return self._groups

    def utilities(self, theta: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Assemble the ``(n, n_zones)`` utility matrix.

        Written for allocation, not for looks. The naive form -- one
        ``np.outer`` per interaction term -- allocated thirteen ``(n, 9)``
        matrices per call, about 620 MB of churn per L-BFGS iteration at three
        seasons, across several hundred iterations and eighteen fits. That is
        enough to take a workstation down, and it did.

        This version writes into a caller-supplied buffer, accumulates the pair
        terms in place, and adds each attribute group only to the columns that
        attribute selects. ``test_utilities_match_the_naive_reference`` asserts
        it is arithmetically identical to the obvious implementation -- a
        performance rewrite of the objective's hot path is exactly the kind of
        change that silently alters what is being optimised.
        """
        n_alt = self.alt_matrix.shape[1]
        u = np.empty((self.n, self.n_zones)) if out is None else out
        u[:] = self.alt_matrix @ theta[:n_alt]

        index = n_alt
        scratch = np.empty((self.n, self.n_zones))
        for matrix in self.pair_matrices.values():
            if theta[index]:
                np.multiply(matrix, theta[index], out=scratch)
                np.add(u, scratch, out=u)
            index += 1

        combined = np.empty(self.n)
        for columns, weights, members in self.attribute_groups():
            if not columns.size:
                continue
            combined[:] = 0.0
            for position, shot in members:
                if theta[position]:
                    combined += theta[position] * shot
            u[:, columns] += combined[:, None] * weights[None, :]
        return u

    def gradient_from_residual(self, residual: np.ndarray) -> np.ndarray:
        """``d loss / d theta`` given ``residual = (p - onehot) / n``."""
        grads: list[np.ndarray] = [residual.sum(axis=0) @ self.alt_matrix]
        for matrix in self.pair_matrices.values():
            grads.append(np.array([float((residual * matrix).sum())]))
        for name, shot in self.inter_shot.items():
            grads.append(np.array([float(shot @ (residual @ self.inter_alt[name]))]))
        return np.concatenate(grads)

    @property
    def penalty_mask(self) -> np.ndarray:
        """Which coefficients the ridge penalty applies to.

        Not the alternative-specific constants: they carry the league's own shot
        mix, and shrinking them toward zero would pull every prediction toward a
        uniform nine-way split that nothing in the data supports.
        """
        mask = np.ones(len(self.term_names))
        mask[: self.alt_matrix.shape[1]] = 0.0
        return mask


def build_selection_design(frame: pl.DataFrame, profiles: SelectionProfiles) -> SelectionDesign:
    """Build the factored design for ``frame`` using train-fitted profiles."""
    assert_no_forbidden_features(list(DESIGN_COLUMNS))
    frame = frame.select(DESIGN_COLUMNS)

    n = frame.height
    n_zones = len(ZONE_IDS)
    zone_index = {zone: i for i, zone in enumerate(ZONE_IDS)}
    y = np.array([zone_index[z] for z in frame["zone_id"].to_list()], dtype=int)

    alt_zones = [z for z in ZONE_IDS if z != _REFERENCE_ZONE]
    alt_matrix = np.zeros((n_zones, len(alt_zones)))
    for j, zone in enumerate(alt_zones):
        alt_matrix[zone_index[zone], j] = 1.0

    log_league = np.log(np.maximum(profiles.league_mix, _EPS))
    shooters = frame["shooter_id"].to_list()
    teams = frame["team_id"].to_list()
    seasons = frame["season"].to_list()

    # An unseen shooter or team gets the league mix, whose log ratio is exactly
    # zero -- so the model falls back to the alternative-specific constants
    # rather than to an arbitrary player's profile.
    zero = np.zeros(n_zones)
    shooter_mix = np.empty((n, n_zones))
    team_mix = np.empty((n, n_zones))
    for i in range(n):
        row = profiles.shooter_mix.get(int(shooters[i]))
        shooter_mix[i] = np.log(np.maximum(row, _EPS)) - log_league if row is not None else zero
        trow = profiles.team_mix.get((int(teams[i]), int(seasons[i]) % 100))
        team_mix[i] = np.log(np.maximum(trow, _EPS)) - log_league if trow is not None else zero

    # --- lineup aggregates ------------------------------------------------
    # Five flat arrays per column rather than 600k Python lists. See
    # `util.lineup_slots` for why.
    for_slots = lineup_slots(frame["lineup_for"])
    against_slots = lineup_slots(frame["lineup_against"])
    spacing = np.zeros(n)
    spacing_min = np.zeros(n)
    teammate_rim = np.zeros(n)
    opp_rim = np.zeros(n)
    opp_three = np.zeros(n)

    for i in range(n):
        shooter = int(shooters[i])
        teammates = [
            player
            for player in (int(slot[i]) for slot in for_slots)
            if player != ABSENT_PLAYER and player != shooter
        ]
        if teammates:
            rates = [
                profiles.player_three_rate.get(t, profiles.league_three_rate) for t in teammates
            ]
            rims = [profiles.player_rim_rate.get(t, profiles.league_rim_rate) for t in teammates]
            spacing[i] = float(np.mean(rates)) - profiles.league_three_rate
            spacing_min[i] = min(rates) - profiles.league_three_rate
            teammate_rim[i] = float(np.mean(rims)) - profiles.league_rim_rate
        defenders = [
            player for player in (int(slot[i]) for slot in against_slots) if player != ABSENT_PLAYER
        ]
        if defenders:
            opp_rim[i] = (
                float(
                    np.mean(
                        [
                            profiles.opp_rim_allowed.get(d, profiles.league_rim_rate)
                            for d in defenders
                        ]
                    )
                )
                - profiles.league_rim_rate
            )
            opp_three[i] = (
                float(
                    np.mean(
                        [
                            profiles.opp_three_allowed.get(d, profiles.league_three_rate)
                            for d in defenders
                        ]
                    )
                )
                - profiles.league_three_rate
            )

    # --- context ----------------------------------------------------------
    seconds = (
        frame["seconds_into_possession"].fill_null(profiles.seconds_mean).to_numpy().astype(float)
    )
    seconds_z = (seconds - profiles.seconds_mean) / profiles.seconds_std
    live = frame["live_ball_start"].fill_null(value=False).to_numpy().astype(float)
    second_chance = frame["is_second_chance"].fill_null(value=False).to_numpy().astype(float)
    period = frame["period"].to_numpy().astype(float)
    secs_left = frame["seconds_remaining"].to_numpy().astype(float)
    clutch = ((period >= 4) & (secs_left <= 300)).astype(float)

    three = zone_attribute("three")
    rim = zone_attribute("rim")

    return SelectionDesign(
        n=n,
        y=y,
        alt_matrix=alt_matrix,
        pair_matrices={"shooter_mix": shooter_mix, "team_mix": team_mix},
        inter_shot={
            "into_possession_x_rim": seconds_z,
            "into_possession_x_three": seconds_z,
            "live_ball_x_rim": live,
            "second_chance_x_rim": second_chance,
            "clutch_x_three": clutch,
            "spacing_x_three": spacing,
            "spacing_min_x_three": spacing_min,
            "teammate_rim_x_rim": teammate_rim,
            "opp_rim_allowed_x_rim": opp_rim,
            "opp_three_allowed_x_three": opp_three,
        },
        inter_alt={
            "into_possession_x_rim": rim,
            "into_possession_x_three": three,
            "live_ball_x_rim": rim,
            "second_chance_x_rim": rim,
            "clutch_x_three": three,
            "spacing_x_three": three,
            "spacing_min_x_three": three,
            "teammate_rim_x_rim": rim,
            "opp_rim_allowed_x_rim": rim,
            "opp_three_allowed_x_three": three,
        },
    )


def _softmax(u: np.ndarray) -> np.ndarray:
    """Row-wise softmax, computed in place.

    ``u`` is overwritten. It is always a scratch buffer owned by the caller, and
    at three seasons each copy avoided here is 48 MB.
    """
    u -= u.max(axis=1, keepdims=True)
    np.exp(u, out=u)
    u /= u.sum(axis=1, keepdims=True)
    return u


@dataclass
class ConditionalLogit:
    """Softmax over zones with one shared coefficient per driver.

    Fitted by L-BFGS on the exact analytic gradient. The gradient is checked
    against finite differences in the test suite -- a hand-derived gradient that
    is subtly wrong does not crash, it converges to the wrong answer and reports
    a plausible log loss.
    """

    coefficients: np.ndarray | None = None
    term_names: tuple[str, ...] = TERM_NAMES
    l2: float = 1e-4
    n_iterations: int = 0
    converged: bool = False
    final_loss: float = float("nan")
    #: Filled by :meth:`compute_standard_errors`, on the served fit only.
    standard_errors: np.ndarray | None = None

    def objective(
        self,
        theta: np.ndarray,
        design: SelectionDesign,
        buffer: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Penalised mean negative log likelihood, and its exact gradient.

        The probability matrix is reused as the residual matrix rather than
        copied: after reading off the chosen-class probabilities, subtracting the
        one-hot in place turns ``p`` into ``p - onehot`` with no second
        allocation. Callers pass ``buffer`` so a whole L-BFGS run works out of
        one ``(n, n_zones)`` array.
        """
        p = _softmax(design.utilities(theta, out=buffer))
        rows = np.arange(design.n)
        chosen = p[rows, design.y]
        loss = float(-np.log(np.maximum(chosen, 1e-300)).mean())

        p[rows, design.y] -= 1.0
        p /= design.n
        grad = design.gradient_from_residual(p)

        mask = design.penalty_mask
        loss += 0.5 * self.l2 * float((mask * theta * theta).sum())
        return loss, grad + self.l2 * mask * theta

    def fit(self, design: SelectionDesign) -> ConditionalLogit:
        from scipy.optimize import minimize

        # One buffer for the entire optimisation. Every call to `objective`
        # writes utilities into it, converts them to probabilities in place, and
        # then to residuals in place.
        buffer = np.empty((design.n, design.n_zones))
        result = minimize(
            self.objective,
            np.zeros(len(design.term_names)),
            args=(design, buffer),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10},
        )
        self.coefficients = np.asarray(result.x, dtype=float)
        self.term_names = design.term_names
        self.n_iterations = int(result.nit)
        self.converged = bool(result.success)
        self.final_loss = float(result.fun)
        return self

    def observed_information(self, design: SelectionDesign, *, step: float = 1e-5) -> np.ndarray:
        """Hessian of the mean **unpenalised** negative log likelihood at the fit.

        Central differences on the analytic gradient, which is the cheap and
        accurate way round: differencing the gradient costs ``2p`` evaluations for
        ``p`` parameters, while differencing the loss twice costs ``2p^2`` and
        loses half the significant digits. Twenty parameters is forty evaluations.

        The penalty is subtracted rather than avoided. ``objective`` returns the
        penalised gradient because that is what the optimiser needs, and the
        penalty's own gradient is exactly ``l2 * mask * theta`` -- so removing it
        is arithmetic, not a second code path that could drift from the first.

        The result is symmetrised. Finite differences are symmetric only up to
        truncation error, and an asymmetric "Hessian" propagates into a covariance
        with negative variances on the diagonal, which surfaces as a nan much
        later and looks like a data problem.
        """
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")

        theta = np.asarray(self.coefficients, dtype=float)
        mask = design.penalty_mask
        n_terms = len(theta)
        buffer = np.empty((design.n, design.n_zones))

        def unpenalised_gradient(point: np.ndarray) -> np.ndarray:
            _, grad = self.objective(point, design, buffer)
            return grad - self.l2 * mask * point

        hessian = np.empty((n_terms, n_terms))
        for j in range(n_terms):
            h = step * max(1.0, abs(theta[j]))
            forward, backward = theta.copy(), theta.copy()
            forward[j] += h
            backward[j] -= h
            hessian[:, j] = (unpenalised_gradient(forward) - unpenalised_gradient(backward)) / (
                2.0 * h
            )
        return 0.5 * (hessian + hessian.T)

    def coefficient_covariance(self, design: SelectionDesign) -> np.ndarray:
        """Asymptotic covariance of the fitted coefficients.

        The **ridge sandwich** ``H^-1 I H^-1``, not the inverse Hessian. The fit
        is penalised, so a plain inverse would describe a different estimator
        than the one that produced these numbers: shrinkage trades bias for
        variance, and the sandwich is what accounts for both sides of it. RAPM in
        this repository uses the same estimator for the same reason.

        ``objective`` returns the *mean* negative log likelihood, so its Hessian
        is per-observation information and the total is ``n`` times larger. The
        ``1 / n`` here is where the sample size enters, and it is why these
        intervals are tight while a five-man lineup's rating is not: 671,251
        attempts is a great deal of evidence about twenty parameters, and almost
        none about any particular combination of five players.
        """
        information = self.observed_information(design)
        penalised = information + self.l2 * np.diag(design.penalty_mask)
        inverse = np.linalg.inv(penalised)
        return inverse @ information @ inverse / design.n

    def compute_standard_errors(self, design: SelectionDesign) -> ConditionalLogit:
        """Attach standard errors. Called on the served fit, not on every fold.

        Deliberately separate from ``fit``. Cross-validation fits this model
        eighteen times per pass, and none of those fits needs a covariance
        matrix -- only the coefficients that actually get served do.
        """
        variances = np.diag(self.coefficient_covariance(design))
        # A negative variance means the sandwich came out indefinite, which is a
        # real signal about a flat direction in the likelihood. Taking an
        # absolute value would launder it into a small, confident-looking
        # standard error, so it becomes a nan and the audit reports no interval.
        self.standard_errors = np.where(variances > 0, np.sqrt(np.abs(variances)), np.nan)
        return self

    def predict_proba(self, design: SelectionDesign) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")
        return _softmax(design.utilities(self.coefficients))

    def coefficient(self, name: str) -> float:
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")
        return float(self.coefficients[self.term_names.index(name)])

    def sign_audit(self) -> dict[str, dict[str, object]]:
        """Compare every fitted coefficient against its pre-registered sign.

        Three verdicts once standard errors are available, not two.

        An audit that only reports *agrees* or *DISAGREES* is weaker than it
        looks. A coefficient of +0.097 whose 95% interval spans zero gets
        recorded as agreeing with a positive prediction, and it agrees with
        nothing -- the data cannot tell. Counting that as a confirmation is the
        same error as reading a null result as a refutation.

        So a term whose interval contains zero is **indeterminate** and counts as
        neither. That can only make the audit harder to pass, which is the
        direction an audit should err in.
        """
        out: dict[str, dict[str, object]] = {}
        for term in SELECTION_TERMS:
            if term.expected_sign is None or term.name not in self.term_names:
                continue
            value = self.coefficient(term.name)
            entry: dict[str, object] = {
                "value": value,
                "expected_sign": term.expected_sign,
                "is_lineup": term.is_lineup,
            }

            error: float | None = None
            if self.standard_errors is not None:
                candidate = float(self.standard_errors[self.term_names.index(term.name)])
                error = candidate if np.isfinite(candidate) else None

            if error is None:
                entry["standard_error"] = None
                entry["verdict"] = "agrees" if np.sign(value) == term.expected_sign else "DISAGREES"
                out[term.name] = entry
                continue

            # 1.96 for a two-sided 95% interval on an asymptotically normal MLE.
            half_width = 1.96 * error
            entry["standard_error"] = error
            entry["z"] = value / error if error > 0 else float("nan")
            entry["ci95"] = [value - half_width, value + half_width]
            if abs(value) <= half_width:
                entry["verdict"] = "indeterminate"
            elif np.sign(value) == term.expected_sign:
                entry["verdict"] = "agrees"
            else:
                entry["verdict"] = "DISAGREES"
            out[term.name] = entry
        return out

    def to_dict(self) -> dict[str, object]:
        coefficients = [] if self.coefficients is None else [float(c) for c in self.coefficients]
        return {
            "term_names": list(self.term_names),
            "coefficients": coefficients,
            "l2": self.l2,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "standard_errors": (
                None
                if self.standard_errors is None
                else [None if not np.isfinite(v) else float(v) for v in self.standard_errors]
            ),
            "sign_audit": self.sign_audit(),
        }


def shooter_mix_prediction(frame: pl.DataFrame, profiles: SelectionProfiles) -> np.ndarray:
    """S1: the shooter's own shrunk mix. A lookup table, and the bar to clear."""
    rows = [
        profiles.shooter_mix.get(int(pid), profiles.league_mix)
        for pid in frame["shooter_id"].to_list()
    ]
    return np.vstack(rows)


@dataclass(frozen=True)
class SelectionMetrics:
    """Scores for one model on one fold.

    Multiclass log loss is the headline, but a nine-way log loss is hard to
    reason about, so the two projections the product actually talks about are
    reported beside it: the probability the shot is a three, and the probability
    it is at the rim. Those are ordinary binary predictions and carry ordinary
    Brier decompositions, which is what makes "does this lineup shoot more
    threes" a checkable question rather than a vibe.
    """

    n: int
    log_loss: float
    top1_accuracy: float
    classwise_ece: float
    three_log_loss: float
    three_brier: float
    three_resolution: float
    three_ece: float
    rim_log_loss: float
    rim_brier: float
    rim_resolution: float

    def to_dict(self) -> dict[str, float]:
        return {
            "n": float(self.n),
            "log_loss": self.log_loss,
            "top1_accuracy": self.top1_accuracy,
            "classwise_ece": self.classwise_ece,
            "three_log_loss": self.three_log_loss,
            "three_brier": self.three_brier,
            "three_resolution": self.three_resolution,
            "three_ece": self.three_ece,
            "rim_log_loss": self.rim_log_loss,
            "rim_brier": self.rim_brier,
            "rim_resolution": self.rim_resolution,
        }


def score_selection(y: np.ndarray, p: np.ndarray) -> SelectionMetrics:
    """Score a predicted zone distribution against the realised zone."""
    from lineupiq.eval.metrics import brier_decomposition, expected_calibration_error, log_loss

    n = len(y)
    rows = np.arange(n)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    # Classwise ECE: the mean over zones of the ordinary binary calibration
    # error for "is it this zone". A model can be well calibrated on the
    # frequent zones and badly wrong on corner threes, and a single aggregate
    # would hide exactly the zone the product cares about most.
    classwise = float(
        np.mean(
            [
                expected_calibration_error((y == j).astype(float), p[:, j], n_bins=10)
                for j in range(p.shape[1])
            ]
        )
    )

    three = zone_attribute("three")
    rim = zone_attribute("rim")
    y_three = three[y]
    y_rim = rim[y]
    p_three = np.clip(p @ three, 1e-12, 1 - 1e-12)
    p_rim = np.clip(p @ rim, 1e-12, 1 - 1e-12)

    three_decomposition = brier_decomposition(y_three, p_three)
    rim_decomposition = brier_decomposition(y_rim, p_rim)

    return SelectionMetrics(
        n=n,
        log_loss=float(-np.log(p[rows, y]).mean()),
        top1_accuracy=float((p.argmax(axis=1) == y).mean()),
        classwise_ece=classwise,
        three_log_loss=log_loss(y_three, p_three),
        three_brier=three_decomposition.brier,
        three_resolution=three_decomposition.resolution,
        three_ece=expected_calibration_error(y_three, p_three),
        rim_log_loss=log_loss(y_rim, p_rim),
        rim_brier=rim_decomposition.brier,
        rim_resolution=rim_decomposition.resolution,
    )


def wide_features(design: SelectionDesign) -> np.ndarray:
    """Flatten the factored design into a per-shot matrix for a GBDT.

    The tree model needs one row per shot, not one per alternative, so the two
    mix terms contribute nine columns each and every interaction contributes its
    shot-level driver. It sees exactly the information the conditional logit
    sees; what it does not have is the constraint that an effect enters as a
    single shared coefficient.
    """
    blocks: list[np.ndarray] = list(design.pair_matrices.values())
    blocks.extend(v.reshape(-1, 1) for v in design.inter_shot.values())
    return np.hstack(blocks)


def wide_feature_names(design: SelectionDesign) -> list[str]:
    names: list[str] = []
    for key in design.pair_matrices:
        names.extend(f"{key}__{zone}" for zone in ZONE_IDS)
    names.extend(design.inter_shot.keys())
    return names


def lineup_wide_indices(design: SelectionDesign) -> tuple[int, ...]:
    """Columns of :func:`wide_features` that carry lineup information."""
    names = wide_feature_names(design)
    return tuple(i for i, name in enumerate(names) if name in LINEUP_TERM_NAMES)


def usable_selection_frame(shots: pl.DataFrame) -> pl.DataFrame:
    """Filter to shots this model may be fitted on.

    A guessed lineup must never become a coefficient, and a shot with no
    possession context has no shot clock, so both are excluded here rather than
    being carried along with imputed values.
    """
    return shots.filter(
        pl.col("lineup_for_hash").is_not_null()
        & pl.col("zone_id").is_not_null()
        & (pl.col("stint_quality") == "VALID")
        & pl.col("has_possession_context")
    )


def summarise_mix(frame: pl.DataFrame) -> dict[str, float]:
    """Observed zone shares, for the report."""
    total = max(frame.height, 1)
    return {
        zone: as_float(frame.filter(pl.col("zone_id") == zone).height / total) for zone in ZONE_IDS
    }
