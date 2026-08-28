/**
 * The selection model, evaluated in the Worker.
 *
 * A conditional logit's prediction is `softmax(Xθ)` over nine zones, and every
 * term is either a per-zone constant, a scalar times a zone indicator, or a
 * per-player vector. So one lineup costs a few dozen multiply-adds and nine
 * exponentials — microseconds against a 10 ms CPU budget, reading nothing but a
 * 210 KB JSON file.
 *
 * That is what makes the counterfactual possible at all. The optimizer accepts
 * any five of ~450 players and `C(450, 5)` is 1.5e11, so no amount of
 * precomputation would have covered it. The gradient-boosted reference model
 * fits better and cannot be served this way; the log-loss gap between them is
 * published rather than hidden.
 *
 * **This file and `services/ml/src/lineupiq/serve/score.py` must agree.** Neither
 * is the reference: both read the same exported contract, and
 * `data/parity/selection.json` holds 507 cases that a vitest suite re-scores
 * here and asserts to 1e-9. A disagreement raises nothing at runtime — it
 * serves a plausible wrong number — so the fixture is the only thing standing
 * between a refactor and a silent lie.
 */

/** The zone whose alternative-specific constant is pinned at zero. */
export const REFERENCE_ZONE = "restricted_area";

export type SelectionProfiles = {
  zones: string[];
  rim: number[];
  three: number[];
  league_mix: number[];
  /** League points per attempt, in `zones` order. */
  zone_points: number[];
  shooter_log_ratio: Record<string, number[]>;
  shooter_weight: Record<string, number>;
  team_log_ratio: Record<string, number[]>;
  player_three_rate: Record<string, number>;
  player_rim_rate: Record<string, number>;
  opp_three_allowed: Record<string, number>;
  opp_rim_allowed: Record<string, number>;
  /**
   * Cluster-robust standard errors on the four rate tables above, clustered on
   * game.
   *
   * Only `compare.ts` reads them, and it refuses any player they do not cover.
   * A comparison's answer is driven by the difference between two players'
   * shooting rates, and the coefficient covariance says nothing about those —
   * so serving a comparison without these would report a model interval as
   * though it were the whole uncertainty.
   *
   * Optional on the type because a snapshot exported before they existed still
   * scores lineups perfectly well; it just cannot compare them.
   */
  player_three_rate_se?: Record<string, number>;
  player_rim_rate_se?: Record<string, number>;
  opp_three_allowed_se?: Record<string, number>;
  opp_rim_allowed_se?: Record<string, number>;
  league_three_rate: number;
  league_rim_rate: number;
  seconds_mean: number;
  seconds_std: number;
};

export type SelectionModel = {
  available: boolean;
  term_names: string[];
  coefficients: number[];
};

/**
 * Per-table, per-player replacements for the four rate tables.
 *
 * This exists so `compare.ts` can differentiate the **served** scorer with
 * respect to a player's shooting rate, exactly the way `plays.ts` already
 * differentiates it with respect to a coefficient. The alternative — a
 * hand-derived softmax gradient — would be a second implementation of the model
 * that could drift from this one, and it would drift silently: a wrong gradient
 * does not throw, it just makes every interval the wrong width.
 *
 * With no overrides the arithmetic is unchanged down to the last bit, which is
 * why `data/parity/selection.json` is untouched by this parameter.
 */
export type RateOverrides = Partial<Record<RateTable, Record<string, number>>>;

/** The four rate tables an override can address. */
export type RateTable =
  "player_three_rate" | "player_rim_rate" | "opp_three_allowed" | "opp_rim_allowed";

export type ScoreRequest = {
  shooterId: number;
  offense: number[];
  defense: number[];
  teamId?: number | null;
  season?: number | null;
  /** Null means "league-average possession", which is the mean, not zero. */
  secondsIntoPossession?: number | null;
  liveBall?: boolean;
  secondChance?: boolean;
  clutch?: boolean;
};

export type ScoreResult = {
  zones: string[];
  mix: number[];
  /** The same shooter with every lineup term at the league average. */
  baselineMix: number[];
  utilities: number[];
  shooterKnown: boolean;
  shooterWeight: number;
  /**
   * The five lineup features, in the order their coefficients appear in the
   * term list: spacing, spacingMin, teammateRim, oppRim, oppThree.
   *
   * Every one is a centred deviation from the league rate, which is why
   * dropping all five gives the league-average lineup. Exposed because
   * `compare.ts` reports *which* of them a swap moved, and recomputing them
   * there would be a second implementation of the aggregation — including the
   * `min`, which is the one most likely to be got subtly wrong twice.
   */
  lineupFeatures: number[];
  /**
   * The shot-mix shift, priced in points per 100 attempts.
   *
   * `sum(deltaShare * leaguePointsPerAttempt) * 100`. This is the number the
   * product is about: "0.27 percentage points more corner threes" is not
   * something anyone can act on, and this is the same fact in units that are.
   *
   * Priced at **league** conversion rates, not the shooter's own. That is the
   * estimand rather than a shortcut — using his rates would fold the two
   * channels back together, so part of the answer would be "he shoots better
   * from there" and part "the lineup got him there". At fixed conversion, all of
   * it is selection.
   */
  pointsPer100: number;
};

function softmax(values: number[]): number[] {
  // Shifted by the max. These utilities never overflow, but an unshifted
  // softmax is a latent bug that surfaces on the one input nobody tried.
  let top = -Infinity;
  for (const v of values) if (v > top) top = v;
  const exps = values.map((v) => Math.exp(v - top));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

/**
 * Mean of `ids` looked up in `table`, falling back to `fallback`.
 *
 * The loop is written out rather than composed from `map`/`reduce` so the
 * summation order is unambiguous. Floating-point addition is not associative,
 * and this has to match Python's left-to-right `sum()` exactly or the parity
 * fixture fails in the last few bits.
 */
/**
 * One player's rate, with an override taking precedence over the table.
 *
 * The fallback is the league rate, and it is why a player below the profile
 * fit's attempt floor scores as exactly league-average rather than throwing.
 * That is correct for scoring one lineup and wrong for comparing two, so
 * `compare.ts` refuses such a player rather than serving the zero this would
 * produce.
 */
function rateOf(
  table: Record<string, number>,
  overrides: Record<string, number> | undefined,
  id: number,
  fallback: number
): number {
  const key = String(id);
  if (overrides !== undefined) {
    const override = overrides[key];
    if (override !== undefined) return override;
  }
  const value = table[key];
  return value === undefined ? fallback : value;
}

function meanRate(
  ids: number[],
  table: Record<string, number>,
  overrides: Record<string, number> | undefined,
  fallback: number
): number {
  let total = 0;
  for (const id of ids) {
    total += rateOf(table, overrides, id, fallback);
  }
  return total / ids.length;
}

function minRate(
  ids: number[],
  table: Record<string, number>,
  overrides: Record<string, number> | undefined,
  fallback: number
): number {
  let smallest = Infinity;
  for (const id of ids) {
    const rate = rateOf(table, overrides, id, fallback);
    if (rate < smallest) smallest = rate;
  }
  return smallest;
}

export function scoreSelection(
  request: ScoreRequest,
  profiles: SelectionProfiles,
  model: SelectionModel,
  rateOverrides?: RateOverrides
): ScoreResult {
  const { zones, rim, three } = profiles;
  const nZones = zones.length;

  // Looked up by name, not by position. The coefficient order is a documented
  // contract, but a contract read through a map fails loudly when it is broken
  // and one read by counting fails silently with plausible numbers.
  const theta = new Map<string, number>();
  model.term_names.forEach((name, i) => theta.set(name, model.coefficients[i] ?? 0));
  const coef = (name: string): number => theta.get(name) ?? 0;

  const shooterKey = String(request.shooterId);
  const known = profiles.shooter_log_ratio[shooterKey];
  const shooterKnown = known !== undefined;
  // Exactly zero, not some arbitrary player: the log ratio of the league mix to
  // itself. The model falls back to its alternative-specific constants.
  const shooterRatio = known ?? new Array<number>(nZones).fill(0);
  const shooterWeight = profiles.shooter_weight[shooterKey] ?? 0;

  let teamRatio = new Array<number>(nZones).fill(0);
  if (request.teamId != null && request.season != null) {
    const twoDigit = String(request.season % 100).padStart(2, "0");
    teamRatio = profiles.team_log_ratio[`${request.teamId}:${twoDigit}`] ?? teamRatio;
  }

  const leagueThree = profiles.league_three_rate;
  const leagueRim = profiles.league_rim_rate;

  const teammates = request.offense.filter((p) => p !== request.shooterId);
  let spacing = 0;
  let spacingMin = 0;
  let teammateRim = 0;
  if (teammates.length > 0) {
    const threeOverride = rateOverrides?.player_three_rate;
    const rimOverride = rateOverrides?.player_rim_rate;
    spacing =
      meanRate(teammates, profiles.player_three_rate, threeOverride, leagueThree) - leagueThree;
    spacingMin =
      minRate(teammates, profiles.player_three_rate, threeOverride, leagueThree) - leagueThree;
    teammateRim = meanRate(teammates, profiles.player_rim_rate, rimOverride, leagueRim) - leagueRim;
  }

  let oppRim = 0;
  let oppThree = 0;
  if (request.defense.length > 0) {
    oppRim =
      meanRate(
        request.defense,
        profiles.opp_rim_allowed,
        rateOverrides?.opp_rim_allowed,
        leagueRim
      ) - leagueRim;
    oppThree =
      meanRate(
        request.defense,
        profiles.opp_three_allowed,
        rateOverrides?.opp_three_allowed,
        leagueThree
      ) - leagueThree;
  }

  const seconds = request.secondsIntoPossession ?? profiles.seconds_mean;
  const secondsZ = (seconds - profiles.seconds_mean) / profiles.seconds_std;
  const live = request.liveBall ? 1 : 0;
  const secondChance = request.secondChance ? 1 : 0;
  const clutch = request.clutch ? 1 : 0;

  const utilities = (withLineup: boolean): number[] => {
    const out: number[] = [];
    for (let z = 0; z < nZones; z += 1) {
      const zone = zones[z] as string;
      const isRim = rim[z] as number;
      const isThree = three[z] as number;
      let u = zone === REFERENCE_ZONE ? 0 : coef(`alt_${zone}`);
      u += coef("shooter_mix") * (shooterRatio[z] as number);
      u += coef("team_mix") * (teamRatio[z] as number);
      u += coef("into_possession_x_rim") * secondsZ * isRim;
      u += coef("into_possession_x_three") * secondsZ * isThree;
      u += coef("live_ball_x_rim") * live * isRim;
      u += coef("second_chance_x_rim") * secondChance * isRim;
      u += coef("clutch_x_three") * clutch * isThree;
      if (withLineup) {
        u += coef("spacing_x_three") * spacing * isThree;
        u += coef("spacing_min_x_three") * spacingMin * isThree;
        u += coef("teammate_rim_x_rim") * teammateRim * isRim;
        u += coef("opp_rim_allowed_x_rim") * oppRim * isRim;
        u += coef("opp_three_allowed_x_three") * oppThree * isThree;
      }
      out.push(u);
    }
    return out;
  };

  const full = utilities(true);
  const mix = softmax(full);
  // Every lineup term is a deviation from the league average, so dropping them
  // *is* the league-average lineup. No second profile is needed.
  const baselineMix = softmax(utilities(false));

  // Written out rather than composed, for the same reason `meanRate` is: the
  // summation order has to match Python's left-to-right `sum()` exactly, or the
  // parity fixture fails in the last few bits.
  let points = 0;
  for (let z = 0; z < nZones; z += 1) {
    const value = profiles.zone_points[z] ?? 0;
    points += ((mix[z] as number) - (baselineMix[z] as number)) * value;
  }

  return {
    zones,
    mix,
    baselineMix,
    utilities: full,
    shooterKnown,
    shooterWeight,
    lineupFeatures: [spacing, spacingMin, teammateRim, oppRim, oppThree],
    pointsPer100: 100 * points,
  };
}
