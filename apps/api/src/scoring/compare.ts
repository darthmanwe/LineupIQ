/**
 * Comparing two lineups for one shooter, in the Worker.
 *
 * **This file and `services/ml/src/lineupiq/serve/compare.py` must agree.**
 * Neither is the reference: both read the same exported contract, and
 * `data/parity/compare.json` holds the cases a vitest suite re-computes here and
 * asserts to 1e-9.
 *
 * The failure this mirror can have is quieter than the scorer's. A version that
 * dropped the profile variance term entirely would reproduce every delta, every
 * share and every rank — it would differ only in how wide the intervals are,
 * and nothing at runtime would notice. That is why the fixture stores the two
 * variance components separately instead of only their sum.
 *
 * What the endpoint is for, and what it is not, is in the Python module's
 * docstring; the short version is that this measures *where* a shot is taken
 * and not whether it goes in, and it is a different estimand from the withheld
 * trade projection rather than a way around it.
 */

import {
  GRADIENT_STEP,
  contrastShares,
  quadraticForm,
  standardError,
  type SelectionModelWithCovariance,
} from "./plays";
import {
  scoreSelection,
  type RateOverrides,
  type RateTable,
  type ScoreRequest,
  type SelectionProfiles,
} from "./selection";

/**
 * The rate tables a swap can move, each with the table holding its error.
 *
 * The order is fixed here rather than taken from object key order, because the
 * profile variance is a sum over these and Python accumulates it in this
 * sequence. Floating-point addition is not associative.
 */
export const RATE_TABLES: ReadonlyArray<readonly [RateTable, string]> = [
  ["player_three_rate", "player_three_rate_se"],
  ["player_rim_rate", "player_rim_rate_se"],
  ["opp_three_allowed", "opp_three_allowed_se"],
  ["opp_rim_allowed", "opp_rim_allowed_se"],
] as const;

/**
 * The five lineup features, in `ScoreResult.lineupFeatures` order, paired with
 * the zone attribute each one's coefficient multiplies.
 *
 * That second element is what makes the omnibus two-dimensional rather than
 * eight, so it is data rather than a comment: every lineup term multiplies
 * either the rim indicator or the three indicator, so a lineup's whole effect
 * on the nine utilities is `a·rim + b·three` and it has exactly two parameters.
 */
export const LINEUP_TERM_ATTRIBUTE: ReadonlyArray<readonly [string, "rim" | "three"]> = [
  ["spacing_x_three", "three"],
  ["spacing_min_x_three", "three"],
  ["teammate_rim_x_rim", "rim"],
  ["opp_rim_allowed_x_rim", "rim"],
  ["opp_three_allowed_x_three", "three"],
] as const;

/** Just the names, in the same order. */
export const LINEUP_TERMS: readonly string[] = LINEUP_TERM_ATTRIBUTE.map(([term]) => term);

const LEAGUE_FALLBACK: Record<RateTable, "league_three_rate" | "league_rim_rate"> = {
  player_three_rate: "league_three_rate",
  player_rim_rate: "league_rim_rate",
  opp_three_allowed: "league_three_rate",
  opp_rim_allowed: "league_rim_rate",
};

export type ComparisonContract = {
  confidence: number;
  critical_value: number;
  omnibus_critical_value: number;
};

export type ZoneComparison = {
  zone: string;
  /** `P(zone | shooter, left) − P(zone | shooter, right)`. Sums to zero. */
  deltaShare: number;
  shareLeft: number;
  shareRight: number;
  pointsPer100: number;
  standardError: number;
  interval: [number, number];
  /** The two components of `standardError ** 2`, published rather than summed away. */
  varianceCoefficients: number;
  varianceProfiles: number;
};

export type MechanismTerm = {
  term: string;
  featureLeft: number;
  featureRight: number;
  featureDelta: number;
  coefficient: number;
  expectedSign: number | null;
  /** `agrees` / `DISAGREES` / `indeterminate`, from the fitted sign audit. */
  verdict: string;
};

/**
 * Did the shot mix move at all, tested on the two parameters a lineup has.
 *
 * `rimShift` and `threeShift` are those parameters: how far the left lineup
 * pulls this shooter toward the rim and toward the arc, relative to the right
 * one, in utility units. They are jointly sufficient for the whole difference —
 * the nine-zone delta is zero if and only if both are.
 */
export type Omnibus = {
  statistic: number;
  degreesOfFreedom: number;
  criticalValue: number;
  distinguishable: boolean;
  /** True when the two lineups predict identically, so there is nothing to test. */
  degenerate: boolean;
  rimShift: number;
  threeShift: number;
  rimShiftError: number;
  threeShiftError: number;
};

export type LineupComparison = {
  zones: ZoneComparison[];
  omnibus: Omnibus;
  mechanism: MechanismTerm[];
  confidence: number;
  criticalValue: number;
  profileVarianceShare: number;
  argminUnstable: boolean;
};

/**
 * Thrown when a lineup contains a player with no fitted shooting rate.
 *
 * Such a player silently inherits the league rate, so the comparison comes back
 * as exactly zero — a number indistinguishable, to a reader, from a real
 * finding that the swap does not matter. The route turns this into a 422 that
 * names the players, the same way it turns a missing covariance into a 503.
 */
export class UnprofiledPlayerError extends Error {
  readonly players: number[];

  constructor(players: number[]) {
    super(`no fitted shooting rate for player(s): ${players.join(", ")}`);
    this.name = "UnprofiledPlayerError";
    this.players = players;
  }
}

type RateKey = { table: RateTable; seTable: string; player: number };

/**
 * Every `(table, player)` whose rate enters either lineup, on a total order.
 *
 * The profile variance is a sum over these, so an unordered iteration would
 * make the answer depend on insertion order — reproducible on one machine and
 * not across two.
 */
function rateKeys(left: ScoreRequest, right: ScoreRequest | null): RateKey[] {
  const seen = new Map<string, RateKey>();
  const add = (table: RateTable, seTable: string, player: number): void => {
    seen.set(`${table}:${player}`, { table, seTable, player });
  };
  for (const request of [left, right]) {
    if (request === null) continue;
    for (const player of request.offense) {
      if (player === request.shooterId) continue;
      add("player_three_rate", "player_three_rate_se", player);
      add("player_rim_rate", "player_rim_rate_se", player);
    }
    for (const player of request.defense) {
      add("opp_three_allowed", "opp_three_allowed_se", player);
      add("opp_rim_allowed", "opp_rim_allowed_se", player);
    }
  }
  const rank = new Map<string, number>();
  RATE_TABLES.forEach(([table], index) => rank.set(table, index));
  return [...seen.values()].sort((a, b) => {
    const byTable = (rank.get(a.table) ?? 0) - (rank.get(b.table) ?? 0);
    return byTable !== 0 ? byTable : a.player - b.player;
  });
}

/**
 * `[deltaRimPull, deltaThreePull]` between two lineups.
 *
 * The two numbers that jointly determine the entire difference. `right = null`
 * gives the league-average lineup, whose offsets are exactly zero by
 * construction — every lineup feature is a centred deviation, so the
 * league-average lineup pulls in neither direction.
 *
 * Read off `ScoreResult.lineupFeatures` and the fitted coefficients rather than
 * re-derived: this is the model's linear index, which the scorer already
 * computes, and the only arithmetic here is the five multiply-adds that group
 * it by zone attribute.
 */
export function contrastOffsets(
  left: ScoreRequest,
  right: ScoreRequest | null,
  profiles: SelectionProfiles,
  model: SelectionModelWithCovariance,
  coefficients: number[],
  rateOverrides?: RateOverrides
): number[] {
  const theta = new Map<string, number>();
  model.term_names.forEach((name, i) => theta.set(name, coefficients[i] ?? 0));

  const offsets = (request: ScoreRequest | null): [number, number] => {
    if (request === null) return [0, 0];
    const features = scoreSelection(
      request,
      profiles,
      { ...model, coefficients },
      rateOverrides
    ).lineupFeatures;
    let rim = 0;
    let three = 0;
    LINEUP_TERM_ATTRIBUTE.forEach(([term, attribute], index) => {
      const contribution = (theta.get(term) ?? 0) * (features[index] as number);
      if (attribute === "rim") rim += contribution;
      else three += contribution;
    });
    return [rim, three];
  };

  const [leftRim, leftThree] = offsets(left);
  const [rightRim, rightThree] = offsets(right);
  return [leftRim - rightRim, leftThree - rightThree];
}

/** `d offset[i] / d θ[j]`, indexed `[offset][term]`. */
export function offsetGradients(
  left: ScoreRequest,
  right: ScoreRequest | null,
  profiles: SelectionProfiles,
  model: SelectionModelWithCovariance
): number[][] {
  const base = model.coefficients;
  const gradients: number[][] = [
    new Array<number>(base.length).fill(0),
    new Array<number>(base.length).fill(0),
  ];
  for (let j = 0; j < base.length; j += 1) {
    const step = GRADIENT_STEP * Math.max(1, Math.abs(base[j] as number));
    const forward = base.slice();
    const backward = base.slice();
    forward[j] = (base[j] as number) + step;
    backward[j] = (base[j] as number) - step;
    const high = contrastOffsets(left, right, profiles, model, forward);
    const low = contrastOffsets(left, right, profiles, model, backward);
    for (let i = 0; i < 2; i += 1) {
      (gradients[i] as number[])[j] = ((high[i] as number) - (low[i] as number)) / (2 * step);
    }
  }
  return gradients;
}

/** `d offset[i] / d rate[k]`, indexed `[key][offset]`. */
export function profileOffsetGradients(
  left: ScoreRequest,
  right: ScoreRequest | null,
  profiles: SelectionProfiles,
  model: SelectionModelWithCovariance,
  keys: RateKey[]
): number[][] {
  const gradients: number[][] = [];
  for (const key of keys) {
    const fallback = profiles[LEAGUE_FALLBACK[key.table]];
    const existing = profiles[key.table][String(key.player)];
    const base = existing === undefined ? fallback : existing;
    const step = GRADIENT_STEP * Math.max(1, Math.abs(base));
    const high = contrastOffsets(left, right, profiles, model, model.coefficients, {
      [key.table]: { [String(key.player)]: base + step },
    });
    const low = contrastOffsets(left, right, profiles, model, model.coefficients, {
      [key.table]: { [String(key.player)]: base - step },
    });
    gradients.push([
      ((high[0] as number) - (low[0] as number)) / (2 * step),
      ((high[1] as number) - (low[1] as number)) / (2 * step),
    ]);
  }
  return gradients;
}

/** `d deltaShare[zone] / d θ[j]`, indexed `[zone][term]`. */
export function shareGradients(
  left: ScoreRequest,
  right: ScoreRequest | null,
  profiles: SelectionProfiles,
  model: SelectionModelWithCovariance
): number[][] {
  const base = model.coefficients;
  const nTerms = base.length;
  const nZones = profiles.zones.length;
  const gradients: number[][] = Array.from({ length: nZones }, () =>
    new Array<number>(nTerms).fill(0)
  );
  for (let j = 0; j < nTerms; j += 1) {
    const step = GRADIENT_STEP * Math.max(1, Math.abs(base[j] as number));
    const forward = base.slice();
    const backward = base.slice();
    forward[j] = (base[j] as number) + step;
    backward[j] = (base[j] as number) - step;
    const high = contrastShares(left, right, profiles, model, forward);
    const low = contrastShares(left, right, profiles, model, backward);
    for (let z = 0; z < nZones; z += 1) {
      (gradients[z] as number[])[j] = ((high[z] as number) - (low[z] as number)) / (2 * step);
    }
  }
  return gradients;
}

/**
 * `d deltaShare[zone] / d rate[k]`, indexed `[key][zone]`.
 *
 * The perturbation goes through the scorer's override map rather than through a
 * rewritten copy of the profiles, so what is differentiated is the served
 * function itself and not a reconstruction of it.
 */
export function profileShareGradients(
  left: ScoreRequest,
  right: ScoreRequest | null,
  profiles: SelectionProfiles,
  model: SelectionModelWithCovariance,
  keys: RateKey[]
): number[][] {
  const nZones = profiles.zones.length;
  const gradients: number[][] = [];
  for (const key of keys) {
    const fallback = profiles[LEAGUE_FALLBACK[key.table]];
    const table = profiles[key.table];
    const existing = table[String(key.player)];
    const base = existing === undefined ? fallback : existing;
    const step = GRADIENT_STEP * Math.max(1, Math.abs(base));
    const high: RateOverrides = { [key.table]: { [String(key.player)]: base + step } };
    const low: RateOverrides = { [key.table]: { [String(key.player)]: base - step } };
    const forward = contrastShares(left, right, profiles, model, model.coefficients, high);
    const backward = contrastShares(left, right, profiles, model, model.coefficients, low);
    const row: number[] = [];
    for (let z = 0; z < nZones; z += 1) {
      row.push(((forward[z] as number) - (backward[z] as number)) / (2 * step));
    }
    gradients.push(row);
  }
  return gradients;
}

/**
 * `v' M⁻¹ v` for symmetric positive-definite `M`, or null if it is not.
 *
 * Factor once and solve rather than inverting: `v' (L L')⁻¹ v` is `‖L⁻¹v‖²`, so
 * the forward substitution is the whole computation. Explicit loops, for the
 * same reason `quadraticForm` uses them.
 *
 * Null does not mean something went wrong. For this matrix it means the two
 * lineups differ by nothing the model can resolve — a state the caller renders
 * rather than an error it handles.
 */
function choleskyQuadratic(matrix: number[][], vector: number[]): number | null {
  const n = matrix.length;
  const lower: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let total = (matrix[i] as number[])[j] as number;
      for (let k = 0; k < j; k += 1) {
        total -= ((lower[i] as number[])[k] as number) * ((lower[j] as number[])[k] as number);
      }
      if (i === j) {
        if (total <= 0) return null;
        (lower[i] as number[])[j] = Math.sqrt(total);
      } else {
        (lower[i] as number[])[j] = total / ((lower[j] as number[])[j] as number);
      }
    }
  }
  const solved = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    let total = vector[i] as number;
    for (let k = 0; k < i; k += 1) {
      total -= ((lower[i] as number[])[k] as number) * (solved[k] as number);
    }
    solved[i] = total / ((lower[i] as number[])[i] as number);
  }
  let accumulated = 0;
  for (const value of solved) accumulated += value * value;
  return accumulated;
}

/**
 * Compare two lineups for one shooter.
 *
 * `right = null` compares against the league-average lineup, which is the same
 * quantity `/lineups/score` already returns per zone — deliberately routed
 * through `contrastShares` so the two cannot drift.
 *
 * The contract is passed in rather than read here, for the reason `rankPlays`
 * gives: it comes from the pre-registered, hash-pinned thresholds file, and a
 * serving module reaching for its own defaults would be a second place the
 * contract lives.
 */
export function compareLineups(
  left: ScoreRequest,
  right: ScoreRequest | null,
  profiles: SelectionProfiles,
  model: SelectionModelWithCovariance,
  contract: ComparisonContract
): LineupComparison {
  const covariance = model.covariance;
  if (covariance === null || covariance === undefined) {
    throw new Error("the selection model was exported without a covariance matrix");
  }

  const zones = profiles.zones;
  const keys = rateKeys(left, right);

  // Every player whose rate enters the answer needs both a rate and an error for
  // it. Missing either means the league fallback is doing the work, and the
  // difference would come back as a zero that looks like a measurement.
  const unprofiled = new Set<number>();
  for (const key of keys) {
    const table = profiles[key.table];
    const errors = profiles[key.seTable as keyof SelectionProfiles] as
      Record<string, number> | undefined;
    const id = String(key.player);
    if (table[id] === undefined || errors === undefined || errors[id] === undefined) {
      unprofiled.add(key.player);
    }
  }
  if (unprofiled.size > 0) {
    throw new UnprofiledPlayerError([...unprofiled].sort((a, b) => a - b));
  }

  const scoredLeft = scoreSelection(left, profiles, model);
  const scoredRight = right === null ? null : scoreSelection(right, profiles, model);
  const reference = scoredRight === null ? scoredLeft.baselineMix : scoredRight.mix;
  const deltas: number[] = [];
  for (let z = 0; z < zones.length; z += 1) {
    deltas.push((scoredLeft.mix[z] as number) - (reference[z] as number));
  }

  const thetaGradients = shareGradients(left, right, profiles, model);
  const rateGradients = profileShareGradients(left, right, profiles, model, keys);
  const rateErrors = keys.map((key) => {
    const table = profiles[key.seTable as keyof SelectionProfiles] as Record<string, number>;
    return table[String(key.player)] as number;
  });

  const comparisons: ZoneComparison[] = [];
  let totalCoefficientVariance = 0;
  let totalProfileVariance = 0;
  for (let z = 0; z < zones.length; z += 1) {
    const gradient = thetaGradients[z] as number[];
    const varianceTheta = quadraticForm(gradient, covariance, gradient);
    let varianceRates = 0;
    for (let k = 0; k < keys.length; k += 1) {
      const slope = (rateGradients[k] as number[])[z] as number;
      const error = rateErrors[k] as number;
      varianceRates += slope * slope * error * error;
    }
    const scale = 100 * (profiles.zone_points[z] ?? 0);
    const squared = scale * scale;
    const varianceCoefficients = squared * varianceTheta;
    const varianceProfiles = squared * varianceRates;
    const points = 100 * (deltas[z] as number) * (profiles.zone_points[z] ?? 0);
    const error = standardError(varianceCoefficients + varianceProfiles);
    totalCoefficientVariance += varianceCoefficients;
    totalProfileVariance += varianceProfiles;
    comparisons.push({
      zone: zones[z] as string,
      deltaShare: deltas[z] as number,
      shareLeft: scoredLeft.mix[z] as number,
      shareRight: reference[z] as number,
      pointsPer100: points,
      standardError: error,
      interval: [
        points - contract.critical_value * error,
        points + contract.critical_value * error,
      ],
      varianceCoefficients,
      varianceProfiles,
    });
  }

  // The omnibus, on the two parameters a lineup actually has: how far it pulls
  // this shooter toward the rim, and how far toward the arc. See the module
  // docstring in the Python mirror for why this is not a test over zones.
  const offsets = contrastOffsets(left, right, profiles, model, model.coefficients);
  const offsetTheta = offsetGradients(left, right, profiles, model);
  const offsetRates = profileOffsetGradients(left, right, profiles, model, keys);
  const matrix: number[][] = [
    [0, 0],
    [0, 0],
  ];
  for (let a = 0; a < 2; a += 1) {
    for (let b = 0; b < 2; b += 1) {
      let value = quadraticForm(offsetTheta[a] as number[], covariance, offsetTheta[b] as number[]);
      for (let k = 0; k < keys.length; k += 1) {
        const error = rateErrors[k] as number;
        value +=
          ((offsetRates[k] as number[])[a] as number) *
          ((offsetRates[k] as number[])[b] as number) *
          error *
          error;
      }
      (matrix[a] as number[])[b] = value;
    }
  }
  const statistic = choleskyQuadratic(matrix, offsets);
  const omnibus: Omnibus = {
    statistic: statistic === null ? 0 : statistic,
    degreesOfFreedom: 2,
    criticalValue: contract.omnibus_critical_value,
    distinguishable: statistic !== null && statistic > contract.omnibus_critical_value,
    degenerate: statistic === null,
    rimShift: offsets[0] as number,
    threeShift: offsets[1] as number,
    rimShiftError: standardError((matrix[0] as number[])[0] as number),
    threeShiftError: standardError((matrix[1] as number[])[1] as number),
  };

  const theta = new Map<string, number>();
  model.term_names.forEach((name, i) => theta.set(name, model.coefficients[i] ?? 0));
  const audit = (model.sign_audit ?? {}) as Record<
    string,
    { expected_sign?: number | null; verdict?: string }
  >;
  const rightFeatures = scoredRight === null ? [0, 0, 0, 0, 0] : scoredRight.lineupFeatures;
  const mechanism: MechanismTerm[] = LINEUP_TERMS.map((term, index) => {
    const entry = audit[term];
    const featureLeft = scoredLeft.lineupFeatures[index] as number;
    const featureRight = rightFeatures[index] as number;
    return {
      term,
      featureLeft,
      featureRight,
      featureDelta: featureLeft - featureRight,
      coefficient: theta.get(term) ?? 0,
      expectedSign: entry?.expected_sign ?? null,
      verdict: entry?.verdict ?? "unaudited",
    };
  });

  const total = totalCoefficientVariance + totalProfileVariance;
  const leftThreeRates = left.offense
    .filter((p) => p !== left.shooterId && profiles.player_three_rate[String(p)] !== undefined)
    .map((p) => profiles.player_three_rate[String(p)] as number)
    .sort((a, b) => a - b);
  const argminUnstable =
    leftThreeRates.length > 1 &&
    (leftThreeRates[1] as number) - (leftThreeRates[0] as number) < 2 * GRADIENT_STEP;

  return {
    zones: comparisons,
    omnibus,
    mechanism,
    confidence: contract.confidence,
    criticalValue: contract.critical_value,
    profileVarianceShare: total > 0 ? totalProfileVariance / total : 0,
    argminUnstable,
  };
}
