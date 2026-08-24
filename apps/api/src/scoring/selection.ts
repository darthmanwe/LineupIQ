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
  shooter_log_ratio: Record<string, number[]>;
  shooter_weight: Record<string, number>;
  team_log_ratio: Record<string, number[]>;
  player_three_rate: Record<string, number>;
  player_rim_rate: Record<string, number>;
  opp_three_allowed: Record<string, number>;
  opp_rim_allowed: Record<string, number>;
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
function meanRate(ids: number[], table: Record<string, number>, fallback: number): number {
  let total = 0;
  for (const id of ids) {
    const value = table[String(id)];
    total += value === undefined ? fallback : value;
  }
  return total / ids.length;
}

function minRate(ids: number[], table: Record<string, number>, fallback: number): number {
  let smallest = Infinity;
  for (const id of ids) {
    const value = table[String(id)];
    const rate = value === undefined ? fallback : value;
    if (rate < smallest) smallest = rate;
  }
  return smallest;
}

export function scoreSelection(
  request: ScoreRequest,
  profiles: SelectionProfiles,
  model: SelectionModel
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
    spacing = meanRate(teammates, profiles.player_three_rate, leagueThree) - leagueThree;
    spacingMin = minRate(teammates, profiles.player_three_rate, leagueThree) - leagueThree;
    teammateRim = meanRate(teammates, profiles.player_rim_rate, leagueRim) - leagueRim;
  }

  let oppRim = 0;
  let oppThree = 0;
  if (request.defense.length > 0) {
    oppRim = meanRate(request.defense, profiles.opp_rim_allowed, leagueRim) - leagueRim;
    oppThree = meanRate(request.defense, profiles.opp_three_allowed, leagueThree) - leagueThree;
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
  return {
    zones,
    mix: softmax(full),
    // Every lineup term is a deviation from the league average, so dropping
    // them *is* the league-average lineup. No second profile is needed.
    baselineMix: softmax(utilities(false)),
    utilities: full,
    shooterKnown,
    shooterWeight,
  };
}
