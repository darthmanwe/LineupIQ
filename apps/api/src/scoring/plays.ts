/**
 * Ranking a lineup's zones by what it is worth in each — and refusing to rank
 * what the data cannot separate.
 *
 * Mirrors `services/ml/src/lineupiq/serve/plays.py`. Neither is the reference;
 * both read the same exported contract and `data/parity/plays.json` holds cases
 * that a vitest suite re-derives here and asserts to 1e-9. An unverified second
 * implementation of a **variance** calculation is worse than none — a wrong
 * interval does not fail, it just quietly ranks things that are tied — so this
 * file did not ship until the fixture did.
 *
 * ## Why the Worker carries a 20×20 matrix
 *
 * Each zone's priced contribution is a smooth function of the twenty
 * coefficients, so the delta method gives its variance as `g' Σ g`. The
 * tempting shortcut is to ship only the twenty standard errors and compare
 * marginal intervals for overlap. That test is wrong, and wrong in the
 * direction that looks careful.
 *
 * Zone shares come out of a softmax and sum to one, so share that shows up at
 * the rim came from somewhere else: two contributions are strongly *negatively*
 * correlated. `Var(a − b) = Var(a) + Var(b) − 2·Cov(a, b)`, and with a large
 * negative covariance the difference is far better determined than either
 * endpoint. Comparing marginal intervals drops that term and calls pairs
 * indistinguishable that separate decisively — refusing to rank what the model
 * genuinely can rank, which is its own way of misleading someone.
 *
 * `diagonalWouldRefuse` counts, per request, how many ranked pairs that
 * shortcut would have given up on. It is reported in the response so the claim
 * is a measurement rather than a comment.
 *
 * ## Why the gradient is finite-differenced here
 *
 * A softmax difference times a constant is differentiable by hand. But a
 * hand-derived gradient is a *second* implementation of the model, and one that
 * drifts silently: nothing crashes, the intervals are just the wrong width.
 * Differencing the served scorer means the gradient cannot describe a different
 * model than the one being served, and it means parity here checks two scorers
 * rather than two transcriptions of one formula. Forty scorer calls, each a few
 * hundred flops, against a 10 ms budget.
 */

import {
  scoreSelection,
  type RateOverrides,
  type ScoreRequest,
  type SelectionModel,
  type SelectionProfiles,
} from "./selection";

/**
 * Relative step for the central difference. Must equal `GRADIENT_STEP` in
 * `plays.py` — a different step is a different number, not a rounding
 * difference, and the parity fixture is what enforces it.
 */
export const GRADIENT_STEP = 1e-4;

/** The pre-registered ranking contract, shipped with the model. */
export type RankingContract = {
  confidence: number;
  /**
   * The two-sided normal quantile for `confidence`, resolved in Python and
   * shipped. The Worker has no inverse normal CDF and adding one would be a
   * second place this number could be wrong.
   */
  critical_value: number;
  min_zone_share: number;
};

export type SelectionModelWithCovariance = SelectionModel & {
  /** Row-major, in `term_names` order. Null before the model is refitted. */
  covariance: number[][] | null;
  ranking?: RankingContract | null;
  /**
   * The pre-registered sign audit, per term. Read only by `compare.ts`, which
   * reports the verdict beside the feature a swap moved -- one of the three
   * terms an offensive swap touches is the one whose predicted sign came back
   * contradicted, and that belongs next to the number rather than in a footnote.
   */
  sign_audit?: Record<string, { expected_sign?: number | null; verdict?: string }> | null;
};

export type Play = {
  zone: string;
  share: number;
  baselineShare: number;
  /** `100 · (share − baselineShare) · leaguePointsPerAttempt(zone)`. */
  pointsPer100: number;
  standardError: number;
  interval: [number, number];
  /**
   * 1-based band index. Zones the data cannot separate share a rank — a shared
   * rank is the honest rendering of a tie, not a coin flip resolved by float
   * comparison.
   */
  rank: number;
};

export type PlayRanking = {
  plays: Play[];
  bands: string[][];
  /**
   * False when every eligible zone landed in one band: the model has a point
   * estimate for each and no basis for putting them in any order. Said out loud
   * rather than implied by list position.
   */
  ordered: boolean;
  excluded: string[];
  confidence: number;
  criticalValue: number;
  diagonalWouldRefuse: number;
  pairsCompared: number;
  /**
   * Indistinguishable pairs that landed in different bands, because bands are
   * contiguous runs of the ranked list. The information contiguity throws away,
   * counted instead of argued about.
   */
  tiesSpanningBands: number;
};

/**
 * Per-zone difference in predicted shot share between two lineups.
 *
 * `right = null` means "the same shooter with every lineup term at the league
 * average", which is `ScoreResult.baselineMix` and needs no second scoring
 * call — every lineup term is a centred deviation, so dropping them *is* the
 * league-average lineup.
 *
 * That is not a special case added for `compare.ts`: it is the quantity
 * `rankPlays` has always ranked. Routing both through one function is what
 * makes `compare(L, leagueAverage)` and `/lineups/score`'s per-zone `delta` the
 * same number by construction rather than by two implementations agreeing.
 */
export function contrastShares(
  left: ScoreRequest,
  right: ScoreRequest | null,
  profiles: SelectionProfiles,
  model: SelectionModel,
  coefficients: number[],
  rateOverrides?: RateOverrides
): number[] {
  const scored = scoreSelection(left, profiles, { ...model, coefficients }, rateOverrides);
  const reference =
    right === null
      ? scored.baselineMix
      : scoreSelection(right, profiles, { ...model, coefficients }, rateOverrides).mix;
  const out: number[] = [];
  for (let z = 0; z < scored.zones.length; z += 1) {
    out.push((scored.mix[z] as number) - (reference[z] as number));
  }
  return out;
}

function contributions(
  request: ScoreRequest,
  profiles: SelectionProfiles,
  model: SelectionModel,
  coefficients: number[]
): number[] {
  const deltas = contrastShares(request, null, profiles, model, coefficients);
  const out: number[] = [];
  for (let z = 0; z < deltas.length; z += 1) {
    const points = profiles.zone_points[z] ?? 0;
    out.push(100 * (deltas[z] as number) * points);
  }
  return out;
}

/** `d contribution[zone] / d θ[j]`, by central differences. Indexed `[zone][term]`. */
export function contributionGradients(
  request: ScoreRequest,
  profiles: SelectionProfiles,
  model: SelectionModel
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
    const high = contributions(request, profiles, model, forward);
    const low = contributions(request, profiles, model, backward);
    for (let z = 0; z < nZones; z += 1) {
      (gradients[z] as number[])[j] = ((high[z] as number) - (low[z] as number)) / (2 * step);
    }
  }
  return gradients;
}

/**
 * `left' Σ right`, accumulated in a fixed order.
 *
 * An explicit double loop rather than anything composed, because Python sums
 * the same terms in the same order and floating-point addition is not
 * associative. This is the same reason `meanRate` in `selection.ts` is a loop.
 */
export function quadraticForm(left: number[], covariance: number[][], right: number[]): number {
  let total = 0;
  for (let i = 0; i < covariance.length; i += 1) {
    const row = covariance[i] as number[];
    let inner = 0;
    for (let j = 0; j < row.length; j += 1) {
      inner += (row[j] as number) * (right[j] as number);
    }
    total += (left[i] as number) * inner;
  }
  return total;
}

// A tiny negative comes out of the delta method when the true variance is near
// zero and the covariance is only numerically positive semi-definite. Clamping
// at zero is right; `Math.abs` would turn a flat direction into a
// confident-looking interval.
export function standardError(quadratic: number): number {
  return quadratic > 0 ? Math.sqrt(quadratic) : 0;
}

export function rankPlays(
  request: ScoreRequest,
  profiles: SelectionProfiles,
  model: SelectionModelWithCovariance,
  contract: RankingContract
): PlayRanking {
  const covariance = model.covariance;
  if (covariance === null) {
    throw new Error("the selection model was exported without a covariance matrix");
  }
  const { critical_value: critical, min_zone_share: minShare } = contract;

  const zones = profiles.zones;
  const scored = scoreSelection(request, profiles, model);
  const contribution = contributions(request, profiles, model, model.coefficients);
  const gradients = contributionGradients(request, profiles, model);

  const errors = zones.map((_, z) =>
    standardError(quadraticForm(gradients[z] as number[], covariance, gradients[z] as number[]))
  );

  const eligible: number[] = [];
  const excluded: string[] = [];
  for (let z = 0; z < zones.length; z += 1) {
    if ((scored.mix[z] as number) >= minShare) eligible.push(z);
    else excluded.push(zones[z] as string);
  }

  // Contribution descending, zone index as a total tiebreaker. Several zones can
  // price identically — a zone the lineup does not move contributes exactly 0 —
  // and an untiebroken sort would order those by whatever the engine did.
  const order = eligible.slice().sort((a, b) => {
    const ca = contribution[a] as number;
    const cb = contribution[b] as number;
    if (ca !== cb) return ca > cb ? -1 : 1;
    return a - b;
  });

  const distinguishable = new Set<string>();
  let diagonalWouldRefuse = 0;
  let pairsCompared = 0;
  for (let ai = 0; ai < order.length; ai += 1) {
    const a = order[ai] as number;
    for (let bi = ai + 1; bi < order.length; bi += 1) {
      const b = order[bi] as number;
      pairsCompared += 1;
      const difference = (contribution[a] as number) - (contribution[b] as number);
      const delta: number[] = [];
      for (let j = 0; j < model.coefficients.length; j += 1) {
        delta.push(
          ((gradients[a] as number[])[j] as number) - ((gradients[b] as number[])[j] as number)
        );
      }
      const joint = standardError(quadraticForm(delta, covariance, delta));
      const separates = Math.abs(difference) > critical * joint;
      if (separates) distinguishable.add(`${a}:${b}`);
      // The naive test: do the marginal intervals overlap? Equivalent to
      // comparing against `z·(se_a + se_b)`, which is never smaller than
      // `z·se(a − b)` and is much larger when the two are negatively correlated.
      const naive =
        Math.abs(difference) > critical * ((errors[a] as number) + (errors[b] as number));
      if (separates && !naive) diagonalWouldRefuse += 1;
    }
  }

  // Bands are maximal **contiguous** runs of the sorted list, extended while the
  // incoming zone is indistinguishable from at least one zone already in the
  // run. Single linkage, restricted to contiguity. A Tukey letter display.
  //
  // Two decisions are packed in here, and both were made the hard way.
  //
  // *Single* linkage rather than breaking at the first separated adjacent pair:
  // a chain of individually-indistinguishable gaps can add up to a gap that is
  // not, and breaking on adjacency would report that sum as a real ordering.
  //
  // *Contiguous* rather than true connected components: the difference test has
  // a per-pair standard error, so a wider gap can separate while a narrower one
  // inside it does not, and unrestricted components can therefore interleave —
  // zone 6 joining the same component as zone 1 while zones 2–5 form their own.
  // That is not renderable as a ranked list and is not coherent as one either.
  // The first version did exactly that and produced rank sequences like
  // 1, 2, 2, 1; the parity suite caught it. Contiguity is what makes `rank`
  // monotone in list position, which is what any reader will assume it is.
  //
  // What contiguity costs is counted rather than assumed: `tiesSpanningBands`.
  const bandOf = new Map<number, number>();
  const bands: number[][] = [];
  for (const zoneIndex of order) {
    const last = bands[bands.length - 1];
    if (last !== undefined && last.some((m) => !distinguishable.has(`${m}:${zoneIndex}`))) {
      last.push(zoneIndex);
    } else {
      bands.push([zoneIndex]);
    }
    bandOf.set(zoneIndex, bands.length - 1);
  }

  let tiesSpanningBands = 0;
  for (let ai = 0; ai < order.length; ai += 1) {
    const a = order[ai] as number;
    for (let bi = ai + 1; bi < order.length; bi += 1) {
      const b = order[bi] as number;
      if (!distinguishable.has(`${a}:${b}`) && bandOf.get(a) !== bandOf.get(b)) {
        tiesSpanningBands += 1;
      }
    }
  }

  const plays: Play[] = order.map((z) => ({
    zone: zones[z] as string,
    share: scored.mix[z] as number,
    baselineShare: scored.baselineMix[z] as number,
    pointsPer100: contribution[z] as number,
    standardError: errors[z] as number,
    interval: [
      (contribution[z] as number) - critical * (errors[z] as number),
      (contribution[z] as number) + critical * (errors[z] as number),
    ],
    rank: (bandOf.get(z) as number) + 1,
  }));

  return {
    plays,
    bands: bands.map((band) => band.map((z) => zones[z] as string)),
    ordered: bands.length > 1,
    excluded,
    confidence: contract.confidence,
    criticalValue: critical,
    diagonalWouldRefuse,
    pairsCompared,
    tiesSpanningBands,
  };
}
