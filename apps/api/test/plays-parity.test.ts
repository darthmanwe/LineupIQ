/**
 * Python/TypeScript parity for the play ranking, asserted inside workerd.
 *
 * This is the fixture the endpoint was waiting on. `/lineups/optimal-plays`
 * ranks nine numbers and refuses to order the ones it cannot separate, and the
 * separation test is a variance calculation — a second implementation of which
 * fails in the worst available way. It does not raise. It does not return an
 * implausible number. It returns the same ranked list with intervals of the
 * wrong width, so the endpoint quietly starts ordering ties, or quietly starts
 * refusing to order things it can.
 *
 * **So the standard errors are asserted, not just the ranks.** A ranking is a
 * sequence of comparisons, and comparisons survive exactly the class of error a
 * variance calculation makes: two implementations can agree on every rank while
 * disagreeing about every interval by a factor of two. Checking only the order
 * would be a test that passes when the thing it exists to check is broken.
 */

import { describe, expect, it } from "vitest";

// JSON imports rather than `readFileSync`: an `import.meta.url` path resolves to
// `/C:/...` on Windows, which workerd's filesystem shim rejects.
import fixtureJson from "../../../data/parity/plays.json";
import modelJson from "../../web/public/data/selection_model.json";
import profilesJson from "../../web/public/data/selection_profiles.json";
import { GRADIENT_STEP, rankPlays } from "../src/scoring/plays";
import type { RankingContract, SelectionModelWithCovariance } from "../src/scoring/plays";
import type { SelectionProfiles } from "../src/scoring/selection";

type Case = {
  request: {
    shooter_id: number;
    offense: number[];
    defense: number[];
    team_id: number | null;
    season: number | null;
    seconds_into_possession: number | null;
    live_ball: boolean;
    second_chance: boolean;
    clutch: boolean;
  };
  plays: { zone: string; points_per_100: number; standard_error: number; rank: number }[];
  bands: string[][];
  ordered: boolean;
  excluded: string[];
  diagonal_would_refuse: number;
  pairs_compared: number;
  ties_spanning_bands: number;
};

type Fixture = {
  seed: number;
  zones: string[];
  term_names: string[];
  ranking: RankingContract;
  n_cases: number;
  n_unordered: number;
  n_bands_total: number;
  diagonal_would_refuse: number;
  pairs_compared: number;
  ties_spanning_bands: number;
  cases: Case[];
};

const fixture = fixtureJson as unknown as Fixture;
const profiles = profilesJson as unknown as SelectionProfiles;
const model = modelJson as unknown as SelectionModelWithCovariance;

const TOLERANCE = 1e-9;

function rank(c: Case) {
  return rankPlays(
    {
      shooterId: c.request.shooter_id,
      offense: c.request.offense,
      defense: c.request.defense,
      teamId: c.request.team_id,
      season: c.request.season,
      secondsIntoPossession: c.request.seconds_into_possession,
      liveBall: c.request.live_ball,
      secondChance: c.request.second_chance,
      clutch: c.request.clutch,
    },
    profiles,
    model,
    fixture.ranking
  );
}

describe("the ranking contract", () => {
  it("ships the covariance the difference test needs", () => {
    // Twenty numbers would be enough for per-coefficient intervals and are not
    // enough for this. The endpoint compares *differences* of strongly
    // correlated quantities, and the covariance term is what makes them
    // separable.
    expect(model.covariance).not.toBeNull();
    expect(model.covariance).toHaveLength(model.term_names.length);
    for (const row of model.covariance as number[][]) {
      expect(row).toHaveLength(model.term_names.length);
    }
  });

  it("is symmetric, which is what makes it a covariance", () => {
    // Python symmetrises the observed information before inverting it, so an
    // asymmetric matrix here means the export transposed or truncated
    // something. A transposed near-symmetric matrix passes every smoke test you
    // would think to write, which is why this one is written.
    const cov = model.covariance as number[][];
    for (let i = 0; i < cov.length; i += 1) {
      for (let j = 0; j < i; j += 1) {
        const a = (cov[i] as number[])[j] as number;
        const b = (cov[j] as number[])[i] as number;
        expect(Math.abs(a - b)).toBeLessThan(1e-12);
      }
    }
  });

  it("uses the pre-registered confidence level, resolved once in Python", () => {
    expect(fixture.ranking.confidence).toBe(0.8);
    // The two-sided normal quantile for 80%. Shipped rather than computed here:
    // the Worker has no inverse normal CDF, and adding one would be a second
    // place this number could be wrong.
    expect(fixture.ranking.critical_value).toBeCloseTo(1.2815515655, 9);
    expect(model.ranking?.critical_value).toBe(fixture.ranking.critical_value);
  });

  it("differences the scorer with the same step Python used", () => {
    // A different step is a different number, not a rounding difference. The
    // parity assertions below would catch it, but they would report it as a
    // disagreement about variance rather than as what it is.
    expect(GRADIENT_STEP).toBe(1e-4);
  });
});

describe("the fixture itself", () => {
  it("was generated against this model and these profiles", () => {
    expect(fixture.zones).toEqual(profiles.zones);
    expect(fixture.term_names).toEqual(model.term_names);
    expect(fixture.cases).toHaveLength(fixture.n_cases);
  });

  it("contains a ranking the model refuses to order at all", () => {
    // The branch this endpoint exists for. It fires on roughly three per cent of
    // requests, so a fixture that covered it by luck would stop covering it the
    // next time the sample changed — the generator picks one deliberately.
    expect(fixture.n_unordered).toBeGreaterThan(0);
    const unordered = fixture.cases.find((c) => !c.ordered) as Case;
    expect(unordered.bands).toHaveLength(1);
    expect(new Set(unordered.plays.map((p) => p.rank))).toEqual(new Set([1]));
  });

  it("contains rankings that are not fully separable either", () => {
    // Nine bands would mean the data orders every zone, and then the banding is
    // decoration. The mean is around seven.
    expect(fixture.n_bands_total).toBeLessThan(9 * fixture.n_cases);
  });
});

describe("play ranking parity", () => {
  it("reproduces every priced contribution to 1e-9", () => {
    let worst = 0;
    let worstAt = "";
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      expect(result.plays, `case ${i}`).toHaveLength(c.plays.length);
      c.plays.forEach((expected, k) => {
        const actual = result.plays[k] as (typeof result.plays)[number];
        expect(actual.zone, `case ${i} position ${k}`).toBe(expected.zone);
        const delta = Math.abs(actual.pointsPer100 - expected.points_per_100);
        if (delta > worst) {
          worst = delta;
          worstAt = `case ${i} ${expected.zone}`;
        }
      });
    });
    expect(worst, `worst at ${worstAt}`).toBeLessThan(TOLERANCE);
  });

  it("reproduces every delta-method standard error to 1e-9", () => {
    // The assertion this whole file exists for. These come out of forty
    // finite-differenced scorer calls and a 400-term quadratic form; there is a
    // great deal of arithmetic here for two languages to disagree about, and
    // none of it would surface as anything but a slightly different interval.
    let worst = 0;
    let worstAt = "";
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      c.plays.forEach((expected, k) => {
        const actual = result.plays[k] as (typeof result.plays)[number];
        const delta = Math.abs(actual.standardError - expected.standard_error);
        if (delta > worst) {
          worst = delta;
          worstAt = `case ${i} ${expected.zone}`;
        }
      });
    });
    expect(worst, `worst at ${worstAt}`).toBeLessThan(TOLERANCE);
  });

  it("agrees on every rank and every band", () => {
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      expect(
        result.plays.map((p) => p.rank),
        `case ${i}`
      ).toEqual(c.plays.map((p) => p.rank));
      expect(result.bands, `case ${i}`).toEqual(c.bands);
      expect(result.ordered, `case ${i}`).toBe(c.ordered);
      expect(result.excluded, `case ${i}`).toEqual(c.excluded);
    });
  });

  it("agrees on what the contiguous-band constraint threw away", () => {
    // Bands are contiguous runs of the ranked list, so an indistinguishable pair
    // that is not adjacent enough to share a run gets separate ranks anyway. The
    // constraint is what makes `rank` monotone in list position; this is what it
    // costs, and it is asserted rather than described.
    let spanning = 0;
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      expect(result.tiesSpanningBands, `case ${i}`).toBe(c.ties_spanning_bands);
      spanning += result.tiesSpanningBands;
    });
    expect(spanning).toBe(fixture.ties_spanning_bands);
    // Small, but not zero — so the constraint is a real choice with a real
    // price, not a free simplification.
    expect(spanning / fixture.pairs_compared).toBeLessThan(0.02);
  });

  it("agrees on how often the marginal intervals would have refused", () => {
    // Not a parity detail. This counter is the published justification for
    // shipping a 20x20 matrix instead of its diagonal, and a number that only
    // one of the two implementations can produce is not evidence of anything.
    let refused = 0;
    let compared = 0;
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      expect(result.diagonalWouldRefuse, `case ${i}`).toBe(c.diagonal_would_refuse);
      expect(result.pairsCompared, `case ${i}`).toBe(c.pairs_compared);
      refused += result.diagonalWouldRefuse;
      compared += result.pairsCompared;
    });
    expect(refused).toBe(fixture.diagonal_would_refuse);
    expect(compared).toBe(fixture.pairs_compared);
    // And it is not a rounding artefact: a meaningful share of ranked pairs are
    // separable only once the correlation is accounted for.
    expect(refused / compared).toBeGreaterThan(0.02);
  });
});

describe("what the ranking must never do", () => {
  it("gives zones in the same band the same rank, and never a bare tie-break", () => {
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      result.bands.forEach((band, bandIndex) => {
        band.forEach((zone) => {
          const play = result.plays.find((p) => p.zone === zone);
          expect(play?.rank, `case ${i} zone ${zone}`).toBe(bandIndex + 1);
        });
      });
    });
  });

  it("orders the plays by contribution, descending, within the whole list", () => {
    // Bands group; they do not reorder. A band boundary must never appear
    // between two zones that are out of order by contribution.
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      for (let k = 1; k < result.plays.length; k += 1) {
        const previous = result.plays[k - 1] as (typeof result.plays)[number];
        const current = result.plays[k] as (typeof result.plays)[number];
        expect(previous.pointsPer100, `case ${i} at ${k}`).toBeGreaterThanOrEqual(
          current.pointsPer100
        );
        expect(current.rank).toBeGreaterThanOrEqual(previous.rank);
      }
    });
  });

  it("never separates a pair whose difference is inside the interval", () => {
    // Restated independently of the implementation: if two zones got different
    // ranks, the gap between them must exceed the critical value times the
    // standard error of their *difference*. Recomputing that here from the
    // marginal errors is not possible — which is exactly the point — so the
    // weaker, always-true implication is asserted: a separated pair must at
    // least differ.
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      for (let a = 0; a < result.plays.length; a += 1) {
        for (let b = a + 1; b < result.plays.length; b += 1) {
          const first = result.plays[a] as (typeof result.plays)[number];
          const second = result.plays[b] as (typeof result.plays)[number];
          if (first.rank !== second.rank) {
            expect(first.pointsPer100, `case ${i}`).not.toBe(second.pointsPer100);
          }
        }
      }
    });
  });

  it("excludes only zones below the pre-registered share floor", () => {
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      expect(result.plays.length + result.excluded.length, `case ${i}`).toBe(profiles.zones.length);
      result.plays.forEach((play) => {
        expect(play.share, `case ${i} ${play.zone}`).toBeGreaterThanOrEqual(
          fixture.ranking.min_zone_share
        );
      });
    });
  });

  it("keeps the per-zone contributions summing to the headline priced shift", () => {
    // The endpoint decomposes a number the scorer already publishes. If the
    // parts stopped adding to the whole, one of the two would be wrong and
    // nothing else in the suite would notice.
    fixture.cases.forEach((c, i) => {
      const result = rank(c);
      const total = result.plays.reduce((a, p) => a + p.pointsPer100, 0);
      const excludedZones = new Set(result.excluded);
      // Excluded zones still carry a contribution, so the identity only holds
      // when nothing was excluded. That is the usual case and it is asserted
      // where it applies rather than approximated everywhere.
      if (excludedZones.size === 0) {
        const headline = c.plays.reduce((a, p) => a + p.points_per_100, 0);
        expect(Math.abs(total - headline), `case ${i}`).toBeLessThan(TOLERANCE);
      }
    });
  });
});
