/**
 * Python ↔ TypeScript parity for the lineup comparison, to 1e-9.
 *
 * This is the quietest of the four parity contracts, and that is why the
 * fixture is shaped the way it is.
 *
 * A mirror that dropped the profile variance entirely would reproduce every
 * delta, every share, every priced contribution and every rank in the other
 * three fixtures. It would differ only in how wide the intervals are — by the
 * factor of two or so the profile term is worth — and nothing at runtime would
 * raise. An interval that is too narrow does not look like a bug; it looks like
 * a confident model. So the two variance components are asserted **separately**
 * rather than through their sum.
 *
 * The refusals are asserted too. A player with no fitted shooting rate scores
 * as exactly zero through the league fallback, which is also what a genuine
 * "this swap changes nothing" answer looks like — so a mirror that forgot to
 * refuse would produce a plausible number, not an error.
 */

import { describe, expect, it } from "vitest";

import fixture from "../../../data/parity/compare.json";
import {
  LINEUP_TERM_ATTRIBUTE,
  UnprofiledPlayerError,
  compareLineups,
  type ComparisonContract,
} from "../src/scoring/compare";
import type { ScoreRequest, SelectionProfiles } from "../src/scoring/selection";
import type { SelectionModelWithCovariance } from "../src/scoring/plays";
import profilesData from "../../web/public/data/selection_profiles.json";
import modelData from "../../web/public/data/selection_model.json";

const TOLERANCE = 1e-9;

const profiles = profilesData as unknown as SelectionProfiles;
const model = modelData as unknown as SelectionModelWithCovariance;
const contract = (fixture as unknown as { comparison: ComparisonContract }).comparison;

type FixtureRequest = {
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

type FixtureCase = {
  label: string;
  left: FixtureRequest;
  right: FixtureRequest | null;
  unprofiled: number[] | null;
  omnibus?: {
    statistic: number;
    degrees_of_freedom: number;
    distinguishable: boolean;
    degenerate: boolean;
    rim_shift: number;
    three_shift: number;
    rim_shift_error: number;
    three_shift_error: number;
  };
  profile_variance_share?: number;
  argmin_unstable?: boolean;
  zones?: Array<{
    zone: string;
    delta_share: number;
    points_per_100: number;
    standard_error: number;
    variance_coefficients: number;
    variance_profiles: number;
  }>;
  mechanism?: Array<{
    term: string;
    feature_delta: number;
    coefficient: number;
    verdict: string;
  }>;
};

const cases = (fixture as unknown as { cases: FixtureCase[] }).cases;

function toRequest(raw: FixtureRequest | null): ScoreRequest | null {
  if (raw === null) return null;
  return {
    shooterId: raw.shooter_id,
    offense: raw.offense,
    defense: raw.defense,
    teamId: raw.team_id,
    season: raw.season,
    secondsIntoPossession: raw.seconds_into_possession,
    liveBall: raw.live_ball,
    secondChance: raw.second_chance,
    clutch: raw.clutch,
  };
}

describe("lineup comparison parity", () => {
  it("has a fixture worth running", () => {
    expect(cases.length).toBeGreaterThan(50);
    expect(contract.omnibus_critical_value).toBeGreaterThan(0);
  });

  it.each(cases.map((c, index) => [index, c.label, c] as const))(
    "case %i (%s)",
    (_index, _label, expected) => {
      const left = toRequest(expected.left) as ScoreRequest;
      const right = toRequest(expected.right);

      if (expected.unprofiled !== null) {
        let thrown: unknown;
        try {
          compareLineups(left, right, profiles, model, contract);
        } catch (error) {
          thrown = error;
        }
        expect(thrown).toBeInstanceOf(UnprofiledPlayerError);
        expect((thrown as UnprofiledPlayerError).players).toEqual(expected.unprofiled);
        return;
      }

      const actual = compareLineups(left, right, profiles, model, contract);
      const zones = expected.zones ?? [];
      expect(actual.zones.length).toBe(zones.length);

      for (let z = 0; z < zones.length; z += 1) {
        const want = zones[z] as (typeof zones)[number];
        const got = actual.zones[z] as (typeof actual.zones)[number];
        expect(got.zone).toBe(want.zone);
        expect(got.deltaShare).toBeCloseTo(want.delta_share, 12);
        expect(Math.abs(got.pointsPer100 - want.points_per_100)).toBeLessThan(TOLERANCE);
        expect(Math.abs(got.standardError - want.standard_error)).toBeLessThan(TOLERANCE);
        // Asserted apart, not through the sum: a mirror that dropped the
        // profile term would still match a summed total to within the
        // coefficient term's share of it, which is exactly the failure this
        // fixture exists to catch.
        expect(Math.abs(got.varianceCoefficients - want.variance_coefficients)).toBeLessThan(
          TOLERANCE
        );
        expect(Math.abs(got.varianceProfiles - want.variance_profiles)).toBeLessThan(TOLERANCE);
      }

      const omnibus = expected.omnibus as NonNullable<FixtureCase["omnibus"]>;
      expect(actual.omnibus.degreesOfFreedom).toBe(omnibus.degrees_of_freedom);
      expect(actual.omnibus.degenerate).toBe(omnibus.degenerate);
      expect(actual.omnibus.distinguishable).toBe(omnibus.distinguishable);
      // A relative tolerance on the statistic, because it is a quadratic form
      // after a triangular solve and its magnitude ranges over four orders
      // across the corpus. It holds at 1e-9 only because the test is two
      // dimensional: the eight-dimensional version this replaced inverted a
      // near-singular matrix and drifted past 1e-9 on one case in ninety-eight.
      const scale = Math.max(1, Math.abs(omnibus.statistic));
      expect(Math.abs(actual.omnibus.statistic - omnibus.statistic) / scale).toBeLessThan(1e-9);
      expect(Math.abs(actual.omnibus.rimShift - omnibus.rim_shift)).toBeLessThan(TOLERANCE);
      expect(Math.abs(actual.omnibus.threeShift - omnibus.three_shift)).toBeLessThan(TOLERANCE);
      expect(Math.abs(actual.omnibus.rimShiftError - omnibus.rim_shift_error)).toBeLessThan(
        TOLERANCE
      );
      expect(Math.abs(actual.omnibus.threeShiftError - omnibus.three_shift_error)).toBeLessThan(
        TOLERANCE
      );

      expect(actual.argminUnstable).toBe(expected.argmin_unstable);
      expect(
        Math.abs(actual.profileVarianceShare - (expected.profile_variance_share as number))
      ).toBeLessThan(TOLERANCE);

      const mechanism = expected.mechanism ?? [];
      expect(actual.mechanism.length).toBe(mechanism.length);
      for (let m = 0; m < mechanism.length; m += 1) {
        const want = mechanism[m] as (typeof mechanism)[number];
        const got = actual.mechanism[m] as (typeof actual.mechanism)[number];
        expect(got.term).toBe(want.term);
        expect(got.verdict).toBe(want.verdict);
        expect(got.coefficient).toBeCloseTo(want.coefficient, 12);
        expect(Math.abs(got.featureDelta - want.feature_delta)).toBeLessThan(TOLERANCE);
      }
    }
  );
});

describe("the fixture covers the branches that matter", () => {
  /**
   * A parity fixture that never exercises a branch cannot prove the branch
   * agrees, and each of these is a branch whose disagreement would be silent.
   */
  it("includes the league-average arm", () => {
    expect(cases.some((c) => c.label === "league_average" && c.right === null)).toBe(true);
  });

  it("includes a refusal", () => {
    expect(cases.some((c) => c.unprofiled !== null && c.unprofiled.length > 0)).toBe(true);
  });

  it("includes a degenerate comparison", () => {
    expect(cases.some((c) => c.omnibus?.degenerate === true)).toBe(true);
  });

  it("includes a case the omnibus separates and one it does not", () => {
    const scored = cases.filter((c) => c.unprofiled === null);
    expect(scored.some((c) => c.omnibus?.distinguishable === true)).toBe(true);
    expect(scored.some((c) => c.omnibus?.distinguishable === false)).toBe(true);
  });
});

describe("invariants the fixture itself must satisfy", () => {
  it("tests two parameters, because a lineup only has two", () => {
    // Every lineup term multiplies either the rim or the three indicator, so a
    // lineup's whole effect on the nine utilities is `a·rim + b·three`. The
    // shares live on an eight-dimensional simplex; the lineup does not.
    expect(LINEUP_TERM_ATTRIBUTE.length).toBe(5);
    for (const [, attribute] of LINEUP_TERM_ATTRIBUTE) {
      expect(["rim", "three"]).toContain(attribute);
    }
    for (const one of cases) {
      if (one.omnibus === undefined) continue;
      expect(one.omnibus.degrees_of_freedom).toBe(2);
    }
  });

  it("has a zero shift exactly when the lineups predict identically", () => {
    for (const one of cases) {
      if (one.omnibus === undefined || one.zones === undefined) continue;
      const moved = one.zones.some((zone) => zone.delta_share !== 0);
      const shifted = one.omnibus.rim_shift !== 0 || one.omnibus.three_shift !== 0;
      expect(shifted).toBe(moved);
    }
  });

  it("keeps every delta on the simplex", () => {
    for (const one of cases) {
      if (one.zones === undefined) continue;
      const total = one.zones.reduce((sum, zone) => sum + zone.delta_share, 0);
      expect(Math.abs(total)).toBeLessThan(1e-12);
    }
  });

  it("returns exactly nothing when a lineup is compared with itself", () => {
    // The placebo identity, asserted on the committed numbers rather than only
    // in the Python suite. Not "close to zero" -- exactly zero, including the
    // standard error, because a difference of a quantity with itself has no
    // sampling variability and any float here came from an asymmetry in the
    // finite differences.
    const identity = cases.find((c) => c.label === "identity");
    expect(identity).toBeDefined();
    for (const zone of (identity as FixtureCase).zones ?? []) {
      expect(zone.delta_share).toBe(0);
      expect(zone.points_per_100).toBe(0);
      expect(zone.standard_error).toBe(0);
      expect(zone.variance_coefficients).toBe(0);
      expect(zone.variance_profiles).toBe(0);
    }
  });

  it("carries a profile variance that is a real share of the total", () => {
    // The headline claim of the whole feature. If this ever collapsed toward
    // zero, the intervals would have quietly become model-only again.
    const share = (fixture as unknown as { mean_profile_variance_share: number })
      .mean_profile_variance_share;
    expect(share).toBeGreaterThan(0.2);
    expect(share).toBeLessThanOrEqual(1);
  });

  it("reports the contradicted pre-registered sign on a term a swap moves", () => {
    const swap = cases.find((c) => c.label === "one_player_swap");
    const spacing = (swap as FixtureCase).mechanism?.find((m) => m.term === "spacing_x_three");
    expect(spacing?.verdict).toBe("DISAGREES");
    expect(spacing?.coefficient).toBeLessThan(0);
  });
});
