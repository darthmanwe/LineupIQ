/**
 * Python/TypeScript parity for the selection scorer, asserted inside workerd.
 *
 * The Worker re-implements the whole served model — nine utilities and a
 * softmax. A disagreement with Python raises nothing: it returns a plausible
 * shot mix that is quietly wrong, on the one endpoint the product is about. So
 * every case is re-scored here against a committed fixture Python generated.
 *
 * **Utilities are asserted, not only the mix.** A softmax is a contraction:
 * implementations that differ in the fourth decimal of a utility can still agree
 * to 1e-9 on the resulting share for the small zones, so comparing after the
 * normalisation is the weaker test. Both are checked.
 *
 * The fixture is the contract. Neither implementation is the reference.
 */

import { describe, expect, it } from "vitest";

// JSON imports rather than `readFileSync`: an `import.meta.url` path resolves to
// `/C:/...` on Windows, which workerd's filesystem shim rejects. esbuild inlines
// these at build time, so it works identically on both platforms.
import fixtureJson from "../../../data/parity/selection.json";
import modelJson from "../../web/public/data/selection_model.json";
import profilesJson from "../../web/public/data/selection_profiles.json";
import { REFERENCE_ZONE, scoreSelection } from "../src/scoring/selection";
import type { SelectionModel, SelectionProfiles } from "../src/scoring/selection";

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
  utilities: number[];
  mix: number[];
  baseline_mix: number[];
  shooter_known: boolean;
  shooter_weight: number;
};

type Fixture = {
  seed: number;
  zones: string[];
  term_names: string[];
  n_cases: number;
  n_unknown_shooters: number;
  cases: Case[];
};

const fixture = fixtureJson as unknown as Fixture;
const profiles = profilesJson as unknown as SelectionProfiles;
const model = modelJson as unknown as SelectionModel;

/** Tight enough that any real arithmetic difference fails it. */
const TOLERANCE = 1e-9;

function score(c: Case) {
  return scoreSelection(
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
    model
  );
}

describe("the fixture itself", () => {
  it("was generated against this model and these profiles", () => {
    // A fixture regenerated after a refit, against a model file that was not,
    // would fail every case with no indication why. Check the shapes first so
    // the failure names the real problem.
    expect(fixture.zones).toEqual(profiles.zones);
    expect(fixture.term_names).toEqual(model.term_names);
    expect(model.coefficients).toHaveLength(model.term_names.length);
    expect(fixture.cases).toHaveLength(fixture.n_cases);
  });

  it("exercises the branches a random draw would miss", () => {
    // A fixture of 500 random five-man draws never contains an unseen shooter,
    // a two-man lineup, or an empty defence — and those are exactly the paths
    // where a fallback can differ between two languages.
    expect(fixture.n_unknown_shooters).toBeGreaterThan(0);
    expect(fixture.cases.some((c) => c.request.offense.length < 5)).toBe(true);
    expect(fixture.cases.some((c) => c.request.defense.length === 0)).toBe(true);
    expect(fixture.cases.some((c) => c.request.live_ball && c.request.clutch)).toBe(true);
    expect(fixture.cases.some((c) => c.request.team_id === 1)).toBe(true);
  });

  it("pins the reference zone at zero utility contribution", () => {
    // If the two sides disagreed about which alternative is pinned, every
    // utility would shift by a constant and every *mix* would still match.
    // This is the one asymmetry the softmax hides completely.
    expect(profiles.zones).toContain(REFERENCE_ZONE);
    expect(model.term_names).not.toContain(`alt_${REFERENCE_ZONE}`);
  });
});

describe("selection scorer parity", () => {
  it("reproduces every utility to 1e-9", () => {
    let worst = 0;
    let worstIndex = -1;
    fixture.cases.forEach((c, i) => {
      const result = score(c);
      c.utilities.forEach((expected, z) => {
        const delta = Math.abs((result.utilities[z] as number) - expected);
        if (delta > worst) {
          worst = delta;
          worstIndex = i;
        }
      });
    });
    expect(worst, `worst disagreement at case ${worstIndex}`).toBeLessThan(TOLERANCE);
  });

  it("reproduces every predicted mix to 1e-9, and each sums to one", () => {
    fixture.cases.forEach((c, i) => {
      const result = score(c);
      c.mix.forEach((expected, z) => {
        expect(
          Math.abs((result.mix[z] as number) - expected),
          `case ${i} zone ${fixture.zones[z]}`
        ).toBeLessThan(TOLERANCE);
      });
      const total = result.mix.reduce((a, b) => a + b, 0);
      expect(Math.abs(total - 1)).toBeLessThan(1e-12);
    });
  });

  it("reproduces the league-average-lineup baseline to 1e-9", () => {
    // The baseline is what the lineup effect is measured against, so an error
    // here would misstate every delta the product reports while leaving the
    // absolute predictions correct.
    fixture.cases.forEach((c, i) => {
      const result = score(c);
      c.baseline_mix.forEach((expected, z) => {
        expect(
          Math.abs((result.baselineMix[z] as number) - expected),
          `case ${i} zone ${fixture.zones[z]}`
        ).toBeLessThan(TOLERANCE);
      });
    });
  });

  it("agrees on which shooters it has never seen", () => {
    fixture.cases.forEach((c, i) => {
      const result = score(c);
      expect(result.shooterKnown, `case ${i}`).toBe(c.shooter_known);
      expect(Math.abs(result.shooterWeight - c.shooter_weight)).toBeLessThan(TOLERANCE);
    });
  });

  it("gives an unseen shooter the league mix exactly", () => {
    // Not "approximately the league mix" — the fallback is a log ratio of zero,
    // so the prediction must be the model's own alternative-specific constants
    // with no player term at all.
    const unseen = fixture.cases.find((c) => !c.shooter_known);
    expect(unseen).toBeDefined();
    const result = score(unseen as Case);
    const withZeroTeam = scoreSelection(
      {
        shooterId: -1,
        offense: (unseen as Case).request.offense,
        defense: (unseen as Case).request.defense,
      },
      profiles,
      model
    );
    // Two different unknown ids must produce identical utilities: the model has
    // no information distinguishing them, and inventing any would be a lie.
    withZeroTeam.utilities.forEach((u, z) => {
      expect(Math.abs(u - (result.utilities[z] as number))).toBeLessThan(TOLERANCE);
    });
  });
});
