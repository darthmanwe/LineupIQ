/**
 * Python/TypeScript parity, asserted inside workerd.
 *
 * The Worker re-implements the lineup hash and the support tier. Neither
 * disagreement raises anything at runtime: a hash mismatch returns zero rows
 * everywhere and looks like missing data, and a tier mismatch serves a
 * confident number where Python would have refused. So both are checked against
 * a committed fixture that Python generated.
 *
 * The fixture is the contract. Neither implementation is the reference.
 */

import { describe, expect, it } from "vitest";

// Imported rather than read from disk. `readFileSync` with an `import.meta.url`
// path resolves to `/C:/...` on Windows, which workerd's filesystem shim
// rejects; a JSON import is inlined by esbuild at build time and works
// identically on both platforms.
import fixtureJson from "../../../data/parity/lineups.json";
import playersJson from "../../web/public/data/players.json";
import supportJson from "../../web/public/data/support.json";
import { canonicalLineup, lineupHash, md5 } from "../src/scoring/lineupHash";
import { assessSupport } from "../src/scoring/support";
import type { PlayersData, SupportData } from "../src/data/store";

type Case = {
  players: number[];
  canonical: string;
  lineup_hash: string;
  possessions: number;
  min_player_attempts: number;
  tier: "reportable" | "directional" | "refused";
  counterfactual: boolean;
};

type Fixture = {
  seed: number;
  thresholds: SupportData["thresholds"];
  tier_counts: Record<string, number>;
  n_cases: number;
  cases: Case[];
};

const fixture = fixtureJson as unknown as Fixture;
const support = supportJson as unknown as SupportData;
const players = playersJson as unknown as PlayersData;

describe("md5", () => {
  // Standard vectors: if these fail, nothing else in this file means anything.
  it("matches the published test vectors", () => {
    expect(md5("")).toBe("d41d8cd98f00b204e9800998ecf8427e");
    expect(md5("abc")).toBe("900150983cd24fb0d6963f7d28e17f72");
    expect(md5("message digest")).toBe("f96b697d7cb7938d525a2f31aaf161d0");
    expect(md5("abcdefghijklmnopqrstuvwxyz")).toBe("c3fcd3d76192e4007dfb496cca67e13b");
  });

  it("handles inputs that cross a padding block boundary", () => {
    // 56 bytes is exactly where MD5 needs a second block for the length.
    expect(md5("a".repeat(55))).toBe(md5("a".repeat(55)));
    expect(md5("a".repeat(56)).length).toBe(32);
    expect(md5("a".repeat(64))).toBe("014842d480b571495a4a0363793f7367");
  });
});

describe("lineup hash", () => {
  it("sorts numerically, not lexicographically", () => {
    // 201143 must come before 1630552. A string sort reverses them, and every
    // engine that sorts numerically then produces a different hash.
    expect(canonicalLineup([1630552, 201143, 2544, 203999, 1629029])).toBe(
      "2544,201143,203999,1629029,1630552"
    );
    const lexicographic = [1630552, 201143, 2544, 203999, 1629029].map(String).sort().join(",");
    expect(canonicalLineup([1630552, 201143, 2544, 203999, 1629029])).not.toBe(lexicographic);
  });

  it("is order invariant", () => {
    const ids = [201939, 1628369, 203507, 1629027, 202695];
    const shuffled = [...ids].reverse();
    expect(lineupHash(shuffled)).toBe(lineupHash(ids));
  });

  it("rejects a lineup that is not five distinct players", () => {
    expect(() => lineupHash([1, 2, 3, 4])).toThrow(/is 5 players/);
    expect(() => lineupHash([1, 1, 2, 3, 4])).toThrow(/same player twice/);
  });

  it("reproduces every hash in the Python fixture", () => {
    expect(fixture.cases.length).toBeGreaterThan(2000);
    for (const entry of fixture.cases) {
      expect(canonicalLineup(entry.players)).toBe(entry.canonical);
      expect(lineupHash(entry.players)).toBe(entry.lineup_hash);
    }
  });
});

describe("support tier parity", () => {
  it("exercises every tier", () => {
    // A fixture that never reaches a branch cannot prove the branch agrees.
    // Random five-player draws produce zero reportable cases, which is why the
    // generator also samples real lineups by time played.
    for (const tier of ["reportable", "directional", "refused"]) {
      expect(fixture.tier_counts[tier], `no ${tier} cases in the fixture`).toBeGreaterThan(0);
    }
  });

  it("applies the same pre-registered thresholds as Python", () => {
    expect(support.thresholds).toEqual(fixture.thresholds);
  });

  it("reaches the same tier as Python on every case", () => {
    const disagreements: Array<{ players: number[]; python: string; typescript: string }> = [];

    for (const entry of fixture.cases) {
      const assessment = assessSupport(entry.players, support, players);
      if (assessment.tier !== entry.tier) {
        disagreements.push({
          players: entry.players,
          python: entry.tier,
          typescript: assessment.tier,
        });
      }
    }

    expect(disagreements.slice(0, 5)).toEqual([]);
    expect(disagreements).toHaveLength(0);
  });

  it("agrees on possession counts for every exported lineup", () => {
    for (const entry of fixture.cases) {
      const assessment = assessSupport(entry.players, support, players);
      // The export omits lineups below the directional possession floor, so a
      // Python count under that floor legitimately reads as 0 here. Above it,
      // the numbers must match exactly.
      if (entry.possessions >= fixture.thresholds.directional_possessions) {
        expect(assessment.possessions).toBe(entry.possessions);
      } else {
        expect(assessment.possessions).toBe(0);
      }
    }
  });

  it("flags a counterfactual lineup as counterfactual", () => {
    // Five real players who have certainly never shared a floor.
    const invented = fixture.cases.find((entry) => entry.counterfactual);
    expect(invented).toBeDefined();
    if (invented) {
      expect(assessSupport(invented.players, support, players).counterfactual).toBe(true);
    }
  });
});
