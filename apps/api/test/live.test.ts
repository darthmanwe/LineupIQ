/**
 * The live routes, against the real exported assets.
 *
 * The assets binding is stubbed with the committed JSON rather than a fixture,
 * so these tests fail if `lineupiq export` produces something the Worker cannot
 * read. That is the coupling worth testing: the export and the reader are in
 * different languages and different repositories' worth of tooling.
 *
 * The refusal cases matter most. A 200 carrying a confident number for a lineup
 * with no evidence is the one failure this project is built to prevent, and it
 * would not raise anywhere -- it would just be wrong.
 */

import { describe, expect, it } from "vitest";

import lineupFixtureJson from "../../../data/parity/lineups.json";
import coverageJson from "../../web/public/data/coverage.json";
import evaluationJson from "../../web/public/data/evaluation.json";
import playerZonesJson from "../../web/public/data/player_zones.json";
import playersJson from "../../web/public/data/players.json";
import selectionJson from "../../web/public/data/selection_model.json";
import selectionProfilesJson from "../../web/public/data/selection_profiles.json";
import snapshotJson from "../../web/public/data/snapshot.json";
import supportJson from "../../web/public/data/support.json";
import zonesJson from "../../web/public/data/zones.json";
import worker from "../src/index";
import { clearAssetCache } from "../src/data/store";

const ASSETS: Record<string, unknown> = {
  "support.json": supportJson,
  "players.json": playersJson,
  "player_zones.json": playerZonesJson,
  "zones.json": zonesJson,
  "snapshot.json": snapshotJson,
  "selection_model.json": selectionJson,
  "selection_profiles.json": selectionProfilesJson,
  "evaluation.json": evaluationJson,
  "coverage.json": coverageJson,
};

const env = {
  ENVIRONMENT: "test",
  SNAPSHOT: "test-snapshot",
  ASSETS: {
    fetch: (request: Request): Promise<Response> => {
      const name = new URL(request.url).pathname.replace("/data/", "");
      const payload = ASSETS[name];
      if (payload === undefined) {
        return Promise.resolve(new Response("not found", { status: 404 }));
      }
      return Promise.resolve(Response.json(payload));
    },
  } as unknown as Fetcher,
};

async function get(path: string): Promise<Response> {
  clearAssetCache();
  return worker.fetch(new Request(`https://test.local${path}`), env);
}

async function post(path: string, body: unknown): Promise<Response> {
  clearAssetCache();
  return worker.fetch(
    new Request(`https://test.local${path}`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
    }),
    env
  );
}

/** Five players certain to have real evidence: the highest-volume ones. */
/**
 * A five-man group that actually cleared the reportable possession floor.
 *
 * Not "the five highest-volume shooters" -- those five have never shared a
 * floor, so that lineup is `directional` and every magnitude comes back null.
 * Only 487 of 49,827 observed groups clear 200 possessions, which is the
 * estimability finding the whole refusal contract exists for; a test that wants
 * to see a number has to go and find one of them.
 */
function reportableLineup(): number[] {
  const fixture = lineupFixtureJson as unknown as {
    cases: Array<{ players: number[]; tier: string }>;
  };
  const found = fixture.cases.find((c) => c.tier === "reportable");
  if (found === undefined) throw new Error("no reportable lineup in the parity fixture");
  return found.players;
}

/** The weakest lineup in the fixture: the tier the refusal contract exists for. */
function refusedLineup(): number[] {
  const fixture = lineupFixtureJson as unknown as {
    cases: Array<{ players: number[]; tier: string }>;
  };
  const found = fixture.cases.find((c) => c.tier === "refused");
  if (found === undefined) throw new Error("no refused lineup in the parity fixture");
  return found.players;
}

function highVolumePlayers(count: number): number[] {
  const players = playersJson as unknown as {
    players: Record<string, { attempts: number }>;
  };
  return Object.entries(players.players)
    .sort((a, b) => b[1].attempts - a[1].attempts)
    .slice(0, count)
    .map(([id]) => Number(id));
}

describe("live metadata routes", () => {
  it("serves the declared seasons", async () => {
    const response = await get("/api/seasons");
    expect(response.status).toBe(200);
    const body = (await response.json()) as { data: { seasons: unknown[] } };
    expect(body.data.seasons).toHaveLength(3);
  });

  it("serves the zone vocabulary the model actually uses", async () => {
    const response = await get("/api/zones");
    const body = (await response.json()) as {
      data: { zones: Array<{ id: string }>; count: number };
    };
    expect(response.status).toBe(200);
    expect(body.data.count).toBe(9);
    // The taxonomy is exported from Python, not restated here. If these ids
    // ever drift, the court heatmap and the model disagree about "corner three".
    expect(body.data.zones.map((z) => z.id)).toContain("corner_three_left");
  });

  it("serves the snapshot fingerprints", async () => {
    const response = await get("/api/meta/snapshot");
    const body = (await response.json()) as {
      data: { n_contracts: number; thresholds_sha256: string };
    };
    expect(body.data.n_contracts).toBeGreaterThan(10);
    expect(body.data.thresholds_sha256).toHaveLength(64);
  });

  it("serves players with their evidence attached", async () => {
    const response = await get("/api/players?limit=5");
    const body = (await response.json()) as {
      data: { players: Array<{ player_id: string; attempts: number }>; total: number };
    };
    expect(body.data.players).toHaveLength(5);
    expect(body.data.total).toBeGreaterThan(400);
    for (const player of body.data.players) {
      expect(player.attempts).toBeGreaterThanOrEqual(0);
    }
  });

  it("filters players by name", async () => {
    const response = await get("/api/players?q=jokic");
    const body = (await response.json()) as { data: { players: Array<{ name: string }> } };
    expect(body.data.players.length).toBeGreaterThan(0);
    for (const player of body.data.players) {
      expect(player.name.toLowerCase()).toContain("jokic");
    }
  });
});

describe("the refusal contract", () => {
  it("refuses a lineup of players with no evidence, with a 422", async () => {
    // Ids that are not in the snapshot at all.
    const response = await post("/api/lineups/support", {
      players: [900000001, 900000002, 900000003, 900000004, 900000005],
    });
    expect(response.status).toBe(422);
    expect(response.headers.get("Content-Type")).toContain("application/problem+json");

    const body = (await response.json()) as {
      code: string;
      what_would_help: string;
      n_possessions: number;
      threshold: number;
    };
    expect(body.code).toBe("INSUFFICIENT_SUPPORT");
    expect(body.n_possessions).toBe(0);
    expect(body.threshold).toBeGreaterThan(0);
    // "Not enough data" without "of what" is not an answer.
    expect(body.what_would_help.length).toBeGreaterThan(20);
  });

  it("answers a counterfactual lineup of known players directionally", async () => {
    // Five real, high-volume players who have never all shared a floor: the
    // player terms have support, the combination does not. A 200 with a null
    // centre is the correct answer, and a refusal would be wrong.
    const response = await post("/api/lineups/support", { players: highVolumePlayers(5) });
    expect(response.status).toBe(200);

    const body = (await response.json()) as {
      data: { estimate_permitted: boolean; what_would_help: string | null };
      meta: { support: { tier: string; counterfactual: boolean } };
    };
    expect(["directional", "reportable"]).toContain(body.meta.support.tier);
    if (body.meta.support.tier === "directional") {
      expect(body.data.estimate_permitted).toBe(false);
      expect(body.data.what_would_help).toBeTruthy();
    }
  });

  it("never returns a point estimate without the support to back it", async () => {
    const response = await post("/api/lineups/support", { players: highVolumePlayers(5) });
    const body = (await response.json()) as {
      data: { estimate_permitted: boolean };
      meta: {
        support: { tier: string; lineup_possessions: number; thresholds: { possessions: number } };
      };
    };
    // The one invariant: permission to report requires clearing the floor.
    if (body.data.estimate_permitted) {
      expect(body.meta.support.lineup_possessions).toBeGreaterThanOrEqual(
        body.meta.support.thresholds.possessions
      );
    }
  });

  it("echoes the thresholds it applied so a caller can check the arithmetic", async () => {
    const response = await post("/api/lineups/support", { players: highVolumePlayers(5) });
    const body = (await response.json()) as {
      meta: { support: { thresholds: { possessions: number; attempts: number } } };
    };
    expect(body.meta.support.thresholds.possessions).toBeGreaterThan(0);
    expect(body.meta.support.thresholds.attempts).toBeGreaterThan(0);
  });

  it("rejects a malformed lineup rather than guessing", async () => {
    for (const body of [{ players: [1, 2, 3] }, { players: [1, 1, 2, 3, 4] }, { nope: true }]) {
      const response = await post("/api/lineups/support", body);
      expect(response.status).toBe(400);
      expect(((await response.json()) as { code: string }).code).toBe("INVALID_LINEUP");
    }
  });

  it("rejects a body that is not JSON", async () => {
    clearAssetCache();
    const response = await worker.fetch(
      new Request("https://test.local/api/lineups/support", { method: "POST", body: "{" }),
      env
    );
    expect(response.status).toBe(400);
    expect(((await response.json()) as { code: string }).code).toBe("MALFORMED_BODY");
  });
});

describe("scoring a counterfactual lineup", () => {
  it("serves a distribution that sums to one, and a delta against the league baseline", async () => {
    const five = highVolumePlayers(5);
    const defence = highVolumePlayers(10).slice(5);
    const response = await post("/api/lineups/score", {
      shooter_id: five[0],
      offense: five,
      defense: defence,
    });
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: {
        shooter: { known: boolean; evidence_weight: number };
        zones: Array<{
          zone_id: string;
          share: number | null;
          baseline_share: number;
          delta: number;
        }>;
      };
      meta: { support: { tier: string }; scoring: { parity_fixture: string } };
    };

    expect(body.data.zones).toHaveLength(9);
    expect(body.data.shooter.known).toBe(true);

    // The baseline is always populated: it needs no lineup evidence, because
    // it is defined as the league-average lineup.
    const baselineTotal = body.data.zones.reduce((a, z) => a + z.baseline_share, 0);
    expect(Math.abs(baselineTotal - 1)).toBeLessThan(1e-9);

    // The deltas must cancel. Shares live on a simplex, so a lineup that pushes
    // a shooter toward threes has to pull him away from something else -- if
    // these summed to anything but zero the model would be creating attempts.
    const deltaTotal = body.data.zones.reduce((a, z) => a + z.delta, 0);
    expect(Math.abs(deltaTotal)).toBeLessThan(1e-9);

    expect(body.meta.scoring.parity_fixture).toBe("data/parity/selection.json");
  });

  it("nulls the magnitude for a lineup below the reportable floor", async () => {
    // The whole refusal contract in one assertion: a directional answer keeps
    // its direction and loses its digits. There is no path that returns a
    // confident share with a caveat attached.
    const five = highVolumePlayers(5);
    const response = await post("/api/lineups/score", {
      shooter_id: five[0],
      offense: five,
      defense: highVolumePlayers(10).slice(5),
    });
    const body = (await response.json()) as {
      data: { zones: Array<{ share: number | null; delta: number }> };
      meta: { support: { tier: string }; warnings: string[] };
    };
    if (body.meta.support.tier !== "reportable") {
      expect(body.data.zones.every((z) => z.share === null)).toBe(true);
      expect(body.data.zones.every((z) => Number.isFinite(z.delta))).toBe(true);
      expect(body.meta.warnings.join(" ")).toMatch(/direction/i);
    }
  });

  it("refuses a shooter who is not on the floor rather than guessing", async () => {
    const five = highVolumePlayers(5);
    const outsider = highVolumePlayers(6)[5];
    const response = await post("/api/lineups/score", {
      shooter_id: outsider,
      offense: five,
      defense: [],
    });
    expect(response.status).toBe(400);
    const body = (await response.json()) as { detail: string };
    expect(body.detail).toMatch(/must be one of the five/);
  });

  it("warns loudly about a shooter it has never seen", async () => {
    const five = highVolumePlayers(4);
    const response = await post("/api/lineups/score", {
      shooter_id: 1,
      offense: [1, ...five],
      defense: [],
    });
    const body = (await response.json()) as {
      data: { shooter: { known: boolean } };
      meta: { warnings: string[] };
    };
    if (response.status === 200) {
      expect(body.data.shooter.known).toBe(false);
      expect(body.meta.warnings.join(" ")).toMatch(/no fitted profile/);
    }
  });

  it("rejects a possession clock outside the shot clock", async () => {
    const five = highVolumePlayers(5);
    const response = await post("/api/lineups/score", {
      shooter_id: five[0],
      offense: five,
      seconds_into_possession: 40,
    });
    expect(response.status).toBe(400);
  });

  it("rejects a four-man offence", async () => {
    const four = highVolumePlayers(4);
    const response = await post("/api/lineups/score", { shooter_id: four[0], offense: four });
    expect(response.status).toBe(400);
  });
});

describe("ranking the plays a lineup creates", () => {
  it("decomposes the priced shift, and the parts add back to the whole", async () => {
    const five = reportableLineup();
    const defence = highVolumePlayers(10).slice(5);
    const body = {
      shooter_id: five[0],
      offense: five,
      defense: defence,
    };

    const [ranked, scored] = await Promise.all([
      post("/api/lineups/optimal-plays", body),
      post("/api/lineups/score", body),
    ]);
    expect(ranked.status).toBe(200);
    expect(scored.status).toBe(200);

    const ranking = (await ranked.json()) as {
      data: {
        ordered: boolean;
        confidence: number;
        plays: Array<{
          zone_id: string;
          rank: number;
          points_per_100: number | null;
          standard_error: number;
          interval: [number, number];
        }>;
        bands: string[][];
        excluded_zones: string[];
      };
      meta: {
        support: { tier: string };
        ranking: {
          pairs_compared: number;
          diagonal_would_refuse: number;
          critical_value: number;
          parity_fixture: string;
        };
      };
    };
    const score = (await scored.json()) as { data: { points_per_100: number | null } };

    expect(ranking.meta.support.tier).toBe("reportable");
    expect(ranking.meta.ranking.parity_fixture).toBe("data/parity/plays.json");
    expect(ranking.data.confidence).toBe(0.8);

    // This endpoint is a decomposition, not a second model. If the parts stopped
    // summing to the headline, one of the two numbers would be wrong and a
    // reader looking at either page alone would never find out.
    const parts = ranking.data.plays.reduce((a, p) => a + (p.points_per_100 ?? 0), 0);
    const whole = score.data.points_per_100 as number;
    expect(ranking.data.excluded_zones).toEqual([]);
    expect(Math.abs(parts - whole)).toBeLessThan(1e-9);
  });

  it("publishes what the covariance bought, per request", async () => {
    // The response carries the count of pairs a marginal-interval-overlap test
    // would have refused to rank. That is the justification for shipping a 20x20
    // matrix to the edge rather than its diagonal, and it belongs where a caller
    // can check it rather than in a comment in the repository.
    const five = highVolumePlayers(5);
    const response = await post("/api/lineups/optimal-plays", {
      shooter_id: five[0],
      offense: five,
      defense: highVolumePlayers(10).slice(5),
    });
    const body = (await response.json()) as {
      meta: {
        ranking: { pairs_compared: number; diagonal_would_refuse: number; critical_value: number };
      };
    };
    expect(body.meta.ranking.pairs_compared).toBeGreaterThan(0);
    expect(body.meta.ranking.diagonal_would_refuse).toBeGreaterThanOrEqual(0);
    expect(body.meta.ranking.critical_value).toBeCloseTo(1.2815515655, 9);
  });

  it("gives tied zones the same rank and says so in a warning", async () => {
    const five = highVolumePlayers(5);
    const response = await post("/api/lineups/optimal-plays", {
      shooter_id: five[0],
      offense: five,
      defense: highVolumePlayers(10).slice(5),
    });
    const body = (await response.json()) as {
      data: {
        ordered: boolean;
        bands: string[][];
        plays: Array<{ zone_id: string; rank: number }>;
      };
      meta: { warnings: string[] };
    };

    // Ranks are 1..bands.length with no gaps: a rank that skipped a number would
    // mean a band existed with nothing in it.
    const ranks = body.data.plays.map((p) => p.rank);
    expect(new Set(ranks).size).toBe(body.data.bands.length);
    expect(Math.max(...ranks)).toBe(body.data.bands.length);

    // And the ranks never decrease down the list. That property is what makes
    // the array renderable as a ranking at all; the first version of the banding
    // produced sequences like 1, 2, 2, 1 and the parity fixture caught it.
    for (let i = 1; i < ranks.length; i += 1) {
      expect(ranks[i]).toBeGreaterThanOrEqual(ranks[i - 1] as number);
    }

    if (body.data.bands.some((band) => band.length > 1)) {
      expect(body.meta.warnings.join(" ")).toMatch(/share a rank/i);
    }
  });

  it("rejects a four-man offence", async () => {
    const four = highVolumePlayers(4);
    const response = await post("/api/lineups/optimal-plays", {
      shooter_id: four[0],
      offense: four,
    });
    expect(response.status).toBe(400);
  });
});

describe("comparing two lineups", () => {
  it("compares a five against the league average and agrees with /lineups/score", async () => {
    // The two endpoints publish the same quantity here -- `mix - baselineMix`
    // per zone -- and they are routed through one function so they cannot
    // drift. This is the assertion that would notice if anybody un-routed it.
    const five = highVolumePlayers(5);
    const body = {
      shooter_id: five[0],
      left: { offense: five, defense: [] },
      right: { preset: "league_average" },
    };
    const [compared, scored] = await Promise.all([
      post("/api/lineups/compare", body),
      post("/api/lineups/score", { shooter_id: five[0], offense: five, defense: [] }),
    ]);
    expect(compared.status).toBe(200);
    expect(scored.status).toBe(200);

    const comparison = (await compared.json()) as {
      data: {
        reference: string;
        right_hash: string | null;
        swap: unknown;
        omnibus: { degrees_of_freedom: number; rim_shift: number; three_shift: number };
        zones: Array<{ zone_id: string; delta_share: number }>;
      };
      meta: { comparison: { profile_variance_share: number; parity_fixture: string } };
    };
    const score = (await scored.json()) as {
      data: { zones: Array<{ zone_id: string; delta: number }> };
    };

    expect(comparison.data.reference).toBe("league_average");
    expect(comparison.data.right_hash).toBeNull();
    expect(comparison.data.swap).toBeNull();
    expect(comparison.data.omnibus.degrees_of_freedom).toBe(2);
    expect(comparison.meta.comparison.parity_fixture).toBe("data/parity/compare.json");

    for (let i = 0; i < score.data.zones.length; i += 1) {
      const a = comparison.data.zones[i] as { zone_id: string; delta_share: number };
      const b = score.data.zones[i] as { zone_id: string; delta: number };
      expect(a.zone_id).toBe(b.zone_id);
      expect(a.delta_share).toBe(b.delta);
    }
  });

  it("names the one player that changed", async () => {
    const pool = highVolumePlayers(6);
    const left = pool.slice(0, 5);
    const right = [...pool.slice(0, 4), pool[5] as number];
    const response = await post("/api/lineups/compare", {
      shooter_id: left[0],
      left: { offense: left, defense: [] },
      right: { offense: right, defense: [] },
    });
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: { swap: { out: string; in: string } | null; reference: string };
    };
    expect(body.data.reference).toBe("lineup");
    expect(body.data.swap).toEqual({ out: String(pool[4]), in: String(pool[5]) });
  });

  it("returns exactly nothing when a lineup is compared with itself", async () => {
    // The placebo identity, served. The trade backtest requires a player
    // swapped for himself to project exactly +0.000 on the grounds that a
    // placebo which drifts means the pipeline is broken; the same standard
    // applies to the same idea at request time.
    const five = highVolumePlayers(5);
    const response = await post("/api/lineups/compare", {
      shooter_id: five[0],
      left: { offense: five, defense: [] },
      right: { offense: five, defense: [] },
    });
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: {
        omnibus: { degenerate: boolean; distinguishable: boolean; statistic: number };
        zones: Array<{ delta_share: number; standard_error: number }>;
      };
      meta: { warnings: string[] };
    };
    expect(body.data.omnibus.degenerate).toBe(true);
    expect(body.data.omnibus.distinguishable).toBe(false);
    for (const zone of body.data.zones) {
      expect(zone.delta_share).toBe(0);
      expect(zone.standard_error).toBe(0);
    }
    expect(body.meta.warnings.join(" ")).toMatch(/nothing to test/i);
  });

  it("refuses a shooter who is not on both floors", async () => {
    // Swapping the shooter himself changes `shooter_mix`, whose coefficient is
    // 0.996 at z = 351. It would swamp the five lineup terms completely, and the
    // result would be a between-player comparison wearing a lineup-effect
    // label. That is a 400, not a caveat.
    const pool = highVolumePlayers(6);
    const response = await post("/api/lineups/compare", {
      shooter_id: pool[4],
      left: { offense: pool.slice(0, 5), defense: [] },
      right: { offense: [...pool.slice(0, 4), pool[5] as number], defense: [] },
    });
    expect(response.status).toBe(400);
    const body = (await response.json()) as { code: string; detail: string };
    expect(body.code).toBe("INVALID_REQUEST");
    expect(body.detail).toMatch(/right/i);
  });

  it("lets the weaker of the two lineups govern the tier", async () => {
    // A difference cannot be better supported than the lineups it is a
    // difference of. Taking the stronger side would let a well-evidenced lineup
    // launder a claim about one nobody has ever seen.
    const strong = highVolumePlayers(5);
    const weak = refusedLineup();

    const response = await post("/api/lineups/compare", {
      shooter_id: strong[0],
      left: { offense: strong, defense: [] },
      right: { offense: [strong[0] as number, ...weak.slice(0, 4)], defense: [] },
    });
    // Either the support gate or the fitted-rate gate fires; both are 422 and
    // both name what is missing. What must not happen is a 200.
    expect(response.status).toBe(422);
    const body = (await response.json()) as { code: string; what_would_help?: string };
    expect(["INSUFFICIENT_SUPPORT", "NO_FITTED_RATE"]).toContain(body.code);
  });

  it("publishes both variance components and says which one dominates", async () => {
    // The endpoint's central claim: most of a comparison's uncertainty is about
    // who these players are, not about the fitted model. If this ever collapsed
    // the intervals would have quietly become model-only again.
    const five = highVolumePlayers(5);
    const response = await post("/api/lineups/compare", {
      shooter_id: five[0],
      left: { offense: five, defense: [] },
      right: { preset: "league_average" },
    });
    const body = (await response.json()) as {
      data: {
        zones: Array<{ variance_share: { coefficients: number; profiles: number } }>;
      };
      meta: { comparison: { profile_variance_share: number } };
    };
    expect(body.meta.comparison.profile_variance_share).toBeGreaterThan(0);
    expect(body.meta.comparison.profile_variance_share).toBeLessThanOrEqual(1);
    for (const zone of body.data.zones) {
      expect(zone.variance_share.profiles).toBeGreaterThanOrEqual(0);
      expect(zone.variance_share.coefficients).toBeGreaterThanOrEqual(0);
    }
  });

  it("reports the mechanism, including the contradicted pre-registered sign", async () => {
    const pool = highVolumePlayers(6);
    const response = await post("/api/lineups/compare", {
      shooter_id: pool[0],
      left: { offense: pool.slice(0, 5), defense: [] },
      right: { offense: [...pool.slice(0, 4), pool[5] as number], defense: [] },
    });
    const body = (await response.json()) as {
      data: {
        mechanism: Array<{
          term: string;
          feature_delta: number;
          expected_sign: number | null;
          verdict: string;
        }>;
      };
    };
    const spacing = body.data.mechanism.find((m) => m.term === "spacing_x_three");
    expect(spacing).toBeDefined();
    expect(spacing?.verdict).toBe("DISAGREES");
    expect(spacing?.expected_sign).toBe(1);
    // An offensive-only swap cannot move the opponent terms.
    const opponent = body.data.mechanism.find((m) => m.term === "opp_rim_allowed_x_rim");
    expect(opponent?.feature_delta).toBe(0);
  });

  it("rejects a malformed reference", async () => {
    const five = highVolumePlayers(5);
    const response = await post("/api/lineups/compare", {
      shooter_id: five[0],
      left: { offense: five, defense: [] },
      right: { preset: "best_available" },
    });
    expect(response.status).toBe(400);
    const body = (await response.json()) as { detail: string };
    expect(body.detail).toMatch(/league_average/);
  });
});

describe("lineup hash endpoint", () => {
  it("returns the numerically-sorted canonical form", async () => {
    const response = await post("/api/lineups/hash", {
      players: [1630552, 201143, 2544, 203999, 1629029],
    });
    const body = (await response.json()) as { data: { canonical: string; lineup_hash: string } };
    expect(body.data.canonical).toBe("2544,201143,203999,1629029,1630552");
    expect(body.data.lineup_hash).toBe("055603fd81221abc574796a1e5d3c08a");
  });
});

describe("model and evaluation routes", () => {
  it("does not report an indeterminate coefficient as a contradiction", async () => {
    // Three verdicts, and conflating two of them overstates the finding in the
    // model's own favour: an interval spanning zero has not contradicted a
    // pre-registered sign, it has failed to resolve it. The distinction is the
    // reason standard errors were added.
    const response = await get("/api/models/selection");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: { sign_audit: Record<string, { verdict: string; ci95?: [number, number] }> };
      meta: { warnings: string[] };
    };
    const rows = Object.entries(body.data.sign_audit ?? {});
    expect(rows.length).toBeGreaterThan(0);

    const contradiction = body.meta.warnings.find((w) => w.includes("contradict"));
    const unresolved = body.meta.warnings.find((w) => w.includes("spanning"));
    for (const [name, row] of rows) {
      if (row.verdict === "indeterminate") {
        expect(contradiction ?? "", name).not.toContain(name);
        expect(unresolved ?? "", name).toContain(name);
      }
      if (row.verdict === "DISAGREES") {
        expect(contradiction ?? "", name).toContain(name);
      }
      // Whenever an interval is served, the verdict must agree with it.
      if (row.ci95) {
        const straddles = row.ci95[0] <= 0 && 0 <= row.ci95[1];
        expect(row.verdict === "indeterminate", name).toBe(straddles);
      }
    }
  });

  it("surfaces a contradicted pre-registered sign as a warning", async () => {
    const response = await get("/api/models/selection");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: { term_names: string[]; coefficients: number[] };
      meta: { warnings: string[] };
    };
    expect(body.data.term_names.length).toBe(body.data.coefficients.length);
    // spacing_x_three contradicts its pre-registered sign, and that is the most
    // interesting thing about this model rather than a footnote.
    expect(body.meta.warnings.join(" ")).toContain("pre-registered");
  });

  it("warns that the trade backtest is underpowered before serving its numbers", async () => {
    const response = await get("/api/eval/model");
    expect(response.status).toBe(200);
    const body = (await response.json()) as { meta: { warnings: string[] } };
    expect(body.meta.warnings.join(" ")).toContain("UNDERPOWERED");
  });

  it("404s an unknown evaluation section and lists the real ones", async () => {
    const response = await get("/api/eval/model?section=whatever-looks-best");
    expect(response.status).toBe(404);
    const body = (await response.json()) as { code: string; available: string[] };
    expect(body.code).toBe("NO_SUCH_SECTION");
    expect(body.available.length).toBeGreaterThan(0);
  });

  it("serves every quality gate with its threshold and verdict", async () => {
    const response = await get("/api/dq/coverage");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: {
        gates: Array<{ name: string; verdict: string; severity: string; detail: string }>;
        n_gates: number;
        n_passing: number;
      };
    };
    expect(body.data.n_gates).toBeGreaterThanOrEqual(12);
    // A gate with no detail is a gate nobody can act on.
    for (const gate of body.data.gates) {
      expect(gate.detail.length).toBeGreaterThan(20);
      expect(["PASS", "WARN", "FAIL"]).toContain(gate.verdict);
    }
    // The served snapshot must pass its own gates, or the endpoint 503s.
    expect(body.data.n_passing).toBe(body.data.n_gates);
  });

  it("serves the retrieval ablation", async () => {
    const response = await get("/api/eval/retrieval");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: { by_corpus: Record<string, Record<string, { recall: number }> | undefined> };
    };
    const recall = (corpus: string): number => {
      const scores = body.data.by_corpus[corpus];
      expect(scores, `no ${corpus} corpus in the ablation`).toBeDefined();
      const bm25 = scores?.bm25;
      expect(bm25, `no bm25 leg for ${corpus}`).toBeDefined();
      return bm25?.recall ?? Number.NaN;
    };
    // The finding itself: bare decimals retrieve far worse than vocabulary.
    expect(recall("full")).toBeGreaterThan(recall("numbers"));
  });
});

describe("per-player zone rates", () => {
  it("ships the raw rate, the shrunk rate, and the weight between them", async () => {
    const [id] = highVolumePlayers(1);
    const response = await get(`/api/players/${id}/zones`);
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: {
        attempts: number;
        zones: Array<{
          zone_id: string;
          attempts: number;
          raw_rate: number;
          shrunk_rate: number;
          shrinkage_weight: number;
        }>;
      };
    };
    expect(body.data.zones.length).toBeGreaterThan(0);
    for (const zone of body.data.zones) {
      // Shrinkage moves a rate toward the prior, so the shrunk value must lie
      // between the raw one and the league rate — never outside both.
      expect(zone.shrinkage_weight).toBeGreaterThanOrEqual(0);
      expect(zone.shrinkage_weight).toBeLessThanOrEqual(1);
      expect(zone.attempts).toBeGreaterThan(0);
      expect(zone.raw_rate).toBeGreaterThanOrEqual(0);
      expect(zone.raw_rate).toBeLessThanOrEqual(1);
    }
    // A high-volume shooter's own evidence should dominate almost everywhere.
    const heavy = body.data.zones.filter((z) => z.shrinkage_weight > 0.9);
    expect(heavy.length).toBeGreaterThan(0);
  });

  it("404s a player with no recorded attempts, without claiming he does not exist", async () => {
    const response = await get("/api/players/999999999/zones");
    expect(response.status).toBe(404);
    const body = (await response.json()) as { detail: string };
    expect(body.detail).toMatch(/not the same as/);
  });

  it("rejects a non-numeric id rather than looking it up", async () => {
    const response = await get("/api/players/lebron/zones");
    expect(response.status).toBe(400);
  });
});

describe("groundedness", () => {
  it("never serves a grounded rate without its controls beside it", async () => {
    const response = await get("/api/eval/groundedness");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      data: {
        templates: Array<{
          template: string;
          grounded_rate: number;
          mean_traceability: number;
          controls: { easy: number; near_miss: number };
        }>;
      };
      meta: { warnings: string[] };
    };
    expect(body.data.templates.length).toBeGreaterThan(0);
    for (const t of body.data.templates) {
      expect(t.controls.easy).toBeGreaterThanOrEqual(0);
      expect(t.controls.near_miss).toBeGreaterThanOrEqual(0);
      // The checker has to be able to tell real evidence from a near miss, or
      // its pass rate is measuring nothing.
      if (t.grounded_rate > 0.5) {
        expect(t.controls.near_miss).toBeLessThan(t.grounded_rate);
      }
    }
  });

  it("warns that perfect traceability is not perfect groundedness", async () => {
    // The single most important sentence the harness produces. If every
    // template traces at 1.00 — including the one written to hallucinate — a
    // reader who sees only that number will draw the wrong conclusion.
    const response = await get("/api/eval/groundedness");
    const body = (await response.json()) as { meta: { warnings: string[] } };
    expect(body.meta.warnings.join(" ")).toMatch(/cannot settle meaning/);
  });

  it("attributes each failure to the check that caught it", async () => {
    const response = await get("/api/eval/groundedness");
    const body = (await response.json()) as {
      data: { templates: Array<{ template: string; failures_by_check: Record<string, number> }> };
    };
    // The hallucinating template must fail, and it must fail somewhere
    // nameable. A harness that reports a low rate without saying which check
    // fired is not diagnostic.
    const hallucinating = body.data.templates.find((t) => t.template === "hallucinating");
    expect(hallucinating).toBeDefined();
    expect(Object.keys(hallucinating?.failures_by_check ?? {}).length).toBeGreaterThan(0);
  });
});

describe("missing assets", () => {
  it("503s rather than pretending, when the snapshot is not deployed", async () => {
    clearAssetCache();
    const response = await worker.fetch(new Request("https://test.local/api/zones"), {
      ENVIRONMENT: "test",
    });
    expect(response.status).toBe(503);
    expect(((await response.json()) as { code: string }).code).toBe("SNAPSHOT_NOT_DEPLOYED");
  });
});
