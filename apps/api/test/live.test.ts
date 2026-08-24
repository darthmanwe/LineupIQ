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

import coverageJson from "../../web/public/data/coverage.json";
import evaluationJson from "../../web/public/data/evaluation.json";
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
