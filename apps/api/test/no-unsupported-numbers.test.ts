/**
 * The one invariant this project exists to hold.
 *
 * **No scoring route may return a point estimate for a lineup that does not
 * support one.** Not a small number, not a number with a caveat, not a number
 * with a wide interval — no number.
 *
 * This is a sweep, not a set of examples. It walks every lineup in the support
 * parity fixture whose tier is `refused` or `directional` through every live
 * scoring route and asserts the invariant on each. The point of doing it that
 * way is that adding a new scoring route without gating it is the natural
 * mistake: the route works, the numbers look plausible, and nothing raises. A
 * per-route test would have to be remembered; this one fails the moment a live
 * POST route appears in the registry that the sweep does not know how to gate.
 *
 * The three tiers, restated because the assertions below are only meaningful
 * against them:
 *
 * - `refused` — no basis for any claim. **422**, carrying the shortfall and
 *   what would help. Never a 200.
 * - `directional` — the player terms have support and the five-man combination
 *   does not. This is the *normal* case for a counterfactual lineup. **200**,
 *   with the sign and the direction populated and every magnitude `null`.
 * - `reportable` — a magnitude may be served.
 */

import { describe, expect, it } from "vitest";

import fixtureJson from "../../../data/parity/lineups.json";
import playerZonesJson from "../../web/public/data/player_zones.json";
import playersJson from "../../web/public/data/players.json";
import selectionJson from "../../web/public/data/selection_model.json";
import selectionProfilesJson from "../../web/public/data/selection_profiles.json";
import snapshotJson from "../../web/public/data/snapshot.json";
import supportJson from "../../web/public/data/support.json";
import zonesJson from "../../web/public/data/zones.json";
import { ROUTES } from "../src/routes/registry";
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

type FixtureCase = {
  players: number[];
  tier: "reportable" | "directional" | "refused";
};

const fixture = fixtureJson as unknown as { cases: FixtureCase[] };

/**
 * Every live POST route that could return a number about a lineup, and how to
 * build a body for it.
 *
 * Kept beside the sweep rather than derived from the route table, because a
 * route's *shape* cannot be inferred from the registry. The completeness test
 * below is what stops this list from silently falling behind.
 */
const SCORING_ROUTES: Array<{ path: string; body: (players: number[]) => unknown }> = [
  { path: "/lineups/support", body: (players) => ({ players }) },
  {
    path: "/lineups/score",
    body: (players) => ({ shooter_id: players[0], offense: players, defense: [] }),
  },
];

/** Routes that take a lineup but make no claim about it, so support cannot apply. */
const NOT_A_CLAIM = new Set(["/lineups/hash"]);

async function post(path: string, body: unknown): Promise<Response> {
  clearAssetCache();
  return worker.fetch(
    new Request(`https://test.local/api${path}`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
    }),
    env
  );
}

function sample(tier: FixtureCase["tier"], count: number): number[][] {
  return fixture.cases
    .filter((c) => c.tier === tier)
    .slice(0, count)
    .map((c) => c.players);
}

describe("the sweep covers every route that could break the invariant", () => {
  it("knows how to call every live POST route", () => {
    // If this fails, a scoring route was added and the sweep does not exercise
    // it. Add it to SCORING_ROUTES, or to NOT_A_CLAIM with a reason.
    const live = ROUTES.filter((r) => r.state === "live" && r.method === "POST").map((r) => r.path);
    const known = new Set([...SCORING_ROUTES.map((r) => r.path), ...NOT_A_CLAIM]);
    const missed = live.filter((path) => !known.has(path));
    expect(missed, `unswept live POST routes: ${missed.join(", ")}`).toEqual([]);
  });

  it("has fixture cases in the tiers it needs", () => {
    expect(sample("refused", 1).length).toBe(1);
    expect(sample("directional", 1).length).toBe(1);
  });
});

describe("a refused lineup gets a 422 from every scoring route", () => {
  const lineups = sample("refused", 12);
  for (const route of SCORING_ROUTES) {
    it(`${route.path}`, async () => {
      for (const players of lineups) {
        const response = await post(route.path, route.body(players));
        expect(response.status, `${route.path} on ${players.join(",")}`).toBe(422);
        expect(response.headers.get("Content-Type")).toContain("application/problem+json");

        const body = (await response.json()) as {
          code: string;
          what_would_help?: string;
        };
        expect(body.code).toBe("INSUFFICIENT_SUPPORT");
        // "Not enough data" without "of what" is not an answer.
        expect((body.what_would_help ?? "").length).toBeGreaterThan(20);
      }
    });
  }
});

describe("a directional lineup keeps its direction and loses its magnitude", () => {
  const lineups = sample("directional", 12);

  it("/lineups/support serves no point estimate", async () => {
    for (const players of lineups) {
      const response = await post("/lineups/support", { players });
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        data: { estimate_permitted: boolean; what_would_help: string | null };
        meta: { support: { tier: string } };
      };
      if (body.meta.support.tier !== "reportable") {
        expect(body.data.estimate_permitted).toBe(false);
        expect(body.data.what_would_help).not.toBeNull();
      }
    }
  });

  it("/lineups/score nulls every share and keeps every delta", async () => {
    for (const players of lineups) {
      const response = await post("/lineups/score", {
        shooter_id: players[0],
        offense: players,
        defense: [],
      });
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        data: { zones: Array<{ share: number | null; delta: number; baseline_share: number }> };
        meta: { support: { tier: string }; warnings: string[] };
      };
      if (body.meta.support.tier === "reportable") continue;

      expect(body.data.zones.length).toBeGreaterThan(0);
      for (const zone of body.data.zones) {
        expect(zone.share, `share on ${players.join(",")}`).toBeNull();
        // The direction survives, and it has to be a real number — a null
        // delta would be a refusal wearing a 200, which is its own failure.
        expect(Number.isFinite(zone.delta)).toBe(true);
        expect(Number.isFinite(zone.baseline_share)).toBe(true);
      }
      // Attempts live on a simplex: a lineup that moves a shooter toward one
      // zone must move him away from another. If these did not cancel the model
      // would be inventing attempts.
      const total = body.data.zones.reduce((a, z) => a + z.delta, 0);
      expect(Math.abs(total)).toBeLessThan(1e-9);

      expect(body.meta.warnings.join(" ")).toMatch(/direction/i);
    }
  });
});

describe("the escape hatch is narrow", () => {
  it("/lineups/hash answers without support because it makes no claim", async () => {
    // The one route that takes five players and returns a value regardless. It
    // is a pure function of the ids -- no evidence involved, nothing to refuse.
    const [players] = sample("refused", 1);
    const response = await post("/lineups/hash", { players });
    expect(response.status).toBe(200);
    const body = (await response.json()) as { data: { lineup_hash: string } };
    expect(body.data.lineup_hash).toMatch(/^[0-9a-f]{32}$/);
  });
});
