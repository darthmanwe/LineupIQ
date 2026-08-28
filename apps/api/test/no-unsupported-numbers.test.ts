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
  {
    path: "/lineups/optimal-plays",
    body: (players) => ({ shooter_id: players[0], offense: players, defense: [] }),
  },
  {
    // Compared against the league average rather than a second five, and the
    // choice is forced by the shape of this list: a body builder is handed one
    // lineup, so any synthesised opponent would carry its own tier and the
    // sweep would stop knowing which side it was testing. The league-average
    // arm has no roster and therefore no support of its own, so the tier here
    // is exactly this lineup's.
    //
    // The two-lineup gate -- that the weaker of the two sides governs -- is
    // covered in `live.test.ts`, where both sides can be chosen.
    path: "/lineups/compare",
    body: (players) => ({
      shooter_id: players[0],
      left: { offense: players, defense: [] },
      right: { preset: "league_average" },
    }),
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

  it("/lineups/compare keeps its direction and loses its priced magnitude", async () => {
    // The same split the other two routes make, on a difference rather than a
    // level: the per-zone `delta_share` is the model's own output and survives,
    // the priced `points_per_100` is the product claim and does not.
    //
    // The omnibus survives too, and for the same reason the ranking does one
    // route down: whether the two lineups separate is a statement about how
    // precisely twenty coefficients were fitted, not about how many possessions
    // these particular five have played.
    for (const players of lineups) {
      const response = await post("/lineups/compare", {
        shooter_id: players[0],
        left: { offense: players, defense: [] },
        right: { preset: "league_average" },
      });
      if (response.status === 422) {
        // Unreachable on the current snapshot -- the profile floor sits below
        // the support floor, so the tier gate catches these lineups first --
        // but asserted rather than assumed, because the day the two floors
        // diverge this is the shape the answer takes.
        const refusal = (await response.json()) as { code: string };
        expect(refusal.code).toBe("NO_FITTED_RATE");
        continue;
      }
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        data: {
          omnibus: { statistic: number; distinguishable: boolean; degrees_of_freedom: number };
          zones: Array<{
            delta_share: number;
            share_left: number | null;
            share_right: number | null;
            points_per_100: number | null;
            interval: [number, number] | null;
            standard_error: number;
            variance_share: { coefficients: number; profiles: number };
          }>;
        };
        meta: { support: { tier: string }; warnings: string[]; comparison?: unknown };
      };
      if (body.meta.support.tier === "reportable") continue;

      expect(body.data.zones.length).toBeGreaterThan(0);
      for (const zone of body.data.zones) {
        expect(zone.points_per_100, `points on ${players.join(",")}`).toBeNull();
        expect(zone.interval, `interval on ${players.join(",")}`).toBeNull();
        expect(zone.share_left).toBeNull();
        expect(zone.share_right).toBeNull();
        expect(Number.isFinite(zone.delta_share)).toBe(true);
        expect(Number.isFinite(zone.standard_error)).toBe(true);
        // Both variance components ship at every tier. They are what the
        // interval would have been made of, and the endpoint's whole argument
        // is that the second one is usually the larger.
        expect(Number.isFinite(zone.variance_share.coefficients)).toBe(true);
        expect(Number.isFinite(zone.variance_share.profiles)).toBe(true);
      }

      // Shares live on a simplex, so the deltas must cancel.
      const total = body.data.zones.reduce((a, z) => a + z.delta_share, 0);
      expect(Math.abs(total)).toBeLessThan(1e-9);

      expect(body.data.omnibus.degrees_of_freedom).toBe(2);
      expect(Number.isFinite(body.data.omnibus.statistic)).toBe(true);
      // The meta block has to actually survive `envelope()`'s whitelist. It was
      // added to the type and to the spread; a block present in one and not the
      // other is dropped silently, which has happened before.
      expect(body.meta.comparison).toBeDefined();
      expect(body.meta.warnings.join(" ")).toMatch(/direction/i);
    }
  });

  it("/lineups/optimal-plays keeps its ranking and loses its magnitudes", async () => {
    // The ranking is the one thing that *survives* a directional tier here, and
    // the distinction is worth stating because it is easy to get backwards.
    //
    // A magnitude ("this lineup is worth +2.1 points per 100 at the rim") is a
    // claim about this five-man group, and below the possession floor there is
    // no evidence for it. The *ordering* is a claim about the model's own
    // precision -- whether two coefficient-driven contributions separate at 80%
    // -- and that comes from 671,251 attempts fitting twenty parameters, not
    // from this lineup's possessions. So the ranks stay and the numbers go.
    for (const players of lineups) {
      const response = await post("/lineups/optimal-plays", {
        shooter_id: players[0],
        offense: players,
        defense: [],
      });
      expect(response.status).toBe(200);
      const body = (await response.json()) as {
        data: {
          plays: Array<{
            rank: number;
            points_per_100: number | null;
            points_direction: string;
            standard_error: number;
            interval: [number, number] | null;
            share: number | null;
          }>;
        };
        meta: { support: { tier: string } };
      };
      if (body.meta.support.tier === "reportable") continue;

      expect(body.data.plays.length).toBeGreaterThan(0);
      for (const play of body.data.plays) {
        expect(play.points_per_100, `points on ${players.join(",")}`).toBeNull();
        expect(play.share).toBeNull();
        // The interval must go too. It is centred on the point estimate, so
        // serving `[lo, hi]` beside a nulled `points_per_100` would hand the
        // refused number straight back as `(lo + hi) / 2` -- a refusal that
        // does not refuse, which is worse than none because it looks correct.
        expect(play.interval, `interval on ${players.join(",")}`).toBeNull();
        // What survives: the rank, the direction, and the interval width. None
        // of those is a magnitude for this lineup.
        expect(play.rank).toBeGreaterThanOrEqual(1);
        expect(["gain", "loss", "flat"]).toContain(play.points_direction);
        expect(Number.isFinite(play.standard_error)).toBe(true);
      }
    }
  });

  it("/lineups/optimal-plays never presents a tie as an ordering", async () => {
    // Two zones the model cannot separate must carry the same rank, and a
    // response with `ordered: false` must have exactly one band. Getting either
    // wrong is precisely the failure this endpoint exists to prevent, and it
    // would look completely ordinary in the payload.
    for (const players of lineups) {
      const response = await post("/lineups/optimal-plays", {
        shooter_id: players[0],
        offense: players,
        defense: [],
      });
      const body = (await response.json()) as {
        data: {
          ordered: boolean;
          bands: string[][];
          plays: Array<{ zone_id: string; rank: number }>;
        };
        meta: { warnings: string[] };
      };
      const ranks = new Set(body.data.plays.map((p) => p.rank));
      expect(ranks.size).toBe(body.data.bands.length);
      if (!body.data.ordered) {
        expect(body.data.bands).toHaveLength(1);
        expect(body.meta.warnings.join(" ")).toMatch(/unordered set/i);
      }
      body.data.bands.forEach((band, i) => {
        for (const zone of band) {
          const play = body.data.plays.find((p) => p.zone_id === zone);
          expect(play?.rank).toBe(i + 1);
        }
      });
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
