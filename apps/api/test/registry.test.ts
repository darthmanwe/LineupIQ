import { describe, expect, it } from "vitest";

// `?raw` inlines the file's text at build time. `readFileSync` is not an option:
// there is no filesystem in workerd, and an `import.meta.url` path resolves to
// `/C:/...` on Windows, which its shim rejects outright.
import overview from "../../web/src/app/page.tsx?raw";
import evidencePage from "../../web/src/app/evidence/page.tsx?raw";
import lineupPage from "../../web/src/app/lineup/page.tsx?raw";
import qualityPage from "../../web/src/app/quality/page.tsx?raw";
import tradePage from "../../web/src/app/trade/page.tsx?raw";

import app from "../src/index";
import { PROBLEM_CONTENT_TYPE } from "../src/http/problem";
import { ROUTES, UNBACKED_ROUTES } from "../src/routes/registry";

const env = { ENVIRONMENT: "test", SNAPSHOT: "unbuilt" };

/** Fill `:param` segments so a declared path can actually be requested. */
function concretePath(path: string): string {
  return `/api${path.replace(/:[^/]+/g, "1628389")}`;
}

async function call(spec: { method: "GET" | "POST"; path: string }): Promise<Response> {
  return app.request(
    concretePath(spec.path),
    spec.method === "POST"
      ? { method: "POST", body: "{}", headers: { "Content-Type": "application/json" } }
      : { method: "GET" },
    env
  );
}

describe("route registry", () => {
  it("serves itself", async () => {
    const res = await app.request("/api", {}, env);
    expect(res.status).toBe(200);

    const body = (await res.json()) as {
      data: { routes: unknown[]; counts: Record<string, number> };
      meta: { warnings: string[] };
    };
    expect(body.data.routes).toHaveLength(ROUTES.length);
    expect(body.meta.warnings.join(" ")).toContain("No data is loaded yet");
  });

  it("advertises every route under the /api prefix", async () => {
    const res = await app.request("/api", {}, env);
    const body = (await res.json()) as { data: { routes: Array<{ path: string }> } };
    for (const route of body.data.routes) {
      expect(route.path.startsWith("/api")).toBe(true);
    }
  });

  it("stamps a request id and echoes a supplied one", async () => {
    const generated = await app.request("/api", {}, env);
    expect(generated.headers.get("X-Request-Id")).toBeTruthy();

    const echoed = await app.request("/api", { headers: { "X-Request-Id": "abc-123" } }, env);
    expect(echoed.headers.get("X-Request-Id")).toBe("abc-123");
  });
});

describe("health", () => {
  it("reports absent bindings without claiming to be unhealthy", async () => {
    const res = await app.request("/api/health", {}, env);
    expect(res.status).toBe(200);

    const body = (await res.json()) as {
      data: { status: string; bindings: Record<string, string> };
    };
    // Nothing is bound yet, which is expected -- not an error.
    expect(body.data.status).toBe("ok");
    expect(body.data.bindings.d1).toBe("absent");
  });
});

/**
 * The guarantee this project is built around: nothing unbacked ever answers 2xx.
 *
 * Walking the registry rather than listing paths means a new endpoint is covered
 * the moment it is declared, including one added in a hurry.
 */
describe("no unbacked route returns success", () => {
  it("has routes to check", () => {
    expect(UNBACKED_ROUTES.length).toBeGreaterThan(0);
  });

  it.each(UNBACKED_ROUTES.map((r) => [`${r.method} ${r.path}`, r] as const))(
    "%s refuses",
    async (_label, spec) => {
      const res = await call(spec);

      expect(res.ok).toBe(false);
      expect(res.status).toBe(spec.state === "withdrawn" ? 410 : 501);
      expect(res.headers.get("Content-Type")).toContain(PROBLEM_CONTENT_TYPE);

      const body = (await res.json()) as Record<string, unknown>;
      expect(body.code).toBe(spec.state === "withdrawn" ? "METRIC_WITHDRAWN" : "NOT_YET_BACKED");
      expect(body.type).toContain("docs/errors.md#");
      expect(typeof body.detail).toBe("string");
    }
  );

  it("every 501 names what will back it and when", async () => {
    for (const spec of UNBACKED_ROUTES.filter((r) => r.state === "planned")) {
      const body = (await (await call(spec)).json()) as Record<string, string>;
      expect(body.milestone, `${spec.path} must name a milestone`).toBeTruthy();
      expect(body.backed_by, `${spec.path} must name its data source`).toBeTruthy();
      expect(body.will_serve, `${spec.path} must say what it will serve`).toBeTruthy();
    }
  });

  it("every 410 says why it is impossible and what to use instead", async () => {
    for (const spec of UNBACKED_ROUTES.filter((r) => r.state === "withdrawn")) {
      const body = (await (await call(spec)).json()) as { detail?: string; instead?: string };
      expect((body.detail ?? "").length, `${spec.path} must explain itself`).toBeGreaterThan(40);
      expect(body.instead, `${spec.path} must offer an alternative`).toBeTruthy();
    }
  });
});

describe("registry declarations are complete", () => {
  it.each(ROUTES.filter((r) => r.state === "planned").map((r) => [r.path, r] as const))(
    "planned route %s declares willServe, milestone and backedBy",
    (_path, spec) => {
      expect(spec.willServe).toBeTruthy();
      expect(spec.milestone).toBeTruthy();
      expect(spec.backedBy).toBeTruthy();
    }
  );

  it.each(ROUTES.filter((r) => r.state === "withdrawn").map((r) => [r.path, r] as const))(
    "withdrawn route %s declares reason and instead",
    (_path, spec) => {
      expect(spec.reason).toBeTruthy();
      expect(spec.instead).toBeTruthy();
    }
  );

  it("has no duplicate method+path pairs", () => {
    const keys = ROUTES.map((r) => `${r.method} ${r.path}`);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe("unknown routes", () => {
  it("404 as a problem document pointing at the registry", async () => {
    const res = await app.request("/api/nope", {}, env);
    expect(res.status).toBe(404);
    expect(res.headers.get("Content-Type")).toContain(PROBLEM_CONTENT_TYPE);

    const body = (await res.json()) as Record<string, unknown>;
    expect(body.code).toBe("NO_SUCH_ROUTE");
    expect(body.registry).toBe("/api");
  });
});

describe("the site does not contradict the registry", () => {
  /**
   * The overview page is the first screen a visitor sees, and it went stale for
   * the entire time this project was being built: it said "Milestone 1 of 8 —
   * every analytics endpoint returns 501" while sixteen endpoints were live and
   * both models were fitted.
   *
   * Every published *number* here is generated from a run log and re-derived in
   * CI. None of that machinery covers prose, and prose is what told a visitor
   * the project was not built. This is the cheapest possible guard on the one
   * claim that actually misleads: the front page saying nothing works while the
   * registry says otherwise.
   */
  /**
   * Comments stripped first. The file's own header describes the stale text in
   * order to explain why this guard exists, and the first version of the guard
   * matched that description and failed — a check that cannot tell a page from
   * a note about the page is not checking the page.
   */
  const rendered = overview.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

  it("does not claim endpoints are unbacked while the registry serves them", () => {
    const liveRoutes = ROUTES.filter((r) => r.state === "live");
    expect(liveRoutes.length).toBeGreaterThan(0);

    // Phrases that are only true of a skeleton. Each one was, at some point,
    // literally on the page.
    for (const stale of [
      /every analytics endpoint returns/i,
      /Milestone \d+ of \d+/i,
      /the skeleton is deployed/i,
    ]) {
      expect(rendered, `overview page still says: ${stale}`).not.toMatch(stale);
    }
  });

  it("does not promise as future work something already built", () => {
    // "Next: ingest three seasons ... and reconstruct which five players were
    // on the floor" survived on the page long after both were done.
    expect(rendered).not.toMatch(/Next:\s*ingest/i);
    expect(rendered).not.toMatch(/What this will do/i);
  });
});

describe("no page says a live endpoint is unbuilt", () => {
  /**
   * The generalisation of the guard above, and it exists because fixing the
   * overview page was not enough: the Lineup page said
   * `POST /api/lineups/optimal-plays` returns 501 the day after that route went
   * live, and the Evidence page listed `GET /api/eval/retrieval` among the
   * endpoints it was waiting on -- a route that has been live and publishing
   * measured recall for some time.
   *
   * All three are the same failure. Prose is the only thing in this repository
   * with no staleness check, and the failure is always in the same direction:
   * the site understates what the API does, so a visitor concludes less is
   * built than is.
   *
   * The check is deliberately narrow. It does not police how a page describes a
   * route; it asserts that no page puts a *live* path next to `501`, `not yet`,
   * `will back` or `not built`, which is the specific sentence that has now gone
   * wrong three times.
   */
  const PAGES: Array<[string, string]> = [
    ["overview", overview],
    ["lineup", lineupPage],
    ["trade", tradePage],
    ["quality", qualityPage],
    ["evidence", evidencePage],
  ];

  const UNBUILT = /501|not yet|will back|not built|arrives in/i;

  const strip = (source: string): string =>
    source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

  const livePaths = ROUTES.filter((r) => r.state === "live").map((r) => r.path);

  it.each(PAGES)("%s page", (name, source) => {
    const rendered = strip(source);
    // Sentences, roughly: split on the boundaries a claim actually lives inside.
    const sentences = rendered.split(/(?<=[.!?])\s+|\n\n/);

    for (const path of livePaths) {
      // Parameterised paths never appear verbatim in prose, and the registry
      // root `/` is a substring of every closing JSX tag on every page -- the
      // first version of this check flagged three pages for containing `</div>`.
      if (path.includes(":") || path.length < 6) continue;
      for (const sentence of sentences) {
        if (!sentence.includes(path)) continue;
        expect(
          UNBUILT.test(sentence),
          `${name} page describes the live route ${path} as unbuilt: ${sentence.trim().slice(0, 160)}`
        ).toBe(false);
      }
    }
  });

  it("finds the live paths it is checking against", () => {
    // A guard that silently checks nothing passes forever. If the registry were
    // ever restructured so `path` stopped being a plain string, every assertion
    // above would vacuously succeed.
    expect(livePaths.filter((p) => !p.includes(":") && p.length >= 6).length).toBeGreaterThan(8);
    expect(livePaths).toContain("/lineups/optimal-plays");
    expect(livePaths).toContain("/eval/retrieval");
  });
});
