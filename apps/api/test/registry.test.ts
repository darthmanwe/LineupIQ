import { describe, expect, it } from "vitest";

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
