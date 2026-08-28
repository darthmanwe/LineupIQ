import { Hono, type Context } from "hono";

import { envelope } from "./http/envelope";
import { notYetBacked, problem, withdrawn } from "./http/problem";
import { mountLive } from "./routes/live";
import { ROUTES, type RouteSpec } from "./routes/registry";

export type Bindings = {
  ENVIRONMENT?: string;
  SNAPSHOT?: string;
  ASSETS?: Fetcher;
  // Added in the milestone that first needs them; see wrangler.toml.
  DB?: D1Database;
  LINEUP_VECTORS?: VectorizeIndex;
  AI?: Ai;
};

/**
 * Mounted under /api so the Next.js export can own /. Requests to static assets
 * never reach this Worker, which is why the free request budget is not a
 * constraint for a demo.
 */
const app = new Hono<{ Bindings: Bindings }>().basePath("/api");

app.use("*", async (c, next) => {
  const requestId = c.req.header("X-Request-Id") ?? crypto.randomUUID();
  c.res.headers.set("X-Request-Id", requestId);
  await next();
  c.res.headers.set("X-Request-Id", requestId);
});

/**
 * Redirect `/api/foo/` to `/api/foo`.
 *
 * Every *page* on this origin ends in a slash, because `output: "export"`
 * emits directories -- so a visitor who has learned that convention types
 * `/api/` and gets a 404 from the one endpoint whose whole job is
 * discoverability. The 404 was at least informative (it names `/api` in a
 * `registry` field), but two halves of one origin disagreeing about slashes is
 * a papercut worth removing rather than documenting.
 *
 * 308 rather than 301: the method and body must survive, or a POST to
 * `/api/lineups/compare/` would silently become a GET.
 */
app.use("*", async (c, next) => {
  const url = new URL(c.req.url);
  if (url.pathname.length > 1 && url.pathname.endsWith("/")) {
    url.pathname = url.pathname.replace(/\/+$/, "");
    return c.redirect(url.toString(), 308);
  }
  await next();
});

mountLive(app);

/** The registry itself. Describes the service, so it is backed today. */
app.get("/", (c) =>
  c.json(
    envelope(
      c,
      {
        service: "lineupiq",
        description:
          "NBA lineup and trade forecasting. Every scored number ships the evidence " +
          "behind it, and the API refuses rather than guessing when there is none.",
        docs: "https://github.com/darthmanwe/LineupIQ",
        counts: {
          live: ROUTES.filter((r) => r.state === "live").length,
          planned: ROUTES.filter((r) => r.state === "planned").length,
          withdrawn: ROUTES.filter((r) => r.state === "withdrawn").length,
        },
        routes: ROUTES.map((r) => ({
          method: r.method,
          path: `/api${r.path === "/" ? "" : r.path}`,
          summary: r.summary,
          state: r.state,
          ...(r.milestone ? { milestone: r.milestone } : {}),
          ...(r.reason ? { reason: r.reason } : {}),
        })),
      },
      {
        snapshot: c.env.SNAPSHOT ?? null,
        warnings:
          ROUTES.some((r) => r.state === "planned") && !c.env.DB
            ? ["No data is loaded yet. Every analytics endpoint returns 501 by design."]
            : [],
      }
    )
  )
);

/** Probes bindings rather than asserting health. An unreachable D1 is not healthy. */
app.get("/health", async (c) => {
  const checks: Record<string, "ok" | "absent" | "error"> = {};

  if (c.env.DB) {
    try {
      await c.env.DB.prepare("SELECT 1").first();
      checks.d1 = "ok";
    } catch {
      checks.d1 = "error";
    }
  } else {
    checks.d1 = "absent";
  }
  checks.vectorize = c.env.LINEUP_VECTORS ? "ok" : "absent";
  checks.ai = c.env.AI ? "ok" : "absent";

  const degraded = Object.values(checks).includes("error");
  return c.json(
    envelope(
      c,
      {
        status: degraded ? "degraded" : "ok",
        environment: c.env.ENVIRONMENT ?? "unknown",
        bindings: checks,
      },
      {
        snapshot: c.env.SNAPSHOT ?? null,
        warnings: degraded ? ["At least one binding is configured but unreachable."] : [],
      }
    ),
    degraded ? 503 : 200
  );
});

/**
 * Mount everything that is not yet backed.
 *
 * Generated from the registry rather than written out, so a route cannot exist
 * in the advertised list while quietly returning something else.
 */
function mountUnbacked(spec: RouteSpec): void {
  const handler = (c: Context<{ Bindings: Bindings }>): Response => {
    if (spec.state === "withdrawn") {
      return withdrawn(c, {
        endpoint: `/api${spec.path}`,
        reason: spec.reason ?? "Not supportable from public data.",
        instead: spec.instead ?? "See the roadmap.",
      });
    }
    return notYetBacked(c, {
      endpoint: `/api${spec.path}`,
      willServe: spec.willServe ?? spec.summary,
      milestone: spec.milestone ?? "unscheduled",
      backedBy: spec.backedBy ?? "unspecified",
    });
  };

  if (spec.method === "GET") app.get(spec.path, handler);
  else app.post(spec.path, handler);
}

// Live routes first: Hono matches in registration order, and the unbacked
// mounts below must never shadow a route that is actually backed.
mountLive(app);

for (const spec of ROUTES) {
  if (spec.state !== "live") mountUnbacked(spec);
}

app.notFound((c) =>
  problem(c, {
    status: 404,
    code: "NO_SUCH_ROUTE",
    title: "No such route",
    detail: `${c.req.method} ${new URL(c.req.url).pathname} is not a route this API serves.`,
    extensions: { registry: "/api" },
  })
);

app.onError((err, c) => {
  console.error("unhandled", err);
  return problem(c, {
    status: 500,
    code: "INTERNAL_ERROR",
    title: "Internal error",
    detail: "The request failed for a reason the API did not anticipate.",
  });
});

export default app;
