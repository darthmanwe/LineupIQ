import type { Context } from "hono";

/**
 * RFC 9457 `application/problem+json`. Every non-2xx response uses this shape,
 * so a client branches on `code` rather than string-matching prose.
 */
export type Problem = {
  type: string;
  title: string;
  status: number;
  code: string;
  detail: string;
  [extension: string]: unknown;
};

const DOCS_BASE = "https://github.com/darthmanwe/LineupIQ";

export const PROBLEM_CONTENT_TYPE = "application/problem+json";

export function problem(
  c: Context,
  init: {
    status: number;
    code: string;
    title: string;
    detail: string;
    extensions?: Record<string, unknown>;
  }
): Response {
  const body: Problem = {
    type: `${DOCS_BASE}/blob/main/docs/errors.md#${init.code.toLowerCase().replace(/_/g, "-")}`,
    title: init.title,
    status: init.status,
    code: init.code,
    detail: init.detail,
    ...init.extensions,
  };

  return c.json(body, init.status as 400, { "Content-Type": PROBLEM_CONTENT_TYPE });
}

/**
 * The answer for every endpoint that has no data behind it yet.
 *
 * A 501 that names what will back it, and when, is worth more than a 200
 * returning a plausible zero. The CLI does the same thing for the same reason:
 * an empty result is indistinguishable from a correct result over empty data.
 */
export function notYetBacked(
  c: Context,
  spec: { endpoint: string; willServe: string; milestone: string; backedBy: string }
): Response {
  return problem(c, {
    status: 501,
    code: "NOT_YET_BACKED",
    title: "Endpoint not yet backed by data",
    detail:
      "This endpoint is declared but has no model or dataset behind it yet. " +
      "It returns 501 rather than a placeholder value.",
    extensions: {
      endpoint: spec.endpoint,
      will_serve: spec.willServe,
      milestone: spec.milestone,
      backed_by: spec.backedBy,
      roadmap: `${DOCS_BASE}#roadmap`,
    },
  });
}

/**
 * For metrics that are not coming, as distinct from not built yet.
 *
 * `410 Gone`, not `501`: defender distance, contest quality and gravity cannot
 * be computed from public play-by-play at all, and putting them on a roadmap
 * would be the same overclaim in a slower form. A client seeing 410 should stop
 * asking.
 */
export function withdrawn(
  c: Context,
  spec: { endpoint: string; reason: string; instead: string }
): Response {
  return problem(c, {
    status: 410,
    code: "METRIC_WITHDRAWN",
    title: "Metric permanently withdrawn",
    detail: spec.reason,
    extensions: {
      endpoint: spec.endpoint,
      instead: spec.instead,
      roadmap: `${DOCS_BASE}#what-this-is-not`,
    },
  });
}

/**
 * Not enough evidence to make the requested claim.
 *
 * Always a problem document, never a 200 with a confident number and a
 * footnote. `what_would_help` is required by convention: "not enough data"
 * without "of what" is not an answer.
 */
export function insufficientSupport(
  c: Context,
  spec: {
    detail: string;
    nPossessions: number;
    threshold: number;
    shortfallPlayers?: Array<{ player_id: string; attempts: number; threshold: number }>;
    whatWouldHelp: string;
  }
): Response {
  return problem(c, {
    status: 422,
    code: "INSUFFICIENT_SUPPORT",
    title: "Not enough possessions to answer this",
    detail: spec.detail,
    extensions: {
      n_possessions: spec.nPossessions,
      threshold: spec.threshold,
      shortfall_players: spec.shortfallPlayers ?? [],
      what_would_help: spec.whatWouldHelp,
    },
  });
}
