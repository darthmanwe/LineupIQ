/**
 * The route registry.
 *
 * Declared as data, not as a hand-maintained list in prose, so three things
 * stay in sync automatically: what `GET /` advertises, what the Worker actually
 * mounts, and what CI walks to assert that nothing unbacked ever returns 2xx.
 *
 * A route moves from `planned` to `live` in exactly one place. Forgetting to
 * update the registry is not possible, because the registry is what mounts it.
 */

export type RouteState =
  /** Backed by real data and serving. */
  | "live"
  /** Declared, returns 501 naming what will back it. */
  | "planned"
  /** Permanently 410. Public data cannot support it. */
  | "withdrawn";

export type RouteSpec = {
  method: "GET" | "POST";
  path: string;
  summary: string;
  state: RouteState;
  /** What it will return once backed. Required for `planned`. */
  willServe?: string;
  /** Milestone that lands it. Required for `planned`. */
  milestone?: string;
  /** The dataset or model behind it. Required for `planned`. */
  backedBy?: string;
  /** Why it can never be served. Required for `withdrawn`. */
  reason?: string;
  /** The honest alternative. Required for `withdrawn`. */
  instead?: string;
};

const M3 = "M3 — EPSA model, calibration and the eval harness";
const M4 = "M4 — the refusal contract";
const M5 = "M5 — trade simulator and counterfactual backtest";
const M6 = "M6 — retrieval and the LLM evaluation harness";

export const ROUTES: readonly RouteSpec[] = [
  // -- meta: live now, because it describes the service rather than the data --
  {
    method: "GET",
    path: "/",
    summary: "This registry: every endpoint and its current state.",
    state: "live",
  },
  {
    method: "GET",
    path: "/health",
    summary: "Liveness, and which bindings are actually reachable.",
    state: "live",
  },
  {
    method: "GET",
    path: "/meta/snapshot",
    summary: "Snapshot id, git sha, row counts, build time.",
    state: "live",
  },
  {
    method: "GET",
    path: "/seasons",
    summary: "Declared season coverage.",
    state: "live",
  },
  {
    method: "GET",
    path: "/zones",
    summary: "Shot-zone vocabulary plus the SVG geometry the court heatmap draws.",
    state: "live",
  },

  // -- players -------------------------------------------------------------
  {
    method: "GET",
    path: "/players",
    summary: "Typeahead over normalised player names.",
    state: "live",
  },
  {
    method: "POST",
    path: "/lineups/support",
    summary: "The refusal contract: how much evidence these five have, and what may be said.",
    state: "live",
  },
  {
    method: "POST",
    path: "/lineups/hash",
    summary: "The order-invariant lineup hash, so a client can check its own.",
    state: "live",
  },
  {
    method: "GET",
    path: "/models/selection",
    summary: "Served selection coefficients and the pre-registered sign audit.",
    state: "live",
  },
  {
    method: "GET",
    path: "/players/:id/zones",
    summary: "Per-zone base rates with sample sizes, no lineup context.",
    state: "live",
    willServe:
      "The raw rate, the empirical-Bayes shrunk rate, and the weight that " +
      "separates them — because a shrunk rate on eleven attempts is mostly the " +
      "league prior and looks exactly like a measurement.",
    milestone: M3,
    backedBy: "data/gold/shot_facts/",
  },

  // -- lineups -------------------------------------------------------------
  {
    method: "POST",
    path: "/lineups/score",
    summary: "How this five-man lineup shifts a shooter's shot selection.",
    state: "live",
    // Not what this route was originally specced to serve, and the change is
    // the finding. It was going to return per-zone EPSA -- how well a lineup
    // makes a player shoot from a given spot. That was built, evaluated
    // against a full baseline ladder, and came back at +0.02% log loss on
    // unseen combinations: nothing. Shot *selection* is where the effect lives,
    // so the route serves a distribution over zones and the delta against the
    // league-average lineup.
    willServe:
      "A predicted distribution over the nine zones, the same shooter's " +
      "league-average-lineup baseline, and the delta between them -- with the " +
      "magnitude nulled out below the reportable possession floor.",
    milestone: M3,
    backedBy: "the closed-form conditional logit, equal to the Python fit to 1e-9",
  },
  {
    method: "POST",
    path: "/lineups/optimal-plays",
    summary: "Zones ranked by their priced contribution, with the ties left tied.",
    state: "live",
    willServe:
      "A ranked list - or an explicitly unordered set when the intervals overlap, " +
      "because ranking indistinguishable options is a way of lying.",
    milestone: M4,
    backedBy:
      "the priced shot-mix contribution, with delta-method intervals from the " +
      "full 20x20 coefficient covariance",
  },
  {
    method: "POST",
    path: "/lineups/allocate",
    summary: "Usage redistribution under a fixed possession budget.",
    state: "planned",
    willServe: "Shot-share allocation discounted by a fitted marginal-efficiency decay curve.",
    milestone: M5,
    backedBy: "the concave allocator, bisection on a single Lagrange multiplier",
  },

  // -- trades --------------------------------------------------------------
  {
    method: "POST",
    path: "/trades/simulate",
    summary: "Before/after lineup deltas for a player swap.",
    state: "planned",
    willServe:
      "Offensive and defensive deltas with Monte-Carlo intervals, under an " +
      "explicitly chosen minutes-reallocation rule that is a visible input.",
    milestone: M5,
    backedBy: "EPSA + defensive RAPM + decay curves",
  },

  // -- evidence ------------------------------------------------------------
  {
    method: "GET",
    path: "/evidence/search",
    summary: "Hybrid BM25 + dense retrieval over lineup documents.",
    state: "planned",
    willServe: "Ranked lineup docs with per-retriever scores and the fused rank.",
    milestone: M6,
    backedBy: "D1 FTS5 + Vectorize, with an offline mirror for CI",
  },

  // -- evaluation: the point of the project --------------------------------
  {
    method: "GET",
    path: "/eval/model",
    summary: "Calibration, Brier decomposition, and the full baseline ladder.",
    state: "live",
    willServe:
      "Every metric from the run log, including `beats_best_baseline` — which is " +
      "allowed to be false and is served either way.",
    milestone: M3,
    backedBy: "services/ml/runs/epsa/*.json",
  },
  {
    method: "GET",
    path: "/eval/retrieval",
    summary: "Recall@10, MRR, nDCG per retriever, plus the template ablation.",
    state: "live",
  },
  {
    method: "GET",
    path: "/eval/groundedness",
    summary: "Numeric traceability, per-check pass rates, both distractor controls.",
    state: "live",
    willServe:
      "Deterministic groundedness scores over the committed narratives, with " +
      "both distractor controls beside them — a grounded rate of 1.00 means " +
      "nothing without them, because a checker that accepts everything scores " +
      "1.00 too.",
    milestone: M6,
    backedBy: "services/ml/runs/groundedness/run.json",
  },
  {
    method: "GET",
    path: "/eval/judge",
    summary: "Judge rubric, Cohen's kappa, and the deterministic baselines' kappa.",
    state: "planned",
    willServe: "Agreement against real hand labels, published whichever way it falls.",
    milestone: M6,
    backedBy: "data/llm_labels/ + services/ml/runs/judge/agreement.json",
  },
  {
    method: "GET",
    path: "/dq/coverage",
    summary: "Stint validity, event coverage, stint-vs-box-score minutes.",
    state: "live",
  },

  // -- permanently withdrawn ----------------------------------------------
  {
    method: "GET",
    path: "/leaderboards/gravity",
    summary: "Off-ball gravity leaderboard.",
    state: "withdrawn",
    reason:
      "Gravity requires player-tracking data. This project uses public play-by-play " +
      "and shot coordinates, which record where the ball went and not where the " +
      "defenders were. No amount of further work here produces this metric.",
    instead: "Use the spacing proxy on /lineups/score, which is computed from 3PT attempt rates.",
  },
  {
    method: "GET",
    path: "/shots/:id/contest",
    summary: "Per-shot contest quality.",
    state: "withdrawn",
    reason:
      "Contest quality needs the nearest defender's distance at release. That is " +
      "tracking data, and it is not in any public feed.",
    instead: "Shot difficulty is inferred from location, context and opponent rim protection.",
  },
] as const;

/** Routes that must never return 2xx while unbacked. Walked by CI. */
export const UNBACKED_ROUTES = ROUTES.filter((r) => r.state !== "live");
