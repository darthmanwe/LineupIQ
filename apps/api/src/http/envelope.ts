import type { Context } from "hono";

/**
 * How much evidence stands behind a number.
 *
 * This is the product's central promise, so it is a required part of the wire
 * format rather than a footnote in the UI. A caller can check the arithmetic
 * itself: the thresholds that were applied are echoed back alongside the counts
 * they were applied to.
 */
export type Support = {
  /** Possessions this exact five-man group played. `0` for a counterfactual lineup. */
  lineup_possessions: number;
  /** Weakest player-level support in the lineup, in shot attempts. */
  min_player_attempts: number;
  /**
   * `reportable` — enough evidence for a point estimate.
   * `directional` — sign and rough magnitude only; `*_point` fields are null.
   * `refused`     — no estimate at all; the response is a 422 problem document.
   */
  tier: "reportable" | "directional" | "refused";
  /** True when these five have never shared the floor in this snapshot. */
  counterfactual: boolean;
  /** Pre-registered, hash-pinned in configs/support_thresholds.json. */
  thresholds: { possessions: number; attempts: number };
};

/**
 * Every successful response carries provenance.
 *
 * Two rules are encoded here. A number never appears without the model's own
 * measured error (`model.primary_*`), and a scored number never appears without
 * the evidence behind it (`support`).
 */
export type Meta = {
  request_id: string;
  snapshot: string | null;
  generated_at: string;
  resolved?: {
    season_id?: string;
    requested_season_id?: string | null;
    /** `exact` or `latest_available`; never a silent substitution. */
    resolution?: "exact" | "latest_available";
  };
  model?: {
    name: string;
    version: string;
    primary_metric?: string;
    primary_value?: number;
    primary_ci?: [number, number] | null;
    card?: string;
  };
  /** Present on every scored response. The refusal contract, machine-readable. */
  support?: Support;
  /**
   * Lets a served number be traced to the exact closed form that produced it,
   * and to the fixture proving the TypeScript and Python implementations agree.
   *
   * `git_sha` is the commit whose run log the coefficients came from. Without it
   * a served number can be traced to an *implementation* but not to a *fit*,
   * and those are different questions: "which code computed this" and "which
   * training run produced the numbers it used".
   */
  scoring?: { closed_form_version: string; parity_fixture: string; git_sha?: string | null };
  /**
   * How a ranked response decided what it was allowed to order.
   *
   * Separate from `support`, because they answer different questions and
   * conflating them is the mistake this block exists to prevent. `support` is
   * about *this lineup's* evidence: how many possessions these five have played,
   * and therefore whether a magnitude may be shown at all. This is about the
   * *model's* precision: whether two priced contributions separate at the
   * pre-registered level, which comes from the fit and not from the lineup. A
   * lineup can be below the possession floor -- no magnitudes -- and still have a
   * perfectly well-determined ordering, and that combination is the normal case.
   *
   * `diagonal_would_refuse` is published rather than kept internal because it is
   * the justification for shipping a 20x20 covariance to the edge instead of its
   * diagonal: it counts the pairs a marginal-interval-overlap test would have
   * declined to rank, on this request.
   */
  ranking?: {
    pairs_compared: number;
    diagonal_would_refuse: number;
    ties_spanning_bands: number;
    critical_value: number;
    method: string;
    parity_fixture: string;
    git_sha?: string | null;
  };
  /**
   * What a lineup comparison's uncertainty is actually made of.
   *
   * Separate from `support`, which is about this lineup's evidence, and from
   * `ranking`, which is about the model's precision. This block answers a third
   * question: of the interval you are being shown, how much comes from the
   * fitted coefficients and how much from the two players' own shooting rates.
   * On a typical comparison the second term is the larger one, and a reader who
   * is not told that would reasonably assume the model was the whole story.
   */
  comparison?: {
    profile_variance_share: number;
    omnibus_critical_value: number;
    degrees_of_freedom: number;
    method: string;
    parity_fixture: string;
    git_sha?: string | null;
  };
  warnings: string[];
};

export type Envelope<T> = { data: T; meta: Meta };

export function envelope<T>(
  c: Context,
  data: T,
  extra: Partial<Omit<Meta, "request_id" | "generated_at">> = {}
): Envelope<T> {
  return {
    data,
    meta: {
      request_id: c.res.headers.get("X-Request-Id") ?? "unknown",
      snapshot: extra.snapshot ?? null,
      generated_at: new Date().toISOString(),
      warnings: extra.warnings ?? [],
      ...(extra.resolved ? { resolved: extra.resolved } : {}),
      ...(extra.model ? { model: extra.model } : {}),
      ...(extra.support ? { support: extra.support } : {}),
      ...(extra.scoring ? { scoring: extra.scoring } : {}),
      ...(extra.ranking ? { ranking: extra.ranking } : {}),
      ...(extra.comparison ? { comparison: extra.comparison } : {}),
    },
  };
}
