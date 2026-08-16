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
   */
  scoring?: { closed_form_version: string; parity_fixture: string };
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
    },
  };
}
