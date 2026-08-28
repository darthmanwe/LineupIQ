"use client";

import { useMemo, useState } from "react";

import {
  CourtHeatmap,
  type MetricSpec,
  type ZoneShape,
  type ZoneValue,
} from "@/components/court/CourtHeatmap";
import { PlayerPicker, type PickablePlayer } from "@/components/PlayerPicker";

/**
 * Two lineups, one shooter, and the difference between them.
 *
 * **What this does not do.** It does not project whether a trade makes a team
 * better. That endpoint returns 501, and the reason is on this page directly
 * below: its own power analysis put the minimum detectable effect at the size
 * of the effects it projects, and its backtest could not beat assuming nothing
 * changed. This asks a narrower question the evidence can carry — given that a
 * shot goes up, does swapping this player change *where* it goes up from.
 *
 * Four decisions, each one a place this component could have flattered the
 * model instead:
 *
 * 1. **The omnibus is read before any zone.** Nine zones tested at 80% will
 *    separate something on most comparisons. So the headline is one test on the
 *    two parameters a lineup actually has, and the per-zone numbers sit under
 *    it rather than beside it.
 * 2. **The variance split is on screen, not in a tooltip.** Most of a
 *    comparison's interval is how well these two players' own shooting rates
 *    are known, not the fitted model. A reader told only "±0.4" would
 *    reasonably assume the model was the whole story.
 * 3. **A contradicted pre-registration is shown where it bites.** The marquee
 *    coefficient came back with the wrong sign, and it is one of the three
 *    terms an offensive swap moves. Hiding that would make the first swap
 *    anybody tries look like a bug.
 * 4. **The court encodes direction only when magnitude is withheld.** A
 *    continuous ramp over a number the contract refuses to report would be the
 *    refusal leaking through the colour scale.
 */

type Player = PickablePlayer;

type ZoneDelta = {
  zone_id: string;
  delta_share: number;
  share_left: number | null;
  share_right: number | null;
  points_per_100: number | null;
  points_direction: "gain" | "loss" | "flat";
  standard_error: number;
  interval: [number, number] | null;
  variance_share: { coefficients: number; profiles: number };
};

type Mechanism = {
  term: string;
  feature_left: number;
  feature_right: number;
  feature_delta: number;
  coefficient: number;
  expected_sign: number | null;
  verdict: string;
};

type CompareResponse = {
  data: {
    reference: "league_average" | "lineup";
    shooter: { player_id: string; name: string | null };
    swap: { out: string; in: string } | null;
    omnibus: {
      statistic: number;
      degrees_of_freedom: number;
      critical_value: number;
      distinguishable: boolean;
      degenerate: boolean;
      rim_shift: number;
      three_shift: number;
      rim_shift_error: number;
      three_shift_error: number;
    };
    zones: ZoneDelta[];
    mechanism: Mechanism[];
  };
  meta: {
    support: { tier: string; lineup_possessions: number; counterfactual: boolean };
    comparison: { profile_variance_share: number; degrees_of_freedom: number };
    warnings: string[];
  };
};

type Problem = {
  title?: string;
  detail?: string;
  code?: string;
  what_would_help?: string;
  n_possessions?: number;
  threshold?: number;
  players?: Array<{ player_id: string; name: string | null; attempts: number }>;
};

/** One percentage point of share each way — the same fixed domain the scorer uses. */
const DOMAIN = 0.01;

const GAIN = "#c74845";
const LOSS = "#2a78d6";

/** Human labels for the five lineup features. */
const TERM_LABEL: Record<string, string> = {
  spacing_x_three: "Mean teammate spacing",
  spacing_min_x_three: "Worst spacer on the floor",
  teammate_rim_x_rim: "Teammate rim pressure",
  opp_rim_allowed_x_rim: "Opponent rim protection",
  opp_three_allowed_x_three: "Opponent perimeter defence",
};

/**
 * The court vocabulary for a difference between two lineups.
 *
 * `ATTEMPT_SHARE` almost fits and says the wrong thing in one place: its
 * secondary clause names the baseline as "a league-average lineup on the
 * floor", which is exactly right for the league-average arm and false when the
 * reference is a second five. Two specs rather than one adaptive one, because
 * the wording is the whole reason the vocabulary is a parameter.
 */
function shareDeltaMetric(reference: "league_average" | "lineup"): MetricSpec {
  const baseline =
    reference === "league_average"
      ? "with a league-average lineup on the floor"
      : "with the comparison five on the floor";
  return {
    valueLabel: "Share",
    countNoun: "possession",
    formatDeviation: (d) => `${d >= 0 ? "+" : "−"}${(Math.abs(d) * 100).toFixed(2)}`,
    // NaN is how the caller says "the API nulled this". Reconstructing the
    // share from the baseline plus the delta would put on screen the exact
    // number the support contract refused to serve.
    formatValue: (v) =>
      Number.isFinite(v.points_per_attempt)
        ? `${(v.points_per_attempt * 100).toFixed(2)}%`
        : "not reportable",
    formatSecondary: (v) =>
      Number.isFinite(v.fg) ? `${(v.fg * 100).toFixed(2)}% ${baseline}` : null,
    secondaryLabel: "Reference",
    formatSecondaryCell: (v) => (Number.isFinite(v.fg) ? `${(v.fg * 100).toFixed(2)}%` : "—"),
    countLabel: "Possessions",
    // No per-zone count exists for a modelled share; the lineup's possessions
    // are the same number in all nine zones and belong in the caption.
    formatCount: () => null,
    // Neither end of this scale is the league. The reference is whatever the
    // right-hand side is, and saying "below league" over a comparison between
    // two chosen fives would be a legend contradicting its own caption.
    lowLabel: "Fewer attempts",
    highLabel: "More attempts",
    formatComparison: (v) =>
      `${v.deviation >= 0 ? "+" : "−"}${(Math.abs(v.deviation) * 100).toFixed(2)} pp vs ${
        reference === "league_average" ? "a league-average lineup" : "the comparison five"
      }`,
  };
}

export function LineupCompare({
  players,
  shapes,
  viewBox,
}: {
  players: Player[];
  shapes: ZoneShape[];
  viewBox: string;
}) {
  const [left, setLeft] = useState<number[]>(() => players.slice(0, 5).map((p) => p.id));
  const [right, setRight] = useState<number[]>(() => [
    ...players.slice(0, 4).map((p) => p.id),
    players[5]?.id ?? 0,
  ]);
  const [reference, setReference] = useState<"league_average" | "lineup">("league_average");
  const [shooter, setShooter] = useState<number>(() => players[0]?.id ?? 0);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [refusal, setRefusal] = useState<Problem | null>(null);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const byId = useMemo(() => new Map(players.map((p) => [p.id, p])), [players]);
  const zoneLabels = useMemo(
    () => Object.fromEntries(shapes.map((shape) => [shape.id, shape.label])),
    [shapes]
  );

  const reportable = result?.meta.support.tier === "reportable";
  const metric = useMemo(
    () => shareDeltaMetric(result?.data.reference ?? "league_average"),
    [result?.data.reference]
  );

  const values: Record<string, ZoneValue | undefined> = useMemo(() => {
    if (!result) return {};
    const out: Record<string, ZoneValue> = {};
    for (const zone of result.data.zones) {
      out[zone.zone_id] = {
        attempts: result.meta.support.lineup_possessions,
        fg: zone.share_right ?? Number.NaN,
        points_per_attempt: zone.share_left ?? Number.NaN,
        deviation: zone.delta_share,
        // Hatched when the whole comparison could not separate the two lineups.
        // The hatch says "the model cannot resolve this", which is exactly what
        // a failed omnibus means, and it is more honest than a pale fill that
        // still reads as a small effect.
        below_floor: !result.data.omnibus.distinguishable,
      };
    }
    return out;
  }, [result]);

  /** Copy the left five and change one slot — the swap the product is about. */
  function swapOne(index: number): void {
    setReference("lineup");
    setRight(left.map((id, i) => (i === index ? (right[i] ?? id) : id)));
  }

  async function compare(): Promise<void> {
    setBusy(true);
    setFailed(null);
    setRefusal(null);
    try {
      const response = await fetch("/api/lineups/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shooter_id: shooter,
          left: { offense: left, defense: [] },
          right:
            reference === "league_average"
              ? { preset: "league_average" }
              : { offense: right, defense: [] },
        }),
      });
      if (response.status === 422 || response.status === 400) {
        setRefusal((await response.json()) as Problem);
        setResult(null);
        return;
      }
      if (!response.ok) {
        setFailed(`The API returned ${response.status}.`);
        setResult(null);
        return;
      }
      setResult((await response.json()) as CompareResponse);
    } catch {
      setFailed(
        "Could not reach the API. This page needs the Worker running behind it — " +
          "`npx wrangler dev` from apps/api, or the deployed origin."
      );
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const duplicatedLeft = new Set(left).size !== left.length;
  const duplicatedRight = reference === "lineup" && new Set(right).size !== right.length;
  const shooterMissing =
    !left.includes(shooter) || (reference === "lineup" && !right.includes(shooter));
  const blocked = duplicatedLeft || duplicatedRight || shooterMissing;

  return (
    <section className="compare">
      <style>{`
        .compare { margin-top: 1.5rem; }
        .compare__grid { display: grid; gap: 1.5rem; }
        @media (min-width: 68rem) { .compare__grid { grid-template-columns: 26rem 1fr; align-items: start; } }
        .compare__panel { border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; }
        .compare__panel h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 0.75rem; }
        .compare__sides { display: grid; gap: 0.75rem; }
        @media (min-width: 30rem) { .compare__sides { grid-template-columns: 1fr 1fr; } }
        .compare__modes { display: flex; gap: 0.35rem; margin: 0 0 0.85rem; flex-wrap: wrap; }
        .compare__modes button { border: 1px solid var(--border); background: var(--bg); color: var(--muted); border-radius: 999px; padding: 0.3rem 0.7rem; font-size: 0.8rem; cursor: pointer; }
        .compare__modes button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); }
        .compare__row { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.75rem; flex-wrap: wrap; }
        .compare button.compare__go { border: 1px solid var(--border); background: var(--text); color: var(--bg); border-radius: 6px; padding: 0.45rem 0.9rem; font-size: 0.85rem; cursor: pointer; }
        .compare button[disabled] { opacity: 0.5; cursor: not-allowed; }
        .compare__swap { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.4rem; }
        .compare__swap button { border: 1px dashed var(--border); background: transparent; color: var(--muted); border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.72rem; cursor: pointer; }
        .compare__verdict { border: 1px solid var(--border); border-radius: var(--radius); padding: 0.9rem 1rem; margin-bottom: 1rem; }
        .compare__verdict[data-moved="false"] { border-left: 3px solid var(--border); }
        .compare__verdict[data-moved="true"] { border-left: 3px solid var(--accent); }
        .compare__headline { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.35rem; }
        .compare__stats { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-top: 0.6rem; }
        .compare__stat { font-size: 0.8rem; color: var(--muted); }
        .compare__stat b { display: block; font-family: var(--mono, monospace); font-size: 1rem; color: var(--text); font-weight: 600; }
        .compare__note { color: var(--muted); font-size: 0.83rem; margin: 0.5rem 0 0; }
        .compare__warn { border-left: 3px solid var(--border); padding: 0.5rem 0 0.5rem 0.85rem; margin: 0.75rem 0 0; font-size: 0.85rem; color: var(--muted); }
        .compare__refusal { border: 1px solid var(--border); border-left: 3px solid ${GAIN}; border-radius: var(--radius); padding: 1rem; }
        .compare__refusal h3 { color: var(--text); text-transform: none; letter-spacing: 0; font-size: 1rem; margin-top: 0; }
        .compare__meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.75rem; font-family: var(--mono, monospace); }
        .compare__empty { color: var(--muted); font-size: 0.9rem; }
        .compare__section { margin-top: 1.5rem; }
        .compare__section h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 0.6rem; }
        .compare__scroll { overflow-x: auto; }
        .mech { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .mech th, .mech td { text-align: left; padding: 0.35rem 0.6rem 0.35rem 0; border-bottom: 1px solid var(--border); white-space: nowrap; }
        .mech th { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 500; }
        .mech td.num { font-family: var(--mono, monospace); text-align: right; }
        .mech tr[data-moved="false"] { color: var(--muted); }
        .mech .verdict { font-size: 0.72rem; border-radius: 999px; padding: 0.1rem 0.45rem; border: 1px solid var(--border); }
        .mech .verdict[data-verdict="DISAGREES"] { border-color: ${GAIN}; color: ${GAIN}; }
        .split { display: flex; height: 0.7rem; border-radius: 999px; overflow: hidden; border: 1px solid var(--border); background: var(--bg); }
        .split span { display: block; }
        .split__key { display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.78rem; color: var(--muted); margin-top: 0.4rem; }
        .split__key i { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 2px; margin-right: 0.3rem; vertical-align: baseline; }
      `}</style>

      <div className="compare__grid">
        <div className="compare__panel">
          <h3>The five you have</h3>
          {[0, 1, 2, 3, 4].map((i) => (
            <PlayerPicker
              key={`left-${i}`}
              label={`Slot ${i + 1}`}
              value={left[i] ?? 0}
              exclude={left}
              fallback={players}
              onChange={(id) => {
                const next = [...left];
                next[i] = id;
                setLeft(next);
                if (!next.includes(shooter)) setShooter(next[0] ?? 0);
                // Keep the untouched slots of the comparison lineup in step, so
                // "swap one player" stays a swap of one player.
                setRight((current) =>
                  current.map((held, j) => (j === i ? held : (next[j] ?? held)))
                );
              }}
            />
          ))}

          <PlayerPicker
            label="Taking the shot"
            value={shooter}
            fallback={left.map((id) => byId.get(id)).filter((p): p is Player => p !== undefined)}
            onChange={setShooter}
          />

          <h3 style={{ marginTop: "1.25rem" }}>Compare against</h3>
          <div className="compare__modes">
            <button
              type="button"
              aria-pressed={reference === "league_average"}
              onClick={() => setReference("league_average")}
            >
              League average
            </button>
            <button
              type="button"
              aria-pressed={reference === "lineup"}
              onClick={() => setReference("lineup")}
            >
              Another five
            </button>
          </div>

          {reference === "league_average" ? (
            <p className="compare__note">
              The same shooter with every lineup term at the league mean. It needs no roster: each
              lineup feature is a deviation from the league rate, so dropping all five
              <em> is</em> the average lineup.
            </p>
          ) : (
            <>
              {[0, 1, 2, 3, 4].map((i) => (
                <PlayerPicker
                  key={`right-${i}`}
                  label={`Slot ${i + 1}`}
                  value={right[i] ?? 0}
                  exclude={right}
                  fallback={players}
                  disabled={left[i] === shooter}
                  onChange={(id) => {
                    const next = [...right];
                    next[i] = id;
                    setRight(next);
                  }}
                />
              ))}
              <p className="compare__note">
                Change one slot to ask what that swap does. The shooter has to stay on both floors —
                swapping him would compare two players rather than two lineups.
              </p>
              <div className="compare__swap">
                {left.map((id, i) =>
                  id === shooter ? null : (
                    <button key={id} type="button" onClick={() => swapOne(i)}>
                      reset slot {i + 1}
                    </button>
                  )
                )}
              </div>
            </>
          )}

          <div className="compare__row">
            <button
              type="button"
              className="compare__go"
              onClick={compare}
              disabled={busy || blocked}
            >
              {busy ? "Comparing…" : "Compare"}
            </button>
          </div>
          {duplicatedLeft && <p className="compare__warn">A lineup cannot repeat a player.</p>}
          {duplicatedRight && (
            <p className="compare__warn">The comparison lineup repeats a player.</p>
          )}
          {shooterMissing && (
            <p className="compare__warn">
              The shooter has to appear in both fives. Pick a player who stays on the floor.
            </p>
          )}
        </div>

        <div>
          {failed && <p className="compare__warn">{failed}</p>}

          {refusal && (
            <div className="compare__refusal">
              <h3>{refusal.title ?? "Refused"}</h3>
              <p>{refusal.detail}</p>
              {refusal.what_would_help && (
                <p>
                  <strong>What would help:</strong> {refusal.what_would_help}
                </p>
              )}
              {refusal.players && refusal.players.length > 0 && (
                <ul className="compare__meta">
                  {refusal.players.map((p) => (
                    <li key={p.player_id}>
                      {p.name ?? p.player_id} — {p.attempts.toLocaleString()} attempts
                    </li>
                  ))}
                </ul>
              )}
              {refusal.n_possessions !== undefined && (
                <p className="compare__meta">
                  {refusal.n_possessions.toLocaleString()} possessions against a floor of{" "}
                  {refusal.threshold?.toLocaleString()}
                </p>
              )}
            </div>
          )}

          {!result && !refusal && !failed && (
            <p className="compare__empty">
              Pick a five and press Compare. The default is Denver&rsquo;s starters against a
              league-average floor.
            </p>
          )}

          {result && (
            <>
              <Verdict result={result} />

              <div className="compare__section">
                <CourtHeatmap
                  shapes={shapes}
                  viewBox={viewBox}
                  values={values}
                  metric={metric}
                  domain={DOMAIN}
                  leagueValue={0}
                  minZoneAttempts={0}
                  caption={
                    (result.data.reference === "league_average"
                      ? "Change in the share of this shooter's attempts, against the same shooter with every lineup term at the league mean. "
                      : "Change in the share of this shooter's attempts, your five against the comparison five. ") +
                    (result.data.omnibus.distinguishable
                      ? "Red is more, blue is fewer; the scale is fixed at one percentage point each way, so two comparisons are directly comparable."
                      : "Every zone is hatched because the two lineups did not separate at all — the colours show which way each zone leaned, not an effect the evidence supports.")
                  }
                />
              </div>

              <DeltaTable zones={result.data.zones} labels={zoneLabels} reportable={reportable} />
              <VarianceSplit result={result} />
              <MechanismTable mechanism={result.data.mechanism} />

              {result.meta.warnings.map((warning) => (
                <p className="compare__warn" key={warning}>
                  {warning}
                </p>
              ))}
              <p className="compare__meta">
                tier: {result.meta.support.tier} · possessions together:{" "}
                {result.meta.support.lineup_possessions.toLocaleString()}
                {result.data.swap &&
                  ` · swapped ${byId.get(Number(result.data.swap.out))?.name ?? result.data.swap.out} for ${
                    byId.get(Number(result.data.swap.in))?.name ?? result.data.swap.in
                  }`}
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

/**
 * The headline, and the only thing that should be read first.
 *
 * A Wald test on the two parameters a lineup has — how far it pulls this
 * shooter toward the rim, and how far toward the arc. Nine per-zone tests at
 * 80% would separate something on most comparisons; this one number is what
 * says whether there is anything to separate.
 */
function Verdict({ result }: { result: CompareResponse }): React.ReactElement {
  const { omnibus } = result.data;
  const moved = omnibus.distinguishable;
  return (
    <div className="compare__verdict" data-moved={moved}>
      <p className="compare__headline">
        {omnibus.degenerate
          ? "These two lineups predict exactly the same shot mix."
          : moved
            ? "The shot mix moves by more than the evidence can explain away."
            : "The shot mix does not move by more than the evidence can explain away."}
      </p>
      <p className="compare__note">
        {omnibus.degenerate
          ? "There is nothing to test — the difference is exactly zero, with exactly zero uncertainty."
          : `Wald χ² = ${omnibus.statistic.toFixed(2)} on ${omnibus.degrees_of_freedom} degrees of freedom, against a pre-registered critical value of ${omnibus.critical_value.toFixed(2)}. Two, not eight, because every lineup term multiplies either the rim indicator or the three indicator — a lineup has exactly two ways to move a shooter.`}
      </p>
      {!omnibus.degenerate && (
        <div className="compare__stats">
          <span className="compare__stat">
            Pull toward the rim
            <b>
              {omnibus.rim_shift >= 0 ? "+" : "−"}
              {Math.abs(omnibus.rim_shift).toFixed(3)} ± {omnibus.rim_shift_error.toFixed(3)}
            </b>
          </span>
          <span className="compare__stat">
            Pull toward the arc
            <b>
              {omnibus.three_shift >= 0 ? "+" : "−"}
              {Math.abs(omnibus.three_shift).toFixed(3)} ± {omnibus.three_shift_error.toFixed(3)}
            </b>
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * Per-zone deltas with intervals, as a table rather than a forest plot.
 *
 * The plot would be the better form if the numbers were independent, and they
 * are not: nine shares on a simplex, moved by two parameters. A row of nine
 * interval bars invites exactly the eyeball comparison between zones that the
 * ranking endpoint exists to refuse, so the ordering here is the fixed zone
 * vocabulary and the intervals are read one at a time.
 */
function DeltaTable({
  zones,
  labels,
  reportable,
}: {
  zones: ZoneDelta[];
  labels: Record<string, string>;
  reportable: boolean;
}): React.ReactElement {
  return (
    <div className="compare__section">
      <h3>Where the attempts move</h3>
      <div className="compare__scroll">
        <table className="mech">
          <thead>
            <tr>
              <th scope="col">Zone</th>
              <th scope="col" style={{ textAlign: "right" }}>
                Change in share
              </th>
              <th scope="col" style={{ textAlign: "right" }}>
                {reportable ? "Points per 100" : "Direction"}
              </th>
              <th scope="col" style={{ textAlign: "right" }}>
                {reportable ? "80% interval" : ""}
              </th>
            </tr>
          </thead>
          <tbody>
            {zones.map((zone) => (
              <tr key={zone.zone_id}>
                <td>{labels[zone.zone_id] ?? zone.zone_id}</td>
                <td className="num" style={{ color: zone.delta_share >= 0 ? GAIN : LOSS }}>
                  {zone.delta_share >= 0 ? "+" : "−"}
                  {(Math.abs(zone.delta_share) * 100).toFixed(2)} pp
                </td>
                <td className="num">
                  {zone.points_per_100 === null
                    ? zone.points_direction === "flat"
                      ? "no shift"
                      : zone.points_direction
                    : `${zone.points_per_100 >= 0 ? "+" : "−"}${Math.abs(zone.points_per_100).toFixed(3)}`}
                </td>
                <td className="num">
                  {zone.interval === null
                    ? ""
                    : `[${zone.interval[0].toFixed(3)}, ${zone.interval[1].toFixed(3)}]`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!reportable && (
        <p className="compare__note">
          These five have not played enough possessions together for a priced magnitude, so the
          points column carries direction only. The share column is the model&rsquo;s own output and
          survives — it is what <code>/api/lineups/score</code> publishes at this tier too.
        </p>
      )}
    </div>
  );
}

/**
 * What the interval is made of.
 *
 * The single most useful thing on this page. A comparison is driven by the
 * difference between two players' shooting rates, and the coefficient
 * covariance says nothing about those — so an interval built from the model
 * alone would be technically correct about the wrong quantity. Typically the
 * larger share of the width is the players, not the model, and a reader who was
 * not told that would assume the opposite.
 */
function VarianceSplit({ result }: { result: CompareResponse }): React.ReactElement {
  const share = result.meta.comparison.profile_variance_share;
  const profiles = Math.round(share * 100);
  const model = 100 - profiles;
  return (
    <div className="compare__section">
      <h3>What the uncertainty is made of</h3>
      <div className="split" role="img" aria-label={`${model}% model, ${profiles}% player rates`}>
        <span style={{ width: `${model}%`, background: LOSS }} />
        <span style={{ width: `${profiles}%`, background: GAIN }} />
      </div>
      <div className="split__key">
        <span>
          <i style={{ background: LOSS }} />
          {model}% the fitted model
        </span>
        <span>
          <i style={{ background: GAIN }} />
          {profiles}% how well these players&rsquo; rates are known
        </span>
      </div>
      <p className="compare__note">
        671,251 attempts is overwhelming evidence about twenty coefficients and almost none about
        any particular pair of players. The second bar is the part a model-only interval would have
        left out.
      </p>
    </div>
  );
}

/** Which of the five lineup features the swap moved, and what was predicted of each. */
function MechanismTable({ mechanism }: { mechanism: Mechanism[] }): React.ReactElement {
  return (
    <div className="compare__section">
      <h3>Why — the terms this swap moved</h3>
      <div className="compare__scroll">
        <table className="mech">
          <thead>
            <tr>
              <th scope="col">Lineup feature</th>
              <th scope="col" style={{ textAlign: "right" }}>
                Change
              </th>
              <th scope="col" style={{ textAlign: "right" }}>
                Coefficient
              </th>
              <th scope="col">Pre-registered sign</th>
            </tr>
          </thead>
          <tbody>
            {mechanism.map((term) => (
              <tr key={term.term} data-moved={term.feature_delta !== 0}>
                <td>{TERM_LABEL[term.term] ?? term.term}</td>
                <td className="num">
                  {term.feature_delta === 0
                    ? "—"
                    : `${term.feature_delta >= 0 ? "+" : "−"}${Math.abs(term.feature_delta).toFixed(4)}`}
                </td>
                <td className="num">{term.coefficient.toFixed(4)}</td>
                <td>
                  <span className="verdict" data-verdict={term.verdict}>
                    {term.expected_sign === null
                      ? "none stated"
                      : term.expected_sign > 0
                        ? "positive"
                        : "negative"}{" "}
                    · {term.verdict}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="compare__note">
        Every coefficient&rsquo;s expected direction was written into the source before the model
        was fitted. <strong>Mean teammate spacing came back contradicted</strong> — more shooting
        around a player moves him <em>toward</em> the rim, not the arc, at −0.474 with a standard
        error of 0.045. The reading is shot-mix substitution: put four shooters on the floor and
        somebody has to attack inside. The prediction stays in the source next to the coefficient
        that refutes it.
      </p>
    </div>
  );
}
