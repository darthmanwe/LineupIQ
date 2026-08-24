"use client";

import { useMemo, useState } from "react";

import {
  ATTEMPT_SHARE,
  CourtHeatmap,
  type ZoneShape,
  type ZoneValue,
} from "@/components/court/CourtHeatmap";

/**
 * The counterfactual, made clickable.
 *
 * Pick five players, pick which one is shooting, optionally pick five
 * defenders, and the served conditional logit returns the shot mix it predicts
 * — plus the same shooter's mix with every lineup term at the league average.
 * The court colours the *difference*, because that is the only quantity the
 * model claims and the absolute mix is dominated by who the shooter is.
 *
 * Three decisions worth naming, because each one is a place this component
 * could have flattered the model instead:
 *
 * 1. **The colour scale is fixed, not fitted to the response.** A domain that
 *    renormalised on every request would paint a 0.1-percentage-point shift
 *    the same crimson as a 5-point one, and every lineup would look decisive.
 *    The domain is one percentage point of share, stated in the caption.
 * 2. **A refusal renders as a refusal.** A 422 gets the shortfall and what
 *    would help, not an empty court that reads as "no effect".
 * 3. **Warnings from the envelope are shown, not summarised.** The API decides
 *    what is uncertain about an answer; a UI that reworded that would
 *    eventually disagree with it.
 */

type Player = { id: number; name: string; attempts: number };

type ScoreZone = {
  zone_id: string;
  share: number | null;
  baseline_share: number;
  delta: number;
};

type ScoreResponse = {
  data: {
    lineup_hash: string;
    shooter: { player_id: string; name: string | null; known: boolean; evidence_weight: number };
    zones: ScoreZone[];
  };
  meta: {
    support: { tier: string; lineup_possessions: number; counterfactual: boolean };
    warnings: string[];
  };
};

type Problem = {
  title?: string;
  detail?: string;
  what_would_help?: string;
  n_possessions?: number;
  threshold?: number;
  shortfall_players?: string[];
};

/**
 * One percentage point of share, each way.
 *
 * Chosen from the model, not from the data being displayed: the fitted lineup
 * coefficients move a zone by fractions of a point, so a one-point domain puts
 * a realistic effect in the middle of the scale and leaves headroom for an
 * extreme lineup. It never changes, so two lineups scored one after the other
 * are directly comparable.
 */
const DOMAIN = 0.01;

export function LineupScorer({
  players,
  shapes,
  viewBox,
}: {
  players: Player[];
  shapes: ZoneShape[];
  viewBox: string;
}) {
  const [offense, setOffense] = useState<number[]>(() => players.slice(0, 5).map((p) => p.id));
  const [shooter, setShooter] = useState<number>(() => players[0]?.id ?? 0);
  const [defense, setDefense] = useState<number[]>(() => players.slice(5, 10).map((p) => p.id));
  const [useDefense, setUseDefense] = useState(true);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [refusal, setRefusal] = useState<Problem | null>(null);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const byId = useMemo(() => new Map(players.map((p) => [p.id, p])), [players]);

  const values: Record<string, ZoneValue | undefined> = useMemo(() => {
    if (!result) return {};
    const out: Record<string, ZoneValue> = {};
    for (const zone of result.data.zones) {
      out[zone.zone_id] = {
        // `attempts` drives the support hatch. The support decision is made per
        // lineup rather than per zone, so every zone carries the lineup's own
        // possession count and they hatch or fill together — which is the truth
        // about where the evidence lives.
        attempts: result.meta.support.lineup_possessions,
        fg: zone.baseline_share,
        points_per_attempt: zone.share ?? zone.baseline_share + zone.delta,
        deviation: zone.delta,
        below_floor: false,
      };
    }
    return out;
  }, [result]);

  async function score(): Promise<void> {
    setBusy(true);
    setFailed(null);
    setRefusal(null);
    try {
      const response = await fetch("/api/lineups/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shooter_id: shooter,
          offense,
          defense: useDefense ? defense : [],
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
      setResult((await response.json()) as ScoreResponse);
    } catch {
      // A static export served without the Worker behind it. Say so plainly
      // rather than leaving a spinner running.
      setFailed(
        "Could not reach the API. This page needs the Worker running behind it — " +
          "`npx wrangler dev` from apps/api, or the deployed origin."
      );
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const slot = (
    ids: number[],
    setIds: (next: number[]) => void,
    index: number,
    label: string
  ): React.ReactElement => (
    <label key={`${label}-${index}`} className="scorer__slot">
      <span className="scorer__slotlabel">{label}</span>
      <select
        value={ids[index] ?? 0}
        onChange={(e) => {
          const next = [...ids];
          next[index] = Number(e.target.value);
          setIds(next);
          // Keep the shooter on the floor. Silently scoring a shooter who was
          // just substituted out would answer a different question than the
          // one on screen.
          if (label === "Offence" && !next.includes(shooter)) setShooter(next[0] ?? 0);
        }}
      >
        {players.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} ({p.attempts.toLocaleString()})
          </option>
        ))}
      </select>
    </label>
  );

  const duplicated = new Set(offense).size !== offense.length;

  return (
    <section className="scorer">
      <style>{`
        .scorer { margin-top: 2rem; }
        .scorer__grid { display: grid; gap: 1.5rem; }
        @media (min-width: 62rem) { .scorer__grid { grid-template-columns: 22rem 1fr; align-items: start; } }
        .scorer__panel { border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; }
        .scorer__panel h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 0.75rem; }
        .scorer__slot { display: block; margin-bottom: 0.5rem; }
        .scorer__slotlabel { display: block; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.15rem; }
        .scorer select { width: 100%; padding: 0.35rem 0.4rem; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--text); font-size: 0.85rem; }
        .scorer__row { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.75rem; flex-wrap: wrap; }
        .scorer button { border: 1px solid var(--border); background: var(--text); color: var(--bg); border-radius: 6px; padding: 0.45rem 0.9rem; font-size: 0.85rem; cursor: pointer; }
        .scorer button[disabled] { opacity: 0.5; cursor: not-allowed; }
        .scorer__toggle { display: flex; gap: 0.4rem; align-items: center; font-size: 0.82rem; color: var(--muted); }
        .scorer__warn { border-left: 3px solid var(--border); padding: 0.5rem 0 0.5rem 0.85rem; margin: 0.75rem 0 0; font-size: 0.85rem; color: var(--muted); }
        .scorer__refusal { border: 1px solid var(--border); border-left: 3px solid #c74845; border-radius: var(--radius); padding: 1rem; }
        .scorer__refusal h3 { color: var(--text); text-transform: none; letter-spacing: 0; font-size: 1rem; }
        .scorer__meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.75rem; font-family: var(--mono, monospace); }
        .scorer__empty { color: var(--muted); font-size: 0.9rem; }
      `}</style>

      <div className="scorer__grid">
        <div className="scorer__panel">
          <h3>The five on the floor</h3>
          {[0, 1, 2, 3, 4].map((i) => slot(offense, setOffense, i, "Offence"))}

          <label className="scorer__slot" style={{ marginTop: "0.75rem" }}>
            <span className="scorer__slotlabel">Taking the shot</span>
            <select value={shooter} onChange={(e) => setShooter(Number(e.target.value))}>
              {offense.map((id) => (
                <option key={id} value={id}>
                  {byId.get(id)?.name ?? id}
                </option>
              ))}
            </select>
          </label>

          <div className="scorer__row">
            <label className="scorer__toggle">
              <input
                type="checkbox"
                checked={useDefense}
                onChange={(e) => setUseDefense(e.target.checked)}
              />
              Score against five defenders
            </label>
          </div>
          {useDefense && [0, 1, 2, 3, 4].map((i) => slot(defense, setDefense, i, "Defence"))}

          <div className="scorer__row">
            <button type="button" onClick={score} disabled={busy || duplicated}>
              {busy ? "Scoring…" : "Score this lineup"}
            </button>
          </div>
          {duplicated && (
            <p className="scorer__warn">
              A lineup cannot repeat a player. Change a slot before scoring.
            </p>
          )}
        </div>

        <div>
          {failed && <p className="scorer__warn">{failed}</p>}

          {refusal && (
            <div className="scorer__refusal">
              <h3>{refusal.title ?? "Refused"}</h3>
              <p>{refusal.detail}</p>
              {refusal.what_would_help && (
                <p>
                  <strong>What would help:</strong> {refusal.what_would_help}
                </p>
              )}
              {refusal.n_possessions !== undefined && (
                <p className="scorer__meta">
                  {refusal.n_possessions.toLocaleString()} possessions against a floor of{" "}
                  {refusal.threshold?.toLocaleString()}
                </p>
              )}
            </div>
          )}

          {result && (
            <>
              <CourtHeatmap
                viewBox={viewBox}
                shapes={shapes}
                values={values}
                leagueValue={0}
                minZoneAttempts={0}
                domain={DOMAIN}
                metric={ATTEMPT_SHARE}
                caption={
                  `Percentage points of ${result.data.shooter.name ?? "this shooter"}'s attempts ` +
                  `that this five-man lineup moves into each zone, against the same shooter with ` +
                  `a league-average lineup. Scale is fixed at ±1.00 point so two lineups are ` +
                  `comparable; these effects are much smaller than that, which is the finding.`
                }
              />
              {result.meta.warnings.map((warning) => (
                <p className="scorer__warn" key={warning}>
                  {warning}
                </p>
              ))}
              <p className="scorer__meta">
                tier: {result.meta.support.tier} · possessions together:{" "}
                {result.meta.support.lineup_possessions.toLocaleString()} · shooter evidence weight:{" "}
                {result.data.shooter.evidence_weight.toFixed(3)} · lineup{" "}
                {result.data.lineup_hash.slice(0, 12)}…
              </p>
            </>
          )}

          {!result && !refusal && !failed && (
            <p className="scorer__empty">
              Pick five players and score them. The court will show the change in shot mix, not the
              mix itself — the absolute distribution is mostly a fact about the shooter, and the
              lineup effect is the part this model is actually claiming.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
