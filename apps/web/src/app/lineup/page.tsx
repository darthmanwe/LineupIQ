"use client";

import playersData from "../../../public/data/players.json";
import zonesData from "../../../public/data/zones.json";
import surfaceData from "../../../public/data/zone_surface.json";
import { CourtHeatmap, type ZoneShape, type ZoneValue } from "@/components/court/CourtHeatmap";
import { LineupScorer } from "@/components/LineupScorer";

/**
 * The Lineup page.
 *
 * What is live: the shot-value surface, at league scale and for two real
 * shooters, drawn on geometry generated from the same constants as the model.
 *
 * Then the counterfactual: pick any five, pick who is shooting, and the served
 * conditional logit returns how that lineup moves his shot mix. The two are
 * different quantities and the page keeps them apart — the surface is about how
 * valuable a zone is, the scorer is about which zone a player ends up in.
 *
 * The two example courts below show what the refusal rendering looks like
 * against real data. At league scale no zone is ever within three orders of
 * magnitude of the attempt floor, so a hatch that never fires would be
 * decoration.
 */

type Surface = {
  league_points_per_attempt: number;
  min_zone_attempts: number;
  zones: Record<string, ZoneValue>;
  examples: Record<
    string,
    { player_id: number; player_name: string; attempts: number; zones: Record<string, ZoneValue> }
  >;
};

const zones = zonesData as { viewBox: string; zones: ZoneShape[]; count: number };

/**
 * The pickable players, highest volume first.
 *
 * Capped at 240. Every id in `players.json` is a real player, but the tail is
 * two-minute call-ups whose fitted mix is almost entirely prior — putting them
 * at the top of a dropdown would invite a confident-looking answer about
 * somebody the model barely saw. The selector still shows each player's attempt
 * count, so the thinness is visible rather than implied.
 */
const PICKABLE = Object.entries(
  (playersData as { players: Record<string, { name: string; attempts: number }> }).players
)
  .map(([id, p]) => ({ id: Number(id), name: p.name, attempts: p.attempts }))
  .sort((a, b) => b.attempts - a.attempts || a.name.localeCompare(b.name))
  .slice(0, 240);
const surface = surfaceData as unknown as Surface;

/** One domain across every court, so two charts side by side are comparable. */
const DOMAIN = Math.max(...Object.values(surface.zones).map((z) => Math.abs(z.deviation)), 0.3);

export default function LineupPage() {
  const high = surface.examples.high_volume;
  const low = surface.examples.low_volume;

  return (
    <main className="page">
      <style>{`
        .page { max-width: 68rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
        .page h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
        .page .lede { color: var(--muted); max-width: 46rem; }
        .courts { display: grid; gap: 2rem; margin-top: 2rem; }
        @media (min-width: 60rem) { .courts--pair { grid-template-columns: 1fr 1fr; } }
        .note { border-left: 3px solid var(--border); padding: 0.6rem 0 0.6rem 1rem; color: var(--muted); font-size: 0.9rem; margin: 1.5rem 0; }
        .note strong { color: var(--text); }
        .findings { font-size: 0.92rem; }
        .findings li { margin-bottom: 0.5rem; }
      `}</style>

      <h1>Shot value by zone</h1>
      <p className="lede">
        Expected points per attempt, relative to the league average of{" "}
        {surface.league_points_per_attempt.toFixed(3)}. The fill is <strong>diverging</strong>{" "}
        because the quantity is a polarity — above or below average. Colouring raw expected points
        with a light-to-dark ramp would just redraw the arc.
      </p>

      <div className="courts">
        <CourtHeatmap
          viewBox={zones.viewBox}
          shapes={zones.zones}
          values={surface.zones}
          leagueValue={surface.league_points_per_attempt}
          minZoneAttempts={surface.min_zone_attempts}
          domain={DOMAIN}
          caption={`League-wide, three seasons. ${Object.values(surface.zones)
            .reduce((total, z) => total + z.attempts, 0)
            .toLocaleString()} attempts.`}
        />
      </div>

      <ul className="findings">
        <li>
          The <strong>restricted area is the most valuable shot on the floor</strong> at +0.239
          points per attempt over league, and <strong>mid-range is the worst</strong> at −0.25 to
          −0.27 — which is the modern consensus, computed here rather than quoted.
        </li>
        <li>
          Corner threes beat above-the-break threes (+0.065 to +0.078 against +0.011 for the wing
          and −0.040 for the top of the key). That gap is the whole reason spacing is discussed in
          terms of corners.
        </li>
        <li>
          Nothing in the pipeline was fitted to produce that ordering. It falls out of{" "}
          <code>made × shot_points</code> grouped by a zone taxonomy derived from coordinates.
        </li>
      </ul>

      <div className="note">
        <strong>Where the geometry comes from.</strong> Every zone&rsquo;s SVG path is generated in
        Python from the same constants as <code>derive_zone</code>, exported in{" "}
        <code>zones.json</code>, and only rendered here. A test walks a dense grid asserting that
        every point inside an outline is a point the model puts in that zone. Restating the arc in
        TypeScript is how a chart ends up colouring a region the model never scored.
      </div>

      <h2>What too little evidence looks like</h2>
      <p className="lede">
        Two real shooters, same scale. Uncertainty lives in the <strong>mark</strong> — a zone below
        the {surface.min_zone_attempts}-attempt floor renders hatched, with a dashed edge and no
        value. Not coloured with a caveat in a tooltip nobody opens: colour claims a magnitude, and
        a hatch declines to.
      </p>

      {high && low && (
        <div className="courts courts--pair">
          <CourtHeatmap
            viewBox={zones.viewBox}
            shapes={zones.zones}
            values={high.zones}
            leagueValue={surface.league_points_per_attempt}
            minZoneAttempts={surface.min_zone_attempts}
            domain={DOMAIN}
            caption={`${high.player_name} — ${high.attempts.toLocaleString()} attempts. Every zone clears the floor.`}
          />
          <CourtHeatmap
            viewBox={zones.viewBox}
            shapes={zones.zones}
            values={low.zones}
            leagueValue={surface.league_points_per_attempt}
            minZoneAttempts={surface.min_zone_attempts}
            domain={DOMAIN}
            caption={`${low.player_name} — ${low.attempts.toLocaleString()} attempts. Every zone is refused.`}
          />
        </div>
      )}

      <h2 style={{ fontSize: "1.3rem", marginTop: "3rem" }}>Score a five-man lineup</h2>
      <p className="lede">
        Not &ldquo;does this lineup make him shoot better&rdquo; — that model was built first and
        the answer was <strong>+0.02%</strong> log loss on unseen combinations, which is nothing.
        This is the question that has an answer: does the lineup change{" "}
        <em>which shots he takes</em>. Any five of {PICKABLE.length} players, including combinations
        that have never played a possession together.
      </p>
      <LineupScorer players={PICKABLE} shapes={zones.zones} viewBox={zones.viewBox} />

      <div className="note">
        <strong>Why the numbers are small.</strong> A team&rsquo;s attempts live on a simplex, so a
        lineup that moves a shooter toward threes has to move him away from something else — the
        deltas sum to zero by construction, and the court shows them at their real size. The fitted
        lineup effect is fractions of a percentage point. It is a real, measurable effect that
        survives a shuffled-lineup control, and it is small; both halves of that are the result.
      </div>

      <div className="note">
        <strong>What is still not built.</strong> The optimizer — search over lineups rather than
        scoring one you chose — and the trade simulator&rsquo;s served deltas.{" "}
        <code>POST /api/lineups/optimal-plays</code> and <code>POST /api/trades/simulate</code>{" "}
        return <code>501</code> naming what will back them.
      </div>
    </main>
  );
}
