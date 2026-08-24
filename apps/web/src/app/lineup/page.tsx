"use client";

import zonesData from "../../../public/data/zones.json";
import surfaceData from "../../../public/data/zone_surface.json";
import { CourtHeatmap, type ZoneShape, type ZoneValue } from "@/components/court/CourtHeatmap";

/**
 * The Lineup page.
 *
 * What is live: the shot-value surface, at league scale and for two real
 * shooters, drawn on geometry generated from the same constants as the model.
 *
 * What is not: picking five players and getting a surface for *them*. That needs
 * the served scorer, and this page says so rather than implying the colours are
 * lineup-specific. The two example courts are here to show what the refusal
 * rendering looks like against real data — at league scale no zone is ever
 * within three orders of magnitude of the attempt floor, so a hatch that never
 * fires would be decoration.
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

      <div className="note">
        <strong>Not built: the per-lineup surface.</strong> Picking five players and getting a
        surface for <em>them</em> needs the served closed-form scorer, and{" "}
        <code>POST /api/lineups/score</code> returns <code>501</code> naming what will back it. What
        is live is <code>POST /api/lineups/support</code>, which answers the prior question —
        whether these five have enough evidence for anything to be said at all. For 99% of five-man
        groups the answer is no.
      </div>
    </main>
  );
}
