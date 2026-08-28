"use client";

/**
 * The court heatmap.
 *
 * Four decisions, in the order they were made:
 *
 * **The fill is diverging, not sequential.** The quantity is expected points per
 * attempt *minus the league average* — a polarity, so it gets two hues either
 * side of a neutral midpoint. Filling raw expected points with a light-to-dark
 * ramp would be the obvious mistake: restricted-area value dwarfs corner-three
 * value, so the ramp would simply redraw the arc and say nothing.
 *
 * **The geometry is not written here.** Every zone's SVG path comes from
 * `zones.json`, generated in Python from the same constants as `derive_zone`. A
 * test walks a dense grid asserting that every point inside an outline is a point
 * the model puts in that zone. Restating the arc in TypeScript is how a chart
 * ends up colouring a region the model never scored.
 *
 * **Uncertainty lives in the mark.** A zone below the attempt floor renders
 * hatched, with a dashed edge and no value — not coloured with a caveat in a
 * tooltip nobody opens. Colour claims a magnitude; a hatch declines to.
 *
 * **Every zone labels its own n.** A shot chart without sample sizes invites the
 * exact small-sample reading this project exists to avoid.
 *
 * Palette: the diverging blue↔red pair, four validated steps per arm. Both arms
 * pass the ordinal checks in both modes (monotone lightness, adjacent ΔL ≥ 0.06,
 * light end clearing 2:1 against the surface, single hue), and the poles separate
 * at CVD ΔE 17.7 light / 13.2 dark against a target of 8.
 */

import { useId, useMemo, useState } from "react";

export type ZoneShape = {
  id: string;
  label: string;
  path: string;
  labelAt: { x: number; y: number };
};

function formatSigned(value: number, digits = 3): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

export type ZoneValue = {
  attempts: number;
  fg: number;
  points_per_attempt: number;
  deviation: number;
  below_floor: boolean;
};

/**
 * What the colours mean.
 *
 * The court is a diverging encoding of *some* signed quantity, and it was
 * originally hard-wired to one: expected points per attempt. Serving the
 * selection model needs a second — the share of a shooter's attempts a lineup
 * moves into each zone — and those are not interchangeable. They differ in
 * units, in the size of an interesting effect (0.2 points versus 0.003 of a
 * share), and in what the sample-size floor is counting.
 *
 * Rather than let the component keep saying "points per attempt" over a chart
 * of something else, the vocabulary is a parameter. Everything a reader sees —
 * the in-zone number, the hover sentence, the table headers, the floor
 * explanation — comes from here.
 */
export type MetricSpec = {
  /** Fills "N vs league" and the table's value column. */
  valueLabel: string;
  /** The unit being counted for the support floor, singular. */
  countNoun: string;
  /** In-zone label and table cell. Deliberately short: it sits on the court. */
  formatDeviation: (deviation: number) => string;
  /** The precise value, for the hover readout and the table. */
  formatValue: (value: ZoneValue) => string;
  /** One clause naming the secondary quantity, or null to omit it. */
  formatSecondary: (value: ZoneValue) => string | null;
  /** Header and cell for the table's secondary column, which must stay narrow. */
  secondaryLabel: string;
  formatSecondaryCell: (value: ZoneValue) => string;
  /** Header for the count column. */
  countLabel: string;
  /**
   * The two ends of the diverging legend.
   *
   * Hard-coded as "Below league" / "Above league" until the court was reused
   * for a comparison between two *lineups*, where neither end is the league at
   * all -- the legend said one thing and the caption underneath said another.
   * A legend that contradicts its own caption is worse than no legend.
   */
  lowLabel: string;
  highLabel: string;
  /**
   * The parenthetical in the hover readout: what this zone's value is being
   * compared *to*.
   *
   * A function rather than a noun, because the honest phrasing differs in kind
   * and not only in wording. The league surface has a real midpoint worth
   * printing ("vs league 1.088"); a comparison between two chosen lineups has a
   * midpoint of exactly zero, and "vs league 0.000" over it would be claiming a
   * comparison that is not happening.
   */
  formatComparison: (value: ZoneValue, leagueValue: number) => string;
  /**
   * The in-zone sample-size label, or `null` to print none.
   *
   * Not every metric has a per-zone sample size, and printing one that does not
   * exist is worse than printing nothing. The league surface genuinely measures
   * each zone separately, so `n=6,176` under a zone is a fact about that zone.
   * The selection surface does not: its value is a model prediction, and the
   * only count in scope is the *lineup's* possessions, which is the same number
   * in all nine zones — and is `0` for a counterfactual five that has never
   * played together, which is the normal case for this product.
   *
   * Rendering `n=0` beneath a predicted `+0.09` says the prediction came from
   * nothing. It came from 671,251 attempts fitting twenty parameters. The
   * lineup's own possession count belongs in the caption, where it is a
   * statement about support rather than about sample size, and where it already
   * appears.
   */
  formatCount: (value: ZoneValue) => string | null;
};

/** Expected points per attempt, the original and still the default. */
export const POINTS_PER_ATTEMPT: MetricSpec = {
  valueLabel: "Exp. pts/attempt",
  countNoun: "attempt",
  formatDeviation: (d) => formatSigned(d, 2),
  formatValue: (v) => v.points_per_attempt.toFixed(3),
  formatSecondary: (v) => `${(v.fg * 100).toFixed(1)}% on ${v.attempts.toLocaleString()} attempts`,
  secondaryLabel: "FG%",
  formatSecondaryCell: (v) => `${(v.fg * 100).toFixed(1)}%`,
  countLabel: "Attempts",
  formatCount: (v) => `n=${v.attempts.toLocaleString()}`,
  lowLabel: "Below league",
  highLabel: "Above league",
  formatComparison: (v, league) => `${formatSigned(v.deviation)} vs league ${league.toFixed(3)}`,
};

/**
 * Share of a shooter's attempts, as the selection model predicts it.
 *
 * Three decimals of a percentage point, because that is the real size of the
 * effect. Rounding it to one decimal would show ±0.0 everywhere and rescaling
 * it would be a lie about magnitude.
 */
export const ATTEMPT_SHARE: MetricSpec = {
  valueLabel: "Share",
  countNoun: "possession",
  formatDeviation: (d) => `${d >= 0 ? "+" : "−"}${(Math.abs(d) * 100).toFixed(2)}`,
  // NaN is how the caller says "the API nulled this". Reconstructing it from the
  // baseline plus the delta would put a number on screen that the support
  // contract refused to serve, which is the one thing none of this may do.
  formatValue: (v) =>
    Number.isFinite(v.points_per_attempt)
      ? `${(v.points_per_attempt * 100).toFixed(2)}%`
      : "not reportable",
  formatSecondary: (v) => `${(v.fg * 100).toFixed(2)}% with a league-average lineup on the floor`,
  secondaryLabel: "Baseline",
  formatSecondaryCell: (v) => `${(v.fg * 100).toFixed(2)}%`,
  countLabel: "Possessions",
  // No per-zone count exists here. See `formatCount` on MetricSpec.
  formatCount: () => null,
  lowLabel: "Fewer attempts",
  highLabel: "More attempts",
  formatComparison: (v) =>
    `${v.deviation >= 0 ? "+" : "−"}${(Math.abs(v.deviation) * 100).toFixed(2)} pp vs a league-average lineup`,
};

export type CourtHeatmapProps = {
  viewBox: string;
  shapes: ZoneShape[];
  values: Record<string, ZoneValue | undefined>;
  /** Expected points per attempt at the midpoint of the scale. */
  leagueValue: number;
  minZoneAttempts: number;
  caption: string;
  /** Largest absolute deviation the scale should reach. Shared across courts so
   *  two charts side by side are comparable rather than each self-normalised. */
  domain?: number;
  /** What the numbers mean. Defaults to expected points per attempt. */
  metric?: MetricSpec;
};

/**
 * Four steps per arm. Blue is below the league average, red above — red for
 * "hot" is the convention every shot chart uses, and both are documented poles
 * of the same diverging pair.
 */
const ARMS = {
  below: ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"],
  above: ["#ea9a93", "#dd716a", "#c74845", "#892b2a"],
} as const;

const ARMS_DARK = {
  below: ["#9ec5f4", "#5598e7", "#2a78d6", "#184f95"],
  above: ["#f1aea8", "#dd716a", "#c74845", "#892b2a"],
} as const;

/** Bucket a signed deviation onto the diverging scale. Index 0 is the midpoint. */
function step(
  deviation: number,
  domain: number
): { arm: "below" | "above" | "mid"; index: number } {
  const magnitude = Math.abs(deviation);
  if (domain <= 0 || magnitude < domain * 0.08) return { arm: "mid", index: 0 };
  // Four bands per arm, so the midpoint band is genuinely "about average"
  // rather than a hue standing in for zero.
  const index = Math.min(3, Math.floor((magnitude / domain) * 4));
  return { arm: deviation > 0 ? "above" : "below", index };
}

export function CourtHeatmap({
  viewBox,
  shapes,
  values,
  leagueValue,
  minZoneAttempts,
  caption,
  domain,
  metric = POINTS_PER_ATTEMPT,
}: CourtHeatmapProps) {
  const patternId = useId();
  const [hovered, setHovered] = useState<string | null>(null);
  const [showTable, setShowTable] = useState(false);

  // Whether this metric prints a per-zone count at all. The hint promised one
  // unconditionally, over a court whose spec returns null for every zone.
  const promisesCounts = shapes.some((shape) => {
    const value = values[shape.id];
    return value !== undefined && metric.formatCount(value) !== null;
  });

  const scaleDomain = useMemo(() => {
    if (domain && domain > 0) return domain;
    const magnitudes = Object.values(values)
      .filter((v): v is ZoneValue => v !== undefined && !v.below_floor)
      .map((v: ZoneValue) => Math.abs(v.deviation));
    return magnitudes.length ? Math.max(...magnitudes) : 1;
  }, [values, domain]);

  const active = hovered ? values[hovered] : undefined;
  const activeShape = shapes.find((s) => s.id === hovered);
  const anyBelowFloor = shapes.some((shape) => {
    const value = values[shape.id];
    return !value || value.below_floor;
  });

  return (
    <figure className="court">
      <style>{`
        .court {
          margin: 0;
          --court-surface: #fcfcfb;
          --court-mid: #f0efec;
          --court-line: #c3c2b7;
          --court-ink: #0b0b0b;
          --court-muted: #52514e;
          --court-hatch: #898781;
        }
        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) .court {
            --court-surface: #1a1a19;
            --court-mid: #383835;
            --court-line: #383835;
            --court-ink: #ffffff;
            --court-muted: #c3c2b7;
            --court-hatch: #898781;
          }
        }
        :root[data-theme="dark"] .court {
          --court-surface: #1a1a19;
          --court-mid: #383835;
          --court-line: #383835;
          --court-ink: #ffffff;
          --court-muted: #c3c2b7;
          --court-hatch: #898781;
        }
        .court svg { display: block; width: 100%; height: auto; background: var(--court-surface); border-radius: var(--radius); }
        .court__zone { stroke: var(--court-surface); stroke-width: 2; transition: opacity 120ms ease; cursor: default; }
        .court__zone--dim { opacity: 0.45; }
        .court__zone--unsupported { stroke: var(--court-muted); stroke-width: 2; stroke-dasharray: 6 4; }
        .court__line { fill: none; stroke: var(--court-line); stroke-width: 2; }
        .court__label { font: 600 13px var(--mono, monospace); fill: var(--court-ink); text-anchor: middle; pointer-events: none; }
        .court__sub { font: 400 11px var(--mono, monospace); fill: var(--court-muted); text-anchor: middle; pointer-events: none; }
        .court figcaption { color: var(--muted); font-size: 0.85rem; margin-top: 0.6rem; }
        .court__legend { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem; font-size: 0.78rem; color: var(--muted); }
        .court__swatches { display: flex; }
        .court__swatch { width: 22px; height: 12px; }
        .court__readout { min-height: 3.2rem; margin-top: 0.5rem; font-size: 0.85rem; color: var(--text); }
        .court__tablebtn { background: none; border: 1px solid var(--border); color: var(--muted); border-radius: 6px; padding: 0.2rem 0.5rem; font-size: 0.75rem; cursor: pointer; }
        .court table { border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-top: 0.6rem; }
        .court th, .court td { text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border); }
        .court td.num { text-align: right; font-family: var(--mono, monospace); }
      `}</style>

      <svg viewBox={viewBox} role="img" aria-label={caption}>
        <defs>
          {/* The accessibility channel: a 45-degree hatch for "not enough
              evidence to colour". Tone-on-tone so it reads as absence rather
              than as another category. */}
          <pattern
            id={`${patternId}-hatch`}
            width="8"
            height="8"
            patternTransform="rotate(45)"
            patternUnits="userSpaceOnUse"
          >
            <rect width="8" height="8" fill="var(--court-surface)" />
            <line x1="0" y1="0" x2="0" y2="8" stroke="var(--court-hatch)" strokeWidth="2" />
          </pattern>
        </defs>

        {shapes.map((shape) => {
          const value = values[shape.id];
          const unsupported = !value || value.below_floor;
          const bucket = value
            ? step(value.deviation, scaleDomain)
            : { arm: "mid" as const, index: 0 };
          const fill = unsupported
            ? `url(#${patternId}-hatch)`
            : bucket.arm === "mid"
              ? "var(--court-mid)"
              : `var(--court-${bucket.arm}-${bucket.index})`;

          return (
            <path
              key={shape.id}
              d={shape.path}
              fillRule="evenodd"
              className={[
                "court__zone",
                unsupported ? "court__zone--unsupported" : "",
                hovered && hovered !== shape.id ? "court__zone--dim" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              style={{ fill }}
              onMouseEnter={() => setHovered(shape.id)}
              onMouseLeave={() => setHovered(null)}
              onFocus={() => setHovered(shape.id)}
              onBlur={() => setHovered(null)}
              tabIndex={0}
              aria-label={
                unsupported
                  ? `${shape.label}: below the ${minZoneAttempts}-${metric.countNoun} floor, no estimate`
                  : `${shape.label}: ${formatSigned(value.deviation)} versus league, ${value.attempts} ${metric.countNoun}s`
              }
            />
          );
        })}

        {/* Court furniture, drawn after the fills so it reads on top. */}
        <g className="court__line">
          <line x1="-250" y1="422.5" x2="250" y2="422.5" />
          <rect x="-80" y="185" width="160" height="190" />
          <circle cx="0" cy="375" r="40" />
        </g>

        {/* Every zone labels its own n — where a per-zone n exists. */}
        {shapes.map((shape) => {
          const value = values[shape.id];
          const unsupported = !value || value.below_floor;
          const count = value ? metric.formatCount(value) : null;
          return (
            <g key={`${shape.id}-label`}>
              <text className="court__label" x={shape.labelAt.x} y={shape.labelAt.y}>
                {unsupported ? "—" : metric.formatDeviation(value.deviation)}
              </text>
              {count !== null && (
                <text className="court__sub" x={shape.labelAt.x} y={shape.labelAt.y + 15}>
                  {count}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <div className="court__legend">
        <span>{metric.lowLabel}</span>
        <span className="court__swatches">
          {[...ARMS.below].reverse().map((_, i) => (
            <span
              key={`b${i}`}
              className="court__swatch"
              style={{ background: `var(--court-below-${3 - i})` }}
            />
          ))}
          <span className="court__swatch" style={{ background: "var(--court-mid)" }} />
          {ARMS.above.map((_, i) => (
            <span
              key={`a${i}`}
              className="court__swatch"
              style={{ background: `var(--court-above-${i})` }}
            />
          ))}
        </span>
        <span>{metric.highLabel}</span>
        {anyBelowFloor && (
          // Only when the hatch can actually fire. A legend entry for a mark
          // that never appears on the chart beside it is decoration, and it
          // teaches a reader to expect a distinction the chart is not making.
          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
            <svg width="22" height="12" aria-hidden>
              <rect width="22" height="12" fill={`url(#${patternId}-hatch)`} />
            </svg>
            below the {minZoneAttempts}-{metric.countNoun} floor — no estimate
          </span>
        )}
        <button
          type="button"
          className="court__tablebtn"
          onClick={() => setShowTable((v) => !v)}
          aria-expanded={showTable}
        >
          {showTable ? "Hide table" : "Show table"}
        </button>
      </div>

      <div className="court__readout" aria-live="polite">
        {active && activeShape ? (
          active.below_floor ? (
            <>
              <strong>{activeShape.label}</strong> — {active.attempts.toLocaleString()}{" "}
              {metric.countNoun}s, below the {minZoneAttempts}-{metric.countNoun} floor. No estimate
              is shown, and that is the answer rather than a missing value.
            </>
          ) : (
            <>
              <strong>{activeShape.label}</strong> — {metric.formatValue(active)} (
              {metric.formatComparison(active, leagueValue)})
              {metric.formatSecondary(active) ? `, ${metric.formatSecondary(active)}` : ""}.
            </>
          )
        ) : (
          <span style={{ color: "var(--muted)" }}>
            Hover or tab through a zone for its value
            {promisesCounts ? " and sample size" : ""}.
          </span>
        )}
      </div>

      {showTable && (
        <table>
          <caption className="sr-only">{caption}</caption>
          <thead>
            <tr>
              <th scope="col">Zone</th>
              <th scope="col">{metric.countLabel}</th>
              <th scope="col">{metric.secondaryLabel}</th>
              <th scope="col">{metric.valueLabel}</th>
              <th scope="col">vs league</th>
            </tr>
          </thead>
          <tbody>
            {shapes.map((shape) => {
              const value = values[shape.id];
              return (
                <tr key={`${shape.id}-row`}>
                  <th scope="row">{shape.label}</th>
                  <td className="num">{value ? value.attempts.toLocaleString() : "0"}</td>
                  <td className="num">{value ? (metric.formatSecondary(value) ?? "—") : "—"}</td>
                  <td className="num">
                    {value && !value.below_floor ? metric.formatValue(value) : "—"}
                  </td>
                  <td className="num">
                    {value && !value.below_floor ? formatSigned(value.deviation) : "no estimate"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      <figcaption>{caption}</figcaption>

      {/* Palette tokens. Declared here so the light/dark swap happens in one
          place and the marks above are written against roles. */}
      <style>{`
        .court {
          ${ARMS.below.map((hex, i) => `--court-below-${i}: ${hex};`).join("\n          ")}
          ${ARMS.above.map((hex, i) => `--court-above-${i}: ${hex};`).join("\n          ")}
        }
        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) .court {
            ${ARMS_DARK.below.map((hex, i) => `--court-below-${i}: ${hex};`).join("\n            ")}
            ${ARMS_DARK.above.map((hex, i) => `--court-above-${i}: ${hex};`).join("\n            ")}
          }
        }
        :root[data-theme="dark"] .court {
          ${ARMS_DARK.below.map((hex, i) => `--court-below-${i}: ${hex};`).join("\n          ")}
          ${ARMS_DARK.above.map((hex, i) => `--court-above-${i}: ${hex};`).join("\n          ")}
        }
      `}</style>
    </figure>
  );
}
