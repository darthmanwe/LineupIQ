"use client";

/**
 * The ranked plays, drawn so that a tie looks like a tie.
 *
 * This is a forest plot, and the form is the argument. A bar chart would encode
 * each contribution as a length from zero and say nothing about how well
 * determined it is, so the reader would compare bar lengths — which is exactly
 * the comparison the API spent a covariance matrix establishing they usually
 * cannot make. A dot with its interval puts the uncertainty in the mark, where
 * it cannot be skipped.
 *
 * Four decisions worth naming:
 *
 * 1. **Ties are drawn, not annotated.** Zones the model cannot separate share a
 *    rank, and they are bracketed into one visible group with a single rank
 *    number. A shared rank rendered as "3, 3" in a column would read as a
 *    formatting quirk; a bracket reads as a claim.
 * 2. **The zero line is the reference, and it is not a gridline.** Every
 *    contribution is signed against the league-average lineup, so zero is a
 *    meaningful value rather than an axis convenience. It gets a real rule and
 *    the colours diverge from it — the same two poles the court uses, so a zone
 *    that reads warm here reads warm there.
 * 3. **The axis is fitted to this response, and says so.** Unlike the court —
 *    where a fixed domain is what makes two lineups comparable — a forest plot
 *    with a fixed domain would flatten every interval to a hairline on the
 *    common case. The caption states the range rather than implying stability
 *    the axis does not have.
 * 4. **No magnitudes means no chart.** Below the reportable floor the API nulls
 *    every number and keeps the ranks, so this renders the bands as text. A
 *    chart drawn from nulls would be a chart of zeros.
 */

type Play = {
  zone_id: string;
  rank: number;
  points_per_100: number | null;
  points_direction: "gain" | "loss" | "flat";
  standard_error: number;
  interval: [number, number] | null;
  share: number | null;
  baseline_share: number;
};

export type PlayRankingResponse = {
  data: {
    ordered: boolean;
    confidence: number;
    plays: Play[];
    bands: string[][];
    excluded_zones: string[];
  };
  meta: {
    support: { tier: string };
    ranking: {
      pairs_compared: number;
      diagonal_would_refuse: number;
      ties_spanning_bands: number;
      critical_value: number;
    };
    warnings: string[];
  };
};

/**
 * The same two poles the court diverges between, deliberately unchanged.
 *
 * A gain is warm and a loss is cool in both views. Two different palettes for
 * the same signed quantity would be a second vocabulary for one fact, and a
 * reader moving between the court and this list would have to relearn it.
 */
const GAIN = "#c74845";
const LOSS = "#2a78d6";

const ROW = 30;
const LABEL_WIDTH = 300;
const VALUE_WIDTH = 156;
const PLOT_WIDTH = 230;
const RANK_WIDTH = 34;

/**
 * Zone names come from `zones.json`, the same file the court is drawn from and
 * the same file Python generates the geometry from.
 *
 * Not a local map of prettier strings. One vocabulary, three consumers: a
 * second copy here would drift, and the drift would be invisible — the chart
 * would simply call a zone something the court does not.
 */
function useZoneLabel(labels: Record<string, string>) {
  return (id: string): string => labels[id] ?? id.replace(/_/g, " ");
}

export function PlayRanking({
  ranking,
  labels,
}: {
  ranking: PlayRankingResponse;
  /** `zone_id` to display name, from `zones.json`. */
  labels: Record<string, string>;
}): React.ReactElement {
  const { plays, bands, ordered, confidence, excluded_zones: excluded } = ranking.data;
  const zoneLabel = useZoneLabel(labels);
  const level = `${Math.round(confidence * 100)}%`;

  // Every magnitude withheld: the tier is below reportable. The ranks survive
  // because they are a statement about the model's precision rather than about
  // this lineup's evidence, so they are what gets rendered.
  const withheld = plays.every((p) => p.points_per_100 === null);

  return (
    <section className="ranking">
      <style>{`
        .ranking { margin-top: 1.75rem; }
        .ranking h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 0.5rem; }
        .ranking__lead { font-size: 0.9rem; color: var(--muted); margin: 0 0 0.9rem; max-width: 46rem; }
        .ranking__scroll { overflow-x: auto; }
        .ranking__note { font-size: 0.82rem; color: var(--muted); margin: 0.85rem 0 0; max-width: 46rem; }
        .ranking__unordered { border: 1px solid var(--warn); background: var(--warn-soft); border-radius: var(--radius); padding: 0.75rem 0.9rem; margin: 0 0 1rem; font-size: 0.9rem; }
        .ranking__band { display: flex; gap: 0.6rem; align-items: baseline; padding: 0.35rem 0; border-top: 1px solid var(--border); }
        .ranking__band:first-child { border-top: 0; }
        .ranking__rank { font-family: var(--mono); font-size: 0.8rem; color: var(--muted); min-width: 2.5rem; }
        .ranking__zones { font-size: 0.92rem; }
        .ranking__tied { color: var(--muted); font-size: 0.82rem; }
      `}</style>

      <h3>Where the lineup moves his attempts</h3>

      {!ordered && (
        <p className="ranking__unordered">
          <strong>This is an unordered set, not a ranking.</strong> No two zones separate at the{" "}
          {level} level, so the order below carries no information. It is listed by point estimate
          because it has to be listed in some order.
        </p>
      )}

      <p className="ranking__lead">
        Each zone&rsquo;s share of the priced shift, in points per 100 attempts, against the same
        shooter with a league-average lineup around him. The interval is a {level} delta-method
        interval; zones bracketed together could not be separated from one another and share a rank.
      </p>

      {withheld ? (
        <BandList bands={bands} plays={plays} labels={labels} />
      ) : (
        <div className="ranking__scroll">
          <Forest plays={plays} bands={bands} level={level} labels={labels} />
        </div>
      )}

      <p className="ranking__note">
        {ranking.meta.ranking.diagonal_would_refuse > 0 ? (
          <>
            Of {ranking.meta.ranking.pairs_compared} pairs compared,{" "}
            <strong>{ranking.meta.ranking.diagonal_would_refuse}</strong> separate only because the
            test uses the covariance <em>between</em> two zones rather than comparing their
            intervals for overlap. Shares sum to one, so what one zone gains another loses and the
            difference is far better determined than either endpoint — which is why the intervals
            below can overlap while the ranks still hold.
          </>
        ) : (
          <>
            On this lineup every pair reached the same verdict either way, so comparing the
            intervals for overlap would have given the same ranking. That is not usually true — it
            is the reason the full covariance is shipped.
          </>
        )}
        {excluded.length > 0 && (
          <>
            {" "}
            {excluded.length === 1 ? "One zone is" : `${excluded.length} zones are`} below the
            volume floor and were not ranked: {excluded.map(zoneLabel).join("; ")}.
          </>
        )}
      </p>
    </section>
  );
}

/** The ranks with no numbers, for a lineup below the reportable floor. */
function BandList({
  bands,
  plays,
  labels,
}: {
  bands: string[][];
  plays: Play[];
  labels: Record<string, string>;
}): React.ReactElement {
  const zoneLabel = useZoneLabel(labels);
  const direction = new Map(plays.map((p) => [p.zone_id, p.points_direction]));
  return (
    <div>
      {bands.map((band, i) => (
        <div className="ranking__band" key={band.join("|")}>
          <span className="ranking__rank">#{i + 1}</span>
          <span className="ranking__zones">
            {band.map((zone) => zoneLabel(zone)).join(", ")}
            {band.length > 1 && <span className="ranking__tied"> — tied</span>}
          </span>
          <span className="ranking__tied">
            {band.map((zone) => direction.get(zone)).every((d) => d === "gain")
              ? "gain"
              : band.map((zone) => direction.get(zone)).every((d) => d === "loss")
                ? "loss"
                : "mixed"}
          </span>
        </div>
      ))}
      <p className="ranking__note">
        These five have not played enough together to size any of these. The ordering comes from the
        fitted model rather than from their possessions, so it survives; the magnitudes do not, and
        are not shown.
      </p>
    </div>
  );
}

function Forest({
  plays,
  bands,
  level,
  labels,
}: {
  plays: Play[];
  bands: string[][];
  level: string;
  labels: Record<string, string>;
}): React.ReactElement {
  const zoneLabel = useZoneLabel(labels);
  const lows = plays.map((p) => (p.interval ? p.interval[0] : (p.points_per_100 ?? 0)));
  const highs = plays.map((p) => (p.interval ? p.interval[1] : (p.points_per_100 ?? 0)));
  // Zero is always in the domain: it is the reference every value is signed
  // against, and a plot that cropped it would hide whether a zone crossed it.
  const min = Math.min(0, ...lows);
  const max = Math.max(0, ...highs);
  const pad = (max - min) * 0.08 || 0.1;
  const lo = min - pad;
  const hi = max + pad;

  const x = (value: number): number => LABEL_WIDTH + ((value - lo) / (hi - lo)) * PLOT_WIDTH;
  const height = plays.length * ROW + 34;
  const width = LABEL_WIDTH + PLOT_WIDTH + VALUE_WIDTH + RANK_WIDTH;
  const zero = x(0);

  const bandStart = new Map<number, number>();
  let cursor = 0;
  bands.forEach((band, i) => {
    bandStart.set(i, cursor);
    cursor += band.length;
  });

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      style={{ maxWidth: `${width}px`, display: "block" }}
      role="img"
      aria-label={`Priced contribution by zone with ${level} intervals`}
    >
      <style>{`
        .forest__zone { font: 12px var(--sans); fill: var(--text); }
        .forest__value { font: 11px var(--mono); fill: var(--muted); }
        .forest__rank { font: 10px var(--mono); fill: var(--muted); }
        .forest__axis { font: 10px var(--mono); fill: var(--muted); }
        .forest__rule { stroke: var(--border); stroke-width: 1; }
        .forest__zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
        .forest__bracket { stroke: var(--muted); stroke-width: 1; fill: none; }
      `}</style>

      {/* The zero rule, drawn before the marks so it sits behind them. It is
          not a gridline: it is the league-average lineup, which is the thing
          every value on this chart is a deviation from. */}
      <line className="forest__zero" x1={zero} y1={8} x2={zero} y2={plays.length * ROW + 6} />
      <text className="forest__axis" x={zero} y={plays.length * ROW + 20} textAnchor="middle">
        0
      </text>
      <text className="forest__axis" x={LABEL_WIDTH} y={plays.length * ROW + 20}>
        {lo.toFixed(2)}
      </text>
      <text
        className="forest__axis"
        x={LABEL_WIDTH + PLOT_WIDTH}
        y={plays.length * ROW + 20}
        textAnchor="end"
      >
        {hi.toFixed(2)}
      </text>
      <text className="forest__axis" x={LABEL_WIDTH} y={plays.length * ROW + 32}>
        points per 100 attempts
      </text>

      {bands.map((band, i) => {
        if (band.length < 2) return null;
        const start = bandStart.get(i) as number;
        const top = start * ROW + 6;
        const bottom = (start + band.length) * ROW - 4;
        // A bracket, not a tint. A tied group is a statement — "the data does
        // not order these" — and a background wash reads as decoration.
        return (
          <path
            key={`bracket-${i}`}
            className="forest__bracket"
            d={`M ${RANK_WIDTH - 4} ${top} h -6 V ${bottom} h 6`}
          />
        );
      })}

      {plays.map((play, i) => {
        const y = i * ROW + ROW / 2;
        const value = play.points_per_100 as number;
        const [low, high] = play.interval as [number, number];
        const colour = value >= 0 ? GAIN : LOSS;
        const bandIndex = bands.findIndex((band) => band.includes(play.zone_id));
        const firstOfBand = bandStart.get(bandIndex) === i;

        return (
          <g key={play.zone_id}>
            {firstOfBand && (
              <text className="forest__rank" x={4} y={y + 3}>
                #{play.rank}
              </text>
            )}
            {/* Left-aligned, growing away from the rank. Right-aligning it put
                a long zone name straight through the rank number. */}
            <text className="forest__zone" x={RANK_WIDTH + 8} y={y + 4}>
              {zoneLabel(play.zone_id)}
            </text>
            {/* 2px line, 8px marker, and a surface ring so an interval crossing
                the zero rule still reads as one mark. */}
            <line
              x1={x(low)}
              y1={y}
              x2={x(high)}
              y2={y}
              stroke={colour}
              strokeWidth={2}
              strokeLinecap="round"
            />
            <circle
              cx={x(value)}
              cy={y}
              r={4.5}
              fill={colour}
              stroke="var(--surface)"
              strokeWidth={1.5}
            />
            <text className="forest__value" x={LABEL_WIDTH + PLOT_WIDTH + 12} y={y + 4}>
              {value >= 0 ? "+" : ""}
              {value.toFixed(3)}
              <tspan dx={6} style={{ opacity: 0.75 }}>
                [{low.toFixed(3)}, {high.toFixed(3)}]
              </tspan>
            </text>
          </g>
        );
      })}
    </svg>
  );
}
