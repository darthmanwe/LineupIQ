"use client";

import coverageData from "../../../public/data/coverage.json";
import evaluationData from "../../../public/data/evaluation.json";

/**
 * The Data Quality & Eval page.
 *
 * This is where the project publishes what it gets wrong, so the ordering is
 * deliberate: the underpowered verdict comes before the numbers it invalidates,
 * and the contradicted pre-registered sign comes before the ladder it sits in.
 * A reader who stops halfway down should have seen the bad news, not the good.
 */

type Gate = {
  name: string;
  measured: number;
  threshold: number;
  comparison: "min" | "max";
  verdict: "PASS" | "FAIL" | "WARN";
  severity: "blocking" | "reported";
  detail: string;
};

type Coverage = {
  gates: Gate[];
  n_gates: number;
  n_passing: number;
  coverage: { shots: number; possessions: number; seasons: number[] };
};

type LadderRow = { log_loss: number; top1_accuracy?: number; n: number };
type Evaluation = {
  available: string[];
  selection?: {
    metrics: Record<string, Record<string, LadderRow>>;
    sign_audit: Record<string, { value: number; expected_sign: number; verdict: string }>;
    controls: Record<string, number>;
  };
  rapm?: {
    reliability: Record<string, number>;
    co_occurrence: { n_flagged: number; n_players: number; ceiling: number };
  };
  trade?: Record<
    string,
    {
      power: { n: number; mde: number; residual_sd: number; verdict: string };
      real: Record<string, number | number[]>;
      placebo: Record<string, number | number[]>;
    }
  >;
};

const coverage = coverageData as unknown as Coverage;
const evaluation = evaluationData as unknown as Evaluation;

function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export default function QualityPage() {
  const trade = evaluation.trade?.inherit;
  const selection = evaluation.selection;
  const rapm = evaluation.rapm;
  const lolo = selection?.metrics?.leave_lineup_out;
  const disagreeing = Object.entries(selection?.sign_audit ?? {}).filter(
    ([, row]) => row.verdict !== "agrees"
  );

  return (
    <main className="page">
      <style>{`
        .page { max-width: 66rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
        .page h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
        .page h2 { font-size: 1.15rem; margin-top: 2.5rem; }
        .lede { color: var(--muted); max-width: 46rem; }
        table { border-collapse: collapse; width: 100%; font-size: 0.86rem; margin-top: 1rem; }
        th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); vertical-align: top; }
        td.num, th.num { text-align: right; font-family: var(--mono); white-space: nowrap; }
        .verdict { font-family: var(--mono); font-size: 0.78rem; padding: 0.05rem 0.35rem; border-radius: 4px; }
        .verdict--pass { color: #0ca30c; }
        .verdict--warn { color: #8a6d1f; }
        .verdict--fail { color: #d03b3b; font-weight: 600; }
        .callout { border: 1px solid var(--border); border-left: 3px solid var(--warn); background: var(--warn-soft); padding: 0.9rem 1rem; border-radius: var(--radius); margin: 1.25rem 0; }
        .callout--bad { border-left-color: #d03b3b; }
        .callout h3 { margin: 0 0 0.4rem; font-size: 0.95rem; }
        .callout p { margin: 0; font-size: 0.9rem; }
        .detail { color: var(--muted); font-size: 0.8rem; }
      `}</style>

      <h1>Data quality &amp; evaluation</h1>
      <p className="lede">
        Everything here is re-derived from committed gold, so no number on this page can disagree
        with the model that produced it. The bad news is listed first on purpose.
      </p>

      {trade && (
        <div className="callout callout--bad">
          <h3>The trade backtest is {trade.power.verdict}</h3>
          <p>
            {trade.power.n} evaluable mid-season moves against a team net-rating noise floor of{" "}
            {trade.power.residual_sd.toFixed(2)} points per 100 gives a minimum detectable effect of{" "}
            <strong>{trade.power.mde.toFixed(2)}</strong> — the same size as the effects the model
            projects. <strong>No accuracy claim follows.</strong> That verdict was computed and
            committed before the result, and the numbers below are published for the placebo
            comparison rather than as evidence the projection works.
          </p>
        </div>
      )}

      {disagreeing.length > 0 && (
        <div className="callout">
          <h3>
            {disagreeing.length} pre-registered sign{disagreeing.length === 1 ? "" : "s"}{" "}
            contradicted
          </h3>
          <p>
            {disagreeing.map(([name, row]) => (
              <span key={name}>
                <code>{name}</code> was written into the source as{" "}
                {row.expected_sign > 0 ? "positive" : "negative"} before the model was fitted. It
                fitted at <strong>{row.value.toFixed(3)}</strong>.{" "}
              </span>
            ))}
            It survives every robustness specification and collapses to{" "}
            {(selection?.controls?.shuffled_lineup_spacing_coefficient ?? 0).toFixed(3)} under the
            negative control, so the effect is real and the hypothesis was wrong. The expectation
            stays in the source as written.
          </p>
        </div>
      )}

      <h2>
        Quality gates — {coverage.n_passing} of {coverage.n_gates} passing
      </h2>
      <p className="lede">
        {coverage.coverage.shots.toLocaleString()} shot attempts and{" "}
        {coverage.coverage.possessions.toLocaleString()} possessions across{" "}
        {coverage.coverage.seasons.length} seasons. Four of these gates exist because of bugs that
        were silent when found.
      </p>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th scope="col">Gate</th>
              <th scope="col" className="num">
                Measured
              </th>
              <th scope="col" className="num">
                Threshold
              </th>
              <th scope="col">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {coverage.gates.map((gate) => (
              <tr key={gate.name}>
                <th scope="row">
                  <code>{gate.name}</code>
                  <div className="detail">{gate.detail}</div>
                </th>
                <td className="num">{percent(gate.measured)}</td>
                <td className="num">
                  {gate.comparison === "min" ? "≥" : "≤"} {percent(gate.threshold)}
                </td>
                <td>
                  <span className={`verdict verdict--${gate.verdict.toLowerCase()}`}>
                    {gate.verdict}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {lolo && (
        <>
          <h2>Shot selection — leave-lineup-out</h2>
          <p className="lede">
            Held-out five-man combinations whose five players were each seen in training. Each model
            is compared against <strong>its own no-lineup counterpart</strong>, so a verdict
            isolates lineup information rather than model class.
          </p>
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col" className="num">
                    Log loss
                  </th>
                  <th scope="col" className="num">
                    Top-1
                  </th>
                  <th scope="col" className="num">
                    vs counterpart
                  </th>
                </tr>
              </thead>
              <tbody>
                {(["S0", "S1", "S2", "S3", "full", "full_gbdt"] as const).map((key) => {
                  const row = lolo[key];
                  if (!row) return null;
                  const counterpart = key === "full" ? "S2" : key === "full_gbdt" ? "S3" : null;
                  const reference = counterpart ? lolo[counterpart] : undefined;
                  return (
                    <tr key={key}>
                      <th scope="row">
                        <code>{key}</code>
                      </th>
                      <td className="num">{row.log_loss.toFixed(5)}</td>
                      <td className="num">
                        {row.top1_accuracy ? row.top1_accuracy.toFixed(4) : "—"}
                      </td>
                      <td className="num">
                        {reference
                          ? `${(((reference.log_loss - row.log_loss) / reference.log_loss) * 100).toFixed(3)}% vs ${counterpart}`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {rapm && (
        <>
          <h2>RAPM reliability</h2>
          <p className="lede">
            Split-half correlation between fits on odd and even games. This, and not cross-validated
            error, is the honest test: possession outcomes are dominated by shot noise, so a ridge
            model can cut CV error while its player coefficients are close to arbitrary.
          </p>
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th scope="col">Side</th>
                  <th scope="col" className="num">
                    Split-half r
                  </th>
                  <th scope="col" className="num">
                    Spearman ρ
                  </th>
                  <th scope="col" className="num">
                    Full-sample
                  </th>
                </tr>
              </thead>
              <tbody>
                {(["off", "def"] as const).map((side) => (
                  <tr key={side}>
                    <th scope="row">{side === "off" ? "Offence" : "Defence"}</th>
                    <td className="num">
                      {(rapm.reliability[`${side}_split_half_r`] ?? 0).toFixed(3)}
                    </td>
                    <td className="num">
                      {(rapm.reliability[`${side}_spearman_rho`] ?? 0).toFixed(3)}
                    </td>
                    <td className="num">
                      {(rapm.reliability[`${side}_full_sample_reliability`] ?? 0).toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="detail">
            Moderate, and published as moderate — three seasons is not enough for RAPM to be
            precise, which is the ceiling on everything downstream including the trade backtest.{" "}
            {rapm.co_occurrence.n_flagged} of {rapm.co_occurrence.n_players} players share more than{" "}
            {(rapm.co_occurrence.ceiling * 100).toFixed(0)}% of their floor time with a single
            teammate; for those the pair&rsquo;s sum is identified and neither coefficient is, so
            they are not served as point estimates.
          </p>
        </>
      )}
    </main>
  );
}
