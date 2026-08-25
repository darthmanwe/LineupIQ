"use client";

import { useState } from "react";

import evaluationData from "../../../public/data/evaluation.json";

/**
 * The Trade page.
 *
 * The minutes-rule selector is the point of this page. How much an arriving
 * player plays is a coaching decision nothing in this repository can observe, so
 * it is a **visible input** rather than a silent assumption — and watching the
 * answer move when you change it is the most honest thing here.
 *
 * What it toggles is a real backtest, run three times with three different rules
 * and committed. It is not a live simulator: `POST /api/trades/simulate` returns
 * `501`, and it is withheld on the power analysis rather than pending on missing
 * machinery. The projection runs; the sample cannot establish that it is right.
 */

type Arm = {
  n: number;
  n_with_signed_projection?: number;
  mean_projected: number;
  mean_observed: number;
  mean_did: number;
  sd_did: number;
  correlation_projected_did: number;
  sign_agreement_vs_did: number;
  sign_agreement_ci: [number, number];
  mean_abs_error_vs_did: number;
  mean_abs_did: number;
};

type Run = {
  minutes_rule: string;
  n_moves: number;
  n_placebo: number;
  residual_sd: number;
  power: {
    n: number;
    mde: number;
    residual_sd: number;
    verdict: string;
    sign_accuracy_ci_half_width: number;
  };
  real: Arm;
  placebo: Arm;
  // Partial on purpose: `variance_decomposition` returns only `n` and
  // `n_without_variance` when no projection had a computable variance, so every
  // read below has to survive the key being absent.
  variance: Partial<Record<string, number>>;
  notes: string[];
};

const RULES = [
  {
    id: "inherit",
    label: "Inherit",
    detail:
      "The arriving player takes exactly the departing player's minutes. The cleanest counterfactual and the least realistic — it assumes the coach slots one for the other with no rotation change at all.",
  },
  {
    id: "historical",
    label: "Historical",
    detail:
      "The arriving player plays at his own recent rate, scaled into the vacated minutes. Wider spread, because two rotations rarely match.",
  },
  {
    id: "conservative",
    label: "Conservative",
    detail:
      "He absorbs only part of the vacated minutes and the rest goes to incumbents. Appropriate when the departing player was a starter and the arrival is not.",
  },
] as const;

const runs = (evaluationData as unknown as { trade?: Record<string, Run> }).trade ?? {};

export default function TradePage() {
  const [rule, setRule] = useState<string>("inherit");
  const run = runs[rule];

  return (
    <main className="page">
      <style>{`
        .page { max-width: 62rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
        .page h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
        .page h2 { font-size: 1.1rem; margin-top: 2.25rem; }
        .lede { color: var(--muted); max-width: 44rem; }
        .rules { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1.5rem 0 0.75rem; }
        .rules button { font: inherit; font-size: 0.85rem; padding: 0.4rem 0.85rem; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; }
        .rules button[aria-pressed="true"] { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); font-weight: 600; }
        .ruledetail { color: var(--muted); font-size: 0.85rem; max-width: 44rem; min-height: 3.2rem; }
        .verdict { border: 1px solid var(--border); border-left: 3px solid #d03b3b; background: var(--warn-soft); padding: 0.9rem 1rem; border-radius: var(--radius); margin: 1.25rem 0; }
        .verdict h3 { margin: 0 0 0.4rem; font-size: 0.95rem; }
        .verdict p { margin: 0; font-size: 0.9rem; }
        table { border-collapse: collapse; width: 100%; font-size: 0.86rem; margin-top: 0.75rem; }
        th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
        td.num { text-align: right; font-family: var(--mono); white-space: nowrap; }
        .detail { color: var(--muted); font-size: 0.82rem; }
        .note { border-left: 3px solid var(--border); padding: 0.6rem 0 0.6rem 1rem; color: var(--muted); font-size: 0.88rem; margin: 1.5rem 0; }
        .note strong { color: var(--text); }
      `}</style>

      <h1>Trade projection, and whether it can be tested</h1>
      <p className="lede">
        A trade projection makes a falsifiable claim: this move changes the receiving team by that
        much. Below is that claim checked against every mid-season move in three seasons.
      </p>

      <h2>The minutes rule is an input, not an assumption</h2>
      <div className="rules" role="group" aria-label="Minutes rule">
        {RULES.map((option) => (
          <button
            key={option.id}
            type="button"
            aria-pressed={rule === option.id}
            onClick={() => setRule(option.id)}
          >
            {option.label}
          </button>
        ))}
      </div>
      <p className="ruledetail">{RULES.find((option) => option.id === rule)?.detail}</p>

      {!run ? (
        <p className="detail">
          No committed backtest for this rule. Run <code>lineupiq backtest --rule {rule}</code>.
        </p>
      ) : (
        <>
          <div className="verdict">
            <h3>Power analysis — {run.power.verdict}</h3>
            <p>
              {run.power.n} evaluable moves against a team net-rating noise floor of{" "}
              {run.power.residual_sd.toFixed(2)} points per 100 gives a minimum detectable effect of{" "}
              <strong>{run.power.mde.toFixed(2)}</strong>, which is the same size as the effects
              projected. Sign accuracy carries a ±
              {(run.power.sign_accuracy_ci_half_width * 100).toFixed(1)}% interval at this n.{" "}
              <strong>No accuracy claim follows from the numbers below.</strong> This verdict was
              computed and committed before the result.
            </p>
          </div>

          <table>
            <thead>
              <tr>
                <th scope="col" />
                <th scope="col" className="num">
                  Real moves
                </th>
                <th scope="col" className="num">
                  Placebo (non-movers)
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">n</th>
                <td className="num">{run.real.n}</td>
                <td className="num">{run.placebo.n}</td>
              </tr>
              <tr>
                <th scope="row">Mean projected delta</th>
                <td className="num">{run.real.mean_projected.toFixed(3)}</td>
                <td className="num">{run.placebo.mean_projected.toFixed(3)}</td>
              </tr>
              <tr>
                <th scope="row">Mean difference-in-differences delta</th>
                <td className="num">{run.real.mean_did.toFixed(3)}</td>
                <td className="num">{run.placebo.mean_did.toFixed(3)}</td>
              </tr>
              <tr>
                <th scope="row">Correlation, projected vs DiD</th>
                <td className="num">{run.real.correlation_projected_did.toFixed(3)}</td>
                <td className="num">undefined</td>
              </tr>
              <tr>
                <th scope="row">Sign agreement</th>
                <td className="num">
                  {(run.real.sign_agreement_vs_did * 100).toFixed(1)}% [
                  {(run.real.sign_agreement_ci[0] * 100).toFixed(0)}–
                  {(run.real.sign_agreement_ci[1] * 100).toFixed(0)}%]
                </td>
                <td className="num">undefined</td>
              </tr>
              <tr>
                <th scope="row">Mean absolute error vs DiD</th>
                <td className="num">
                  <strong>{run.real.mean_abs_error_vs_did.toFixed(2)}</strong>
                </td>
                <td className="num">—</td>
              </tr>
              <tr>
                <th scope="row">Mean absolute DiD swing</th>
                <td className="num">{run.real.mean_abs_did.toFixed(2)}</td>
                <td className="num">
                  <strong>{run.placebo.mean_abs_did.toFixed(2)}</strong>
                </td>
              </tr>
            </tbody>
          </table>

          <p className="detail">
            The two bold numbers are the comparison that settles it. Projection error on real moves
            is {run.real.mean_abs_error_vs_did.toFixed(2)} points per 100; a team&rsquo;s rating
            swings {run.placebo.mean_abs_did.toFixed(2)} across an arbitrary mid-season cutoff with{" "}
            <em>no roster change at all</em>.{" "}
            <strong>The projection does not beat assuming no change.</strong>
          </p>

          <div className="note">
            <strong>Why the placebo columns say &ldquo;undefined&rdquo;.</strong> Swapping a player
            for himself projects exactly {run.placebo.mean_projected.toFixed(3)} — the identity
            holding, and if it drifted every number above would be measuring a pipeline bug. But
            <code> sign(0)</code> matches neither +1 nor −1, so sign agreement and correlation have
            no value on an arm whose projection is zero by construction. An earlier version reported
            0.0% there, which reads as catastrophic failure and means nothing.
          </div>

          {run.variance.mean_minutes_variance_share !== undefined && (
            <div className="note">
              <strong>Where the uncertainty lives.</strong> The minutes rule carries{" "}
              {((run.variance.mean_minutes_variance_share ?? 0) * 100).toFixed(0)}% of a
              projection&rsquo;s variance on average and dominates it in{" "}
              {((run.variance.share_where_minutes_dominates ?? 0) * 100).toFixed(0)}% of cases;{" "}
              {((run.variance.share_interval_includes_zero ?? 0) * 100).toFixed(0)}% of 80%
              intervals contain zero. The plan predicted the minutes assumption would dominate — it
              does not, and the player estimates are the larger term. The design&rsquo;s guess about
              where the uncertainty lived was wrong, and the decomposition is published rather than
              the guess.
            </div>
          )}

          {run.notes.map((note) => (
            <p key={note} className="detail">
              <em>Caveat: {note}</em>
            </p>
          ))}
        </>
      )}

      <div className="note">
        <strong>Withheld, not pending.</strong> <code>POST /api/trades/simulate</code> returns{" "}
        <code>501</code>, and the reason is the table above rather than missing machinery — the
        projection runs, and its own power analysis says the smallest effect this sample could
        detect is the same size as the effects it projects. What is here is the backtest of that
        machinery against moves that actually happened, which is the part that says whether the
        projection would be worth serving. It says not yet.
      </div>
    </main>
  );
}
