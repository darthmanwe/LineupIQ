import { NotYetBacked } from "@/components/NotYetBacked";

export default function QualityPage() {
  return (
    <NotYetBacked
      page="Data Quality & Eval"
      willServe="Every data-quality gate with its threshold, measured value and verdict; the shot model's reliability diagram and Brier decomposition against the full baseline ladder; how many players' defensive RAPM intervals include zero; and the LLM judge's agreement with real hand labels next to a cheap deterministic baseline's."
      milestone="M2 (data quality) then M3 (calibration) and M6 (LLM eval)"
      backedBy="services/ml/runs/**/*.json — the same run logs the README is generated from"
      endpoints={[
        "GET /api/dq/coverage",
        "GET /api/eval/model",
        "GET /api/eval/judge",
        "GET /api/eval/groundedness",
      ]}
      refusal="This page is where the project publishes what it gets wrong. The API serves a beats_best_baseline flag that is allowed to be false, and the counterfactual backtest publishes its minimum detectable effect before its result — so an underpowered finding reads as 'not testable at this sample size' rather than a lucky number."
    />
  );
}
