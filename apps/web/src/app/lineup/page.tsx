import { NotYetBacked } from "@/components/NotYetBacked";

export default function LineupPage() {
  return (
    <NotYetBacked
      page="Lineup Optimizer"
      willServe="Pick five players and see a court heatmap of expected points per shot opportunity by zone, the top-k actions ranked by EPSA, and a scouting narrative grounded in retrieved comparable lineups."
      milestone="M3 — EPSA model, calibration and the eval harness"
      backedBy="data/gold/shot_facts/ via the closed-form scorer"
      endpoints={["POST /api/lineups/score", "POST /api/lineups/optimal-plays", "GET /api/zones"]}
      refusal="Zones below the support floor render hatched and unlabelled rather than coloured. If the top-1 and top-3 actions have overlapping 80% intervals, the list is shown unordered — ranking options you cannot distinguish is the most common way a model like this lies."
    />
  );
}
