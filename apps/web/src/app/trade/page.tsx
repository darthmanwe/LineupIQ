import { NotYetBacked } from "@/components/NotYetBacked";

export default function TradePage() {
  return (
    <NotYetBacked
      page="Trade Simulator"
      willServe="Swap a player and see the rotation before and after, with offensive and defensive deltas, per-player zone-share shifts, and Monte-Carlo intervals decomposed by where the uncertainty actually comes from."
      milestone="M5 — trade simulator and counterfactual backtest"
      backedBy="EPSA + defensive RAPM + fitted usage decay curves"
      endpoints={["POST /api/trades/simulate", "POST /api/lineups/allocate"]}
      refusal="Almost every post-trade lineup is counterfactual — those five have never played together, so there is no direct evidence at all. The team-level net impact is reported only when enough projected minutes come from lineups with real support; otherwise the headline number is replaced by a card naming which players' possession counts fall short."
    />
  );
}
