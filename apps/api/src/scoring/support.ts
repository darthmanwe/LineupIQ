/**
 * The refusal contract, evaluated.
 *
 * Three outcomes and the boundaries between them are the product:
 *
 * - `reportable`  — enough evidence for a point estimate.
 * - `directional` — the player-level terms have support but the five-man
 *                   combination does not. A 200 with a null centre and a real
 *                   interval. This is the *normal* case for a trade lineup, and
 *                   it is why the answer is not a refusal.
 * - `refused`     — no basis at all. A 422 problem document.
 *
 * What must never happen is a 200 with a confident number and a footnote.
 *
 * The thresholds are not decided here. They are read from the exported,
 * pre-registered, hash-pinned file, and the hash is echoed in every response so
 * a caller can check that the floors applied are the floors published.
 */

import { lineupHash } from "./lineupHash";
import type { PlayersData, SupportData } from "../data/store";

export type Tier = "reportable" | "directional" | "refused";

export type SupportAssessment = {
  lineupHash: string;
  possessions: number;
  minPlayerAttempts: number;
  tier: Tier;
  counterfactual: boolean;
  /** Players below the directional attempt floor, with their shortfall. */
  shortfallPlayers: Array<{ player_id: string; attempts: number; threshold: number }>;
  unknownPlayers: string[];
  thresholds: { possessions: number; attempts: number };
  thresholdsSha256: string;
};

export function assessSupport(
  playerIds: readonly number[],
  support: SupportData,
  players: PlayersData
): SupportAssessment {
  const hash = lineupHash(playerIds);
  const row = support.lineups[hash];

  // Absent from the export means below the directional possession floor, which
  // for every decision made here is the same as never having played together.
  const possessions = row ? row[0] : 0;
  const counterfactual = row === undefined;

  const unknownPlayers: string[] = [];
  let minPlayerAttempts = Number.POSITIVE_INFINITY;
  const shortfallPlayers: SupportAssessment["shortfallPlayers"] = [];

  for (const id of playerIds) {
    const key = String(id);
    const player = players.players[key];
    if (!player) {
      unknownPlayers.push(key);
      minPlayerAttempts = 0;
      continue;
    }
    minPlayerAttempts = Math.min(minPlayerAttempts, player.attempts);
    if (player.attempts < support.thresholds.directional_attempts) {
      shortfallPlayers.push({
        player_id: key,
        attempts: player.attempts,
        threshold: support.thresholds.directional_attempts,
      });
    }
  }
  if (!Number.isFinite(minPlayerAttempts)) minPlayerAttempts = 0;

  let tier: Tier;
  if (
    possessions >= support.thresholds.reportable_possessions &&
    minPlayerAttempts >= support.thresholds.reportable_attempts
  ) {
    tier = "reportable";
  } else if (minPlayerAttempts >= support.thresholds.directional_attempts) {
    tier = "directional";
  } else {
    tier = "refused";
  }

  return {
    lineupHash: hash,
    possessions,
    minPlayerAttempts,
    tier,
    counterfactual,
    shortfallPlayers,
    unknownPlayers,
    thresholds: {
      possessions: support.thresholds.reportable_possessions,
      attempts: support.thresholds.reportable_attempts,
    },
    thresholdsSha256: support.thresholds_sha256,
  };
}

/**
 * What would make this answerable. Required, not optional.
 *
 * "Not enough data" without "of what, and how much more" is not an answer. This
 * turns the refusal into something a caller can act on.
 */
export function whatWouldHelp(assessment: SupportAssessment): string {
  if (assessment.unknownPlayers.length) {
    return (
      `${assessment.unknownPlayers.length} of these player ids are not in the ` +
      "snapshot at all. Check the ids against /api/players."
    );
  }
  if (assessment.shortfallPlayers.length) {
    const worst = assessment.shortfallPlayers.reduce((a, b) => (a.attempts < b.attempts ? a : b));
    return (
      `Player ${worst.player_id} has ${worst.attempts} recorded attempts against a floor ` +
      `of ${worst.threshold}. A player-level estimate needs roughly ${
        worst.threshold - worst.attempts
      } more, which is a few weeks of rotation minutes.`
    );
  }
  return (
    `This five-man group has ${assessment.possessions} possessions against a reporting ` +
    `floor of ${assessment.thresholds.possessions}. Either play them together more, or ` +
    "accept the directional answer, which uses the player-level terms only."
  );
}
