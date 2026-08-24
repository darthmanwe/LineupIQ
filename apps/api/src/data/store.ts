/**
 * Reading the exported data the Worker serves.
 *
 * Everything comes from static assets written by `lineupiq export` and
 * committed. Two consequences worth naming.
 *
 * First, the Worker does no data engineering. It reads a lookup table and does
 * arithmetic. That is deliberate: anything the Worker computed would be a
 * second implementation of something Python already does, and the two would
 * drift.
 *
 * Second, the assets are fetched through the `ASSETS` binding and cached in
 * module scope for the lifetime of the isolate. A cold start pays the parse
 * once; later requests pay nothing. The support table is the reason the export
 * only includes lineups above the directional floor -- a group below it is
 * refused anyway, and the refusal needs no row.
 */

import type { Bindings } from "../index";

export type SupportRow = {
  possessions: number;
  minPlayerAttempts: number;
};

export type SupportData = {
  thresholds: {
    reportable_possessions: number;
    reportable_attempts: number;
    directional_possessions: number;
    directional_attempts: number;
  };
  thresholds_sha256: string;
  n_observed_lineups: number;
  n_exported: number;
  lineups: Record<string, [number, number]>;
};

export type PlayerRow = {
  name: string;
  attempts: number;
  off_rapm?: number;
  def_rapm?: number;
  off_se?: number;
  def_se?: number;
  off_includes_zero?: boolean;
  def_includes_zero?: boolean;
  possessions?: number;
};

export type PlayersData = { players: Record<string, PlayerRow>; count: number };

export type ZonesData = { zones: Array<{ id: string; label: string }>; count: number };

export type SnapshotData = {
  contracts: Record<string, { rows: number; content_sha256: string }>;
  n_contracts: number;
  thresholds_sha256: string;
};

export type SelectionModelData = {
  available: boolean;
  reason?: string;
  git_sha?: string;
  term_names?: string[];
  coefficients?: number[];
  observed_mix?: Record<string, number>;
  sign_audit?: Record<string, { value: number; expected_sign: number; verdict: string }>;
  n_shots?: number;
  seasons?: number[];
};

/** Thrown when an asset the caller needs is not deployed. */
export class MissingAsset extends Error {
  constructor(readonly asset: string) {
    super(`asset ${asset} is not deployed`);
    this.name = "MissingAsset";
  }
}

const cache = new Map<string, unknown>();

async function read<T>(env: Bindings, name: string): Promise<T> {
  const hit = cache.get(name);
  if (hit !== undefined) return hit as T;

  if (!env.ASSETS) throw new MissingAsset(name);

  // The host is arbitrary: the assets binding routes by path only. Using a
  // fixed placeholder keeps the URL valid without pretending to know the
  // deployed hostname.
  const response = await env.ASSETS.fetch(new Request(`https://assets.local/data/${name}`));
  if (!response.ok) throw new MissingAsset(name);

  const parsed = (await response.json()) as T;
  cache.set(name, parsed);
  return parsed;
}

export const loadSupport = (env: Bindings): Promise<SupportData> =>
  read<SupportData>(env, "support.json");

export const loadPlayers = (env: Bindings): Promise<PlayersData> =>
  read<PlayersData>(env, "players.json");

export const loadZones = (env: Bindings): Promise<ZonesData> => read<ZonesData>(env, "zones.json");

export const loadSnapshot = (env: Bindings): Promise<SnapshotData> =>
  read<SnapshotData>(env, "snapshot.json");

export const loadSelectionModel = (env: Bindings): Promise<SelectionModelData> =>
  read<SelectionModelData>(env, "selection_model.json");

/** Exposed for tests, which need a clean isolate between cases. */
export function clearAssetCache(): void {
  cache.clear();
}
