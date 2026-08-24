/**
 * The routes that are actually backed.
 *
 * Mounted from the registry, so a route cannot advertise itself as `live` here
 * and return 501, or the reverse. Every scored response carries a `support`
 * block; that is not decoration, it is the contract.
 */

import type { Context, Hono } from "hono";

import {
  loadEvaluation,
  loadPlayers,
  loadSelectionModel,
  loadSnapshot,
  loadSupport,
  loadZones,
} from "../data/store";
import { MissingAsset } from "../data/store";
import { envelope } from "../http/envelope";
import { insufficientSupport, problem } from "../http/problem";
import { assessSupport, whatWouldHelp } from "../scoring/support";
import { LINEUP_SIZE, lineupHash } from "../scoring/lineupHash";
import type { Bindings } from "../index";

type App = Hono<{ Bindings: Bindings }>;

/** Declared coverage. The one place scope is stated, mirroring `seasons.py`. */
const SEASONS = [
  { start_year: 2022, label: "2022-23" },
  { start_year: 2023, label: "2023-24" },
  { start_year: 2024, label: "2024-25" },
] as const;

function assetMissing(c: Context<{ Bindings: Bindings }>, error: unknown): Response {
  if (error instanceof MissingAsset) {
    return problem(c, {
      status: 503,
      code: "SNAPSHOT_NOT_DEPLOYED",
      title: "Data snapshot is not deployed",
      detail:
        `The Worker is running but ${error.asset} was not found in the deployed assets. ` +
        "Run `lineupiq export` and redeploy.",
      extensions: { asset: error.asset },
    });
  }
  throw error;
}

/** Parses and validates a five-player request body. */
function parseLineup(body: unknown): { ids: number[] } | { error: string } {
  if (typeof body !== "object" || body === null) return { error: "body must be a JSON object" };
  const raw = (body as { players?: unknown }).players;
  if (!Array.isArray(raw)) return { error: "`players` must be an array of player ids" };
  if (raw.length !== LINEUP_SIZE) {
    return { error: `\`players\` must contain exactly ${LINEUP_SIZE} ids, got ${raw.length}` };
  }
  const ids: number[] = [];
  for (const value of raw) {
    const id = typeof value === "number" ? value : Number.parseInt(String(value), 10);
    if (!Number.isInteger(id)) return { error: `\`${String(value)}\` is not a player id` };
    ids.push(id);
  }
  if (new Set(ids).size !== ids.length) return { error: "a lineup cannot repeat a player" };
  return { ids };
}

export function mountLive(app: App): void {
  app.get("/seasons", (c) =>
    c.json(
      envelope(
        c,
        {
          seasons: SEASONS,
          modelled_game_types: ["regular", "playoffs", "playin"],
          note:
            "The two upstream mirrors label the same season with different years. Season " +
            "is decoded from GAME_ID and asserted at build time, never taken from a filename.",
        },
        { snapshot: c.env.SNAPSHOT ?? null }
      )
    )
  );

  app.get("/zones", async (c) => {
    try {
      const zones = await loadZones(c.env);
      return c.json(
        envelope(c, zones, {
          snapshot: c.env.SNAPSHOT ?? null,
          warnings: [],
        })
      );
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  app.get("/meta/snapshot", async (c) => {
    try {
      const snapshot = await loadSnapshot(c.env);
      return c.json(envelope(c, snapshot, { snapshot: c.env.SNAPSHOT ?? null }));
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  app.get("/players", async (c) => {
    try {
      const players = await loadPlayers(c.env);
      const query = (c.req.query("q") ?? "").trim().toLowerCase();
      const limit = Math.min(Number.parseInt(c.req.query("limit") ?? "50", 10) || 50, 500);

      const rows = Object.entries(players.players)
        .filter(([, row]) => !query || row.name.toLowerCase().includes(query))
        .map(([id, row]) => ({ player_id: id, ...row }))
        .sort((a, b) => (b.possessions ?? 0) - (a.possessions ?? 0))
        .slice(0, limit);

      return c.json(
        envelope(
          c,
          { players: rows, total: players.count, returned: rows.length },
          {
            snapshot: c.env.SNAPSHOT ?? null,
            warnings: rows.some((r) => r.off_rapm === undefined)
              ? ["Some players have no RAPM estimate in this snapshot."]
              : [],
          }
        )
      );
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  /**
   * The refusal contract, as an endpoint.
   *
   * Returns 422 when there is no basis at all, and 200 with `tier:
   * "directional"` and a null point estimate when the player terms have support
   * but the five-man combination does not. Never a 200 with a confident number
   * and a footnote.
   */
  app.post("/lineups/support", async (c) => {
    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      return problem(c, {
        status: 400,
        code: "MALFORMED_BODY",
        title: "Body is not JSON",
        detail: 'Send `{"players": [id, id, id, id, id]}`.',
      });
    }

    const parsed = parseLineup(body);
    if ("error" in parsed) {
      return problem(c, {
        status: 400,
        code: "INVALID_LINEUP",
        title: "Invalid lineup",
        detail: parsed.error,
      });
    }

    try {
      const [support, players] = await Promise.all([loadSupport(c.env), loadPlayers(c.env)]);
      const assessment = assessSupport(parsed.ids, support, players);

      if (assessment.tier === "refused") {
        return insufficientSupport(c, {
          detail:
            "These five players do not have enough recorded evidence for any estimate, " +
            "not even a directional one.",
          nPossessions: assessment.possessions,
          threshold: assessment.thresholds.possessions,
          shortfallPlayers: assessment.shortfallPlayers,
          whatWouldHelp: whatWouldHelp(assessment),
        });
      }

      return c.json(
        envelope(
          c,
          {
            lineup_hash: assessment.lineupHash,
            players: parsed.ids.map((id) => ({
              player_id: String(id),
              name: players.players[String(id)]?.name ?? null,
              attempts: players.players[String(id)]?.attempts ?? 0,
            })),
            // Null on purpose when the combination is not reportable. A caller
            // that renders this field gets nothing to render, which is the
            // intended outcome.
            possessions_together: assessment.possessions,
            estimate_permitted: assessment.tier === "reportable",
            what_would_help: assessment.tier === "reportable" ? null : whatWouldHelp(assessment),
          },
          {
            snapshot: c.env.SNAPSHOT ?? null,
            support: {
              lineup_possessions: assessment.possessions,
              min_player_attempts: assessment.minPlayerAttempts,
              tier: assessment.tier,
              counterfactual: assessment.counterfactual,
              thresholds: assessment.thresholds,
            },
            scoring: {
              closed_form_version: "selection-conditional-logit-v1",
              parity_fixture: "data/parity/lineups.json",
            },
            warnings: assessment.counterfactual
              ? ["These five have never shared the floor in this snapshot."]
              : [],
          }
        )
      );
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  /** The selection model's served coefficients and its own sign audit. */
  app.get("/models/selection", async (c) => {
    try {
      const model = await loadSelectionModel(c.env);
      if (!model.available) {
        return problem(c, {
          status: 503,
          code: "MODEL_NOT_DEPLOYED",
          title: "Selection model is not deployed",
          detail: model.reason ?? "No selection run log was exported.",
        });
      }
      const disagreements = Object.entries(model.sign_audit ?? {})
        .filter(([, row]) => row.verdict !== "agrees")
        .map(([name]) => name);

      return c.json(
        envelope(c, model, {
          snapshot: c.env.SNAPSHOT ?? null,
          model: {
            name: "shot-selection",
            version: model.git_sha ?? "unknown",
            primary_metric: "multiclass log loss (leave-lineup-out)",
            card: "docs/modeling.md",
          },
          // Surfaced as a warning rather than buried in the payload: a
          // coefficient that contradicts its pre-registered sign is the most
          // interesting thing about this model, not a footnote.
          warnings: disagreements.length
            ? [
                `${disagreements.length} coefficient(s) contradict their pre-registered ` +
                  `sign: ${disagreements.join(", ")}. See docs/modeling.md.`,
              ]
            : [],
        })
      );
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  /**
   * Published evaluation results.
   *
   * `?section=` narrows it; without one the whole set comes back, because the
   * point of this endpoint is that a reader can see every measurement including
   * the unflattering ones.
   */
  app.get("/eval/model", async (c) => {
    try {
      const evaluation = await loadEvaluation(c.env);
      const section = c.req.query("section");
      if (section && !(section in evaluation)) {
        return problem(c, {
          status: 404,
          code: "NO_SUCH_SECTION",
          title: "No such evaluation section",
          detail: `Known sections: ${evaluation.available.join(", ")}.`,
          extensions: { available: evaluation.available },
        });
      }
      const body = section ? { [section]: evaluation[section] } : evaluation;

      const trade = evaluation.trade as
        Record<string, { power?: { verdict?: string } }> | undefined;
      const underpowered = trade
        ? Object.values(trade).some((run) => run.power?.verdict === "UNDERPOWERED")
        : false;

      return c.json(
        envelope(c, body, {
          snapshot: c.env.SNAPSHOT ?? null,
          // Surfaced, not buried. The trade backtest's own verdict is that no
          // accuracy claim is supported at this sample size, and a client
          // reading these numbers needs to know that before it reads them.
          warnings: underpowered
            ? [
                "The trade backtest is UNDERPOWERED: its minimum detectable effect is the " +
                  "same size as the effects it projects. No accuracy claim follows from " +
                  "those numbers.",
              ]
            : [],
        })
      );
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  app.get("/eval/retrieval", async (c) => {
    try {
      const evaluation = await loadEvaluation(c.env);
      const retrieval = evaluation.retrieval;
      if (!retrieval) {
        return problem(c, {
          status: 503,
          code: "EVALUATION_NOT_DEPLOYED",
          title: "Retrieval evaluation is not deployed",
          detail: "Run `lineupiq retrieval ablation` and `lineupiq export`, then redeploy.",
        });
      }
      return c.json(envelope(c, retrieval, { snapshot: c.env.SNAPSHOT ?? null }));
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  /** Exposed so a client can verify the hash it computes matches the server's. */
  app.post("/lineups/hash", async (c) => {
    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      return problem(c, {
        status: 400,
        code: "MALFORMED_BODY",
        title: "Body is not JSON",
        detail: 'Send `{"players": [id, id, id, id, id]}`.',
      });
    }
    const parsed = parseLineup(body);
    if ("error" in parsed) {
      return problem(c, {
        status: 400,
        code: "INVALID_LINEUP",
        title: "Invalid lineup",
        detail: parsed.error,
      });
    }
    return c.json(
      envelope(c, {
        lineup_hash: lineupHash(parsed.ids),
        canonical: [...parsed.ids].sort((a, b) => a - b).join(","),
        note:
          "Ids are sorted NUMERICALLY before hashing. A lexicographic sort puts " +
          "1630552 before 201143 and yields a different hash on any engine that " +
          "sorts numerically.",
      })
    );
  });
}
