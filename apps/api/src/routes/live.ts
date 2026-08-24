/**
 * The routes that are actually backed.
 *
 * Mounted from the registry, so a route cannot advertise itself as `live` here
 * and return 501, or the reverse. Every scored response carries a `support`
 * block; that is not decoration, it is the contract.
 */

import type { Context, Hono } from "hono";

import {
  loadCoverage,
  loadEvaluation,
  loadPlayers,
  loadSelectionModel,
  loadPlayerZones,
  loadSelectionProfiles,
  loadSnapshot,
  loadSupport,
  loadZones,
} from "../data/store";
import { MissingAsset } from "../data/store";
import { envelope } from "../http/envelope";
import { insufficientSupport, problem } from "../http/problem";
import { assessSupport, whatWouldHelp } from "../scoring/support";
import { scoreSelection } from "../scoring/selection";
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

type ScoreRequestInput = {
  shooterId: number;
  offense: number[];
  defense: number[];
  teamId: number | null;
  season: number | null;
  secondsIntoPossession: number | null;
  liveBall: boolean;
  secondChance: boolean;
  clutch: boolean;
};

/**
 * Parses a scoring request.
 *
 * Two rules worth stating because they are easy to get wrong in the other
 * direction. The shooter **must** be one of the five: this is a model of which
 * shot he takes given who is around him, and scoring him against a floor he is
 * not on is a question the model was never fitted to answer. And
 * `seconds_into_possession` left out means "league-average possession", which is
 * the fitted mean -- not zero. Zero is a fast-break, and defaulting to it would
 * quietly assert something strong about shot selection.
 */
function parseScoreRequest(body: unknown): ScoreRequestInput | { error: string } {
  if (typeof body !== "object" || body === null) return { error: "body must be a JSON object" };
  const raw = body as Record<string, unknown>;

  const shooterId =
    typeof raw.shooter_id === "number"
      ? raw.shooter_id
      : Number.parseInt(String(raw.shooter_id), 10);
  if (!Number.isInteger(shooterId)) return { error: "`shooter_id` must be a player id" };

  const ids = (value: unknown, field: string, required: boolean): number[] | { error: string } => {
    if (value === undefined || value === null) {
      return required ? { error: `\`${field}\` is required` } : [];
    }
    if (!Array.isArray(value)) return { error: `\`${field}\` must be an array of player ids` };
    const out: number[] = [];
    for (const item of value) {
      const id = typeof item === "number" ? item : Number.parseInt(String(item), 10);
      if (!Number.isInteger(id)) return { error: `\`${String(item)}\` is not a player id` };
      out.push(id);
    }
    if (new Set(out).size !== out.length) return { error: `\`${field}\` cannot repeat a player` };
    return out;
  };

  const offense = ids(raw.offense, "offense", true);
  if ("error" in offense) return offense;
  if (offense.length !== LINEUP_SIZE) {
    return { error: `\`offense\` must contain exactly ${LINEUP_SIZE} ids, got ${offense.length}` };
  }
  if (!offense.includes(shooterId)) {
    return {
      error:
        "`shooter_id` must be one of the five in `offense`. This model predicts which " +
        "shot a player takes given who is on the floor with him; a shooter who is not " +
        "on the floor is not a question it can answer.",
    };
  }

  const defense = ids(raw.defense, "defense", false);
  if ("error" in defense) return defense;
  if (defense.length !== 0 && defense.length !== LINEUP_SIZE) {
    return {
      error: `\`defense\` must be omitted or contain exactly ${LINEUP_SIZE} ids, got ${defense.length}`,
    };
  }

  const optionalInt = (value: unknown): number | null => {
    if (value === undefined || value === null) return null;
    const parsed = typeof value === "number" ? value : Number.parseInt(String(value), 10);
    return Number.isInteger(parsed) ? parsed : null;
  };

  let seconds: number | null = null;
  if (raw.seconds_into_possession !== undefined && raw.seconds_into_possession !== null) {
    const parsed = Number(raw.seconds_into_possession);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 24) {
      return { error: "`seconds_into_possession` must be between 0 and 24" };
    }
    seconds = parsed;
  }

  return {
    shooterId,
    offense,
    defense,
    teamId: optionalInt(raw.team_id),
    season: optionalInt(raw.season),
    secondsIntoPossession: seconds,
    liveBall: raw.live_ball === true,
    secondChance: raw.second_chance === true,
    clutch: raw.clutch === true,
  };
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

  /**
   * Score a counterfactual: what shot mix does this shooter take on this floor?
   *
   * This is the endpoint the project is about, and what it returns is a
   * *distribution over zones*, not a points estimate. That is a deliberate
   * change of question. The conversion model -- does a lineup make a player
   * shoot better from the same spot -- was built first, fully evaluated, and
   * came back at +0.02% log loss against a passing negative control. It is not
   * a real effect. Shot **selection** is: +0.08% on unseen five-man
   * combinations. Lineup construction operates on which shots get taken, so
   * that is what is served.
   *
   * `delta` is the interesting field. It is this lineup's mix minus the same
   * shooter's mix with every lineup term at the league average, so it isolates
   * the part of the prediction the five-man combination is responsible for.
   * Those numbers are small -- fractions of a percentage point -- and they are
   * served at their real size rather than rescaled into looking impressive.
   *
   * Support gating is the same contract as `/lineups/support`: a refused tier
   * gets a 422 with what would help, and a directional tier gets a 200 whose
   * `delta` is populated but whose `mix` carries an explicit warning. There is
   * no path that returns a confident number with a footnote.
   */
  app.post("/lineups/score", async (c) => {
    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      return problem(c, {
        status: 400,
        code: "MALFORMED_BODY",
        title: "Body is not JSON",
        detail:
          'Send `{"shooter_id": id, "offense": [5 ids], "defense": [5 ids]}`. ' +
          "`defense`, `team_id`, `season` and the context flags are optional.",
      });
    }

    const request = parseScoreRequest(body);
    if ("error" in request) {
      return problem(c, {
        status: 400,
        code: "INVALID_REQUEST",
        title: "Invalid scoring request",
        detail: request.error,
      });
    }

    try {
      const [model, profiles, support, players] = await Promise.all([
        loadSelectionModel(c.env),
        loadSelectionProfiles(c.env),
        loadSupport(c.env),
        loadPlayers(c.env),
      ]);

      if (!model.available || !model.term_names || !model.coefficients) {
        return problem(c, {
          status: 503,
          code: "MODEL_NOT_FITTED",
          title: "The selection model has not been fitted",
          detail: model.reason ?? "No selection run log is committed.",
        });
      }

      const assessment = assessSupport(request.offense, support, players);
      if (assessment.tier === "refused") {
        return insufficientSupport(c, {
          detail:
            "These five players do not have enough recorded evidence to support any " +
            "estimate of how they change this shooter's shot selection.",
          nPossessions: assessment.possessions,
          threshold: assessment.thresholds.possessions,
          shortfallPlayers: assessment.shortfallPlayers,
          whatWouldHelp: whatWouldHelp(assessment),
        });
      }

      const result = scoreSelection(
        {
          shooterId: request.shooterId,
          offense: request.offense,
          defense: request.defense,
          teamId: request.teamId,
          season: request.season,
          secondsIntoPossession: request.secondsIntoPossession,
          liveBall: request.liveBall,
          secondChance: request.secondChance,
          clutch: request.clutch,
        },
        profiles,
        { available: true, term_names: model.term_names, coefficients: model.coefficients }
      );

      const warnings: string[] = [];
      if (!result.shooterKnown) {
        warnings.push(
          `Player ${request.shooterId} has no fitted profile in this snapshot. ` +
            "The prediction is the league baseline for this lineup, not a prediction about him."
        );
      }
      if (result.shooterWeight < 0.5) {
        warnings.push(
          `Only ${(result.shooterWeight * 100).toFixed(0)}% of this shooter's mix is his own ` +
            "evidence; the rest is the league prior. Read the shape, not the digits."
        );
      }
      if (assessment.tier === "directional") {
        warnings.push(
          "This five-man combination is below the reportable possession floor. The " +
            "direction of each delta is supported; its magnitude is not."
        );
      }
      if (assessment.counterfactual) {
        warnings.push("These five have never shared the floor in this snapshot.");
      }

      return c.json(
        envelope(
          c,
          {
            lineup_hash: assessment.lineupHash,
            shooter: {
              player_id: String(request.shooterId),
              name: players.players[String(request.shooterId)]?.name ?? null,
              known: result.shooterKnown,
              // The Dirichlet-multinomial shrinkage weight, shipped beside the
              // value it applies to rather than buried in a model card.
              evidence_weight: result.shooterWeight,
            },
            // The headline. Nulled below the reportable floor like every other
            // magnitude, and its sign survives as `points_direction` -- which is
            // what a directional tier means: the direction is supported, the
            // size is not.
            points_per_100: assessment.tier === "reportable" ? result.pointsPer100 : null,
            points_direction:
              result.pointsPer100 > 0 ? "gain" : result.pointsPer100 < 0 ? "loss" : "flat",
            zones: result.zones.map((zone, z) => ({
              zone_id: zone,
              // Null when the magnitude is not supported. A caller that renders
              // this gets nothing to render, which is the intended outcome.
              share: assessment.tier === "reportable" ? (result.mix[z] as number) : null,
              baseline_share: result.baselineMix[z] as number,
              delta: (result.mix[z] as number) - (result.baselineMix[z] as number),
            })),
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
              parity_fixture: "data/parity/selection.json",
              git_sha: model.git_sha ?? null,
            },
            warnings,
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
      const audit = Object.entries(model.sign_audit ?? {});
      // Three verdicts, and conflating two of them was a real bug here. A term
      // whose interval spans zero has not contradicted its pre-registered sign
      // — the data cannot sign it either way — and reporting that as a
      // contradiction overstates the finding in the model's own favour, by
      // making it look like it produced more surprises than it did.
      const disagreements = audit
        .filter(([, row]) => row.verdict === "DISAGREES")
        .map(([name]) => name);
      const indeterminate = audit
        .filter(([, row]) => row.verdict === "indeterminate")
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
          warnings: [
            ...(disagreements.length
              ? [
                  `${disagreements.length} coefficient(s) contradict their pre-registered ` +
                    `sign: ${disagreements.join(", ")}. See docs/modeling.md.`,
                ]
              : []),
            ...(indeterminate.length
              ? [
                  `${indeterminate.length} coefficient(s) have a 95% interval spanning ` +
                    `zero and are neither confirmed nor contradicted: ` +
                    `${indeterminate.join(", ")}. Do not read a sign off them.`,
                ]
              : []),
          ],
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

  /**
   * One player's per-zone conversion, with the shrinkage weight beside it.
   *
   * No lineup context, deliberately — that is what `/lineups/score` is for, and
   * conflating the two would let a reader attribute a shooter's own tendency to
   * whoever happened to be on the floor.
   *
   * Every rate ships three numbers: the raw one, the shrunk one, and the weight
   * that separates them. A shrunk rate on its own is indistinguishable from a
   * measurement, and on eleven attempts it is mostly the league prior. Zones the
   * player has never shot from are absent rather than filled with the prior.
   */
  app.get("/players/:id/zones", async (c) => {
    const id = c.req.param("id");
    if (!/^[0-9]+$/.test(id)) {
      return problem(c, {
        status: 400,
        code: "INVALID_PLAYER_ID",
        title: "Invalid player id",
        detail: "A player id is a positive integer.",
      });
    }

    try {
      const [zones, players] = await Promise.all([loadPlayerZones(c.env), loadPlayers(c.env)]);
      const rows = zones.players[id];
      if (!rows) {
        return problem(c, {
          status: 404,
          code: "NO_SUCH_PLAYER",
          title: "No recorded attempts for that player",
          detail:
            `Player ${id} has no shots in this snapshot. That is not the same as ` +
            "a player who does not exist, and the API does not claim to know which.",
        });
      }

      const total = rows.reduce((a, r) => a + r.attempts, 0);
      const warnings: string[] = [];
      const thin = rows.filter((r) => r.shrinkage_weight < 0.5);
      if (thin.length > 0) {
        warnings.push(
          `${thin.length} of ${rows.length} zones carry less than half their weight from ` +
            `this player's own attempts: ${thin.map((r) => r.zone_id).join(", ")}. Those ` +
            "rates are mostly the league prior."
        );
      }

      return c.json(
        envelope(
          c,
          {
            player_id: id,
            name: players.players[id]?.name ?? null,
            attempts: total,
            league_zone_rate: zones.league_zone_rate,
            zones: rows,
          },
          { snapshot: c.env.SNAPSHOT ?? null, warnings }
        )
      );
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  /**
   * The groundedness harness's results, controls included.
   *
   * The controls are not optional context. A grounded rate of 1.00 proves
   * nothing by itself — a checker that accepts every narrative also scores 1.00
   * — so the response refuses to serve a rate without the two distractors
   * beside it: an easy one (score against the next lineup's evidence) and a
   * near-miss (same lineup, one player swapped, so almost every number is still
   * nearly right). The near-miss number is the one that means something.
   *
   * `mean_traceability` is served with a caveat attached rather than as a
   * headline. It is 1.00 across every template including the deliberately
   * hallucinating one, which is the finding: arithmetic settles where a number
   * came from and cannot settle what it was used to say.
   */
  app.get("/eval/groundedness", async (c) => {
    try {
      const evaluation = await loadEvaluation(c.env);
      const groundedness = evaluation.groundedness as
        | {
            by_template: Record<
              string,
              {
                grounded_rate: number;
                mean_traceability: number;
                control_easy_grounded_rate: number;
                control_near_miss_grounded_rate: number;
                failures_by_check: Record<string, number>;
                checks: string[];
                n: number;
              }
            >;
            n_documents: number;
            n_below_floor: number;
          }
        | undefined;

      if (!groundedness) {
        return problem(c, {
          status: 503,
          code: "NOT_MEASURED",
          title: "The groundedness harness has not been run",
          detail: "No run log at services/ml/runs/groundedness/run.json.",
        });
      }

      const templates = Object.entries(groundedness.by_template);
      const warnings: string[] = [];

      // A control that scores as well as the real thing means the checker is
      // not checking. Assert it here rather than trusting the run log's author.
      const blind = templates.filter(
        ([, t]) => t.control_near_miss_grounded_rate >= t.grounded_rate && t.grounded_rate > 0
      );
      if (blind.length > 0) {
        warnings.push(
          `The near-miss control scores as high as the real evidence for: ` +
            `${blind.map(([name]) => name).join(", ")}. Those rates are not evidence ` +
            `the checker works.`
        );
      }

      if (templates.every(([, t]) => t.mean_traceability === 1)) {
        warnings.push(
          "Numeric traceability is 1.00 on every template, including the one written to " +
            "hallucinate. Every number in those narratives is quoted correctly from the " +
            "evidence and used to say something the evidence does not support. Arithmetic " +
            "settles provenance; it cannot settle meaning."
        );
      }

      return c.json(
        envelope(
          c,
          {
            n_documents: groundedness.n_documents,
            n_below_floor: groundedness.n_below_floor,
            templates: templates.map(([name, t]) => ({
              template: name,
              n: t.n,
              checks: t.checks,
              grounded_rate: t.grounded_rate,
              mean_traceability: t.mean_traceability,
              controls: {
                easy: t.control_easy_grounded_rate,
                near_miss: t.control_near_miss_grounded_rate,
              },
              failures_by_check: t.failures_by_check,
            })),
          },
          { snapshot: c.env.SNAPSHOT ?? null, warnings }
        )
      );
    } catch (error) {
      return assetMissing(c, error);
    }
  });

  /**
   * Every data-quality gate, with its threshold and verdict.
   *
   * A failing blocking gate makes the whole response a 503: an API that serves
   * numbers derived from data it knows is broken is worse than one that stops.
   */
  app.get("/dq/coverage", async (c) => {
    try {
      const coverage = await loadCoverage(c.env);
      const failing = coverage.gates.filter(
        (gate) => gate.verdict === "FAIL" && gate.severity === "blocking"
      );
      if (failing.length) {
        return problem(c, {
          status: 503,
          code: "DATA_QUALITY_FAILURE",
          title: "A blocking data-quality gate is failing",
          detail:
            "The served snapshot does not pass its own quality gates, so every number " +
            "derived from it is suspect. This is a stop, not a warning.",
          extensions: { failing: failing.map((gate) => gate.name) },
        });
      }
      return c.json(
        envelope(c, coverage, {
          snapshot: c.env.SNAPSHOT ?? null,
          warnings: coverage.gates
            .filter((gate) => gate.verdict === "WARN")
            .map((gate) => `${gate.name}: ${gate.detail}`),
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
