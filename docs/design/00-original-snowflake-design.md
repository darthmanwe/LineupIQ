> **SUPERSEDED — kept verbatim, deliberately.**
>
> This is the original pre-implementation Snowflake design, unedited. It is retained
> because the reasoning is worth reading and because a superseded document plus a
> written rationale is more honest than a quietly-updated one.
>
> Sections 3 (Cortex services) and 5 (Streamlit in Snowflake) were replaced under a
> portability constraint: the demo has to be publicly clickable, free, and runnable
> from a clean clone, and a $400/30-day Enterprise trial satisfies none of those.
> [`01-portable-rearchitecture.md`](01-portable-rearchitecture.md) states what
> replaced each service, why, **and what was lost**.
>
> Several technical claims in here are also wrong, and the rearchitecture document
> names them rather than burying them — including one arithmetic error in the
> headline EPSA formula and one internal contradiction about the model.

---

# LineupIQ -- System Design Document

**Author**: LineupIQ Team
**Audience**: Snowflake Data Engineer (Senior) for architecture review
**Status**: Pre-implementation draft

---

## 1. Problem Statement and System Overview

### What this system does

LineupIQ is a decision-support tool for NBA front offices. It answers a single
compound question:

> Given any 5-man lineup on a team, what plays (shot zones and action types) are
> most optimal for each player, and how would swapping a player via trade change
> the lineup's expected offensive and defensive output?

The system is not a live game tool. It operates on historical play-by-play data to
produce batch-scored results that are then explored interactively through a Streamlit
in Snowflake application.

### Why this is hard

NBA lineups interact non-linearly. A player's shooting efficiency from a given court
zone depends on who else is on the floor (spacing, gravity, defensive matchups). Most
5-man combinations play fewer than 200 possessions per season, creating severe
small-sample problems. Trade impact forecasting compounds this: you must predict
efficiency for lineups that have never existed, redistribute shot attempts under a
fixed possession budget, and account for both offensive and defensive effects.

### Architecture overview

**Layer 1 -- Data Ingestion (external to Snowflake)**

Three sources feed into the `BRONZE` schema via `PUT` to internal stage then
`COPY INTO`:

- Kaggle PBP + Shots (1996-2024) -- play-by-play events and shot details with court coordinates
- Synergy Play Types (2012-2025) -- per-player play-type efficiency breakdowns
- NBA API Metadata -- player names, positions, team history for join alignment

**Layer 2 -- Snowflake Data Platform (`LINEUPIQ` database)**

```
BRONZE ──> SILVER ──> GOLD ──> ML
  │                     │
  │                     └──> RAG (Cortex Search index source)
  │
  └──> STAGING (internal file stage + file formats)
```

- `BRONZE` -- Raw CSV landing (all-VARCHAR). Load metadata on every row. Ingest logging and reject tracking.
- `SILVER` -- Typed and conformed events. Stint reconstruction (5-man lineups attached to every event). Transition classification.
- `GOLD` -- Model-ready tables: `SHOT_FACTS` (one row per shot attempt with full lineup context, EPSA, zone, play type), `PLAY_TYPE_FACTS` (Synergy data), `STINT_DOCS` (natural-language narratives for Cortex Search).
- `ML` -- Model output tables: `EP_PREDICTIONS`, `OPTIMAL_PLAYS`, `DEFENSIVE_RATINGS`, `TRADE_SCENARIOS`, `TRADE_IMPACT_RESULTS`.
- `RAG` -- Cortex Search service objects (search service over `STINT_DOCS`).
- `EVAL` -- Evaluation harness tables (golden questions, eval runs, regression results).

**Layer 3 -- ML Compute (local via Snowpark Python)**

Gold tables are pulled locally via Snowpark. Four model components run, then
batch-write results back to `ML` schema:

- EPSA Model -- P(make) + P(foul) prediction with lineup-context embeddings. Produces per-player, per-zone, per-lineup expected points.
- Defensive RAPM -- Ridge regression on stint data estimating each player's defensive contribution.
- Usage Redistribution -- Constrained optimization allocating shot attempts across a lineup under a fixed possession budget.
- Trade Simulator -- Combines EPSA, defensive RAPM, and usage models to project before/after lineup deltas for player swap scenarios.

**Layer 4 -- Cortex AI Services (Snowflake-managed)**

Three Cortex services sit between the data/ML layers and the application:

- Cortex Search -- Hybrid semantic + keyword search over `STINT_DOCS`. Retrieves comparable historical lineup stints as evidence for trade forecasts.
- Cortex Analyst -- NL-to-SQL over structured result tables via a YAML semantic model. Users ask natural-language questions about EPSA results and trade impacts.
- AI_COMPLETE -- LLM function producing scouting-style narrative reports with structured JSON output from optimizer results, trade deltas, and retrieved evidence.

**Layer 5 -- Streamlit in Snowflake (application)**

Four app pages, all reading from `ML` and `GOLD` schemas through the Cortex
service layer:

- Lineup Optimizer -- zone-level EPSA heatmap, top-k optimal actions, scouting narrative
- Trade Simulator -- rotation lineups before/after, offensive/defensive deltas, net impact, confidence tiers
- Evidence / Comps -- free-text search over historical stints via Cortex Search
- Data Quality + Eval -- stint validity, coverage rates, model calibration, RAPM correlation

### Snowflake services used

| Service | Role in system | Why this service |
|---------|---------------|------------------|
| **Cortex Search** | Retrieve comparable historical lineup stints to ground trade forecasts with evidence | Need semantic + keyword hybrid search over natural-language stint narratives; Cortex Search provides managed embedding, indexing, and low-latency serving without external vector DB infra |
| **Cortex Analyst** | Natural-language queries over structured result tables (EPSA predictions, trade impacts) | Users ask questions like "which lineups improved most after adding Player X" -- Cortex Analyst translates to SQL via semantic model, no custom NL-to-SQL pipeline needed |
| **AI_COMPLETE** | Generate scouting-style narrative reports with structured JSON from model outputs | Turns numeric optimizer output into readable analysis; structured output mode enforces JSON schema for reliable app rendering |
| **Feature Store** | Govern ML feature definitions (shot context, lineup embeddings, spacing proxies) | Centralizes feature logic so training and inference use identical transformations; point-in-time correctness via ASOF JOIN |
| **Model Registry** | Version and serve the EPSA model + defensive RAPM model | Enables model lifecycle management, rollback, and batch inference at scale from SQL |
| **Tasks** | Orchestrate ML pipeline refresh and Cortex Search index rebuild | Triggered after model re-scoring (not for data ingestion -- the base data is a static historical load) |

---

## 2. Data Architecture and Pipeline

### Data sources

| Source | What it provides | Volume estimate | Refresh |
|--------|-----------------|-----------------|---------|
| Kaggle `brains14482` NBA/WNBA PBP + shots | Play-by-play events from 3 providers + shot details with x/y coordinates. 1996-2024. | ~6-8M PBP rows, ~4-5M shot rows across all seasons | Static historical dump; no ongoing feed |
| Synergy play-type data (DomSamangy GitHub) | Per-player, per-season play-type stats (isolation, PnR, spot-up, etc.) from NBA.com/Synergy. 2012-2025. | ~50K rows (player-season-play_type grain) | Annual manual refresh |
| `nba_api` | Player metadata: names, positions, team history for join alignment | ~5K active + historical player records | One-time pull for metadata alignment |

Key constraint: no player-tracking data (Second Spectrum). All spatial inference is
from shot coordinates and pbp context, not player movement.

### Schema design (medallion)

**`BRONZE`** -- Raw CSV landing. All columns loaded as VARCHAR. Each table adds
`_loaded_at TIMESTAMP_NTZ`, `_source_file STRING`, `_row_hash STRING` metadata.

| Table | Source | Grain |
|-------|--------|-------|
| `PBP_STATS` | stats.nba.com PBP | One row per game event |
| `PBP_DATA` | data.nba.com PBP | One row per game event (2016+) |
| `PBP_PBPSTATS` | pbpstats.com PBP | One row per possession |
| `SHOTS` | stats.nba.com shot detail | One row per shot attempt |
| `SYNERGY_PLAY_TYPES` | Synergy via GitHub | One row per player-season-play_type |
| `INGEST_RUNS` | System | One row per ingestion execution |
| `REJECT_LOGS` | System | One row per rejected record |

Ingestion uses `COPY INTO` from internal named stage `@STAGING.RAW_FILES` with
`CSV_KAGGLE` file format, `ON_ERROR='CONTINUE'`, pre-validated with
`VALIDATION_MODE`.

**`SILVER`** -- Typed, conformed, deduplicated. Core transformation: **stint
reconstruction** (attaching the 5-man lineup to every event).

| Table | Content |
|-------|---------|
| `EVENTS_TYPED` | All PBP events cast to proper types, deduped on `(game_id, event_num)` |
| `SUBSTITUTIONS` | Parsed sub-in / sub-out events |
| `STINTS` | `(game_id, period, stint_id, start_event, end_event, home_lineup_ids[5], away_lineup_ids[5], stint_quality)` |
| `EVENTS_ENRICHED` | Each event joined to its stint -- every row carries full lineup context |

Stint reconstruction is tiered:
- **Tier 1** (2000+): pbpstats data with pre-ordered events. Best quality.
- **Tier 2** (1996-2000): Parse sub events from stats.nba.com. Heuristic for period-start lineups (first 5 players appearing before first substitution). Handle overtime, ejections, technical fouls.
- **Tier 3** (fallback): Stints that fail validation get flagged `IMPUTED` or `QUARANTINED`. Imputation from box-score minutes where possible.

Cross-validation target: stint-derived player minutes within 2% of box-score totals.

Additional silver transforms:
- **Transition flag**: `is_transition = TRUE` when shot occurs within first 10 seconds of shot clock and possession didn't start after a stoppage. Transition possessions produce ~1.12 PPP vs ~0.95 PPP for half-court -- critical confound if unclassified.

**`GOLD`** -- Analytics-ready, model-input tables.

| Table | Grain | Key columns |
|-------|-------|-------------|
| `SHOT_FACTS` | One row per shot attempt | `shooter_id`, `zone_id`, `shot_type`, `made`, `points`, `and_one`, `foul_on_shot`, `ft_made`, `epsa`, `lineup_for_hash`, `lineup_against_hash`, `lineup_for_ids ARRAY`, `lineup_against_ids ARRAY`, `is_transition`, `is_clutch`, `period`, `clock_sec`, `score_margin_bucket`, `stint_quality`, `season`, `era_bucket` |
| `PLAY_TYPE_FACTS` | One row per player-season-play_type (2012+) | `player_id`, `season`, `play_type`, `possessions`, `ppp`, `fg_pct`, `tov_freq`, `foul_drawn_pct` |
| `STINT_DOCS` | One row per stint | `doc_id`, `game_id`, `stint_id`, `text` (natural-language narrative), metadata columns for Cortex Search filtering |

Lineup hashing: **order-invariant**. Pattern:
`MD5(ARRAY_TO_STRING(ARRAY_SORT(player_id_array), ','))`. Same 5 players in any
permutation produce an identical hash. All joins on lineup identity use this hash.

Era tagging: `era_bucket` column on every gold row. Values: `PRE_2004` (hand-checking
era), `2004_2013` (post-rule-change), `2014_PLUS` (3-point revolution). ML training
uses era-aware weighting.

### Data quality gates

| Check | Table | Threshold | Action on failure |
|-------|-------|-----------|-------------------|
| Stint has exactly 5 players per team | `STINTS` | 100% for `VALID` stints | Flag as `IMPUTED` or `QUARANTINED` |
| Events assigned to stints | `EVENTS_ENRICHED` | >99% coverage | Investigate, extend fallback logic |
| Sub parity (every in has matching out) | `SUBSTITUTIONS` | >98% | Log to `REJECT_LOGS` |
| Stint minutes vs box-score minutes | `STINTS` vs external | <2% discrepancy | Cross-validate with `nba-on-court` |
| Ingest row counts vs source | `INGEST_RUNS` | No unexplained drops | Alert and block downstream |

### Observability

`BRONZE.INGEST_RUNS` logs every load: run_id, source, file count, row count, start/end
timestamps, status. `BRONZE.REJECT_LOGS` captures per-row failures with error
descriptions. Downstream DQ results are stored in `EVAL` schema for dashboard
rendering.

---

## 3. ML Outputs, Cortex Integration, and Deployment

### What the ML engine produces

All models run locally via Snowpark Python pull-to-local pattern, with results
batch-written back to `ML` schema tables.

| Output table | Grain | Key columns | Produced by |
|-------------|-------|-------------|-------------|
| `EP_PREDICTIONS` | player x lineup_hash x zone x shot_type | `p_make`, `p_foul`, `epsa`, `epsa_ci_low`, `epsa_ci_high` | EPSA model (LightGBM + calibration + conformal intervals) |
| `OPTIMAL_PLAYS` | player x lineup_hash, top-k | `rank`, `zone_id`, `play_type`, `epsa`, `confidence_tier` | Play optimizer (ranks actions by EPSA per lineup) |
| `DEFENSIVE_RATINGS` | player | `def_rapm`, `def_rapm_se` | Defensive RAPM (ridge regression on stint data) |
| `TRADE_SCENARIOS` | scenario_id | `team_id`, `outgoing_player_ids`, `incoming_player_ids` | User-defined input |
| `TRADE_IMPACT_RESULTS` | scenario x lineup | `off_ep_delta`, `def_rating_delta`, `net_impact`, `minutes_weighted_impact`, `confidence_tier` | Trade simulator (usage-budget-constrained optimization) |

The EPSA model uses **EPSA = P(make) * points + P(foul_on_shot) * E[FT_points]**
rather than naive `FG% * points`. This captures ~15-20% of scoring value from free
throws drawn on shot attempts.

Lineup context is encoded via learned **player embeddings** with pairwise interaction
dot products and a **spacing proxy** (teammates' 3PT attempt rate). Cold-start players
inherit archetype embeddings from k-means clustering on career box-score profiles.

Usage redistribution is modeled as a **constrained optimization** over a fixed
possession budget (team FGA is roughly constant), so adding a high-usage player
correctly reduces others' shot volume.

### Cortex Search

- **Indexed table**: `GOLD.STINT_DOCS`
- **Document content**: Purpose-built natural-language narratives per stint. Each doc includes team names, all 10 player names, season, game date, period, score context, stint duration, key actions described in plain English, plus structured tags (offensive rating, defensive rating, pace, dominant play types).
- **Why this matters**: Cortex Search quality is bounded by document quality. Generic "event log" text retrieves poorly. Documents are engineered for queries like "lineups with a stretch-5 and elite perimeter scorer in close fourth-quarter games."
- **Service creation**: `CREATE CORTEX SEARCH SERVICE` over `STINT_DOCS.text` with metadata columns for filtered search.
- **Refresh**: Triggered by Task after model re-scoring (not scheduled -- base data is static). Service is **suspended during inactive development** to avoid serving-cost accrual (charged per GB/month of indexed data while active).

### Cortex Analyst

- **Semantic model**: YAML definition (~200+ lines) or `CREATE SEMANTIC VIEW` covering:
  - Logical tables: `SHOT_FACTS`, `EP_PREDICTIONS`, `OPTIMAL_PLAYS`, `TRADE_IMPACT_RESULTS`, `STINTS`
  - Dimensions: `player_name`, `team`, `season`, `zone_id`, `play_type`, `confidence_tier`, `era_bucket`
  - Measures: `epsa`, `off_ep_delta`, `def_rating_delta`, `net_impact`, `possession_count`
  - Relationships: `player_id` joins across tables, `lineup_hash` joins, `game_id` joins
  - Verified queries: 5-10 golden question/SQL pairs to anchor generation quality
- **Exposed via**: REST API endpoint called from Streamlit app for NL question answering over structured data.

### AI_COMPLETE (scouting narratives)

- **Input**: Optimizer output (top plays + EPSA) + trade deltas + retrieved Cortex Search comp snippets
- **Output**: Structured JSON with schema enforcement:
  ```
  {summary, offensive_impact, defensive_impact, key_lineup_changes[],
   confidence, evidence_stints[]}
  ```
  Plus a human-readable scouting paragraph.
- **Model selection**: `mistral-7b` / `snowflake-arctic` for development iteration; `claude-3-5-sonnet` or `llama3.1-70b` for production-quality narratives.
- **Prompt stability**: Prompts are versioned in code. 2-3 few-shot examples anchored in each prompt. Regression tested against 10 golden input/output pairs asserting JSON schema compliance and key claim presence.

### Streamlit in Snowflake app

Four pages, all reading from `ML` and `GOLD` schemas:

| Page | Inputs | Outputs | Cortex services used |
|------|--------|---------|---------------------|
| **Lineup Optimizer** | Player + 4 teammates + 5 opponents | Zone-level EPSA heatmap, top-k optimal actions, play-type breakdown, scouting narrative | AI_COMPLETE, Cortex Search |
| **Trade Simulator** | Team, outgoing player(s), incoming player(s) | Rotation lineups before/after, offensive EP delta, defensive delta, net impact, confidence tier, per-player zone shifts | AI_COMPLETE, Cortex Search, Cortex Analyst |
| **Evidence / Comps** | Free-text search for lineup archetypes | Retrieved historical stints with metadata | Cortex Search |
| **Data Quality + Eval** | None (dashboard) | Stint validity rates, coverage, calibration curve, RAPM correlation, eval pass rates | None (direct SQL) |

### Cost posture

| Resource | Config | Rationale |
|----------|--------|-----------|
| `LINEUPIQ_WH_XS` | X-Small, `AUTO_SUSPEND=60s` | General compute. ~1 credit/hr when active. |
| `LINEUPIQ_WH_S` | Small, `AUTO_SUSPEND=60s` | Cortex Search index builds only. ~2 credits/hr. |
| Resource monitor | 50 credits/month hard cap, notify at 50/75/90% | Prevents surprise bills on trial/personal account. |
| Cortex Search | Suspend when not actively testing | Serving costs accrue per GB/month even with zero queries. |
| AI_COMPLETE model choice | `mistral-7b` for dev, larger models for production | Per-token cost varies 10-100x between models. |

### Implementation plan (bullet summary)

1. Complete manual setup (accounts, packages, Snowflake objects, cost guardrails).
2. Ingest one season (2022-23) through bronze/silver/gold as validation slice.
3. Validate stint reconstruction against `nba-on-court` and box-score minutes.
4. Expand to full history after DQ gates pass.
5. Train EPSA model with temporal CV. Validate calibration and RAPM correlation.
6. Build usage redistribution and defensive RAPM models.
7. Implement trade scenario simulator with possession-budget-aware deltas.
8. Engineer STINT_DOCS and deploy Cortex Search service.
9. Author Cortex Analyst semantic model with verified queries.
10. Wire AI_COMPLETE scouting narratives with structured JSON output.
11. Build Streamlit app (4 pages).
12. Deploy evaluation harness (shot model metrics, retrieval precision, prompt regression).

### Known limitations

- No tracking data: shot difficulty inferred from context, not observed defender positions.
- Shot selection endogeneity: model assumes observed shot mix reflects opportunity, not just player choice.
- Lineup synergies are approximated via pairwise embedding interactions, not higher-order effects.
- Defensive model (RAPM) is coarser than the offensive model.
- Usage-efficiency curves approximate diminishing returns at the player level, not per-zone.
