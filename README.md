# LineupIQ

**Given any five NBA players, which shots each should take — with the possession count
behind every number, and an explicit refusal when there isn't one.**

[![CI](https://github.com/darthmanwe/LineupIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/darthmanwe/LineupIQ/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Status: milestones 1–4 of 8.** Three seasons are ingested, lineups are reconstructed
> and validated against box-score minutes, and the shot model is fitted and scored against
> a full baseline ladder. The trade simulator and the retrieval/LLM layer are not built;
> their endpoints return `501 NOT_YET_BACKED` naming what will back them. No number in
> this README is typed by hand — every one is rendered from a run log by
> `lineupiq report render`, and CI fails if a committed block goes stale.

---

## The headline finding, stated up front

**Lineup context barely moves shot outcomes, and this project measures how little.**

On leave-lineup-out — held-out five-man combinations whose players were all seen during
training — knowing the other four players on the floor improves log loss by **+0.019%**
for the served model and **+0.078%** for the unconstrained one. Set against a negative
control that passes and a baseline ladder containing everything _except_ lineup
information, that is indistinguishable from nothing.

This is not the result the project set out to find. It is reported first because a model
that adds 0.02% over a lookup table is a model that adds 0.02% over a lookup table, and
the ablation was pre-registered so that this outcome would be publishable rather than
quietly dropped. What the data does support is well-calibrated shot estimates with honest
uncertainty — which is why the refusal contract, not the lineup term, is the part worth
reading.

## What it does

Pick any five players. LineupIQ estimates what each should shoot and from where, given who
else is on the floor, then projects how a trade changes it.

| Page                    | What it answers                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Lineup Optimizer**    | Court heatmap of expected points per shot by zone, and the top-k actions ranked by EPSA.                       |
| **Trade Simulator**     | Rotation before and after a swap, with offensive and defensive deltas under an explicitly chosen minutes rule. |
| **Evidence / Comps**    | Free-text search over historical lineup documents, with a retriever toggle and published retrieval metrics.    |
| **Data Quality & Eval** | Every quality gate, the model's calibration, and what it gets wrong.                                           |

## Why this is hard

Not the modelling — the sample size.

A lineup's offensive rating has a standard error of roughly `115/√n` per 100 possessions,
where _n_ is possessions played together. Most five-man lineups play well under 200
possessions in a season. At that sample the measurement noise on a single lineup is about
as large as the entire spread between good and bad lineups.

Two things follow, and they shape the whole project:

1. **A single lineup's observed rating is not a target you can score a model against.**
   Validation has to pool across lineups, or move down to the shot level where Bernoulli
   observations are plentiful.
2. **The headline product claim is counterfactual by construction.** Forecasting a trade
   means predicting lineups that have never existed. That cannot be validated by
   calibration curves on lineups that have.

<!-- lineupiq:begin id=results.estimability -->
| | Value |
|---|---|
| Distinct five-man offensive lineups | 49,827 |
| Median possessions per lineup | 7 |
| Lineups clearing the 200-possession reporting floor | 485 (1.0%) |
| Lineups above 500 possessions | 129 (0.3%) |
| Share of played time covered by reportable lineups | 50.4% |

At the 200-possession floor a lineup's offensive rating carries a standard error
of roughly +/-8 per 100 possessions, against a true between-lineup spread of about 6-8.
So **99.0% of lineups cannot support a point estimate at all** -- which
is why the refusal contract is a feature of the API rather than an error path.
<!-- lineupiq:end id=results.estimability -->

## What was built

<!-- lineupiq:begin id=results.dataquality -->
| | Value |
|---|---|
| Seasons | 2022-23, 2023-24, 2024-25 |
| Shot attempts | 698,314 |
| Resolved to a complete five-man lineup | 698,006 (99.96%) |
| Lineup solved cleanly (training-grade) | 672,772 (96.34%) |
| Stints reconstructed | 117,766 |
<!-- lineupiq:end id=results.dataquality -->

Play-by-play does not record who is on the floor — only substitutions and who did things.
Recovering the five-man lineup for every event means replaying each period forward from a
starting five that is never stated.

The reconstruction is validated against **box-score minutes**, the one genuinely
independent check available here: the box score comes from a different system, and minutes
played is a physical quantity. A lineup reconstruction can agree with another _derived_
lineup file and still be wrong in the same way; it cannot disagree with the clock and be
right. Derived minutes land within about one second per player-game across ~84,000
player-games, and a player the box score says did not play must derive exactly zero
minutes — that is a hard failure, not a tolerance miss.

Two details carry most of the accuracy:

- **`EVENTNUM` is not chronological.** Sorting by it produces hundreds of impossible states
  per season. Sorting by the game clock instead cut invariant violations by ~91%.
- **Events tied on the clock must be evaluated tolerantly.** A player very often fouls and
  is substituted on the same tick; insisting on one order rejects the true starting five.
  That single detail moved exact solves from 39% to 98%.

## The possession layer, and why it exists

The negative result above is a **target mismatch**, not a modelling failure. Spacing does
not make a player shoot better from the corner; it gets him more open corner threes instead
of contested mid-range. Lineup effects live in shot *selection* and in what a possession is
worth — not in conversion once a shot is taken. Measuring conversion and concluding "lineups
don't matter" answers the wrong question well.

So the foundation was rebuilt at possession grain, which is where those effects can actually
be measured and what any trade projection has to rest on:

<!-- lineupiq:begin id=results.possessions -->
| | Value |
|---|---|
| Possessions | 774,467 |
| Attributed to a five-man lineup | 100.00% |
| Agreement with the independent lineup oracle | 89.95% |
| ... restricted to possessions not starting on a substitution | **97.60%** |
| Possessions starting on a substitution (attribution ambiguous) | 14.3% |
| Points per possession, transition | 1.183 |
| Points per possession, half-court | 1.071 |

The oracle is a second lineup reconstruction, written independently in
another language. Away from substitution boundaries the two agree at the
same rate our period-start solver reports exact solutions. About one
possession in seven begins on the exact second of a substitution, where
there are two defensible answers and no way to choose between them; those
are flagged in the data rather than silently trusted.
<!-- lineupiq:end id=results.possessions -->

## What it refuses to answer

This is the part worth reading, and it is deliberately above the architecture section.

The API has two distinct refusal mechanisms, and it is exact about which fires when:

- **`422 INSUFFICIENT_SUPPORT`** — the claim itself has no basis. A problem document, never
  a 200. It carries the possession count, the threshold, _which_ players fall short, and
  what would help. "Not enough data" without "of what" is not an answer.
- **`200` with `tier: "directional"` and a null point estimate** — the player-level terms
  have support but the lineup-interaction term does not. This is the normal case for a
  post-trade lineup. The interval is populated; the centre mark is not.

Never a 200 with a confident number and a footnote.

The thresholds are pre-registered and hash-pinned in
[`support_thresholds.json`](services/ml/src/lineupiq/configs/support_thresholds.json)
_before_ any lineup-level result was computed, and CI asserts the hash is unchanged — so
loosening a floor to make a demo look better fails the build.

Some things are refused permanently rather than pending. `/api/leaderboards/gravity`
returns `410 METRIC_WITHDRAWN`: gravity needs player-tracking data, this project uses
public play-by-play, and no amount of further work here produces it. `410` rather than
`501` is the point — a client that sees it should stop asking.

## Results

Every number below is rendered from a run log by `lineupiq report render`, and CI fails if
a committed block is stale. Nothing here is typed by hand, and the verdict column is
allowed to say _loses_.

<!-- lineupiq:begin id=results.model -->
**Leave-lineup-out -- unseen five-man combinations** -- n = 406,723 shots

| Model | Log loss | Brier | Resolution | ECE | Cal. slope | Verdict |
|---|---|---|---|---|---|---|
| B0 - league zone mean | 0.66035 | 0.23385 | 0.01564 | 0.0096 | 0.991 |  |
| B1 - shooter x zone (shrunk) | 0.65784 | 0.23269 | 0.01704 | 0.0092 | 0.970 |  |
| B2 - B1 + context, no lineup | 0.65696 | 0.23231 | 0.01734 | 0.0107 | 0.941 |  |
| B3 - additive GBDT, no lineup | 0.65251 | 0.23035 | 0.01945 | 0.0150 | 0.926 |  |
| **full - served closed form** | 0.65683 | 0.23226 | 0.01737 | 0.0105 | 0.939 | +0.019% vs B2 |
| **full - unconstrained GBDT** | 0.65200 | 0.23014 | 0.01961 | 0.0141 | 0.932 | +0.078% vs B3 |

**Walk-forward -- later games** -- n = 404,712 shots

| Model | Log loss | Brier | Resolution | ECE | Cal. slope | Verdict |
|---|---|---|---|---|---|---|
| B0 - league zone mean | 0.65895 | 0.23316 | 0.01556 | 0.0061 | 1.000 |  |
| B1 - shooter x zone (shrunk) | 0.65712 | 0.23231 | 0.01678 | 0.0088 | 0.976 |  |
| B2 - B1 + context, no lineup | 0.65608 | 0.23186 | 0.01715 | 0.0093 | 0.941 |  |
| B3 - additive GBDT, no lineup | 0.65991 | 0.23278 | 0.01792 | 0.0255 | 0.770 |  |
| **full - served closed form** | 0.65609 | 0.23187 | 0.01715 | 0.0103 | 0.938 | -0.003% vs B2 |
| **full - unconstrained GBDT** | 0.65646 | 0.23186 | 0.01813 | 0.0233 | 0.821 | +0.524% vs B3 |

**Cost of the serving constraint:** the closed form the Worker evaluates is 0.74% worse in log loss than the unconstrained gradient-boosted fit on unseen lineups. That is the price of exact Python<->TypeScript parity inside a 10 ms CPU budget, and it is published rather than absorbed.

**Negative control:** with lineup context randomly permuted across shots, the model's log-loss gain over B1 is +0.000796 -- indistinguishable from zero, so the lineup features are not leaking. Control passes.

_Generated from run `dcbcb33` on Windows, seed 20260815, 672,772 shots across 3 seasons._
<!-- lineupiq:end id=results.model -->

### How to read the ladder

Each model is compared against **its own no-lineup counterpart**, not against the best
baseline overall. Comparing the logistic `full` against the boosted `B3` would conflate two
differences at once — model class and lineup information — and let a model-class effect be
reported as a lineup effect. `full` vs `B2` and `full_gbdt` vs `B3` each differ in exactly
one thing: whether the four lineup columns are zeroed.

## Architecture

```
Operator machine (local, free)                    Cloudflare (one Worker)
──────────────────────────────                    ───────────────────────
shufinskiy/nba_data  ─┐
sportsdataverse       ├─> bronze ─> silver ─> gold ──> D1        ─┐
nba_api (optional)   ─┘            (stints)   COMMITTED           ├─> Hono API
                                      │                Vectorize ─┤   /api/*
                          services/ml │                Workers AI ┘      │
                          shot model · calibration · eval                ▼
                                      │                        Next.js static export
                                      └──> run logs ──> README + model cards
```

**The serving constraint, and what it costs.** Workers give 10 ms CPU per request, and the
optimizer accepts any 5 of ~450 players — about 1.5×10¹¹ combinations, so nothing can be
precomputed. The model is therefore split at the lineup boundary: everything depending only
on shooter × zone × season is precomputed offline, and only the lineup terms evaluate at
request time, as a closed form over per-player vectors.

That constraint has a price, and the price is measured and published in the results table
above rather than absorbed silently.

## Quickstart

No API key. No network after the first build. No Cloudflare account. No Snowflake account.

```bash
git clone https://github.com/darthmanwe/LineupIQ && cd LineupIQ

# Python: data and modelling
cd services/ml && uv sync --extra dev
uv run pytest                    # offline and free by default
uv run lineupiq seasons          # declared scope, stated in exactly one place
uv run lineupiq verify           # re-derive every gold checksum + run the DQ gates
uv run lineupiq train --verify   # refit and assert the committed metrics reproduce
uv run lineupiq support          # the pre-registered refusal thresholds

# TypeScript: API and web
cd ../.. && npm ci
npm --workspace apps/api run test     # runs inside workerd
npm --workspace apps/web run build    # static export
npm run dev                           # http://127.0.0.1:8787
```

`lineupiq build` re-ingests from upstream (~88 MB, a couple of minutes). Everything else
runs against committed gold with no network.

Tests that cost money or need a network are behind markers and deselected by default:
`pytest -m net`, `-m repro`, `-m snowflake`, `-m llm`.

## Snowflake

This began as a Snowflake-native design and still supports Snowflake: the medallion schema
names and grains are unchanged, so `SELECT * FROM GOLD.SHOT_FACTS` is valid against either
backend. But **nothing in the demo path touches it**, because a $400/30-day trial cannot
host a portfolio demo.

The original design is kept verbatim in
[`docs/design/00-original-snowflake-design.md`](docs/design/00-original-snowflake-design.md),
unedited, alongside
[`01-portable-rearchitecture.md`](docs/design/01-portable-rearchitecture.md) — which states
what replaced each service, **what was lost**, and eight specific errors in the original,
including an arithmetic mistake in its headline formula and a contradiction about the model.

## What this is not

- **No tracking data.** Shot difficulty is inferred from location and context, not observed
  defender position. There is no gravity metric and no contest quality.
- **EPSA is currently points per shot _attempt_, not per shot _opportunity_.** The free-throw
  component — shooting fouls that never produce a field-goal attempt — is specified but not
  yet built, so the metric understates the value of drawing fouls. Named precisely rather
  than presented as complete.
- **Three seasons.** Nothing here generalises across rule eras, and the era-bucketing in the
  original design is deliberately not built: over this window the column would be constant,
  and shipping a weighting scheme driven by a constant column is the same category of
  overclaim as fabricating data.
- **Lineup synergy is pairwise and low-rank by construction.** That constraint is what makes
  the closed form servable. It is a real limitation, not a free lunch — and on this data it
  turns out to cost almost nothing, because the effect it was constraining is itself nearly
  absent.
- **Shot-selection endogeneity is not solved.** The model treats observed shot mix as
  opportunity; some of it is choice.
- **Nothing here is causal.**

## Roadmap

|     | Milestone                                                                      | State                                                                                |
| --- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| M1  | Skeleton: route registry, refusal plumbing, CI, both toolchains                | **done**                                                                             |
| M2  | Ingest 3 seasons; stint reconstruction validated against box-score minutes     | **done**                                                                             |
| M3  | Shot model, calibration, baseline ladder, leakage assertions, negative control | **done**                                                                             |
| M4  | Pre-registered support thresholds and the refusal contract                     | **partial** — thresholds and tiering built; API wiring and court heatmap outstanding |
| M5  | Trade simulator and the counterfactual backtest                                | not started                                                                          |
| M6  | Retrieval and the LLM evaluation harness                                       | not started                                                                          |
| M7  | Snowflake adapter                                                              | not started                                                                          |
| M8  | Results generated from run logs                                                | **done** — media capture and deploy outstanding                                      |

---

MIT · Kutlu Mizrak · [github.com/darthmanwe/LineupIQ](https://github.com/darthmanwe/LineupIQ)
