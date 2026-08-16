# Portable Rearchitecture

**Status**: active design
**Supersedes**: [`00-original-snowflake-design.md`](00-original-snowflake-design.md)
**Date**: 2026-08-15

---

## The constraint

The original design is Snowflake-native end to end: medallion schemas in a `LINEUPIQ`
database, three Cortex services, and a Streamlit in Snowflake app. It is coherent, and
for a team that already runs Snowflake it would be a reasonable build.

It fails three requirements this project actually has:

1. **It cannot be seen.** Streamlit in Snowflake is not publicly shareable. There is no
   URL to hand someone.
2. **It expires.** A Snowflake Enterprise trial is $400 of credit over 30 days. A demo
   that dies a month after it is built is not a demo.
3. **It cannot be reproduced.** A Kaggle account gate plus a warehouse dependency means
   nobody can clone the repository and get the numbers back.

The rearchitecture keeps the data model and replaces the runtime. The medallion schema
names and grains are preserved exactly, so the Snowflake path stays real â€” it is just no
longer load-bearing.

---

## What replaced each service

Each entry names what was lost, not just what was gained. A migration document
that only lists wins is a sales pitch.

### Snowflake medallion (`BRONZE`/`SILVER`/`GOLD`/`ML`)

**Replaced by** local parquet + DuckDB views, driven by one `TableSpec` registry
that also emits the Snowflake DDL.

Same table names, same grains, so `SELECT * FROM GOLD.SHOT_FACTS` is valid on
both. Runs free on a laptop.

**Lost:** warehouse-scale compute. At three seasons the corpus is ~1.75M events —
a scale where DuckDB is simply faster than a network round trip. This would be a
real loss at 1996–2024.

### Cortex Search

**Replaced by** hybrid BM25 (SQLite FTS5 in D1) + dense retrieval (Vectorize,
384-d) with RRF fusion, plus an offline `rank_bm25` + `hnswlib` mirror for CI.

Managed embedding and serving become two free primitives. The offline mirror is
what lets retrieval evaluation reproduce from a clean clone with no account.

**Lost:** managed index lifecycle. Corpus embeddings are built offline and
uploaded, so changing the corpus is a deploy step rather than a `REFRESH`. It
also introduces a parity risk between local and Workers AI embeddings, which
`verify-embeddings` exists to catch.

### Cortex Analyst (NL→SQL via a YAML semantic model)

**Cut.** Replaced by a bounded intent resolver: natural language → typed
`{intent, slots}` → parameterized SQL selected by id. The model never emits SQL.

Model-authored SQL against a serving database is a correctness and security
liability. The bounded version also has a *better* evaluation story — execution
accuracy against a golden set, with abstention counted separately from failure.

**Lost:** genuine open-ended querying. Eight intents cover the questions the four
pages already answer; anything else returns `422 UNRESOLVED_INTENT` with the
supported list. That is a real capability reduction, chosen deliberately.

### `AI_COMPLETE` structured narratives

**Replaced by** the Anthropic SDK with a Pydantic-enforced output schema and a
content-addressed response cache committed to the repository.

The demo costs $0 and is byte-deterministic. The cache is readable JSON, so
anyone can see exactly what was asked and what came back.

**Lost:** in-warehouse execution. Narratives are generated offline and served
from cache, so they are not live — stated plainly in the README rather than
implied away.

### Feature Store

**Replaced by** `features/asof.py`: a features table keyed
`(player_id, as_of_date)` with a backward as-of join.

Same point-in-time semantics, no Enterprise dependency.

**Lost:** governed, shared feature definitions. With one consumer that is
overhead rather than value.

### Model Registry

**Replaced by** run logs, committed model artifacts, and a `train --verify` gate.

Reproducibility enforced by refitting and comparing is stronger than versioned
storage, because it fails when the numbers move rather than when someone forgets
to bump a version.

**Lost:** rollback and lineage tooling. Replaced by git.

### Streamlit in Snowflake

**Replaced by** a Next.js static export plus Hono on Cloudflare Workers, in one
deploy.

Verified always-on free tier. Streamlit Community Cloud sleeps after 12h idle,
Render after 15 minutes, HF Spaces after 48h; Fly.io removed its free allowance.

**Lost:** zero-config auth and the trivial `session.sql()` data path. Replaced by
an explicit API, which the project needed anyway.

### Tasks (orchestration)

**Replaced by** a CLI (`lineupiq build`) plus CI.

The base data is a static historical load; there is nothing to schedule.

**Lost:** nothing. This was over-engineering for a batch corpus.
---

## The serving constraint, and what it costs

Cloudflare Workers give 10 ms CPU per request. The Lineup Optimizer accepts any 5 of
~450 players â€” C(450,5) â‰ˆ 1.5Ã—10Â¹Â¹ combinations, so nothing can be precomputed.

The model is therefore **split at the lineup boundary**. Everything depending only on
shooter Ã— zone Ã— season is a gradient-boosted prediction baked into a precomputed
`a[i,z]` term offline; only the lineup-interaction terms evaluate at request time, as a
closed form over precomputed per-player vectors (~500 float ops).

Hoops_Lab's rule â€” _"the Worker does no arithmetic"_ â€” cannot transfer here. It is
replaced by something stronger: `serve/parity.py` scores 2,000 random lineups in Python,
and a vitest suite inside `workerd` re-scores each in TypeScript and asserts agreement to
1e-9. CI gates every push on it. Train/serve skew is eliminated by proof rather than by
convention.

**The cost is published, not hidden.** The served closed form is benchmarked against the
unconstrained joint model on the shot-level holdout, and the log-loss gap goes in the
README. Constraining a model to fit a runtime is a normal engineering trade; not
measuring what it cost would not be.

---

## Corrections to the original design

These are errors in `00-original-snowflake-design.md`, listed so the delta is auditable.

**1. The EPSA formula is arithmetically wrong.** Section 3 gives
`EPSA = P(make)Â·points + P(foul_on_shot)Â·E[FT_points]`. This double-counts and-1s (the
field goal _and_ the free throw both score) and omits the missed-shot-plus-foul branch,
which is not a field-goal attempt at all. The corrected decomposition over four mutually
exclusive outcomes:

```
EPSA = (1 âˆ’ p_foul)Â·p_make_nfÂ·pts
     + p_foulÂ·[ p_and1Â·(pts + ft_pct) + (1 âˆ’ p_and1)Â·ft_pctÂ·n_ft ]
```

The _framing_ â€” that free throws drawn on shot attempts are a large share of scoring
value and `FG% Ã— points` misses them â€” is correct and is why this metric is worth
building.

**2. The model specification contradicts itself.** Line 198 specifies LightGBM; line 208
specifies learned player embeddings with pairwise interaction dot products. Gradient-boosted
trees do not learn embeddings, and this mechanism is load-bearing for the entire
lineup-interaction claim. Resolved as three stages: ridge on stints (which _is_ RAPM, and
strips out everything additive), a residual bilinear synergy term fitted by deterministic
L-BFGS, and a GBDT consuming the synergy scalars as ordinary numeric columns.

**3. There was no validation plan for the headline claim.** The design correctly notes
that trade forecasting requires predicting lineups that have never existed, then proposes
an evaluation harness covering shot calibration, RAPM correlation, retrieval precision,
and prompt regression â€” none of which test that. Replaced by leave-lineup-out,
leave-pair-out, and a three-tier historical trade backtest with a pre-registered power
analysis.

**4. `STINT_DOCS` is the wrong retrieval granularity.** The design specifies one document
per stint, then correctly observes two paragraphs later that retrieval quality is bounded
by document quality. A stint is ~90 seconds and ~4 possessions; it has no stable
statistical content, so its embedding encodes noise. Documents are built at
`(lineup_hash, team, season)` grain instead.

**5. The transition flag is not computable as specified.** The rule is "shot clock < 10s
and possession didn't start after a stoppage," but the shot clock is not in the feed.
Replaced with possession-start-type plus seconds-into-possession, validated against an
independent possessions source. The design also asserts 1.12 vs 0.95 PPP â€” that gets
computed, not quoted.

**6. `era_bucket` weighting is inert at this scope.** Over 2022â€“25 the column is constant.
It is kept for schema parity and future backfill, but no era-weighting mechanism is
built. Shipping a weighting scheme driven by a constant column would be the same category
of overclaim as fabricating data.

**7. `PLAY_TYPE_FACTS` has no join key.** It is specified as keyed on `player_id`, but the
Synergy source provides `PLAYER`/`TEAM` name strings. Resolved by sourcing play types from
`nba_api.synergyplaytypes`, which returns `PLAYER_ID` natively â€” eliminating the join
rather than fuzzy-matching it.

**8. The stated coverage is inconsistent.** The design claims 1996â€“2024; the Kaggle slug it
names says 1996â€“2021. Neither is this project's scope. Coverage is declared once in
`seasons.py` and asserted at build time against the `GAME_ID` prefix.

---

## What was kept

The parts of the original design that were right, and survive unchanged:

- **The medallion layering and every table name and grain.** `SHOT_FACTS`, `STINTS`,
  `EVENTS_ENRICHED`, `PLAY_TYPE_FACTS` are as specified.
- **Order-invariant lineup hashing** via `MD5(ARRAY_TO_STRING(ARRAY_SORT(ids), ','))`,
  with one correction: the sort must be numeric, not lexicographic, or the hash differs
  across engines.
- **Tiered stint reconstruction with explicit quality flags.** `VALID` / `IMPUTED` /
  `QUARANTINED` â€” a stint that cannot be determined produces a null lineup and a flag,
  never a guess.
- **The data-quality gates and their thresholds**, including the <2% stint-versus-box-score
  minutes check, which is the genuinely independent validation in the whole pipeline.
- **The cost posture**, applied to a different runtime: everything free, with the spend
  path hard-capped in code rather than by a resource monitor.
- **The Known Limitations section.** It was candid, and its content carries forward into
  the README's "What this is not."
