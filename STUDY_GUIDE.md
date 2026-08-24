# LineupIQ — Study Guide

**Who this is for: me, before an interview.** It is the thing to reread the night before,
and the map for a live walkthrough. It covers what every design decision was, why the
alternative was rejected, what the ML actually does, and which files to open in what order.

There is a **presentation script** at the end. It is written to be spoken.

Every number below is reproduced by the repository. Where a number is quoted here it also
appears in a generated README block or a run log, and `lineupiq report check` fails the
build if the two disagree.

---

## Part 1 — The thirty-second version

> LineupIQ answers "what should these five players shoot, and what does a trade change?"
> from public play-by-play. The engineering headline is that play-by-play never says who is
> on the floor, so the five-man lineup for every event is reconstructed and validated
> against box-score minutes. The ML headline is that I pre-registered what the model should
> find, and published the two places it didn't.

If you remember only three things:

1. **The first model asked the wrong question.** `P(make | shot taken)` finds lineup context
   worth +0.019%. Moving the target to `P(zone | shooter, lineup)` — which shot gets taken —
   quadruples that. Spacing does not make a better corner shooter; it produces a corner
   three instead of a contested pull-up.
2. **A pre-registered coefficient came out backwards, and it stayed published.**
   `spacing_x_three` was written into the source as positive before fitting. It fitted at
   **−0.474**, survived three robustness specifications, and collapsed to −0.017 under the
   negative control — so it is real, and it is the opposite of the hypothesis.
3. **The trade backtest's honest answer is "underpowered".** 146 real mid-season moves
   against a noise floor of 4.31 points per 100 gives a minimum detectable effect of 1.00 —
   the same size as the effects projected. That verdict was computed and committed before
   the result.

---

## Part 2 — Where the project came from

The repository started as **288 lines of design document and zero lines of code**. The
design was good — the EPSA framing, order-invariant lineup hashing, the transition confound,
tiered stint reconstruction — but it had three properties fatal to a portfolio piece:

| Problem                                    | Consequence                                                |
| ------------------------------------------ | ---------------------------------------------------------- |
| Streamlit-in-Snowflake                     | Cannot be shared publicly. A reviewer cannot click a link. |
| Snowflake Enterprise trial, $400 / 30 days | The demo dies a month after it is built.                   |
| Kaggle account gate + warehouse dependency | Nobody can clone and run it.                               |

### Decision 1: portable core, Snowflake as an optional adapter

**Rejected:** building on Snowflake and hoping the trial lasts.
**Chose:** polars + DuckDB + parquet as the real pipeline; Snowflake DDL generated from the
same schema registry and kept entirely off the demo path.

The consequence that matters: **gold is committed to the repo** (~27 MB). A clean clone
reproduces every published number offline — no network, no API key, no Cloudflare account,
no Snowflake. That is a claim the CI actually tests, and it is why `.gitignore` ignores only
`data/bronze/` and `data/silver/`.

### Decision 2: swap the data source

**Rejected:** Kaggle `brains14482` — account gate, and the slug says 1996-2021.
**Chose:** `shufinskiy/nba_data` (GitHub, Apache-2.0, no auth) for play-by-play and shots,
plus sportsdataverse releases as independent oracles. Same upstream data, no gate, pinned by
commit SHA rather than `main` so a vanished upstream is a _stale_ build rather than a broken
one.

### Decision 3: deploy on day two, not at the end

Every later milestone flips a `501` to a `200` on a URL that already exists. The route
registry is declared as data, so what `GET /api` advertises, what the Worker mounts, and
what CI walks are the same list — a route cannot claim to be live and return 501.

---

## Part 3 — The data engineering

This is the part that is hard, and the part most worth walking through.

### The core problem

**Play-by-play never records who is on the floor.** It records substitutions and it records
who did things. Recovering the five-man lineup for every event means replaying each period
forward from a starting five that is never stated anywhere.

### Finding 1: `EVENTNUM` is not chronological

Sorting events by `(game_id, period, EVENTNUM)` produced **435 invariant violations across
120 games** — states with six players on the floor, or a player recorded as acting after
being substituted out.

Sorting by `(game_id, period, seconds_remaining DESC, EVENTNUM ASC)` produced **40**. A 91%
reduction from one line of code.

_Why:_ `EVENTNUM` is an insertion order in the source system, not a game clock.

### Finding 2: per-slot side assignment

`PERSON{n}TYPE` marks which side each player in an event belongs to (4 = home, 5 = away).
The first implementation assigned every assertion in an event to `PLAYER1`'s side.

**Exact period-start solve rate: 0.04%.**

The bug: on a steal, block, or foul, `PLAYER2` and `PLAYER3` are on the _opposing_ team.
Giving each slot its own `side` from its own `PERSON{n}TYPE` took it to **38.81%**.

### Finding 3: the tied-clock window — the single biggest win

Still 38.81%. The remaining failure: **a player very often fouls and is substituted on the
same clock tick.** Insisting on one ordering within a tied-clock cluster rejects the true
starting five.

The fix is a tolerant replay: within a cluster of events sharing a clock second, a player
counts as on-court if he was on-court _before_ the cluster **or** enters during it.

```python
permitted = before | on_court
for a in cluster:
    if a.kind == "EV" and a.player_id not in permitted:
        violations += 1
```

**38.81% → 97.75%.** One insight, a factor of 2.5.

### Finding 4: the possession window is not a possession

This one is the best story, because it corrected something I had already published.

The upstream feed's `start_seconds_remaining` and `end_seconds_remaining` are the clock at a
possession's **first and last recorded event**, not at the changes of hands that bound it.
From the first period of the first game in the corpus:

```
possession 1   720 -> 695   (ends on a turnover)
possession 2   675 -> 675   (the other team, one recorded event)
```

Twenty seconds missing between them, and the second possession's window is a single instant.
Corpus-wide, **45% of possessions had `start == end`**.

Four silent consequences:

- `possession_seconds` was not a duration. Median **2 seconds**.
- The published transition/half-court split was computed on that non-duration.
- 4.4% of shots fell into a gap between windows and could not be placed at all.
- 563 possessions whose first recorded event was at clock zero were dropped entirely.

**The fix:** derive the start from the _previous_ possession's last recorded event (the
change of hands), and from the period's opening clock for the first possession. Derive it
**before** `count_as_possession` filtering, or a dropped possession's time gets spliced onto
its neighbour.

**How I knew it was right** — three independent checks, none of them fitted:

| Check                    | Result                                                       |
| ------------------------ | ------------------------------------------------------------ |
| Median possession length | **14.0s** (mean 14.7s). The NBA's actual figure.             |
| PPP by duration bucket   | 1.452 / 1.217 / 1.056 / 0.807 for 0–7 / 8–14 / 15–24 / 25+ s |
| Oracle agreement         | **89.95% → 95.08%**                                          |

That last row **revised an earlier conclusion of mine.** I had diagnosed the 10-point oracle
disagreement as convention ambiguity at substitution boundaries. About half of it was this
bug. The remaining ~5% does behave like genuine ambiguity — agreement away from boundaries
is 97.4%, matching the period-start solver's own exact-solve rate.

### Finding 5: `OffMadeShot` is not a live-ball start

I had `live_ball_start` include `OffMadeShot`. Median possession length by start type:

| Start type            | Median length |
| --------------------- | ------------- |
| `OffLiveBallTurnover` | 7s            |
| `OffMissedShot`       | 10s           |
| `OffDeadball`         | 16s           |
| `OffMadeShot`         | **17s**       |
| `OffTimeout`          | 18s           |

After the opponent scores you inbound from the baseline with the clock stopped — it behaves
exactly like a timeout. The split is where the durations separate, not a judgement call.

### Finding 6: duration is outcome-contaminated

A possession ends on a made shot **at the shot**, but on a miss **at the rebound** a beat
later. So its length is partly determined by whether the shot went in.

The magnitude is not subtle: **shots that end their possession convert at 93.3%; shots that
do not, at 1.3%** — a non-terminal shot is almost by definition a miss that was rebounded.

So `possession_seconds`, `transition` and `possession_points` are in `FORBIDDEN_FEATURES`.
`seconds_into_possession` is safe — it is the clock at the moment the shot goes up, before
the outcome exists — and it is the model's shot-clock proxy, which the feed does not carry.

**The guard is structural, not advisory.** `build_selection_design` narrows the frame to a
whitelist before computing anything, and the whitelist is itself checked against
`FORBIDDEN_FEATURES`. A forbidden column can neither be read nor added to the list without
failing.

### Finding 7: the two feeds keep the clock at different resolutions

Play-by-play parses a `MM:SS` string to a whole second; the possession feed keeps tenths. A
shot at 501 against a possession ending at 501.4 is the same event.

| Tolerance | Shot coverage |
| --------- | ------------- |
| 0s        | 95.841%       |
| **1s**    | **99.739%**   |
| 2s        | 99.749%       |
| 3s        | 99.759%       |

A boundary problem that resolves at exactly one unit of quantisation and then stops improving
is a rounding artefact, not a tuning parameter. One second is the coarser feed's own
resolution.

### The lineup hash

`MD5(numerically-sorted, comma-joined player ids)`, and **numerically** is load-bearing.
Sorting the string forms puts `1630552` before `201143` because `"1" < "2"`, so any engine
that sorts numerically produces a different hash. A mismatch raises nothing — it returns zero
rows, everywhere, and looks like missing data.

It is implemented three times (Python, TypeScript, DuckDB) and a committed fixture asserts
all of them agree on 2,604 cases including deliberately adversarial id sets.

### Validation philosophy

**Box-score minutes is the one genuinely independent check.** It comes from a different
system and minutes played is a physical quantity. A lineup reconstruction can agree with
another _derived_ lineup file and still be wrong the same way; it cannot disagree with the
clock and be right.

Result: mean |Δ| ~1 second per player-game across ~84,000 player-games, and a player the box
score says did not play must derive exactly zero minutes — a hard failure, not a tolerance
miss.

**Failures are flagged, never dropped.** Every stint carries `VALID` / `IMPUTED` /
`QUARANTINED` plus `lineup_method` and `lineup_confidence`. A guess never becomes a
coefficient: training filters to `VALID` only.

---

## Part 4 — The ML

### The two targets, and why the first was wrong

| Model         | Target                                | Lineup effect (leave-lineup-out)          |
| ------------- | ------------------------------------- | ----------------------------------------- |
| Conversion    | `P(make \| shooter, zone, lineup)`    | +0.019% served                            |
| **Selection** | `P(zone \| shooter, lineup, context)` | **+0.082% served, +0.092% unconstrained** |

Measuring conversion and concluding "lineups don't matter" answers the wrong question well.
The effect lives in shot **selection**.

### Why conditional logit, not multinomial logistic

A multinomial fit gives each of nine zones its own coefficient vector. "Spacing shifts
attempts toward threes" then arrives as a pattern across 45 numbers that has to be eyeballed.

In a conditional logit each hypothesis is **one shared coefficient** on a shot-level driver
interacted with a zone attribute. `spacing_x_three` _is_ the hypothesis, with a sign and a
magnitude and a standard error.

Second reason: ~20 parameters instead of ~200. When the effect being measured might be zero,
there is far less room for the model to manufacture one.

Third: the closed form is trivially servable — nine dot products and a softmax, microseconds
against a 10 ms Worker budget.

### The factored design (an engineering decision inside the model)

A conditional logit's natural design is `(n_shots, n_zones, n_terms)` — 6.3M rows by 20
columns for three seasons. Almost none of it is needed:

- an alternative-specific constant varies only by **zone**
- an interaction is the **outer product** of a shot-level vector and a zone-level vector
- only the two mix terms vary along both axes

Held that way the whole design is a few `(n, 9)` matrices and some vectors, and every
gradient stays a matrix product.

The hot path then collapses ten `np.outer` calls into **one masked add per distinct zone
attribute**, using the identity:

```
Σ_k θ_k b_k[i] a_k[z]  =  Σ_groups ( Σ_{k∈group} θ_k b_k[i] ) a_group[z]
```

**93s → 41s per fit, peak allocation 620 MB → 167 MB.** A test asserts the optimised path is
arithmetically identical to the naive one, because a performance rewrite of the function
being optimised is exactly where a speedup becomes a different model.

### Pre-registered signs — the methodological centrepiece

Every coefficient's expected direction is written into `SELECTION_TERMS` **in the source,
before fitting**. Terms whose direction is genuinely unpredictable carry `expected_sign=None`
rather than a guess, because claiming a direction after seeing the fit makes the audit
worthless.

**Result: 9 of 10 agreed. The marquee one did not.**

`spacing_x_three` — "teammates who shoot threes make _this_ player shoot more threes" — was
pre-registered as **positive**. It fitted at **−0.474**.

Robustness:

| Specification                                       | Coefficient |
| --------------------------------------------------- | ----------- |
| All shooters                                        | −0.485      |
| High-volume shooters only (shrinkage weight ≥ 0.97) | −0.515      |
| Spacing centred **within shooter**                  | −0.447      |
| **Negative control** (lineups randomly reassigned)  | **−0.017**  |

The within-shooter row removes between-player variation entirely and asks only what happens
when _this_ player gets more spacing than he usually has. The control row is the decisive
one: the coefficient dies under shuffling, so it is measuring lineups.

**Interpretation — shot-mix substitution.** A team's attempts live on a simplex. If everyone
shot more threes when surrounded by shooters, the mix would run away. Put four shooters on
the floor and somebody has to attack the rim, and for a given player that somebody is more
often him.

Corroborating detail: `spacing_min_x_three` — the _worst_ spacer on the floor — stays
**positive**. Raising the floor of spacing pushes toward threes; raising the mean pulls this
particular shooter inside. Two coefficients separating in opposite directions is a sign the
parameterisation is doing real work rather than fitting noise.

The pre-registered expectation was wrong. It stays in the source as written, next to the
coefficient that contradicts it.

### Why the negative control is on the coefficient, not just the metric

The usual control checks that a pooled metric collapses under shuffling. But this model makes
a **directional** claim, and a coefficient can stay large while a log loss goes flat. So the
control reports both, and the coefficient going −0.474 → −0.017 is the number that settles it.

### RAPM

Ridge on 747,352 possessions: one row per possession, ten indicator columns (five offence,
five defence), points scored as the target.

Three decisions carry it:

**Separate penalties for offence and defence.** Offensive production is concentrated in a few
players per possession; defensive credit is diffuse. One shrinkage over-shrinks whichever is
which. Selected: λ_off = 2,000, λ_def = 4,000.

**Folds grouped by game, never by possession.** Two possessions from the same game share
lineups, opponent, rest, altitude and that night's shooting variance. Splitting between them
lets the model see its own answer and selects a λ far too small — the classic way a ridge
model is reported as better than it is.

**Reliability by split half, not fit quality.** This is the important one. Ridge always
improves in-sample fit as λ falls, and cross-validated error on possession outcomes is
dominated by shot noise — a model can cut CV error while its player coefficients are close to
arbitrary. Fitting odd and even games separately and correlating the two vectors cannot be
fooled that way.

|         | Split-half r | Spearman ρ | Full-sample (Spearman-Brown) |
| ------- | ------------ | ---------- | ---------------------------- |
| Offence | +0.394       | +0.327     | +0.565                       |
| Defence | +0.422       | +0.359     | +0.594                       |

**Moderate, and published as moderate.** Three seasons is not enough for RAPM to be precise.
That number is more useful to a reader than the leaderboard.

Other diagnostics, all published: effective df 643.5 of 1,541 columns; condition number
215.4; **51 of 770 players flagged as non-identified** because more than 85% of their floor
time is shared with a single teammate — for those the pair's _sum_ is identified and neither
coefficient is, so they are not served as point estimates.

Face validity does hold: Jokić first by a clear margin, then Gilgeous-Alexander and
Antetokounmpo, Draymond Green leading defence. Nothing was fitted to produce that ordering.
But the **possession count** next to each name is the column that matters, and it exists
because a first version printed zeros and hid the one reserve whose +5.5 came off 8,098
possessions against everyone else's 20,000-plus.

### Ridge standard errors — the detail worth knowing

`Var(β̂) = σ² A⁻¹ G A⁻¹` where `A = G + D`, the ridge **sandwich**. The easy mistake is the
OLS form `σ² A⁻¹`, which ignores the penalty's effect on the sampling distribution and
reports intervals that are too narrow for exactly the low-minute players whose estimates are
mostly prior.

And for a trade contrast: `Var(a − b) = Var(a) + Var(b) − 2 Cov(a, b)`. The covariance term
matters — two players who share floor time compete for the same credit, so their estimates
are negatively correlated, and dropping it _understates_ the uncertainty of precisely the
quantity a trade projection is.

### The trade simulator and its backtest

**The minutes rule is a visible input.** How much an arriving player plays is a coaching
decision nothing in this repository can observe, so it is a named parameter, printed next to
every number it produced and returned in the API response.

**The power analysis is computed and committed before any result.**

|                                     | Value            |
| ----------------------------------- | ---------------- |
| Evaluable mid-season moves          | 146              |
| Team net-rating noise (sd)          | 4.31 per 100     |
| **Minimum detectable effect**       | **1.00 per 100** |
| Effects the model actually projects | ~1.0 per 100     |
| Verdict                             | **UNDERPOWERED** |

The MDE is the same size as the effects being claimed. That is not a result to work around;
it is the result.

Design details that matter:

- **Training is strictly pre-move.** One filter — `game_id < cutoff` — does it for every
  team and player. The mover's _pre_-move possessions must stay in: they are the only
  evidence his coefficient has.
- **Difference-in-differences.** Teams that trade are usually underperforming, so their
  rating improves afterwards whether the trade helped or not. The comparison is against teams
  that made no move over the same stretch.
- **Placebo arm.** The identical machinery on players who did _not_ move. Swapping a player
  for himself projects exactly +0.000 — the identity holding. If it drifted, every real
  number beside it would be measuring a pipeline bug.

**The result:** sign agreement 49.3% [41%, 57%] — a coin flip. Correlation with the DiD delta
−0.040. And the sharpest comparison: **projection error 3.74 against a placebo swing of
2.66** — the projection does not beat assuming no change.

**One correction to the plan.** It predicted the minutes assumption would dominate the
projection's variance. It does not — 13% on average, and the player estimates are the larger
term. The design's guess about where the uncertainty lived was wrong, and the decomposition
is published rather than the guess.

### The estimability finding (why the refusal contract exists)

A lineup's offensive rating has a standard error of about `115/√n` per 100 possessions. At
200 possessions that is ±8.1, against a true between-lineup spread of roughly 6–8.

**At the possession counts real lineups accumulate, measurement noise is as large as the
entire signal.**

|                                     | Value      |
| ----------------------------------- | ---------- |
| Distinct five-man offensive lineups | 49,827     |
| Median possessions per lineup       | 7          |
| Clearing the 200-possession floor   | 485 (1.0%) |
| Above 500 possessions               | 129 (0.3%) |

**99% of lineups cannot support a point estimate at all.** That is why the refusal contract
is a feature of the API rather than an error path, and why the thresholds are
**pre-registered and hash-pinned** — loosening a floor to make a demo look better is a build
failure, not a judgement call.

Three outcomes, and the boundaries are the product:

- **`reportable`** — enough evidence for a point estimate.
- **`directional`** — 200 with a **null centre** and a real interval. The player terms have
  support, the combination does not. This is the normal case for a trade lineup.
- **`refused`** — 422 problem document, carrying `n_possessions`, `threshold`,
  `shortfall_players[]` and `what_would_help`. "Not enough data" without "of what" is not an
  answer.

Never a 200 with a confident number and a footnote.

### The court heatmap — a chart that cannot disagree with the model

Four decisions, and the first is the one people get wrong.

**The fill is diverging, not sequential.** The quantity is expected points per attempt _minus
the league average_ — a polarity. Restricted-area value dwarfs corner-three value, so a
light-to-dark ramp on raw expected points would simply redraw the arc and say nothing.

**The geometry is generated in Python, not restated in TypeScript.** Every zone's SVG path
comes from the same constants as `derive_zone`, and a test walks a dense grid asserting that
every point inside an outline is a point the model puts in that zone — 104,229 samples, 100%
agreement off the boundaries. Restating the arc in TypeScript is how a chart ends up
colouring a region the model never scored.

That test earned its place twice: it found 2,245 grid points' worth of **gap** in the upper
corners (the top-three outline cut straight from the arc to the viewBox corner instead of
following the 45° lines out), and a **misplaced label anchor** — the wing-three label sat at
court (−215, 225), where |x| < y, which is top-of-the-key territory.

**Uncertainty lives in the mark.** A zone below the attempt floor renders hatched, with a
dashed edge and no value — not coloured with a caveat in a tooltip nobody opens. Colour
claims a magnitude; a hatch declines to.

Because at league scale no zone is within three orders of magnitude of the floor, the page
ships **two real shooters** side by side rather than a mechanism that never fires: Anthony
Edwards (5,447 attempts, every zone clears) against Trey Alexander (41 attempts, every zone
refused).

**The palette was validated, not eyeballed.** Four steps per arm of the diverging blue↔red
pair, each arm passing the ordinal checks in both modes; poles separating at CVD ΔE 17.7
light / 13.2 dark against a target of 8. The red arm was generated to match the documented
blue arm's OKLCH lightness, so the two arms are perceptually balanced rather than picked.

The surface itself reproduces the modern consensus without being fitted to it: restricted
area +0.239, corner threes +0.065 to +0.078, wing three +0.011, top of the key −0.040, and
mid-range worst at −0.25 to −0.27.

### Retrieval — the design document's own claim, measured

The design document warns that a stint is too short to carry stable statistical content, then
proposes indexing per-stint documents. A stint is 90 seconds and four possessions; its
embedding encodes noise.

Documents sit at `(lineup_hash, team, season)` grain above a possession floor and carry four
things on purpose: **names and role vocabulary** (a query for "stretch big" can only match if
those words exist), **comparatives rather than bare numbers**, **closed-vocabulary style
tags**, and **caveats travelling with the number**.

Then it is measured against the two obvious alternatives on identical facts — 2,410
documents, 45 queries, Recall@10 / MRR / nDCG@10 with BM25:

| Corpus                                           | Recall@10 | MRR   | nDCG@10 |
| ------------------------------------------------ | --------- | ----- | ------- |
| `events` — per-stint log (the original proposal) | 0.398     | 0.417 | 0.395   |
| `numbers` — the same facts as bare decimals      | **0.064** | 0.090 | 0.062   |
| `full` — names, roles, comparatives, caveats     | **0.973** | 1.000 | 0.981   |

**Document design is worth a factor of fifteen.** A query is made of words and a decimal has
no words to match.

**BM25 alone beats the hybrid on two of three metrics** (MRR 1.000 vs 0.939, nDCG 0.981 vs
0.945); the hybrid wins only Recall@10. Reported rather than buried — the corpus is a closed
vocabulary and named entities, which is what lexical matching is best at. Rank fusion pulls
more relevant documents into the top ten and dilutes what sits at the top.

Two honesty notes are part of the output, not the commentary: relevance judgements are
**derived programmatically, not hand-graded** (they measure retrieval of stated facts, a
weaker claim than semantic relevance), and the dense leg is **TF-IDF + truncated SVD, not a
neural embedding model** — chosen because it runs offline from a clean clone.

### Groundedness — and what arithmetic cannot check

Deterministic, offline, no model call, so its verdicts are reproducible and CI needs no key.

**The limit is the finding.** Arithmetic settles _provenance_ and cannot settle _meaning_. A
checker can prove every number in a narrative appears in the evidence and be perfectly
satisfied by a sentence quoting the right number for the wrong quantity.

The sibling project measured exactly that: its regex traced 1,027 of 1,027 tokens, raised no
flags, and scored **Cohen's κ = 0.00** against human labels — not a broken checker, but a
detector with no positives, which cannot agree beyond chance.

So the checks split in two. Numeric traceability is cheap and nearly always satisfied. The
four **semantic** checks catch real errors: an invented zone, a player who was not on the
floor, a direction stated backwards, and — the hard one — **a point estimate asserted for a
lineup whose tier forbids one**, where every number is correct and the sentence is still
wrong.

Two negative controls, because a checker that accepts everything also scores 1.00: re-score
against another lineup's evidence (easy), and against the same lineup with one player swapped
(near-miss). The near-miss is the honest number.

**Measured, over 200 lineup documents and three narrative templates:**

| Narrative                             | Grounded   | Numeric traceability | Easy control | Near-miss control |
| ------------------------------------- | ---------- | -------------------- | ------------ | ----------------- |
| faithful                              | **100.0%** | 100.0%               | 0.5%         | 0.5%              |
| overclaiming — _only correct numbers_ | **50.0%**  | 100.0%               | 0.5%         | 0.0%              |
| hallucinating                         | **1.5%**   | 100.0%               | 0.0%         | 0.5%              |

Read the second row. Traceability is **100%** — every figure appears in the evidence — and
half the narratives are still wrong, because 100 of them assert a point estimate for a lineup
below the reporting floor. A harness reporting only traceability would score that row at 100%
and publish it as a pass.

Getting the first row to 100% cost two bug fixes, both found by running it: the checker
flagged the "100" in "points per 100 possessions" as an ungrounded number, which would fail
every correct narrative; and its name extractor could not parse Caldwell-Pope,
Gilgeous-Alexander or Hardaway Jr. — 36 false positives on correct prose. A checker that flags
correct prose is worse than no checker, because the noise buries the real failures.

---

## Part 5 — Architecture

```
Operator machine (local, free)                Cloudflare (one Worker)
──────────────────────────────                ───────────────────────
shufinskiy/nba_data ─┐
sportsdataverse ─────┤
                     ▼
              bronze (gitignored)
                     │  content-addressed, resumable, manifest-backed
                     ▼
              silver (gitignored)
                     │  typed events, stints, period solutions
                     ▼
         ┌──── gold (COMMITTED, ~27 MB) ────┐
         │  shot_facts, stints, dim_player, │
         │  possession_facts, player_rapm   │
         │  + _contracts/*.json fingerprints│
         └──────────────┬───────────────────┘
                        │  lineupiq export
                        ▼
              apps/web/public/data/*.json ──────► Workers Assets
                        │                              │
                        │                         Hono on Workers
                        │                         /api/* (10 ms CPU)
                        ▼                              │
              sql/snowflake/*.ddl                      ▼
              (generated, off the demo path)      Next.js static export
```

### Why the model is split at the lineup boundary

Two requirements collide. The model wants a boosted stack; the free always-on host gives
**10 ms CPU per request**, and the optimizer accepts any 5 of ~450 players — C(450,5) ≈
1.5×10¹¹, so nothing can be precomputed.

Resolution: everything depending only on shooter × zone × season is baked offline. Only the
lineup-interaction terms evaluate at request time — nine dot products and a softmax.

**The cost is published, not absorbed.** The served closed form is benchmarked against the
unconstrained gradient-boosted fit and the log-loss gap goes in the README.

### Why parity is proved, not assumed

The Worker re-implements the lineup hash and the support tier. Neither disagreement raises
anything: a hash mismatch returns zero rows everywhere and looks like missing data; a tier
mismatch serves a confident number where Python would have refused.

So Python writes its answers for 2,604 cases to a committed fixture and a vitest suite
running **inside workerd** asserts TypeScript reproduces them. All three tiers are exercised
— and getting `reportable` cases into the fixture required sampling _real_ lineups, because
random five-player draws produce **zero** of them.

### The generated-numbers rule

Every published number lives between `<!-- lineupiq:begin id=... -->` sentinels.
`lineupiq report render` rewrites only the marked blocks from run logs; `report check`
re-renders into memory and fails on any difference. **Humans own the prose; the tool owns
every number.**

This exists because of an observed failure in the sibling repository: its `--verify` compared
refits against a run log but never against its README, so the build stayed green for months
while the published table described an older model.

---

## Part 6 — The crash, and the hardening

Worth telling because it is an honest operational story.

**A training run drove the machine into swap and took the desktop down.**

**Root cause:** the conditional logit's objective allocated **thirteen 671,255 × 9 float64
matrices per L-BFGS iteration** (~620 MB of churn), across ~190 iterations × 18 fits, while
also holding a 150 MB dense feature matrix _and_ a full copy of it. Launched in the
background alongside `npm test`.

**Four layers now, because fixing one hot path is not a guarantee about the next model:**

| Layer                | What it does                                                                   |
| -------------------- | ------------------------------------------------------------------------------ |
| Fixed hot path       | Ten outer products → one masked add per zone attribute; buffers reused         |
| **OS-enforced cap**  | Windows Job Object, `JOB_OBJECT_LIMIT_PROCESS_MEMORY`, default 6 GB            |
| Measured pre-flight  | Reads actual peak from the job object and refuses if already over half the cap |
| Bounded thread pools | OMP/OPENBLAS/MKL/POLARS capped at 4 in `__init__.py`, before numpy loads       |

The cap **refuses to run** if it cannot be applied; `--allow-uncapped` is an explicit
override. Verified by a subprocess test, not assumed.

**Two bugs found while building it:**

1. **ctypes truncated the handle.** `ctypes` types an undeclared return value as a 32-bit
   int, which truncates a 64-bit Windows `HANDLE`; every subsequent Job Object call failed
   with `ERROR_INVALID_HANDLE` — which reads exactly like a permissions problem and is not
   one.
2. **My memory estimate modelled the wrong thing.** It predicted 314 MB and the process had
   already committed **3,723 MB before fitting anything** — because the possession-context
   join was an equi-join that fanned out to ~8M rows before filtering. Replacing it with an
   **as-of join** (linear merge on elapsed time) took peak to 1,685 MB with identical
   coverage. _An estimate that models only the part you wrote is worse than no estimate._

Also added: progress output and **partial run-log checkpoints after each CV split**, so a
20-minute job interrupted at minute 18 keeps its finished folds. That earned its place
immediately — the very next run was killed and the walk-forward results survived.

---

## Part 7 — Bugs I found in my own work

Keep this list. It is the most credible part of the story.

| Bug                                                   | How it was caught                         | Why it mattered                                        |
| ----------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| Possession windows were event spans, not durations    | Median duration of 2s is not a possession | 45% zero-length; published split was on a non-duration |
| Oracle disagreement blamed on convention              | Fixing the start moved 89.95% → 95.08%    | Half the disagreement was my bug                       |
| `OffMadeShot` treated as live-ball                    | Median length 17s, same as a timeout      | Wrong transition classification                        |
| Duration used as a feature                            | 93.3% vs 1.3% conversion split            | Outcome leakage into a selection model                 |
| Offence attribution defaulted to "away"               | Reading the fall-through branch           | Mis-attributed, _and_ made facts depend on the oracle  |
| `content_hash` segfaulted                             | `0xC0000005` on 1.5M nested-list rows     | Iterating as Python objects exhausted memory           |
| RAPM leaderboard printed zero possessions             | Reading the output                        | Hid a +5.5 built on 8,098 possessions                  |
| `nan` silently became `0.0`                           | Variance decomposition read "0%"          | Reported a decomposition never computed                |
| Placebo sign agreement read 0.0%                      | `sign(0)` matches neither ±1              | A meaningless statistic on the arm that matters        |
| Exported JSON contained bare `NaN`                    | Worker test could not parse its fixture   | **Invalid JSON — a 500 in production**                 |
| Name extractor could not parse hyphenated names       | 36 false positives on faithful narratives | A checker flagging correct prose buries real failures  |
| Groundedness flagged "per 100" as ungrounded          | Writing the test                          | Would fail every correct narrative                     |
| Parity fixture had zero `reportable` cases            | The generator's own warning               | A branch never exercised cannot be proved to agree     |
| First ablation conflated model class with lineup info | Reading the ladder                        | Reported a model-class effect as a lineup effect       |
| Court heatmap had 2,245 grid points of gap            | The geometry agreement test               | Upper corners belonged to no zone                      |
| Wing-three label sat in top-of-the-key                | The label-anchor test                     | A chart mislabelling its own regions                   |
| `points_per_attempt` averaged the shot's face value   | Every zone came out 2.000 or 3.000        | A "surface" that was just the arc redrawn              |
| UTF-8 double-encoding in three files                  | `Ã‚Â±8` in the rendered README            | My own PowerShell patching through a legacy code page  |

The last one is why I stopped using shell string-patching on non-ASCII files, and why the
report renderer is pure ASCII.

---

## Part 8 — What is not built, and what I would do next

**Not built** (endpoints return `501` naming what will back them):

- Live narrative generation — the writer/judge pair, the committed content-addressed cache,
  and human-labelled judge agreement. **A language model has never been called by this
  repository.**
- The `/ask` intent resolver.
- Hand-graded retrieval queries (40 of them). Programmatic judgements are the honest
  substitute and are labelled as such.
- The Workers AI dense retrieval leg.
- The **per-lineup** surface: picking five players and scoring for them needs the served
  closed-form scorer. The court heatmap itself is live, at league scale and for two worked
  examples. The trade and evidence pages remain static shells.
- Playwright media capture, and the deploy itself — `wrangler login` is an interactive
  browser OAuth flow.

**What I would do next, in order:**

1. **A minutes model.** The trade projection's biggest weakness is not the player estimates —
   it is that minutes are assumed. But the variance decomposition says minutes carry only
   13%, so this is a _correctness_ improvement rather than a precision one.
2. **More seasons.** Every underpowered result here is underpowered because of sample size.
   RAPM reliability at 0.4 and an MDE of 1.00 both improve with `√n`, and the pipeline is
   already season-parameterised.
3. **Shot-selection at possession grain.** The selection model conditions on an attempt
   happening. Modelling _whether_ a possession produces an attempt, and where, would close
   the loop to points per possession — which is what the product actually claims.

**What I would cut if asked to ship faster:** the Snowflake adapter and the `/ask` resolver.
Neither is on the demo path.

---

## Part 9 — Presentation script

Roughly 12 minutes with the files open. Timings are for pacing, not precision.

### Opening (1 min) — no files

> "LineupIQ forecasts what a five-man NBA lineup should shoot and what a trade changes, from
> public play-by-play. I want to show you two things: a data-engineering problem that took
> five iterations to get right, and a modelling result that came out the opposite of what I
> predicted — which I published rather than reframed.
>
> Everything reproduces from a clean clone with no network, no API key, and no cloud account.
> Gold is committed and every published number is generated from a run log, so the README
> cannot drift from the model."

### Beat 1 (2 min) — the reconstruction · open `services/ml/src/lineupiq/transform/stints.py`

> "Play-by-play never says who is on the floor. It says who was substituted and who did
> things. So the five-man lineup for every event has to be reconstructed by replaying each
> period forward from a starting five that is never stated.
>
> Three findings took the exact-solve rate from 0.04% to 97.75%."

Point at the `side` field on `CourtAssertion`:

> "First: on a steal or a foul, `PLAYER2` is on the _opposing_ team. Assigning every
> assertion to `PLAYER1`'s side gave me 0.04%. Per-slot sides took it to 38.8%."

Scroll to `_replay` and the `permitted = before | on_court` line:

> "Second, and this is the big one — players constantly foul and get substituted on the same
> clock tick. Insisting on one ordering inside a tied-clock cluster rejects the true starting
> five. Being tolerant within the cluster took it from 38.8% to 97.75%. One insight, factor
> of two and a half."

Then `transform/events.py`, `canonical_order`:

> "Third: `EVENTNUM` is not chronological — it is insertion order in the source system.
> Sorting by it gave 435 impossible states over 120 games. Sorting by the game clock gave
> 40."

### Beat 2 (2.5 min) — the possession bug · open `services/ml/src/lineupiq/transform/possessions.py`

> "This is the story I'd most want you to hear, because it corrected something I had already
> published."

Read the `possession_windows` docstring aloud — the `720 -> 695 / 675 -> 675` example.

> "The feed's start and end are the clock at a possession's first and last _recorded event_,
> not the changes of hands. Forty-five percent of possessions had start equal to end. So
> `possession_seconds` was not a duration — median two seconds — and I had already published
> a transition/half-court split computed on it.
>
> I derived the start from the previous possession's last event instead. Three checks, none
> of them fitted: median possession length came out at fourteen seconds, which is the
> league's actual number. Points per possession fell monotonically with possession length.
> And agreement with an independent lineup oracle went from 89.95% to 95.08%.
>
> That last one revised my own earlier conclusion. I had blamed the disagreement on
> convention ambiguity at substitution boundaries. About half of it was this bug."

Then jump to `FORBIDDEN_FEATURES` in `eval/leakage.py`:

> "And once duration was real, I realised it was unusable as a feature. A possession ends on
> a make _at the shot_ and on a miss _at the rebound_. Shots that end their possession
> convert at 93%; shots that don't, at 1%. So duration is in the forbidden list — and the
> guard is structural: the design matrix is narrowed to a whitelist before anything is
> computed, and the whitelist itself is checked against the forbidden list."

### Beat 3 (3 min) — the pre-registered sign · open `services/ml/src/lineupiq/models/selection.py`

> "My first model asked whether a shot goes in, and found lineup context worth 0.019%. That
> is a target mismatch, not a null result. Spacing doesn't make you a better corner shooter —
> it gets you a corner three instead of a contested pull-up. So I moved the target to _which_
> shot gets taken."

Scroll to `SELECTION_TERMS` / `LINEUP_TERMS`:

> "It's a conditional logit rather than a multinomial one, so each hypothesis is a single
> shared coefficient rather than a pattern across 45 numbers. And every coefficient's
> expected sign is written down here, in the source, before fitting."

Point at `spacing_x_three`, `expected_sign=1`:

> "This is the marquee one: teammates who shoot threes should make _this_ player shoot more
> threes. I wrote `+1`.
>
> It fitted at **minus** 0.474."

Then the README's sign-audit table:

> "Nine of ten agreed. This one didn't, and it survives everything: high-volume shooters only,
> minus 0.515. Centred within shooter, which removes between-player variation entirely, minus
> 0.447. And under the negative control — same five players randomly reassigned to other
> attempts — it collapses to minus 0.017. So it's measuring lineups, and it's the opposite of
> my hypothesis.
>
> The reading is shot-mix substitution. A team's attempts live on a simplex; if everyone shot
> more threes when surrounded by shooters the mix would run away. Put four shooters out there
> and somebody has to attack the rim.
>
> And the corroboration is in the neighbouring coefficient: `spacing_min_x_three`, the
> _worst_ spacer on the floor, stays positive. Raising the floor pushes toward threes;
> raising the mean pulls this shooter inside."

_(If they push back — "isn't this just role assignment?" — the within-shooter row is the
answer: it strips between-player variation and the effect gets stronger, not weaker.)_

### Beat 4 (2 min) — the underpowered verdict · open the README trade block

> "The trade simulator is the product's headline claim, so it gets a backtest against 146
> real mid-season moves. The power analysis is computed and committed _before_ any result."

Point at the MDE row:

> "Team net rating swings 4.31 points per 100 across an arbitrary mid-season cutoff. That
> gives a minimum detectable effect of 1.00 — the same size as the effects I'm projecting.
> Verdict: underpowered. No accuracy claim follows.
>
> And the placebo arm settles it. I run the identical machinery on players who _didn't_ move.
> Swapping a player for himself projects exactly zero, which is the identity holding. But
> those placebos still swing 2.66 points per 100 — and my projection error on real moves is
> 3.74. So the projection does not beat assuming no change.
>
> I'd rather show you that than a lucky 62% sign accuracy on sixty trades."

### Beat 5 (1.5 min) — the refusal contract · open `apps/api/test/live.test.ts`

> "Ninety-nine percent of five-man lineups never play 200 possessions together. At that
> sample the measurement noise is as large as the entire spread between good and bad lineups.
> So refusing is a feature, not an error path."

Point at the 422 test:

> "Three outcomes. Reportable gets a point estimate. Directional gets a 200 with a _null_
> centre and a real interval — that's the normal case for a trade lineup, where the player
> terms have support and the combination doesn't. And refused gets a 422 carrying
> `what_would_help`, because 'not enough data' without 'of what' isn't an answer.
>
> Never a 200 with a confident number and a footnote. The thresholds are pre-registered and
> hash-pinned, so loosening one to make a demo look better is a build failure."

Then `apps/api/test/parity.test.ts`:

> "And the Worker re-implements the tier decision and the lineup hash, so Python writes its
> answers for 2,604 cases to a committed fixture and this suite — running inside workerd —
> asserts TypeScript reproduces them. Getting `reportable` cases into that fixture needed real
> lineups; random five-player draws produce zero of them."

### Close (1 min) — no files

> "Two things I'd point at as the actual engineering judgement.
>
> One: a training run took my desktop down. The fix wasn't just the hot path — it was a
> hard OS-level memory cap that refuses to run if it can't be applied, and a _measured_
> pre-flight check. My original estimate said 314 MB; the process had already committed 3.7
> GB before fitting anything, because a join was fanning out to eight million rows. An
> estimate that models only the part you wrote is worse than no estimate.
>
> Two: the list of bugs I found in my own published numbers is in the study guide, and it's
> the part I'd want you to read. The possession bug had already shipped. The oracle
> conclusion had already been written down. Both got corrected in public."

### Questions to expect

| Question                                             | Answer                                                                                                                                                                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "Why not just use a neural embedding for retrieval?" | It would need a model download, so the evaluation stops being reproducible offline. LSA is labelled as LSA everywhere and the deployed Workers AI leg is stated as unmeasured.                                     |
| "Isn't 0.08% a trivial effect?"                      | Yes, and that's reported as the headline. The interesting result is directional, not magnitude — and the negative control is what makes the direction trustworthy.                                                 |
| "Why MD5?"                                           | It's an identity function, not a security primitive, and it's what the Python side and the Snowflake expression use. The risk isn't the digest — it's the sort order, which is numeric and covered by the fixture. |
| "RAPM reliability of 0.4 is low."                    | It is, and it's published instead of the leaderboard. Three seasons isn't enough. That's the honest ceiling on everything downstream, which is why the trade backtest is underpowered too.                         |
| "What would you do differently?"                     | Start at possession grain. I built a shot-conversion model first and it answered the wrong question well.                                                                                                          |

---

## Appendix — file map for a walkthrough

**Open in this order.** Every one of these is worth reading aloud from.

| #   | File                                                  | Why                                                                        |
| --- | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| 1   | `README.md`                                           | Headline finding, estimability table, generated results                    |
| 2   | `services/ml/src/lineupiq/transform/stints.py`        | The reconstruction. `side`, `_replay`, the tolerant window                 |
| 3   | `services/ml/src/lineupiq/transform/events.py`        | `canonical_order` — `EVENTNUM` is not chronological                        |
| 4   | `services/ml/src/lineupiq/transform/possessions.py`   | `possession_windows` — the best bug story                                  |
| 5   | `services/ml/src/lineupiq/eval/leakage.py`            | `FORBIDDEN_FEATURES` and why each entry is there                           |
| 6   | `services/ml/src/lineupiq/models/selection.py`        | `SELECTION_TERMS`, the pre-registered signs, the factored design           |
| 7   | `services/ml/src/lineupiq/models/rapm.py`             | Separate λ, game-grouped folds, split-half reliability, the ridge sandwich |
| 8   | `services/ml/src/lineupiq/eval/backtest_trade.py`     | Pre-move training, DiD, the placebo arm                                    |
| 9   | `services/ml/src/lineupiq/models/moves.py`            | `power_analysis` and `CLAIMED_EFFECT_PER_100`                              |
| 10  | `services/ml/src/lineupiq/models/support.py`          | The three tiers                                                            |
| 11  | `apps/api/test/live.test.ts`                          | The 422, and the never-a-point-estimate invariant                          |
| 12  | `apps/api/test/parity.test.ts`                        | Cross-language parity over 2,604 cases                                     |
| 13  | `services/ml/src/lineupiq/transform/zone_geometry.py` | Court geometry, generated from the model's own constants                   |
| 14  | `apps/web/src/components/court/CourtHeatmap.tsx`      | Diverging fill, hatch below the floor, every zone labels its n             |
| 15  | `services/ml/src/lineupiq/runtime.py`                 | The memory cap, and the ctypes handle bug                                  |
| 16  | `services/ml/tests/test_support.py`                   | The pinned threshold hash — the pre-registration, enforced                 |
| 17  | `services/ml/src/lineupiq/report/render.py`           | Why numbers are generated, never typed                                     |
| 18  | `docs/modeling.md`                                    | Every correction, with its magnitude                                       |
| 19  | `.github/workflows/ci.yml`                            | 13 jobs, all offline and free                                              |

**Quick commands to run live:**

```bash
cd services/ml
uv run pytest -q                      # 151 tests, offline, no key
uv run lineupiq verify                # 13 contracts, 12 gates
uv run lineupiq report check          # the README is not stale
uv run lineupiq seasons               # scope, declared once
cd ../.. && npm --workspace apps/api run test   # 59 tests inside workerd
```
