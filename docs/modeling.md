# Modelling notes

Why the models are shaped the way they are, and what had to be corrected on the way.
Numbers quoted here are reproduced by `lineupiq verify`, `lineupiq train` and
`lineupiq selection` from committed gold.

---

## Two targets, and why the first one was the wrong question

The original design had one model: `P(make | shot, lineup)`. It was built, scored against a
full baseline ladder, and the honest answer came back — on unseen five-man combinations,
knowing the other four players improves log loss by **+0.02%** for the served closed form
and **+0.06%** for the unconstrained gradient-boosted fit. Against a passing negative
control, that is nothing.

That result is correct and it is not interesting, because it answers a question nobody was
asking. Spacing does not make a player a better corner shooter. **It gets him a corner three
instead of a contested pull-up.** Conditioning on "a shot was taken from here" throws away
the entire channel through which lineup construction operates.

So there are two targets:

| Model                    | Target                                   | What a lineup effect would look like             |
| ------------------------ | ---------------------------------------- | ------------------------------------------------ |
| Conversion (`train`)     | `P(make \| shooter, zone, lineup)`       | Better shooting from the same spot. Barely exists. |
| Selection (`selection`)  | `P(zone \| shooter, lineup, context)`    | A different distribution of attempts.            |

Reporting the first without the second would have been a real result presented as if it
settled a question it never touched.

### Form of the selection model

A **conditional logit** over the nine zones, not a multinomial logistic regression.

A multinomial fit gives each zone its own coefficient vector, so a claim like "spacing
shifts attempts toward threes" has to be read off a pattern across 45 numbers. In a
conditional logit the same claim is one shared coefficient on a shot-level driver
interacted with a zone attribute: `spacing_x_three` is the hypothesis, with a sign and a
magnitude. Roughly twenty parameters instead of two hundred also matters when the effect
under test might be zero — there is much less room for the model to manufacture one.

The design is stored **factored**, never materialised. A conditional logit's natural design
is `(n_shots, n_zones, n_terms)`, which for three seasons is 6.3M rows by 20 columns.
Almost none of it is needed: an alternative-specific constant varies only by zone, and an
interaction is the outer product of a shot-level vector and a zone-level vector. Only the
two mix terms vary along both axes. Held that way the design is a few `(n, 9)` matrices and
some vectors, and every gradient stays a matrix product.

The gradient is derived by hand and **checked against finite differences in the test
suite**. A wrong analytic gradient does not raise: L-BFGS follows it to a nearby point,
reports success, and prints a plausible log loss.

### Pre-registered signs

Every coefficient's expected direction is written into `SELECTION_TERMS` in the source,
before fitting. Terms whose direction is genuinely not predictable in advance carry
`expected_sign=None` rather than a guess — claiming a direction after seeing the fit would
make the whole audit worthless.

This is not decoration. On the first full fit **nine of ten pre-registered signs agreed and
the marquee one did not**: `spacing_x_three`, the "shooters around you mean more threes for
you" hypothesis, came out negative. Three specifications later it is still negative:

| Specification                                          | `spacing_x_three` |
| ------------------------------------------------------ | ----------------- |
| All shooters                                           | −0.485            |
| High-volume shooters only (shrinkage weight ≥ 0.97)    | −0.515            |
| Spacing centred **within shooter**                     | −0.565            |

The last row is the cleanest identification available: it removes the between-player
component entirely and asks only what happens when *this* player gets more spacing than he
usually has. The effect gets stronger, not weaker, so it is not an artefact of shrinkage or
of roster construction.

The substantive reading is **shot-mix substitution**. A team's attempts live on a simplex —
if everyone shot more threes when surrounded by shooters, the mix would run away. Put four
shooters on the floor and somebody has to attack the rim, and for a given player that
somebody is more often him. Note that `spacing_min_x_three` — the *worst* spacer on the
floor — stays positive: raising the floor of spacing does push toward threes, while raising
the mean pulls this particular shooter inside. Two coefficients separating in opposite
directions is a sign the parameterisation is doing real work.

The pre-registered expectation was wrong. It stays in the source as written.

### Standard errors, a third verdict, and a prediction of mine that was wrong

The audit could originally say two things: *agrees* or *DISAGREES*. That is weaker than it
looks, because a coefficient of +0.097 whose interval spans zero would be recorded as agreeing
with a positive prediction while agreeing with nothing at all. So the served fit now carries a
covariance, and a coefficient whose 95% interval spans zero is **indeterminate** — counted as
neither, which can only make the audit harder to pass.

**I built it expecting two terms to fall into that bucket, and neither did.** The reasoning was
that `spacing_min_x_three` at +0.097 and `live_ball_x_rim` at +0.087 are small, so they were
probably not distinguishable from zero. That confuses a coefficient's *size* with its
*precision*:

| Term                        | Coefficient |     SE |     z | Verdict |
| --------------------------- | ----------- | ------ | ----- | ------- |
| `shooter_mix`               |     +0.9958 | 0.0028 | 351.3 | agrees  |
| `into_possession_x_rim`     |     −0.3691 | 0.0048 | −77.0 | agrees  |
| `second_chance_x_rim`       |     +0.2529 | 0.0063 |  40.1 | agrees  |
| `team_mix`                  |     +0.3420 | 0.0098 |  35.0 | agrees  |
| `opp_three_allowed_x_three` |     +1.7604 | 0.0531 |  33.2 | agrees  |
| `opp_rim_allowed_x_rim`     |     +1.4821 | 0.0468 |  31.7 | agrees  |
| `teammate_rim_x_rim`        |     −0.6780 | 0.0414 | −16.4 | agrees  |
| `live_ball_x_rim`           |     +0.0871 | 0.0062 |  14.0 | agrees  |
| `spacing_x_three`           |     −0.4740 | 0.0453 | −10.5 | **DISAGREES** |
| `spacing_min_x_three`       |     +0.0968 | 0.0243 |   4.0 | agrees  |

Nothing is indeterminate. The smallest `|z|` in the model is 4.0. At 671,251 attempts against
twenty parameters there is simply an enormous amount of evidence about each one, and a
coefficient of +0.087 sits fourteen standard errors from zero.

Two things follow, and they are the reason this was worth building even though the bucket came
back empty.

**The pre-registration failure is not a marginal call.** `spacing_x_three` is ten and a half
standard errors below zero. It was not a coefficient that wandered across the axis; the
hypothesis is decisively wrong in the stated direction.

**And significance says nothing about magnitude.** Every coefficient here is overwhelmingly
significant, and the whole shot-mix effect is still worth a standard deviation of 0.19 points
per 100 attempts. Those two facts are not in tension — they are the same fact seen from two
sides, and reporting only the first is how a p-value becomes an overclaim. The pricing table
is the honest companion to this one, and neither should be read alone.

The estimator is the **ridge sandwich** `H⁻¹ I H⁻¹`, not the inverse Hessian. The fit is
penalised (`l2 = 1e-4` on the non-constant terms), so a plain inverse would describe a
different estimator than the one that produced these numbers — shrinkage trades bias for
variance and the sandwich accounts for both sides. RAPM in this repository already uses it, for
the same reason.

`I` comes from central differences on the **analytic** gradient rather than second differences
of the loss: `2p` evaluations instead of `2p²`, keeping the significant digits the other route
throws away. The L2 term is removed arithmetically — its gradient is exactly `l2 · mask · θ` —
rather than through a second code path that could drift from the optimiser's. The result is
symmetrised, because finite differences are symmetric only to truncation error and an
asymmetric Hessian produces negative variances that surface as a nan much later and look like
a data problem. It is computed on the served fit only; cross-validation fits this model
eighteen times per pass and none of those needs a covariance.

**None of the six tests asserts a value.** A covariance never raises, and one wrong by a factor
of two still looks like a standard error. So they assert properties instead: the
finite-difference Hessian agrees with a second-difference Hessian of the loss, an independent
route sharing no code beyond the objective; the standard errors shrink as `1/√n`, which fails
by orders of magnitude if the `1/n` is missing or applied twice; the covariance is symmetric
and positive definite; and on random data, where every lineup coefficient is truly zero, the
audit returns `indeterminate` rather than crediting whichever sign the noise produced. That
last test is why the empty bucket here can be trusted as a finding rather than suspected as a
bug.

---

## Corrections to the possession layer

The possession layer was rebuilt after the conversion model's null result, and then
corrected twice more. Both corrections were found by checking derived quantities against
things already known, not by any test failing.

### 1. The feed's possession window is not a possession

`start_seconds_remaining` and `end_seconds_remaining` are the clock at a possession's
**first and last recorded event**, not at the changes of hands that bound it. From the first
period of the first game in the corpus:

```
possession 1   720 -> 695   (ends on a turnover)
possession 2   675 -> 675   (the other team, one recorded event)
```

Twenty seconds are missing between them, and the second possession's recorded window is a
single instant. Corpus-wide, **45% of possessions had `start == end`**.

Consequences, all silent:

- `possession_seconds` was not a duration. Median 2s.
- The published transition/half-court split was computed on that non-duration.
- 4.4% of shots fell into a gap between windows and could not be placed at all.
- Possessions whose first recorded event was at clock 0 were dropped by the stint join
  entirely — 563 of them.

**Fix.** The change of hands is the previous possession's last recorded event, so the start
is derived from that, and from the period's opening clock for the first possession of a
period. The window must be derived **before** `count_as_possession` filtering, or a dropped
possession's time is spliced onto its neighbour.

**Check.** Derived durations have a median of 14.0s and a mean of 14.7s. The NBA has
averaged close to fourteen seconds a possession for years. And PPP falls monotonically with
possession length — 1.452 at 0–7s, 1.217 at 8–14s, 1.056 at 15–24s, 0.807 beyond — which is
the early-offence gradient every basketball source reports. Nothing was fitted to produce
either.

The oracle agreement improved as a side effect, which is the strongest evidence the new
start is the right one: attributing possessions to stints at the change of hands rather than
at the first recorded event moved agreement with the independent reconstruction from
**89.95% to 95.08%**, and cut the share of possessions landing ambiguously on a substitution
from 14.3% to 9.0%.

That also revises an earlier conclusion in this repository. The 10-point disagreement was
diagnosed as convention ambiguity at substitution boundaries; about half of it was a wrong
possession start. The remaining ~5% does behave like genuine ambiguity — agreement away from
boundaries is 97.4%, matching the period-start solver's exact-solve rate.

### 2. `OffMadeShot` is not a live-ball start

`live_ball_start` originally included `OffMadeShot`. Median possession length by start type
says otherwise:

| Start type            | Median length |
| --------------------- | ------------- |
| `OffLiveBallTurnover` | 7s            |
| `OffMissedShot`       | 10s           |
| `OffDeadball`         | 16s           |
| `OffMadeShot`         | 17s           |
| `OffTimeout`          | 18s           |

A possession beginning after the opponent scores is inbounded from the baseline with the
clock stopped, and it behaves exactly like a timeout. Live-ball starts are a defensive
rebound or a steal, and the split is where the durations separate rather than a judgement
call.

### 3. Duration is outcome-contaminated

A possession ends on a made shot **at the shot**, but on a miss **at the rebound** a second
or two later. So possession length is partly determined by whether the shot went in, and
anything derived from it inherits that.

The magnitude is not subtle: shots that end their possession convert at **93.3%**, shots
that do not at **1.3%** — because a non-terminal shot is, almost by definition, a miss that
was rebounded.

So `possession_seconds`, `transition` and `possession_points` are in
`FORBIDDEN_FEATURES`. They remain attached for reporting, and the two PPP framings are
published side by side: the duration-based transition split with its bias stated, and PPP by
**start type**, which is fixed before the offence does anything and cannot be contaminated
this way.

`seconds_into_possession` is safe — it is the clock at the moment the shot goes up, before
the outcome exists — and it is the model's shot-clock proxy, which the feed does not carry.

The guard is structural rather than advisory: `build_selection_design` narrows the frame to
a whitelist (`DESIGN_COLUMNS`) before computing anything, and that whitelist is itself
checked against `FORBIDDEN_FEATURES`. A forbidden column can neither be read nor added to
the list without failing.

### 4. The two feeds keep the clock at different resolutions

Play-by-play parses a `MM:SS` string to a whole second; the possession feed keeps tenths. A
shot recorded at 501 against a possession ending at 501.4 is the same event, and matching
exactly dropped 4% of shots.

Coverage against tolerance:

| Tolerance | Coverage |
| --------- | -------- |
| 0s        | 95.841%  |
| **1s**    | **99.739%** |
| 2s        | 99.749%  |
| 3s        | 99.759%  |

A boundary problem that resolves at exactly one unit of quantisation and then stops
improving is a rounding artefact, not a tuning parameter. One second is the coarser feed's
own resolution, and `shot_possession_context_coverage` gates it at 99%.

---

## Gates added because of the above

Every correction above is now something CI can catch:

| Gate                                      | Threshold | What it would have caught                    |
| ----------------------------------------- | --------- | -------------------------------------------- |
| `possession_length_plausible`             | ≥ 95%     | 45% of possessions measuring zero seconds    |
| `possession_oracle_agreement`             | ≥ 93%     | attribution drifting off the change of hands |
| `possession_oracle_agreement_unambiguous` | ≥ 96%     | the same, away from substitution boundaries  |
| `shot_possession_context_coverage`        | ≥ 99%     | shots silently dropped by a boundary mismatch |

---

## Running the reproducibility gate for the first time

`train --verify` refits the conversion model from committed gold and asserts every metric
reproduces to 1e-6. That claim was in the README before anything had executed it, and
executing it crashed. Four rounds of diagnosis; the first three were wrong.

Rounds one to three found real inefficiencies, all kept:

- Every fold was materialised up front, so four copies of the corpus were resident at once.
  The generator is now iterated lazily.
- `build_features` read the lineup columns with `.to_list()` — 600k Python lists of five ints
  per column per fold, roughly 120 MB of small objects whose churn the allocator never gave
  back. `util.lineup_slots` reads five flat integer arrays through polars' `list.get`
  instead, and zone strings go through categorical codes.
- The shuffled-lineup control round-tripped two columns through Python before a `gather`
  that works directly on the Series.

**None of them was the cause.** Each fix relocated the fault. What settled it was noticing
that the faults kept landing on lines that cannot fault — a bare `for i in range(n):`, a
dict lookup, and finally `_logit` reporting
`TypeError: unsupported operand type(s) for /: 'type' and 'float'` for a dictionary whose
every value is built by a literal `float(...)` call. No execution of that source puts a type
object there.

Reproducing the workload with the project taken out of it — numpy arrays, Python dicts, a
plain loop, no polars, no scikit-learn, no memory cap — produced an access violation on one
run and `IndexError: invalid index to scalar variable` on a 700,000-element array on the
next. The machine's own logs then showed a kernel-mode bugcheck (`0x1E`, `0xC0000005`), four
more minidumps predating this repository by a month, and a WHEA corrected machine-check
reported by Processor Core on an i9-13900KF with non-ECC memory.

So the crash is a hardware fault that sustained memory pressure exposes, and nothing in this
repository can fix it. **The reproducibility claim is therefore verified in CI, on Linux
runners — `repro.yml`, not this workstation.**

The per-row arithmetic was left untouched through all of it, deliberately: same operations,
same order, same values, so the committed metrics still mean what they meant and `--verify`
remains a test of reproducibility rather than of whichever refactor came last.

One genuine consequence for the memory cap: a fixed 6 GB default was wrong in both
directions — generous on a laptop, and tight enough on a 64 GB workstation to block a
legitimate refit and read as a reproducibility failure. It now derives from physical RAM, a
quarter of it, clamped to 4–24 GB.

---

## The reproducibility tolerance, and what it is allowed to be loose about

Two tolerances, because two kinds of metric.

`log_loss`, `brier`, `uncertainty`, `calibration_slope` and `top1_accuracy` are smooth
functions of the predictions. They reproduce to **1e-6** and have no excuse not to.

ECE and the Brier decomposition's reliability and resolution terms sort predictions into bins
and aggregate within them, which makes them *discontinuous in the predictions*: a value on a
bin edge moves by 1e-16 — ordinary variation between two platforms' matrix multiplies — and
lands in the next bin. Measured, on identical folds: `log_loss` and `brier` held to 1e-6 while
`ece` moved 2.5e-4. They get **1e-3**, which is still below the estimator's own sampling error
at this sample size, and the drift report says which bound it applied.

**The classification was wrong twice, both times because it was a list.** An exact-name set
held the selection model's nineteen per-zone-group variants (`three_ece`, `rim_resolution`,
`classwise_ece`) to 1e-6 and failed the gate on nothing; it also included `skill_score`, which
is `1 - brier/uncertainty` and smooth. The rule is now derived from the estimator — any
underscore-separated part of the name matching `ece`, `reliability` or `resolution` — so a
metric added under a new prefix classifies itself.

---

## Order dependence, and why a seed is not reproducibility

Running the gate on a second platform found the same bug shape in five places. It is worth
naming precisely: **a sort whose key has ties, applied to the output of a `group_by`, which
makes no ordering promise.** Nothing raises. Every individual number stays plausible.

| Where               | Tied key                                        | What it changed                                       |
| ------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| `eval/splits.py`    | list order out of `group_by`                     | cross-validation fold membership                      |
| `serve/parity.py`   | stint seconds                                    | 30 of 2,604 parity cases swapped                      |
| `retrieval/docs.py` | possession count                                 | which 200 documents the groundedness harness scored   |
| `serve/export.py`   | attempt count                                    | which player ships as the low-volume worked example   |
| `io/gold.py`        | `unique(keep="first")` with no `maintain_order`   | which of a player's rows survives                     |

The retrieval one had already moved a published number: the hallucinating template's grounded
rate was 0.015 on one machine and 0.010 on another, from `player_scope` failures of 197 versus
198. That figure was in the README.

Every site now sorts on a total key — the tied column plus an id or a hash. The regenerated
numbers move slightly and the conclusions do not: the full corpus still beats the per-stint
event log by 15x on Recall@10, BM25 still beats the hybrid on MRR and nDCG@10, and the
hallucinating template still scores about 1%.

**The generalisable point.** `np.random.default_rng(SEED)` was doing exactly what it
promised: producing a fixed permutation of *positions*. Nothing in the code established what
was *at* each position. A pinned seed over an unpinned ordering is not reproducible, and from
inside a single machine the two are indistinguishable — which is why this held for months and
then failed on the first run somewhere else.

`tests/test_order_independence.py` and `tests/test_split_determinism.py` pin the property
rather than the values: permute the input rows, and the derived order must not move.
