# Modelling notes

Why the models are shaped the way they are, and what had to be corrected on the way.
Numbers quoted here are reproduced by `lineupiq verify`, `lineupiq train` and
`lineupiq selection` from committed gold.

---

## Two targets, and why the first one was the wrong question

The original design had one model: `P(make | shot, lineup)`. It was built, scored against a
full baseline ladder, and the honest answer came back — on unseen five-man combinations,
knowing the other four players improves log loss by **+0.019%** for the served closed form
and **+0.078%** for the unconstrained gradient-boosted fit. Against a passing negative
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
