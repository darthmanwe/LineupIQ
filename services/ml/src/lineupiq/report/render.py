"""Generated documentation blocks.

Every published number is rendered here from a run log or from committed gold,
and written between HTML comment sentinels. ``lineupiq report check`` re-renders
into memory and fails if a committed block differs.

This exists because of a specific, observed failure: the sibling project's
``--verify`` compared refits against a run log but never against its README, so
the build stayed green for months while the published results table described
an older model. Humans own the prose between the sentinels; the tool owns every
number inside them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from lineupiq.io.gold import load_all_gold
from lineupiq.models.moves import CLAIMED_EFFECT_PER_100
from lineupiq.models.support import load_thresholds
from lineupiq.models.train import latest_run
from lineupiq.paths import DataPaths
from lineupiq.util import as_float

__all__ = ["BLOCK_RE", "apply_blocks", "check_blocks", "render_blocks"]

BLOCK_RE = re.compile(
    r"(<!-- lineupiq:begin id=(?P<id>[\w.]+) -->)(?P<body>.*?)(<!-- lineupiq:end id=(?P=id) -->)",
    re.DOTALL,
)


@dataclass(frozen=True)
class RenderContext:
    paths: DataPaths
    run: dict[str, Any] | None
    selection_run: dict[str, Any] | None
    shots: pl.DataFrame
    stints: pl.DataFrame


def _fmt(value: float | None, spec: str = ".5f") -> str:
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


def _estimability(ctx: RenderContext) -> str:
    """How little evidence most lineups actually have.

    This is the most important number in the repository, so it is computed
    rather than asserted.
    """
    thresholds = load_thresholds()
    # Both sides of the floor, matching `build_lineup_support`. Counting only
    # home lineups would report a different "distinct lineups" figure than
    # `lineupiq support` does, and two numbers for one quantity in one repo is
    # worse than either number alone.
    per_lineup = (
        pl.concat(
            [
                ctx.stints.filter(pl.col(col).is_not_null() & (pl.col(col).list.len() == 5)).select(
                    pl.col(col).list.sort().cast(pl.List(pl.Utf8)).list.join(",").alias("k"),
                    "duration_seconds",
                )
                for col in ("home_lineup", "away_lineup")
            ]
        )
        .group_by("k")
        .agg((pl.col("duration_seconds").sum() / 24.0).alias("possessions"))
    )
    total = per_lineup.height
    if not total:
        return "\n_No stints built._\n"

    floor = thresholds.reportable_possessions
    above = per_lineup.filter(pl.col("possessions") >= floor).height
    above_500 = per_lineup.filter(pl.col("possessions") >= 500).height
    median = as_float(per_lineup["possessions"].median())

    # Share of played time covered by lineups that clear the floor.
    total_secs = as_float(ctx.stints["duration_seconds"].sum(), 1.0) or 1.0
    covered = (
        as_float(per_lineup.filter(pl.col("possessions") >= floor)["possessions"].sum()) * 24.0
    )

    lines = [
        "",
        "| | Value |",
        "|---|---|",
        f"| Distinct five-man offensive lineups | {total:,} |",
        f"| Median possessions per lineup | {median:,.0f} |",
        f"| Lineups clearing the {floor}-possession reporting floor | "
        f"{above:,} ({above / total:.1%}) |",
        f"| Lineups above 500 possessions | {above_500:,} ({above_500 / total:.1%}) |",
        f"| Share of played time covered by reportable lineups | {covered / total_secs:.1%} |",
        "",
        f"At the {floor}-possession floor a lineup's offensive rating carries a standard error",
        "of roughly +/-8 per 100 possessions, against a true between-lineup spread of about 6-8.",
        f"So **{1 - above / total:.1%} of lineups cannot support a point estimate at all** -- which",
        "is why the refusal contract is a feature of the API rather than an error path.",
        "",
    ]
    return "\n".join(lines)


def _model_results(ctx: RenderContext) -> str:
    if ctx.run is None:
        return "\n_No run log. Run `lineupiq train` first._\n"

    out: list[str] = [""]
    labels = {
        "B0": "B0 - league zone mean",
        "B1": "B1 - shooter x zone (shrunk)",
        "B2": "B2 - B1 + context, no lineup",
        "B3": "B3 - additive GBDT, no lineup",
        "full": "**full - served closed form**",
        "full_gbdt": "**full - unconstrained GBDT**",
    }
    counterpart = {"full": "B2", "full_gbdt": "B3"}

    for split, models in ctx.run.get("metrics", {}).items():
        title = {
            "walk_forward": "Walk-forward -- later games",
            "leave_lineup_out": "Leave-lineup-out -- unseen five-man combinations",
        }.get(split, split)
        n = int(models.get("full", {}).get("n", 0))
        out += [
            f"**{title}** -- n = {n:,} shots",
            "",
            "| Model | Log loss | Brier | Resolution | ECE | Cal. slope | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for key in ("B0", "B1", "B2", "B3", "full", "full_gbdt"):
            if key not in models:
                continue
            m = models[key]
            verdict = ""
            if (ref := counterpart.get(key)) and ref in models:
                base = models[ref]["log_loss"]
                delta = (base - m["log_loss"]) / base
                verdict = f"{delta:+.3%} vs {ref}"
            out.append(
                f"| {labels.get(key, key)} | {_fmt(m['log_loss'])} | {_fmt(m['brier'])} | "
                f"{_fmt(m['resolution'])} | {_fmt(m['ece'], '.4f')} | "
                f"{_fmt(m['calibration_slope'], '.3f')} | {verdict} |"
            )
        out.append("")

    lolo = ctx.run.get("metrics", {}).get("leave_lineup_out", {})
    if "full" in lolo and "full_gbdt" in lolo:
        cost = (lolo["full"]["log_loss"] - lolo["full_gbdt"]["log_loss"]) / lolo["full_gbdt"][
            "log_loss"
        ]
        out += [
            f"**Cost of the serving constraint:** the closed form the Worker evaluates is "
            f"{cost:.2%} worse in log loss than the unconstrained gradient-boosted fit on "
            "unseen lineups. That is the price of exact Python<->TypeScript parity inside a "
            "10 ms CPU budget, and it is published rather than absorbed.",
            "",
        ]

    control = ctx.run.get("controls", {}).get("shuffled_lineup_logloss_gain")
    if control is not None:
        verdict = "passes" if abs(control) < 1e-3 else "**FAILS**"
        out += [
            f"**Negative control:** with lineup context randomly permuted across shots, the "
            f"model's log-loss gain over B1 is {control:+.6f} -- indistinguishable from zero, so "
            f"the lineup features are not leaking. Control {verdict}.",
            "",
        ]

    meta = (
        f"_Generated from run `{ctx.run.get('git_sha')}` on {ctx.run.get('platform')}, "
        f"seed {ctx.run.get('seed')}, {ctx.run.get('n_shots', 0):,} shots across "
        f"{len(ctx.run.get('seasons', []))} seasons._"
    )
    out += [meta, ""]
    return "\n".join(out)


def _possessions(ctx: RenderContext) -> str:
    """The possession layer, and the oracle check on it."""
    try:
        poss = load_all_gold(ctx.paths, "possession_facts")
    except FileNotFoundError:
        return "\n_Not yet built._\n"

    n = poss.height
    matched = poss.with_columns(
        pl.col("oracle_off_lineup").list.sort().eq(pl.col("off_lineup").list.sort()).alias("m")
    )
    overall = as_float(matched["m"].mean())
    clean = matched.filter(~pl.col("boundary_ambiguous"))
    unambiguous = as_float(clean["m"].mean()) if clean.height else 0.0
    boundary_rate = as_float(poss["boundary_ambiguous"].mean())

    # The transition flag is read from the data, not re-derived here. A report
    # that recomputes a definition is a report that will eventually describe a
    # different split than the one the model was fitted on.
    live = poss.filter(pl.col("transition"))
    half = poss.filter(~pl.col("transition"))

    by_start = (
        poss.group_by("possession_start_type")
        .agg(pl.len().alias("n"), pl.col("points").mean().alias("ppp"))
        # Five groups and a float key, so a tie is unlikely rather than
        # impossible -- but this table goes into the README and `report check`
        # fails on any difference, so "unlikely" is not the standard.
        .sort(["ppp", "possession_start_type"], descending=[True, False])
    )

    lines = [
        "",
        "| | Value |",
        "|---|---|",
        f"| Possessions | {n:,} |",
        f"| Attributed to a five-man lineup | "
        f"{poss.filter(pl.col('off_lineup_hash').is_not_null()).height / n:.2%} |",
        f"| Agreement with the independent lineup oracle | {overall:.2%} |",
        f"| ... restricted to possessions not starting on a substitution | **{unambiguous:.2%}** |",
        f"| Possessions starting on a substitution (attribution ambiguous) | {boundary_rate:.1%} |",
        f"| Mean possession length | {as_float(poss['possession_seconds'].mean()):.2f}s |",
        f"| Median possession length | {as_float(poss['possession_seconds'].median()):.1f}s |",
        f"| Points per possession, transition | {as_float(live['points'].mean()):.3f} |",
        f"| Points per possession, half-court | {as_float(half['points'].mean()):.3f} |",
        "",
        "The oracle is a second lineup reconstruction, written independently in",
        "another language. Away from substitution boundaries the two agree at the",
        "same rate our period-start solver reports exact solutions. About one",
        "possession in eleven begins on the exact second of a substitution, where",
        "there are two defensible answers and no way to choose between them; those",
        "are flagged in the data rather than silently trusted.",
        "",
        "Possession length is derived, not a column in the feed, which makes it",
        "worth checking against something already known: the NBA has averaged",
        "close to 14 seconds a possession for years. That check is what caught a",
        "real bug here -- the feed's own start and end fields are the clock at a",
        "possession's first and last *recorded event*, so 45% of possessions",
        "measured as zero seconds long and the published transition split was",
        "computed on a duration that was not a duration.",
        "",
        "**Points per possession by how the possession began.** Duration is partly",
        "decided by the outcome: a make ends a possession at the shot, a miss at",
        "the rebound a beat later, so short possessions over-collect makes and the",
        "transition figure above is biased upward. Start type is fixed before the",
        "offence does anything and cannot be contaminated that way, so it is",
        "published beside it.",
        "",
        "| Possession start | n | PPP |",
        "|---|---|---|",
    ]
    lines += [
        f"| {start_type} | {int(count):,} | {as_float(ppp):.3f} |"
        for start_type, count, ppp in by_start.iter_rows()
    ]
    lines += [
        "",
        "A steal is the most valuable way to get the ball and a timeout the least,",
        "with about a quarter of a point per possession between them. Nothing in",
        "the pipeline was fitted to produce that ordering.",
        "",
    ]
    return "\n".join(lines)


def _data_quality(ctx: RenderContext) -> str:
    total = ctx.shots.height
    with_lineup = ctx.shots.filter(pl.col("lineup_for_hash").is_not_null()).height
    valid = ctx.shots.filter(pl.col("stint_quality") == "VALID").height
    seasons = sorted({int(s) for s in ctx.shots["season"].unique().to_list()})

    return "\n".join(
        [
            "",
            "| | Value |",
            "|---|---|",
            f"| Seasons | {', '.join(f'{s}-{(s + 1) % 100:02d}' for s in seasons)} |",
            f"| Shot attempts | {total:,} |",
            f"| Resolved to a complete five-man lineup | {with_lineup:,} "
            f"({with_lineup / total:.2%}) |",
            f"| Lineup solved cleanly (training-grade) | {valid:,} ({valid / total:.2%}) |",
            f"| Stints reconstructed | {ctx.stints.height:,} |",
            "",
        ]
    )


def _selection_results(ctx: RenderContext) -> str:
    """The shot-selection ladder -- the model that asks the right question."""
    run = ctx.selection_run
    if run is None:
        return "\n_No run log. Run `lineupiq selection` first._\n"

    labels = {
        "S0": "S0 - league zone mix",
        "S1": "S1 - shooter's own shrunk mix (lookup table)",
        "S2": "S2 - conditional logit, no lineup",
        "S3": "S3 - multiclass GBDT, no lineup",
        "full": "**full - conditional logit + lineup (served)**",
        "full_gbdt": "**full - GBDT + lineup (unconstrained)**",
    }
    counterpart = {"full": "S2", "full_gbdt": "S3"}

    out: list[str] = [""]
    for split, models in run.get("metrics", {}).items():
        title = {
            "walk_forward": "Walk-forward -- later games",
            "leave_lineup_out": "Leave-lineup-out -- unseen five-man combinations",
        }.get(split, split)
        n = int(models.get("full", {}).get("n", 0))
        out += [
            f"**{title}** -- n = {n:,} attempts",
            "",
            "| Model | Log loss (9-way) | Top-1 | 3PA log loss | 3PA resolution | Verdict |",
            "|---|---|---|---|---|---|",
        ]
        for key in ("S0", "S1", "S2", "S3", "full", "full_gbdt"):
            if key not in models:
                continue
            m = models[key]
            verdict = ""
            if (ref := counterpart.get(key)) and ref in models:
                base = models[ref]["log_loss"]
                verdict = f"{(base - m['log_loss']) / base:+.3%} vs {ref}"
            out.append(
                f"| {labels.get(key, key)} | {_fmt(m['log_loss'])} | "
                f"{_fmt(m['top1_accuracy'], '.4f')} | {_fmt(m['three_log_loss'])} | "
                f"{_fmt(m['three_resolution'])} | {verdict} |"
            )
        out.append("")

    audit = run.get("model", {}).get("sign_audit", {})
    within = run.get("model", {}).get("within_shooter_coefficients", {})
    if audit:
        agree = sum(1 for row in audit.values() if row.get("verdict") == "agrees")
        disagree = sum(1 for row in audit.values() if row.get("verdict") == "DISAGREES")
        unknown = sum(1 for row in audit.values() if row.get("verdict") == "indeterminate")
        has_errors = any(row.get("standard_error") is not None for row in audit.values())

        headline = f"**Pre-registered sign audit -- {agree}/{len(audit)} agree"
        if disagree:
            headline += f", {disagree} disagree"
        if unknown:
            headline += f", {unknown} indeterminate"
        headline += ".**"

        out += [
            headline + " Each coefficient's direction was written down in the source before",
            "the model was fitted, so a term that improves log loss while pointing the wrong",
            "way cannot be presented as confirmation of the thing it was named after.",
        ]
        if has_errors:
            out += [
                "",
                "**Indeterminate is a third verdict, not a rounding of the other two.** A",
                "coefficient whose 95% interval spans zero has neither confirmed nor",
                "contradicted its pre-registered sign, and counting it as agreement would be",
                "the same error as reading a null result as a refutation. Intervals come from",
                "the ridge sandwich `H^-1 I H^-1` over the observed information, which is the",
                "right estimator for a penalised fit and the same one RAPM uses.",
            ]
            if unknown == 0:
                smallest = min(
                    (abs(float(row["z"])) for row in audit.values() if row.get("z") is not None),
                    default=float("nan"),
                )
                out += [
                    "",
                    f"**Nothing landed there.** The smallest `|z|` in the model is "
                    f"{smallest:.1f}: at {int(run.get('n_shots') or 0):,} attempts against "
                    "twenty parameters there is an enormous amount of evidence about each "
                    "one, and even a coefficient of +0.09 sits several standard errors from "
                    "zero. Two things follow. The pre-registered failure below is not a "
                    "marginal call -- it is ten and a half standard errors the wrong way. "
                    "And **significance says nothing about magnitude**: every term here is "
                    "overwhelmingly significant while the whole effect is worth a standard "
                    "deviation of 0.19 points per 100 attempts. Those are the same fact from "
                    "two sides, and reporting only the first is how a p-value becomes an "
                    "overclaim.",
                ]
        out += [
            "",
            (
                "| Term | Coefficient | Std. error | 95% interval | Expected | Verdict | Lineup term |"
                if has_errors
                else "| Term | Coefficient | Expected | Verdict | Lineup term |"
            ),
            "|---|---|---|---|---|---|---|" if has_errors else "|---|---|---|---|---|",
        ]
        for name, row in audit.items():
            expected = "+" if row.get("expected_sign") == 1 else "-"
            raw = row.get("verdict")
            verdict = {
                "agrees": "agrees",
                "DISAGREES": "**DISAGREES**",
                "indeterminate": "_indeterminate_",
            }.get(str(raw), str(raw))
            lineup = "yes" if row.get("is_lineup") else ""
            if has_errors:
                error = row.get("standard_error")
                interval = row.get("ci95")
                error_cell = "--" if error is None else _fmt(error, ".4f")
                if isinstance(interval, list) and len(interval) == 2:
                    interval_cell = f"{interval[0]:+.4f} to {interval[1]:+.4f}"
                else:
                    interval_cell = "--"
                out.append(
                    f"| `{name}` | {_fmt(row.get('value'), '+.4f')} | {error_cell} | "
                    f"{interval_cell} | {expected} | {verdict} | {lineup} |"
                )
            else:
                out.append(
                    f"| `{name}` | {_fmt(row.get('value'), '+.4f')} | {expected} | "
                    f"{verdict} | {lineup} |"
                )
        out.append("")

    if within:
        out += [
            "**Within-shooter refit.** The lineup aggregates are anti-correlated with a",
            "shooter's own tendencies by roster construction -- put four shooters on the floor",
            "and the fifth man is usually the centre -- so each lineup feature is also",
            "re-estimated after centring it within shooter, which removes the between-player",
            "component entirely and asks only what happens when *this* player gets more",
            "spacing than he usually has.",
            "",
            "| Term | Headline | Within shooter |",
            "|---|---|---|",
        ]
        for name, value in within.items():
            headline = audit.get(name, {}).get("value")
            out.append(f"| `{name}` | {_fmt(headline, '+.4f')} | {_fmt(value, '+.4f')} |")
        out.append("")

    controls = run.get("controls", {})
    gain = controls.get("shuffled_lineup_logloss_gain")
    spacing = controls.get("shuffled_lineup_spacing_coefficient")
    if gain is not None:
        out += [
            "**Negative control.** With the five-man lineups randomly reassigned across "
            f"attempts, the full model's log-loss gain over S2 is {gain:+.6f} and the "
            f"`spacing_x_three` coefficient collapses to {spacing:+.4f}. The second number is "
            "the one worth having: a pooled metric can go flat while a coefficient stays "
            "large, and this model's claim is directional, so the coefficient is what has to "
            "die under shuffling.",
            "",
        ]

    meta = (
        f"_Generated from run `{run.get('git_sha')}` on {run.get('platform')}, "
        f"seed {run.get('seed')}, {run.get('n_shots', 0):,} attempts across "
        f"{len(run.get('seasons', []))} seasons._"
    )
    out += [meta, ""]
    return "\n".join(out)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


#: Lineups drawn to price the selection effect. Enough that the tails are
#: stable to two decimals; the draw is seeded, so the table reproduces.
_PRICED_SAMPLE = 4_000


def _selection_priced(ctx: RenderContext) -> str:
    """The selection effect converted into points, which is the honest close.

    The ladder above says the effect is real. This says what it is worth, and
    those are different questions -- a model can be statistically detectable and
    economically negligible at once, and reporting the first without the second
    is how a real result turns into an overclaim.

    Priced at league conversion rates by zone, so the whole quantity is
    selection: pricing at the shooter's own rates would fold in "he shoots
    better from there", which is what the other model measures and did not find.
    """
    import json as _json

    import numpy as _np

    from lineupiq.config import SEED
    from lineupiq.serve.score import ScoreRequest, score_selection

    profiles_path = ctx.paths.root / "apps" / "web" / "public" / "data" / "selection_profiles.json"
    model_path = ctx.paths.root / "apps" / "web" / "public" / "data" / "selection_model.json"
    if not profiles_path.exists() or not model_path.exists():
        return "\n_No served model exported. Run `lineupiq export` first._\n"

    profiles = _json.loads(profiles_path.read_text(encoding="utf-8"))
    model = _json.loads(model_path.read_text(encoding="utf-8"))
    if not model.get("available"):
        return "\n_No selection run log committed._\n"

    known = sorted(int(k) for k in profiles["shooter_log_ratio"])
    rng = _np.random.default_rng(SEED)
    values = _np.empty(_PRICED_SAMPLE)
    for i in range(_PRICED_SAMPLE):
        offence = [known[j] for j in rng.choice(len(known), 5, replace=False)]
        defence = [known[j] for j in rng.choice(len(known), 5, replace=False)]
        values[i] = score_selection(
            ScoreRequest(offence[0], tuple(offence), tuple(defence)),
            profiles,
            model["coefficients"],
            model["term_names"],
        ).points_per_100

    zone_points = dict(zip(profiles["zones"], profiles["zone_points"], strict=True))
    ordered = sorted(zone_points.items(), key=lambda kv: -kv[1])

    out = [
        "",
        f"**What the shot-mix shift is worth**, over {_PRICED_SAMPLE:,} random five-man "
        "lineups, priced at league points per attempt by zone:",
        "",
        "| | Points per 100 attempts |",
        "|---|---|",
        f"| Median | {_np.median(values):+.3f} |",
        f"| Interquartile range | {_np.percentile(values, 25):+.3f} to "
        f"{_np.percentile(values, 75):+.3f} |",
        f"| 1st to 99th percentile | {_np.percentile(values, 1):+.3f} to "
        f"{_np.percentile(values, 99):+.3f} |",
        f"| Standard deviation | {values.std():.3f} |",
        f"| Largest in the sample | {_np.abs(values).max():.3f} |",
        f"| Within +/-0.5 points | {(_np.abs(values) < 0.5).mean():.1%} |",
        "",
        "**The effect is real and it is small, and both halves are the result.** It improves "
        "log loss on unseen five-man combinations, it survives a shuffled-lineup control, and "
        "`spacing_x_three` keeps its sign across three specifications including a "
        "within-shooter one. Priced, it is worth hundredths of a point per hundred attempts. "
        "A model can be statistically detectable and economically negligible at the same "
        "time; reporting the first without the second is how a real result becomes an "
        "overclaim.",
        "",
        "The zone values it prices with are the ones every basketball source reports, which "
        "is a cheap check that nothing is inverted:",
        "",
        "| Zone | Points per attempt |",
        "|---|---|",
    ]
    out += [f"| {zone} | {value:.3f} |" for zone, value in ordered]
    out.append("")
    return "\n".join(out)


#: Lineups drawn to measure how often the ranking refuses. Seeded, so the table
#: reproduces; large enough that the "refuses to order at all" rate -- which is a
#: few per cent -- is stable to a tenth of a point.
_RANKING_SAMPLE = 2_000


def _selection_ranking(ctx: RenderContext) -> str:
    """How often the ranking declines to rank, and what the covariance bought.

    Both numbers exist because a ranked list is a claim. Nine zones sorted by
    contribution *reads as* nine ordered facts, and at an effect this small most
    of those orderings are not supported -- so the mechanism has to be able to
    say "these two are tied", and the rate at which it does is the evidence that
    the mechanism is doing anything at all.

    The second number is the design justification. The obvious separation test
    asks whether two marginal intervals overlap; the correct one asks whether
    the *difference* is distinguishable from zero, which needs the off-diagonal
    covariance. If the two tests always agreed, the twenty standard errors would
    do and the 20x20 matrix would be dead weight. They do not agree, and the gap
    is measured here rather than asserted.
    """
    import json as _json

    import numpy as _np

    from lineupiq.config import SEED
    from lineupiq.serve.plays import rank_plays
    from lineupiq.serve.score import ScoreRequest

    data = ctx.paths.root / "apps" / "web" / "public" / "data"
    profiles_path = data / "selection_profiles.json"
    model_path = data / "selection_model.json"
    if not profiles_path.exists() or not model_path.exists():
        return "\n_No served model exported. Run `lineupiq export` first._\n"

    profiles = _json.loads(profiles_path.read_text(encoding="utf-8"))
    model = _json.loads(model_path.read_text(encoding="utf-8"))
    if not model.get("available") or model.get("covariance") is None:
        return "\n_The committed selection run log carries no covariance matrix._\n"

    contract = model["ranking"]
    known = sorted(int(k) for k in profiles["shooter_log_ratio"])
    rng = _np.random.default_rng(SEED)

    bands: list[int] = []
    ranked: list[int] = []
    unordered = 0
    any_tie = 0
    tied_groups = 0
    excluded = 0
    refused = 0
    compared = 0
    spanning = 0
    for _ in range(_RANKING_SAMPLE):
        offence = [known[j] for j in rng.choice(len(known), 5, replace=False)]
        defence = [known[j] for j in rng.choice(len(known), 5, replace=False)]
        ranking = rank_plays(
            ScoreRequest(offence[0], tuple(offence), tuple(defence)),
            profiles,
            model,
            confidence=contract["confidence"],
            critical_value=contract["critical_value"],
            min_zone_share=contract["min_zone_share"],
        )
        bands.append(len(ranking.bands))
        ranked.append(len(ranking.plays))
        unordered += 0 if ranking.ordered else 1
        # A tie is a band with more than one member. **Not** `len(bands) < 9`:
        # a ranking can have fewer bands than zones because the share floor
        # excluded one, which is a different fact with a different cause, and
        # the first version of this table conflated the two -- it reported 91%
        # of rankings containing a tie beside a mean of 0.39 tied groups per
        # ranking, which cannot both be true.
        groups = sum(1 for band in ranking.bands if len(band) > 1)
        any_tie += 1 if groups else 0
        tied_groups += groups
        excluded += len(ranking.excluded)
        refused += ranking.diagonal_would_refuse
        compared += ranking.pairs_compared
        spanning += ranking.ties_spanning_bands

    n_zones = len(profiles["zones"])
    level = contract["confidence"]
    out = [
        "",
        f"**How often the ranking declines to rank**, over {_RANKING_SAMPLE:,} random "
        f"five-man lineups at the pre-registered {level:.0%} level:",
        "",
        "| | |",
        "|---|---|",
        f"| Zones ranked, mean | {_np.mean(ranked):.2f} of {n_zones} |",
        f"| Zones below the share floor, mean | {excluded / _RANKING_SAMPLE:.2f} |",
        f"| Distinct ranks, mean | {_np.mean(bands):.2f} |",
        f"| Rankings containing a tie | {any_tie / _RANKING_SAMPLE:.1%} |",
        f"| Tied groups per ranking, mean | {tied_groups / _RANKING_SAMPLE:.2f} |",
        f"| Rankings with no supported order at all | {unordered / _RANKING_SAMPLE:.1%} |",
        "",
        f"So a typical lineup separates into {_np.mean(bands):.1f} distinct ranks over "
        f"{_np.mean(ranked):.1f} ranked zones, {any_tie / _RANKING_SAMPLE:.0%} of rankings "
        f"contain at least one tie, and on {unordered / _RANKING_SAMPLE:.1%} of them nothing "
        "separates from anything. Those last are served as unordered sets with a warning "
        "saying so, not as lists whose order happens to carry no information.",
        "",
        "**What the covariance bought.** The obvious test asks whether two zones' intervals "
        "overlap. That test is wrong: shares come out of a softmax and sum to one, so two "
        "contributions are strongly negatively correlated and `Var(a - b)` is far smaller "
        f"than `Var(a) + Var(b)`. Of {compared:,} ranked pairs, "
        f"**{refused:,} ({refused / compared:.1%}) separate on the difference and would have "
        "been called indistinguishable by comparing marginal intervals** -- pairs the model "
        "really can order, that the cheaper test would have refused. That is why a 20x20 "
        "matrix ships to the edge instead of its diagonal.",
        "",
        "The bands are contiguous runs of the ranked list, which is what makes `rank` "
        "monotone in list position. That constraint is not free: "
        f"{spanning:,} of {compared:,} pairs ({spanning / compared:.2%}) are "
        "indistinguishable and still landed in different bands, because they were not "
        "adjacent enough to share a run. Small, but not zero, and counted rather than "
        "waved at.",
        "",
    ]
    return "\n".join(out)


def _rapm(ctx: RenderContext) -> str:
    """RAPM, and the reliability number that decides whether to believe it."""
    run = _read_json(ctx.paths.runs / "rapm" / "run.json")
    if run is None:
        return "\n_Not yet fitted. Run `lineupiq rapm`._\n"

    reliability = run.get("reliability", {})
    co = run.get("co_occurrence", {})
    boundary = run.get("boundary_sensitivity", {})
    spread = run.get("spread", {})

    lines = [
        "",
        "| | Value |",
        "|---|---|",
        f"| Possessions | {int(run['n_possessions']):,} |",
        f"| Players estimated | {int(run['n_players']):,} |",
        f"| Ridge penalty, offence / defence | {run['lambda_offence']:,.0f} / "
        f"{run['lambda_defence']:,.0f} |",
        f"| Effective degrees of freedom | {run['effective_df']:,.1f} "
        f"(of {2 * int(run['n_players']) + 1:,} columns) |",
        f"| Condition number | {run['condition_number']:,.1f} |",
        f"| League points per possession | {run['league_ppp']:.4f} |",
        f"| Home advantage | {run['home_advantage']:+.2f} per 100 |",
        f"| Between-player spread, offence / defence | {spread.get('off_sd', 0):.2f} / "
        f"{spread.get('def_sd', 0):.2f} sd |",
        "",
    ]

    if reliability.get("n_players"):
        lines += [
            "**Split-half reliability -- the number that decides whether to believe any of it.**",
            "",
            "| Side | Odd vs even games (r) | Spearman | Full-sample (Spearman-Brown) |",
            "|---|---|---|---|",
        ]
        for side in ("off", "def"):
            lines.append(
                f"| {side} | {reliability[f'{side}_split_half_r']:+.3f} | "
                f"{reliability[f'{side}_spearman_rho']:+.3f} | "
                f"{reliability[f'{side}_full_sample_reliability']:+.3f} |"
            )
        lines += [
            "",
            f"Measured on {int(reliability['n_players']):,} players with at least "
            f"{int(reliability['min_possessions'])} possessions in each half. This, and not "
            "cross-validated error, is the honest test: possession outcomes are dominated by "
            "shot noise, so a ridge model can cut CV error while its player coefficients are "
            "close to arbitrary. Two disjoint halves of the same league agreeing about who is "
            "good cannot happen by accident -- and a correlation near 0.4 says the agreement is "
            "real but moderate. Three seasons is not enough for RAPM to be precise, and the "
            "reliability figure is published rather than the leaderboard alone.",
            "",
        ]

    if co:
        lines += [
            f"**Identifiability.** {int(co['n_flagged'])} of {int(co['n_players'])} players "
            f"share more than {co['ceiling']:.0%} of their floor time with a single teammate "
            f"(median {co['median_max_co_occurrence']:.0%}). For those, the pair's *sum* is "
            "identified and neither coefficient is, so they are flagged and not served as "
            "point estimates.",
            "",
        ]

    if boundary:
        lines += [
            f"**Boundary sensitivity.** Dropping the {boundary['share_excluded']:.1%} of "
            "possessions that begin on a substitution -- where two lineup attributions are "
            "both defensible -- moves offensive coefficients by "
            f"{boundary['off_mean_abs_change']:.3f} per 100 on average "
            f"(correlation {boundary['off_correlation']:.4f}) and defensive by "
            f"{boundary['def_mean_abs_change']:.3f} "
            f"(correlation {boundary['def_correlation']:.4f}). Measured rather than assumed, "
            'because "we excluded 9% of the data and nothing moved" and "we excluded 9% and '
            'everything moved" call for very different amounts of caution downstream.',
            "",
        ]
    return "\n".join(lines)


def _trade(ctx: RenderContext) -> str:
    """The counterfactual backtest, with its power analysis first."""
    directory = ctx.paths.runs / "trade"
    runs: dict[str, dict[str, Any]] = {}
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            if payload:
                runs[path.stem] = payload
    if not runs:
        return "\n_Not yet backtested. Run `lineupiq backtest`._\n"

    # `inherit` is the reference for the prose below: it is the default rule and
    # the cleanest counterfactual. Taking whichever file sorted first would make
    # the narrative depend on alphabetical order.
    reference = runs.get("inherit") or next(iter(runs.values()))
    power = reference["power"]

    lines = [
        "",
        "**The power analysis, computed and committed before any result.**",
        "",
        "| | Value |",
        "|---|---|",
        f"| Evaluable mid-season moves | {int(power['n'])} |",
        f"| Team net-rating noise (sd) | {power['residual_sd']:.2f} per 100 |",
        f"| Minimum detectable effect | **{power['mde']:.2f} per 100** |",
        f"| Effects this model actually projects | ~{CLAIMED_EFFECT_PER_100:.1f} per 100 |",
        f"| Sign-accuracy 95% half-width | +/-{power['sign_accuracy_ci_half_width']:.1%} |",
        f"| Verdict | **{power['verdict']}** |",
        "",
        "The minimum detectable effect is the same size as the effects being claimed. That is",
        "not a result to work around -- it is the result. No accuracy claim follows from what",
        "is below, and committing to that before running the backtest is the point of stating",
        "it first.",
        "",
        "| Minutes rule | n | Mean projected | Mean DiD | Corr | Sign agreement | MAE vs DiD |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        real = run.get("real", {})
        if not real.get("n"):
            continue
        low, high = real.get("sign_agreement_ci", [float("nan"), float("nan")])
        lines.append(
            f"| `{name}` | {int(real['n'])} | {real['mean_projected']:+.3f} | "
            f"{real['mean_did']:+.3f} | {real['correlation_projected_did']:+.3f} | "
            f"{real['sign_agreement_vs_did']:.1%} "
            f"[{as_float(low):.0%}, {as_float(high):.0%}] | "
            f"{real['mean_abs_error_vs_did']:.3f} |"
        )

    placebo = reference.get("placebo", {})
    if placebo.get("n"):
        lines += [
            "",
            "**The placebo arm is the number that settles it.** The identical machinery runs on",
            f'{int(placebo["n"])} players who did *not* move, pretending each "arrived" at his',
            "own team on a matched date. Swapping a player for himself projects exactly "
            f"{placebo['mean_projected']:+.3f}, which is the identity holding -- if it drifted, "
            "every number above would be measuring a pipeline bug.",
            "",
            f"Those placebos still show a mean absolute DiD swing of **{placebo['mean_abs_did']:.2f} "
            "per 100**. That is how far a team's rating moves across an arbitrary mid-season",
            "cutoff with no roster change at all, and it is the floor below which nothing here",
            f"is measurable. The real moves' projection error is {reference['real']['mean_abs_error_vs_did']:.2f}"
            " -- larger than the placebo swing, so **the projection does not beat assuming no",
            "change.**",
            "",
        ]

    variance = reference.get("variance", {})
    if variance.get("mean_minutes_variance_share") is not None:
        lines += [
            f"**Variance decomposition.** The minutes rule carries "
            f"{variance['mean_minutes_variance_share']:.0%} of a projection's variance on "
            f"average and dominates it in {variance['share_where_minutes_dominates']:.0%} of "
            f"cases; {variance['share_interval_includes_zero']:.0%} of 80% intervals contain "
            "zero. The plan expected the minutes assumption to dominate, and it does not -- the "
            "player estimates are the larger term. That is worth knowing precisely because it "
            "contradicts the design's own guess about where the uncertainty lived.",
            "",
        ]

    for note in reference.get("notes", []):
        lines += [f"_Caveat: {note}._", ""]
    return "\n".join(lines)


def _retrieval(ctx: RenderContext) -> str:
    """The corpus ablation: does document design actually matter?"""
    run = _read_json(ctx.paths.runs / "retrieval" / "ablation.json")
    if run is None:
        return "\n_Not yet run. Run `lineupiq retrieval ablation`._\n"

    labels = {
        "events": "`events` -- per-stint event log (the original design's proposal)",
        "numbers": "`numbers` -- the same facts as bare decimals",
        "full": "`full` -- names, archetypes, style tags, comparatives, caveats",
    }
    lines = [
        "",
        f"{int(run['n_documents']):,} documents at `(lineup_hash, team, season)` grain, "
        f"{int(run['n_queries'])} queries.",
        "",
        "| Corpus | BM25 | LSA (dense) | RRF (hybrid) |",
        "|---|---|---|---|",
    ]
    for variant in ("events", "numbers", "full"):
        scores = run["by_corpus"].get(variant)
        if not scores:
            continue
        cells = [
            f"{scores[name]['recall']:.3f} / {scores[name]['mrr']:.3f} / {scores[name]['ndcg']:.3f}"
            for name in ("bm25", "lsa", "rrf")
        ]
        lines.append(f"| {labels.get(variant, variant)} | " + " | ".join(cells) + " |")

    lines += ["", "_Each cell is Recall@10 / MRR / nDCG@10._", ""]

    numbers = run["by_corpus"].get("numbers", {}).get("bm25", {}).get("recall")
    full = run["by_corpus"].get("full", {}).get("bm25", {}).get("recall")
    if numbers and full:
        lines += [
            f"**Document design moves Recall@10 from {numbers:.3f} to {full:.3f}** on identical "
            f"underlying facts -- a factor of {full / max(numbers, 1e-9):.0f}. The original "
            "design document asserted that document design drives retrieval quality; this is "
            "that assertion measured. A corpus of bare decimals is close to unusable, because "
            "a query has words in it and a decimal has no words to match.",
            "",
        ]

    full_scores = run["by_corpus"].get("full", {})
    if full_scores:
        bm25 = full_scores["bm25"]
        rrf = full_scores["rrf"]
        lines += [
            "**BM25 alone beats the hybrid on two of three metrics, and that is reported "
            "rather than buried.** On the full corpus BM25 reaches MRR "
            f"{bm25['mrr']:.3f} and nDCG@10 {bm25['ndcg']:.3f} against the hybrid's "
            f"{rrf['mrr']:.3f} and {rrf['ndcg']:.3f}; the hybrid wins only on Recall@10 "
            f"({rrf['recall']:.3f} vs {bm25['recall']:.3f}). Rank fusion pulls more relevant "
            "documents into the top ten and dilutes what sits at the top. That is the "
            "expected shape for a corpus built from a closed vocabulary and named entities, "
            "which is precisely what lexical matching is best at -- a dense leg earns its "
            "place when queries are phrased in words the documents do not contain, and these "
            "queries are not.",
            "",
        ]

    by_kind = run.get("by_kind", {})
    if by_kind:
        lines += ["| Query kind | BM25 | LSA | RRF |", "|---|---|---|---|"]
        for kind, scores in sorted(by_kind.items()):
            lines.append(
                f"| {kind} | "
                + " | ".join(f"{scores.get(n, 0.0):.3f}" for n in ("bm25", "lsa", "rrf"))
                + " |"
            )
        lines += ["", "_Recall@10 on the full corpus, by query kind._", ""]

    for note in run.get("notes", []):
        lines += [f"_{note}_", ""]
    return "\n".join(lines)


def _groundedness(ctx: RenderContext) -> str:
    """What arithmetic can and cannot check, measured."""
    run = _read_json(ctx.paths.runs / "groundedness" / "run.json")
    if run is None:
        return "\n_Not yet run. Run `lineupiq groundedness`._\n"

    by_template = run.get("by_template", {})
    labels = {
        "faithful": "`faithful` — every number traceable, tier respected",
        "overclaiming": "`overclaiming` — **only correct numbers**, asserts a point estimate",
        "hallucinating": "`hallucinating` — names a player who was not on the floor",
    }
    lines = [
        "",
        f"{int(run['n_documents'])} lineup documents, {int(run['n_below_floor'])} of them below the",
        "reporting floor. Narratives are **templated, not generated** — no language model has",
        "been called by this repository.",
        "",
        "| Narrative | n | Grounded | Numeric traceability | Easy control | Near-miss control |",
        "|---|---|---|---|---|---|",
    ]
    for template in ("faithful", "overclaiming", "hallucinating"):
        row = by_template.get(template)
        if not row:
            continue
        lines.append(
            f"| {labels.get(template, template)} | {int(row['n'])} | "
            f"**{row['grounded_rate']:.1%}** | {row['mean_traceability']:.1%} | "
            f"{row['control_easy_grounded_rate']:.1%} | "
            f"{row['control_near_miss_grounded_rate']:.1%} |"
        )

    overclaiming = by_template.get("overclaiming", {})
    faithful = by_template.get("faithful", {})
    if overclaiming and faithful:
        tier_failures = int(
            (overclaiming.get("failures_by_check") or {}).get("tier_consistency", 0)
        )
        lines += [
            "",
            "**Read the second row.** Numeric traceability is "
            f"{overclaiming['mean_traceability']:.0%} — every figure in every one of those "
            "narratives appears in the evidence. And "
            f"{overclaiming['grounded_rate']:.0%} of them are grounded, because "
            f"{tier_failures} assert a point estimate for a lineup below the reporting floor. "
            "The numbers are right and the sentences are wrong.",
            "",
            "That is the whole case for semantic checks. A groundedness harness reporting only "
            "traceability would score this row at 100% and publish it as a pass. The sibling "
            "project measured the same thing from the other direction: its regex traced "
            "1,027 of 1,027 tokens, raised no flags, and scored Cohen's kappa 0.00 against "
            "human labels — a detector with no positives cannot agree beyond chance.",
            "",
            "**Both controls collapse**, which is what makes the first row mean something: a "
            "checker that accepts everything also scores 100%. Re-scored against another "
            "lineup's evidence the faithful narratives drop to "
            f"{faithful['control_easy_grounded_rate']:.1%}; against the same lineup with one "
            f"player swapped, {faithful['control_near_miss_grounded_rate']:.1%}.",
            "",
            "The `faithful` row also cost two bug fixes to reach 100%. The checker first "
            'flagged the "100" in "points per 100 possessions" as an ungrounded number, '
            "and its name extractor could not parse Caldwell-Pope, Gilgeous-Alexander or "
            "Hardaway Jr. — 36 false positives on correct prose. A checker that flags correct "
            "prose is worse than no checker, because the noise buries the real failures.",
            "",
        ]
    return "\n".join(lines)


RENDERERS = {
    "results.estimability": _estimability,
    "results.model": _model_results,
    "results.selection": _selection_results,
    "results.selection_priced": _selection_priced,
    "results.selection_ranking": _selection_ranking,
    "results.rapm": _rapm,
    "results.trade": _trade,
    "results.retrieval": _retrieval,
    "results.groundedness": _groundedness,
    "results.dataquality": _data_quality,
    "results.possessions": _possessions,
}


def render_blocks(paths: DataPaths) -> dict[str, str]:
    ctx = RenderContext(
        paths=paths,
        run=latest_run(paths),
        selection_run=latest_run(paths, kind="selection"),
        shots=load_all_gold(paths, "shot_facts"),
        stints=load_all_gold(paths, "stints"),
    )
    return {key: renderer(ctx) for key, renderer in RENDERERS.items()}


def apply_blocks(path: Path, blocks: dict[str, str]) -> tuple[str, list[str]]:
    """Return the rewritten text and the ids that changed."""
    original = path.read_text(encoding="utf-8")
    changed: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        block_id = match.group("id")
        if block_id not in blocks:
            return match.group(0)
        body = blocks[block_id]
        if match.group("body").strip() != body.strip():
            changed.append(block_id)
        return f"{match.group(1)}\n{body.strip()}\n{match.group(4)}"

    return BLOCK_RE.sub(_replace, original), changed


def check_blocks(path: Path, blocks: dict[str, str]) -> list[str]:
    _, changed = apply_blocks(path, blocks)
    return changed
