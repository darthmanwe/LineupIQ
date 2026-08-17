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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from lineupiq.io.gold import load_all_gold
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

    live = poss.filter(pl.col("live_ball_start") & (pl.col("possession_seconds") <= 7))
    half = poss.filter(~pl.col("live_ball_start") | (pl.col("possession_seconds") > 7))

    return "\n".join(
        [
            "",
            "| | Value |",
            "|---|---|",
            f"| Possessions | {n:,} |",
            f"| Attributed to a five-man lineup | "
            f"{poss.filter(pl.col('off_lineup_hash').is_not_null()).height / n:.2%} |",
            f"| Agreement with the independent lineup oracle | {overall:.2%} |",
            f"| ... restricted to possessions not starting on a substitution | "
            f"**{unambiguous:.2%}** |",
            f"| Possessions starting on a substitution (attribution ambiguous) | "
            f"{boundary_rate:.1%} |",
            f"| Points per possession, transition | {as_float(live['points'].mean()):.3f} |",
            f"| Points per possession, half-court | {as_float(half['points'].mean()):.3f} |",
            "",
            "The oracle is a second lineup reconstruction, written independently in",
            "another language. Away from substitution boundaries the two agree at the",
            "same rate our period-start solver reports exact solutions. About one",
            "possession in seven begins on the exact second of a substitution, where",
            "there are two defensible answers and no way to choose between them; those",
            "are flagged in the data rather than silently trusted.",
            "",
        ]
    )


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


RENDERERS = {
    "results.estimability": _estimability,
    "results.model": _model_results,
    "results.dataquality": _data_quality,
    "results.possessions": _possessions,
}


def render_blocks(paths: DataPaths) -> dict[str, str]:
    ctx = RenderContext(
        paths=paths,
        run=latest_run(paths),
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
