"""The ``lineupiq`` command line.

Commands that are not yet backed by data say so and exit non-zero. They do not
print a plausible-looking zero, and they do not succeed quietly. This mirrors
the API, where an unbacked route returns 501 naming what will back it -- the
same honesty rule applied to both surfaces.
"""

from __future__ import annotations

import io
import sys
from typing import NoReturn

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from lineupiq import __version__
from lineupiq.paths import DataPaths, RepoRootNotFound
from lineupiq.seasons import MODELLED_GAME_TYPES, SEASON_COVERAGE, Season


def _force_utf8_stdout() -> None:
    """Make Rich output survive a Windows console.

    Windows still defaults stdout to a legacy code page, and the first box-drawing
    character raises UnicodeEncodeError mid-render -- so the failure looks like a
    crash in whatever command was running.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() not in {
            "utf-8",
            "utf8",
        }:
            stream.reconfigure(encoding="utf-8", errors="replace")


_force_utf8_stdout()

app = typer.Typer(
    name="lineupiq",
    help="NBA lineup and trade forecasting. Offline and free by default.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _not_yet_backed(what: str, milestone: str, backed_by: str) -> NoReturn:
    """Refuse a command that has no data behind it yet.

    Named after the API's 501 registry, and for the same reason: a command that
    prints an empty table is indistinguishable from one that ran correctly
    against an empty dataset.
    """
    console.print(f"[bold yellow]NOT YET BACKED[/] -- {what}")
    console.print(f"  Arrives in : [cyan]{milestone}[/]")
    console.print(f"  Will read  : {backed_by}")
    console.print("\nNothing was computed. This is a deliberate refusal, not a failure.")
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


@app.command()
def seasons() -> None:
    """Show the declared season coverage.

    This is the single place scope is stated. README, model cards and the design
    docs must agree with it, and CI checks that they do.
    """
    table = Table(title="Declared coverage", header_style="bold")
    table.add_column("Season")
    table.add_column("GAME_ID digits", justify="center")
    table.add_column("shufinskiy file", justify="center")
    table.add_column("sportsdataverse file", justify="center")
    for season in SEASON_COVERAGE:
        table.add_row(
            season.label,
            season.two_digit,
            f"…_{season.shufinskiy_year()}.tar.xz",
            f"…_{season.sportsdataverse_year()}.parquet",
        )
    console.print(table)
    console.print(
        "\n[dim]The two mirrors label the same season with different years. "
        "Season is decoded from GAME_ID and asserted, never taken from a filename.[/]"
    )
    console.print(f"[dim]Modelled game types: {', '.join(sorted(MODELLED_GAME_TYPES))}[/]")


@app.command()
def paths() -> None:
    """Show resolved repository paths and which of them exist."""
    try:
        p = DataPaths.discover()
    except RepoRootNotFound as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Resolved paths", header_style="bold")
    table.add_column("Name")
    table.add_column("Path")
    table.add_column("Exists", justify="center")
    for name in ("root", "bronze", "silver", "gold", "contracts", "parity", "runs"):
        path = getattr(p, name)
        table.add_row(
            name,
            str(path.relative_to(p.root) if path != p.root else "."),
            "[green]yes[/]" if path.exists() else "[dim]no[/]",
        )
    console.print(table)


@app.command()
def build(
    season: int | None = typer.Option(
        None, help="Season start year, e.g. 2023. Omit to build every declared season."
    ),
    no_oracles: bool = typer.Option(
        False, "--no-oracles", help="Skip the validation mirrors (faster, less checked)."
    ),
) -> None:
    """Build bronze -> silver for one season, or all declared seasons."""
    from lineupiq.build import build_season

    paths = DataPaths.discover()
    targets = [Season(season)] if season else list(SEASON_COVERAGE)

    table = Table(title="Stint reconstruction", header_style="bold")
    table.add_column("Season")
    table.add_column("Games", justify="right")
    table.add_column("Events", justify="right")
    table.add_column("Stints", justify="right")
    table.add_column("EXACT", justify="right")
    table.add_column("Resolved", justify="right")

    minutes = Table(title="Derived minutes vs box score", header_style="bold")
    minutes.add_column("Season")
    minutes.add_column("Player-games", justify="right")
    minutes.add_column("Mean |Δ|", justify="right")
    minutes.add_column("p95 |Δ|", justify="right")
    minutes.add_column("Total Δ", justify="right")
    minutes.add_column("DNP w/ mins", justify="right")

    for target in targets:
        console.print(f"[dim]building {target.label}…[/]")
        report = build_season(target, paths, oracles=not no_oracles)
        table.add_row(
            target.label,
            f"{report.n_games:,}",
            f"{report.n_events:,}",
            f"{report.n_stints:,}",
            f"{report.exact_rate:.2%}",
            f"{report.resolved_rate:.2%}",
        )
        if (m := report.minutes) is not None:
            minutes.add_row(
                target.label,
                f"{m.n_player_games:,}",
                f"{m.mean_abs_delta:.1f}s",
                f"{m.p95_abs_delta:.0f}s",
                f"{m.pct_total_discrepancy:.3%}",
                ("[red]" if m.n_dnp_with_minutes else "[green]")
                + str(m.n_dnp_with_minutes)
                + "[/]",
            )
        console.print(
            f"  shots={report.n_shots:,}  lineup-coverage={report.shot_lineup_coverage:.2%}"
            f"  3PT-agreement={report.zone_agreement:.3%}"
        )
        if report.missing_sources:
            console.print(f"  [yellow]absent upstream:[/] {', '.join(report.missing_sources)}")

    console.print(table)
    console.print(minutes)


@app.command()
def contracts() -> None:
    """Re-derive and write a contract for every committed gold partition."""
    from lineupiq.io.gold import refresh_contracts

    paths = DataPaths.discover()
    written = refresh_contracts(paths)
    if not written:
        console.print("[yellow]No gold partitions found. Run `lineupiq build` first.[/]")
        raise typer.Exit(code=2)
    for name in written:
        console.print(f"  wrote contract [cyan]{name}[/]")
    console.print(f"\n{len(written)} contracts written to {paths.contracts}")


@app.command()
def verify() -> None:
    """Re-derive every committed gold checksum and run the DQ gates. Offline."""
    from lineupiq.io.gold import GOLD_TABLES, available_seasons, load_all_gold, load_gold
    from lineupiq.validate.checks import run_gates
    from lineupiq.validate.contracts import verify_table

    paths = DataPaths.discover()
    if not paths.gold.exists():
        console.print("[red]No gold layer. Run `lineupiq build` first.[/]")
        raise typer.Exit(code=2)

    failures: list[str] = []

    # --- contracts -------------------------------------------------------
    contract_table = Table(title="Data contracts", header_style="bold")
    contract_table.add_column("Table")
    contract_table.add_column("Partition")
    contract_table.add_column("Rows", justify="right")
    contract_table.add_column("Status")

    checked = 0
    for table in GOLD_TABLES:
        for season in available_seasons(paths, table):
            partition = f"season={season.start_year}"
            path = paths.contracts / f"{table}__{partition}.json"
            frame = load_gold(paths, table, season)
            if not path.exists():
                contract_table.add_row(table, partition, f"{frame.height:,}", "[yellow]absent[/]")
                failures.append(f"{table} {partition}: no committed contract")
                continue
            drifts = verify_table(frame, path)
            checked += 1
            if drifts:
                contract_table.add_row(
                    table, partition, f"{frame.height:,}", "[red]DRIFT[/] " + "; ".join(drifts)
                )
                failures.extend(f"{table} {partition}: {d}" for d in drifts)
            else:
                contract_table.add_row(table, partition, f"{frame.height:,}", "[green]match[/]")

    console.print(contract_table)

    # --- quality gates ---------------------------------------------------
    tables = {
        "shot_facts": load_all_gold(paths, "shot_facts"),
        "stints": load_all_gold(paths, "stints"),
    }
    gate_table = Table(title="Quality gates", header_style="bold")
    gate_table.add_column("Gate")
    gate_table.add_column("Measured", justify="right")
    gate_table.add_column("Threshold", justify="right")
    gate_table.add_column("Verdict")

    for result in run_gates(tables):
        style = {"PASS": "green", "FAIL": "red", "WARN": "yellow"}[result.verdict]
        gate_table.add_row(
            result.name,
            f"{result.measured:.4%}",
            f"{'>=' if result.comparison == 'min' else '<='} {result.threshold:.2%}",
            f"[{style}]{result.verdict}[/]",
        )
        if not result.passed and result.severity == "blocking":
            failures.append(f"gate {result.name}: {result.measured:.4%} vs {result.threshold:.2%}")

    console.print(gate_table)

    if failures:
        console.print(f"\n[bold red]{len(failures)} failure(s)[/]")
        for failure in failures:
            console.print(f"  - {failure}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold green]OK[/] -- {checked} contracts verified, all gates passed.")


@app.command()
def train(
    check: bool = typer.Option(
        False, "--verify", help="Refit and assert the committed metrics reproduce."
    ),
    no_controls: bool = typer.Option(
        False, "--no-controls", help="Skip the shuffled-lineup negative control (faster)."
    ),
) -> None:
    """Fit the shot model and score it against the full baseline ladder."""
    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.train import (
        TOLERANCE,
        compare_to_committed,
        latest_run,
        train_and_evaluate,
        write_run_log,
    )

    paths = DataPaths.discover()
    shots = load_all_gold(paths, "shot_facts")
    console.print(f"[dim]fitting on {shots.height:,} shots…[/]")

    log = train_and_evaluate(shots, run_controls=not no_controls)

    for split, models in log.metrics.items():
        table = Table(
            title=f"{split}  (n={int(models.get('full', {}).get('n', 0)):,})", header_style="bold"
        )
        table.add_column("Model")
        table.add_column("Log loss", justify="right")
        table.add_column("Brier", justify="right")
        table.add_column("Resolution", justify="right")
        table.add_column("ECE", justify="right")
        table.add_column("Slope", justify="right")
        table.add_column("vs B3")

        # Each model is compared to its own no-lineup counterpart, so the
        # verdict isolates lineup information rather than model class.
        counterpart = {"full": "B2", "full_gbdt": "B3"}
        for key in ("B0", "B1", "B2", "B3", "full", "full_gbdt"):
            if key not in models:
                continue
            m = models[key]
            if (ref := counterpart.get(key)) and ref in models:
                base = models[ref]["log_loss"]
                delta = (base - m["log_loss"]) / base
                verdict = (
                    f"[green]+{delta:.3%} vs {ref}[/]"
                    if delta > 0
                    else f"[red]{delta:.3%} vs {ref}[/]"
                )
            else:
                verdict = ""
            table.add_row(
                key,
                f"{m['log_loss']:.5f}",
                f"{m['brier']:.5f}",
                f"{m['resolution']:.5f}",
                f"{m['ece']:.4f}",
                f"{m['calibration_slope']:.3f}",
                verdict,
            )
        console.print(table)

    if log.controls:
        gain = log.controls.get("shuffled_lineup_logloss_gain", 0.0)
        ok = abs(gain) < 1e-3
        console.print(
            f"\n[bold]Negative control[/] -- shuffled-lineup log-loss gain: "
            f"{gain:+.6f} "
            + (
                "[green]PASS[/] (no signal from shuffled context)"
                if ok
                else "[red]FAIL -- lineup features help on shuffled data, which is leakage[/]"
            )
        )

    if check:
        committed = latest_run(paths)
        if committed is None:
            console.print("[yellow]No committed run log to verify against.[/]")
            raise typer.Exit(code=2)
        drifts = compare_to_committed(log, committed)
        if drifts:
            console.print(f"\n[bold red]{len(drifts)} metric(s) moved > {TOLERANCE}[/]")
            for drift in drifts[:20]:
                console.print(f"  - {drift}")
            raise typer.Exit(code=1)
        console.print("\n[bold green]--verify OK[/] -- every metric reproduced.")
        return

    path = write_run_log(log, paths)
    console.print(f"\nrun log written to [cyan]{path.name}[/]")


report_app = typer.Typer(help="Generate documentation blocks from run logs.")
app.add_typer(report_app, name="report")


@report_app.command("render")
def report_render() -> None:
    """Rewrite every generated block in the README from the latest run log."""
    from lineupiq.report.render import apply_blocks, render_blocks

    paths = DataPaths.discover()
    blocks = render_blocks(paths)
    readme = paths.root / "README.md"

    text, changed = apply_blocks(readme, blocks)
    readme.write_text(text, encoding="utf-8", newline="\n")

    if changed:
        for block_id in changed:
            console.print(f"  updated [cyan]{block_id}[/]")
        console.print(f"\n{len(changed)} block(s) rewritten in README.md")
    else:
        console.print("README.md already current.")


@report_app.command("check")
def report_check() -> None:
    """Fail if any committed block is stale. Run this in CI."""
    from lineupiq.report.render import check_blocks, render_blocks

    paths = DataPaths.discover()
    stale = check_blocks(paths.root / "README.md", render_blocks(paths))
    if stale:
        console.print(f"[bold red]{len(stale)} stale block(s):[/] {', '.join(stale)}")
        console.print("Run `lineupiq report render` and commit the result.")
        raise typer.Exit(code=1)
    console.print("[green]README is current.[/]")


@app.command()
def support() -> None:
    """Show the pre-registered support thresholds and how many lineups clear them."""
    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.support import build_lineup_support, load_thresholds, thresholds_hash

    paths = DataPaths.discover()
    thresholds = load_thresholds()
    table = build_lineup_support(load_all_gold(paths, "stints"), load_all_gold(paths, "shot_facts"))

    total = table.height
    reportable = table.filter(
        (pl.col("possessions") >= thresholds.reportable_possessions)
        & (pl.col("min_player_attempts") >= thresholds.reportable_attempts)
    ).height

    out = Table(title="Lineup support", header_style="bold")
    out.add_column("Tier")
    out.add_column("Rule")
    out.add_column("Lineups", justify="right")
    out.add_column("Share", justify="right")
    out.add_row(
        "reportable",
        f">= {thresholds.reportable_possessions} poss and "
        f">= {thresholds.reportable_attempts} attempts",
        f"{reportable:,}",
        f"{reportable / total:.2%}" if total else "-",
    )
    out.add_row("all observed", "any", f"{total:,}", "100.00%")
    console.print(out)
    console.print(f"\nthresholds sha256: [dim]{thresholds_hash()}[/]")
    console.print(
        "[dim]Pre-registered before any lineup-level result was computed; CI asserts "
        "this hash is unchanged.[/]"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
