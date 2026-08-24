"""The ``lineupiq`` command line.

Commands that are not yet backed by data say so and exit non-zero. They do not
print a plausible-looking zero, and they do not succeed quietly. This mirrors
the API, where an unbacked route returns 501 naming what will back it -- the
same honesty rule applied to both surfaces.
"""

from __future__ import annotations

import io
import sys
from typing import Any, NoReturn

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from lineupiq import __version__
from lineupiq.paths import DataPaths, RepoRootNotFound
from lineupiq.runtime import DEFAULT_MEMORY_CAP_GB
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


def _guard_memory(cap_gb: float, *, allow_uncapped: bool) -> None:
    """Impose a hard memory ceiling before any heavy command allocates.

    Two limits, reported together because they guard different failure modes.

    The **memory cap** bounds what this repository can ask of a machine. Past it
    an allocation fails and Python raises MemoryError, which is a stack trace
    instead of a slow slide into swap. If it cannot be applied the command
    refuses rather than running unprotected -- "we tried to limit it" is not a
    limit.

    The **thread count** is the hardware-safety limit, and it is read back live
    rather than assumed: every numeric library fixes its pool size at import, so
    a variable set too late silently does nothing, and a safety limit that
    silently failed is worse than none. Four threads of thirty-two means this
    project never asks for the sustained all-core load that a 13th-generation
    Intel part is worst at.
    """
    from lineupiq.runtime import cap_process_memory, thread_pool_report

    if cap_gb <= 0:
        if not allow_uncapped:
            console.print(
                "[bold red]Refusing to run uncapped.[/] Pass --allow-uncapped to override."
            )
            raise typer.Exit(code=2)
        console.print(
            f"[yellow]Running with no memory cap, by request.[/] "
            f"[dim]Threads: {thread_pool_report()}.[/]"
        )
        return

    result = cap_process_memory(cap_gb)
    if result.applied:
        console.print(f"[dim]{result}; threads: {thread_pool_report()}[/]")
        return

    console.print(f"[bold red]{result}[/]")
    if not allow_uncapped:
        console.print(
            "Refusing to run unprotected. Pass [cyan]--allow-uncapped[/] to accept the risk."
        )
        raise typer.Exit(code=2)
    console.print("[yellow]Continuing unprotected, by request.[/]")


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
    memory_cap_gb: float = typer.Option(
        DEFAULT_MEMORY_CAP_GB, help="Hard ceiling on process memory. 0 disables the cap."
    ),
    allow_uncapped: bool = typer.Option(
        False, "--allow-uncapped", help="Proceed even if the memory cap cannot be applied."
    ),
) -> None:
    """Build bronze -> silver for one season, or all declared seasons."""
    _guard_memory(memory_cap_gb, allow_uncapped=allow_uncapped)

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
        if (pp := report.possessions) is not None:
            console.print(
                f"  possessions={pp.n_possessions:,}  lineup-coverage={pp.lineup_coverage:.2%}"
                f"  oracle-agreement={pp.oracle_agreement:.2%}"
                f" (unambiguous {pp.oracle_agreement_unambiguous:.2%},"
                f" {pp.boundary_ambiguous_rate:.1%} at a sub boundary)"
            )
            console.print(
                f"  PPP transition={pp.transition_ppp:.3f} vs half-court={pp.halfcourt_ppp:.3f}"
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
    from lineupiq.io.gold import (
        GOLD_TABLES,
        POOLED_GOLD_TABLES,
        POOLED_PARTITION,
        available_seasons,
        load_all_gold,
        load_gold,
        load_pooled_gold,
    )
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
    checks: list[tuple[str, str, object]] = [
        (table, f"season={season.start_year}", season)
        for table in GOLD_TABLES
        for season in available_seasons(paths, table)
    ]
    checks += [
        (table, POOLED_PARTITION, None)
        for table in POOLED_GOLD_TABLES
        if (paths.gold / table / POOLED_PARTITION / "part.parquet").exists()
    ]

    for table, partition, season in checks:
        path = paths.contracts / f"{table}__{partition}.json"
        frame = (
            load_pooled_gold(paths, table) if season is None else load_gold(paths, table, season)  # type: ignore[arg-type]
        )
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
    from lineupiq.features.shot_context import attach_possession_context

    shot_facts = load_all_gold(paths, "shot_facts")
    possession_facts = load_all_gold(paths, "possession_facts")
    tables = {
        "shot_facts": shot_facts,
        "stints": load_all_gold(paths, "stints"),
        "possession_facts": possession_facts,
        # Derived here rather than committed: it is a join between two gold
        # tables that are both already contract-checked, so nothing is taken on
        # trust that verify has not already verified.
        "shots_with_context": attach_possession_context(shot_facts, possession_facts),
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
    memory_cap_gb: float = typer.Option(
        DEFAULT_MEMORY_CAP_GB, help="Hard ceiling on process memory. 0 disables the cap."
    ),
    allow_uncapped: bool = typer.Option(
        False, "--allow-uncapped", help="Proceed even if the memory cap cannot be applied."
    ),
) -> None:
    """Fit the shot model and score it against the full baseline ladder."""
    _guard_memory(memory_cap_gb, allow_uncapped=allow_uncapped)

    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.train import (
        BINNED_TOLERANCE,
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
            console.print(
                f"\n[bold red]{len(drifts)} metric(s) moved beyond tolerance[/] "
                f"[dim]({TOLERANCE:g}, or {BINNED_TOLERANCE:g} for binned estimators)[/]"
            )
            for drift in drifts[:20]:
                console.print(f"  - {drift}")
            raise typer.Exit(code=1)
        console.print("\n[bold green]--verify OK[/] -- every metric reproduced.")
        return

    path = write_run_log(log, paths)
    console.print(f"\nrun log written to [cyan]{path.name}[/]")


@app.command()
def selection(
    check: bool = typer.Option(
        False, "--verify", help="Refit and assert the committed metrics reproduce."
    ),
    no_controls: bool = typer.Option(
        False, "--no-controls", help="Skip the shuffled-lineup negative control (faster)."
    ),
    memory_cap_gb: float = typer.Option(
        DEFAULT_MEMORY_CAP_GB, help="Hard ceiling on process memory. 0 disables the cap."
    ),
    allow_uncapped: bool = typer.Option(
        False, "--allow-uncapped", help="Proceed even if the memory cap cannot be applied."
    ),
) -> None:
    """Fit the shot-selection model -- P(zone | shooter, lineup, context).

    The conversion model asks whether a shot goes in and finds that lineups
    barely matter. This asks which shot gets taken, which is where a lineup
    effect would actually live.
    """
    _guard_memory(memory_cap_gb, allow_uncapped=allow_uncapped)

    from lineupiq.features.shot_context import attach_possession_context, context_coverage
    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.selection import SELECTION_TERMS, usable_selection_frame
    from lineupiq.models.train import (
        BINNED_TOLERANCE,
        TOLERANCE,
        compare_to_committed,
        latest_run,
        write_run_log,
    )
    from lineupiq.models.train_selection import (
        COUNTERPART,
        SELECTION_LADDER,
        train_and_evaluate_selection,
    )
    from lineupiq.runtime import MemoryBudget, peak_process_memory_bytes

    paths = DataPaths.discover()
    shots = attach_possession_context(
        load_all_gold(paths, "shot_facts"), load_all_gold(paths, "possession_facts")
    )
    console.print(
        f"[dim]{shots.height:,} shots, possession context on {context_coverage(shots):.2%}[/]"
    )

    # Pre-flight, in two parts.
    #
    # The estimate covers the model's own arrays. The *measured* peak covers
    # everything else -- and everything else turned out to be the larger half:
    # a first version of this command was estimated at 314 MB and had already
    # committed 3.7 GB before fitting anything, because the possession-context
    # join fanned out to eight million rows. An estimate that models only the
    # part you wrote is worse than no estimate, so both are printed and the
    # measured number is the one that gates.
    budget = MemoryBudget(
        n_rows=usable_selection_frame(shots).height,
        n_zones=9,
        n_pair_matrices=2,
        n_wide_columns=28,
    )
    console.print(f"[dim]model arrays: {budget.describe()}[/]")

    measured = peak_process_memory_bytes()
    if measured is not None:
        console.print(f"[dim]measured peak after loading: {measured / 1e6:.0f} MB[/]")
        if memory_cap_gb > 0 and measured > 0.5 * memory_cap_gb * 1e9:
            console.print(
                f"[bold red]Already at {measured / 1e9:.1f} GB of a {memory_cap_gb:.1f} GB cap "
                "before fitting.[/] Raise --memory-cap-gb or narrow the season coverage."
            )
            raise typer.Exit(code=2)

    log = train_and_evaluate_selection(
        shots,
        run_controls=not no_controls,
        paths=paths,
        progress=lambda message: console.print(f"[dim]{message}[/]"),
    )
    console.print(f"[dim]fitted on {log.n_shots:,} shots across {log.n_lineups:,} lineups[/]")

    for split, models in log.metrics.items():
        table = Table(
            title=f"{split}  (n={int(models.get('full', {}).get('n', 0)):,} shots)",
            header_style="bold",
        )
        table.add_column("Model")
        table.add_column("Log loss", justify="right")
        table.add_column("Top-1", justify="right")
        table.add_column("3PA log loss", justify="right")
        table.add_column("3PA resol.", justify="right")
        table.add_column("Classwise ECE", justify="right")
        table.add_column("Verdict")
        for key in SELECTION_LADDER:
            if key not in models:
                continue
            m = models[key]
            verdict = ""
            if (ref := COUNTERPART.get(key)) and ref in models:
                base = models[ref]["log_loss"]
                delta = (base - m["log_loss"]) / base
                colour = "green" if delta > 0 else "red"
                verdict = f"[{colour}]{delta:+.3%} vs {ref}[/]"
            table.add_row(
                key,
                f"{m['log_loss']:.5f}",
                f"{m['top1_accuracy']:.4f}",
                f"{m['three_log_loss']:.5f}",
                f"{m['three_resolution']:.5f}",
                f"{m['classwise_ece']:.5f}",
                verdict,
            )
        console.print(table)

    audit = log.model.get("sign_audit", {})
    if audit:
        detail = {t.name: t.detail for t in SELECTION_TERMS}
        within = log.model.get("within_shooter_coefficients", {})
        signs = Table(title="Pre-registered sign audit", header_style="bold")
        signs.add_column("Term")
        signs.add_column("Coefficient", justify="right")
        signs.add_column("Expected", justify="center")
        signs.add_column("Verdict")
        signs.add_column("Lineup", justify="center")
        for name, row in audit.items():
            agrees = row["verdict"] == "agrees"
            signs.add_row(
                name,
                f"{float(row['value']):+.4f}",
                "+" if row["expected_sign"] == 1 else "-",
                "[green]agrees[/]" if agrees else "[red]DISAGREES[/]",
                "yes" if row["is_lineup"] else "",
            )
        console.print(signs)

        disagreed = [n for n, r in audit.items() if r["verdict"] == "DISAGREES"]
        console.print(
            f"[bold]{len(audit) - len(disagreed)}/{len(audit)}[/] pre-registered signs agree."
        )
        for name in disagreed:
            console.print(f"  [yellow]{name}[/] -- {detail.get(name, '')}")
            if name in within:
                console.print(
                    f"    within-shooter refit: [bold]{within[name]:+.4f}[/] "
                    "(between-player variation removed)"
                )

    if log.controls:
        gain = log.controls.get("shuffled_lineup_logloss_gain", 0.0)
        spacing = log.controls.get("shuffled_lineup_spacing_coefficient", 0.0)
        console.print("\n[bold]Negative control[/] -- lineups randomly reassigned:")
        console.print(
            f"  log-loss gain of full over S2 : {gain:+.6f} "
            + ("[green]PASS[/]" if abs(gain) < 1e-3 else "[red]FAIL -- this is leakage[/]")
        )
        console.print(
            f"  spacing_x_three coefficient   : {spacing:+.4f} "
            + (
                "[green]PASS[/] (collapses toward zero)"
                if abs(spacing) < 0.1
                else "[red]FAIL -- survives shuffling, so it is not a lineup effect[/]"
            )
        )

    if check:
        committed = latest_run(paths, kind="selection")
        if committed is None:
            console.print("[yellow]No committed run log to verify against.[/]")
            raise typer.Exit(code=2)
        drifts = compare_to_committed(log, committed)
        if drifts:
            console.print(
                f"\n[bold red]{len(drifts)} metric(s) moved beyond tolerance[/] "
                f"[dim]({TOLERANCE:g}, or {BINNED_TOLERANCE:g} for binned estimators)[/]"
            )
            for drift in drifts[:20]:
                console.print(f"  - {drift}")
            raise typer.Exit(code=1)
        console.print("\n[bold green]--verify OK[/] -- every metric reproduced.")
        return

    path = write_run_log(log, paths, kind="selection")
    console.print(f"\nrun log written to [cyan]{path.name}[/]")


@app.command()
def rapm(
    folds: int = typer.Option(5, help="Game-grouped folds for lambda selection."),
    memory_cap_gb: float = typer.Option(
        DEFAULT_MEMORY_CAP_GB, help="Hard ceiling on process memory. 0 disables the cap."
    ),
    allow_uncapped: bool = typer.Option(
        False, "--allow-uncapped", help="Proceed even if the memory cap cannot be applied."
    ),
) -> None:
    """Fit RAPM on possessions -- the additive player-effect model.

    A trade delta is a difference of player effects, so without these the trade
    simulator would be a lookup table with opinions. Split-half reliability is
    printed alongside the fit, because that -- not cross-validated error on
    possession outcomes -- is what says whether the coefficients mean anything.
    """
    _guard_memory(memory_cap_gb, allow_uncapped=allow_uncapped)

    import json

    from lineupiq.io.gold import load_all_gold, write_pooled_gold
    from lineupiq.models.rapm import fit_rapm

    paths = DataPaths.discover()
    possessions = load_all_gold(paths, "possession_facts")
    players = load_all_gold(paths, "dim_player")
    names = dict(zip(players["player_id"].to_list(), players["player_name"].to_list(), strict=True))

    console.print(f"[dim]fitting on {possessions.height:,} possessions…[/]")
    report = fit_rapm(possessions, n_folds=folds)
    fit = report.fit

    summary = Table(title="RAPM fit", header_style="bold")
    summary.add_column("")
    summary.add_column("Value", justify="right")
    summary.add_row("Possessions", f"{fit.n_possessions:,}")
    summary.add_row("Players", f"{len(fit.players):,}")
    summary.add_row("lambda offence", f"{fit.lambda_offence:,.0f}")
    summary.add_row("lambda defence", f"{fit.lambda_defence:,.0f}")
    summary.add_row("Effective df", f"{fit.effective_df:,.1f}")
    summary.add_row("Condition number", f"{fit.condition_number:,.1f}")
    summary.add_row("League PPP", f"{fit.league_ppp:.4f}")
    summary.add_row("Home advantage (per 100)", f"{fit.home_advantage:+.2f}")
    summary.add_row("Offensive spread (sd)", f"{report.spread['off_sd']:.2f}")
    summary.add_row("Defensive spread (sd)", f"{report.spread['def_sd']:.2f}")
    console.print(summary)

    reliability = report.reliability
    if reliability.get("n_players"):
        rel = Table(title="Split-half reliability (odd vs even games)", header_style="bold")
        rel.add_column("Side")
        rel.add_column("Pearson r", justify="right")
        rel.add_column("Spearman", justify="right")
        rel.add_column("Full-sample (Spearman-Brown)", justify="right")
        for side in ("off", "def"):
            rel.add_row(
                side,
                f"{reliability[f'{side}_split_half_r']:+.3f}",
                f"{reliability[f'{side}_spearman_rho']:+.3f}",
                f"{reliability[f'{side}_full_sample_reliability']:+.3f}",
            )
        console.print(rel)
        console.print(
            f"[dim]{reliability['n_players']} players with "
            f">= {reliability['min_possessions']} possessions in both halves. This, not "
            "cross-validated error, is the number that says whether RAPM measures "
            "anything: possession outcomes are dominated by shot noise, so a model can "
            "cut CV error while its player coefficients are close to arbitrary.[/]"
        )

    co = report.co_occurrence
    console.print(
        f"\n[bold]Identifiability:[/] {co['n_flagged']} of {co['n_players']} players share "
        f"more than {co['ceiling']:.0%} of their floor time with a single teammate "
        f"(median {co['median_max_co_occurrence']:.0%}). For those, the pair's *sum* is "
        "identified and neither coefficient is -- so they are not served as point estimates."
    )

    if report.boundary_sensitivity:
        b = report.boundary_sensitivity
        console.print(
            f"[bold]Boundary sensitivity:[/] dropping the {b['share_excluded']:.1%} of "
            "possessions that begin on a substitution moves offensive coefficients by "
            f"{b['off_mean_abs_change']:.3f} per 100 on average (correlation "
            f"{b['off_correlation']:.4f}) and defensive by {b['def_mean_abs_change']:.3f} "
            f"(correlation {b['def_correlation']:.4f})."
        )

    # Real possession counts, not zeros. The count is the first thing a reader
    # needs: a +3.5 offensive coefficient off 400 possessions and one off 6,000
    # are different claims, and a leaderboard that hides the difference invites
    # exactly the small-sample reading this project exists to avoid.
    coefficients = report.fit.to_frame(report.appearances)
    frame = coefficients.with_columns(
        pl.col("player_id")
        .map_elements(lambda p: names.get(int(p), str(p)), return_dtype=pl.Utf8)
        .alias("player")
    )
    top = Table(title="Top 15 by total RAPM (per 100 possessions)", header_style="bold")
    top.add_column("Player")
    top.add_column("Off", justify="right")
    top.add_column("Def", justify="right")
    top.add_column("Total", justify="right")
    top.add_column("Poss", justify="right")
    for row in frame.head(15).iter_rows(named=True):
        top.add_row(
            row["player"],
            f"{row['off_rapm']:+.2f}",
            f"{row['def_rapm']:+.2f}",
            f"{row['total_rapm']:+.2f}",
            f"{row['possessions']:,}",
        )
    console.print(top)

    directory = paths.runs / "rapm"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_possessions": fit.n_possessions,
        "n_players": len(fit.players),
        "lambda_offence": fit.lambda_offence,
        "lambda_defence": fit.lambda_defence,
        "effective_df": fit.effective_df,
        "condition_number": fit.condition_number,
        "league_ppp": fit.league_ppp,
        "home_advantage": fit.home_advantage,
        "cv_mse": fit.cv_mse,
        "spread": report.spread,
        "reliability": reliability,
        "co_occurrence": co,
        "boundary_sensitivity": report.boundary_sensitivity,
        "lambda_trace": report.lambda_trace,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (directory / "run.json").write_text(f"{text}\n", encoding="utf-8", newline="\n")

    # Player coefficients are a gold-grade artefact: the trade simulator reads
    # them, so they are committed and contract-checked like any other table.
    # The partition is "pooled" rather than a season -- RAPM is fitted across the
    # whole corpus at once, and a per-season partition would imply three
    # independent fits that do not exist.
    written = write_pooled_gold(paths, "player_rapm", coefficients)
    console.print(f"\nwrote [cyan]{directory / 'run.json'}[/] and gold [cyan]{written}[/]")
    console.print("[dim]Run `lineupiq contracts` to fingerprint the new partition.[/]")


@app.command()
def export() -> None:
    """Write every artefact the Worker serves, as committed JSON.

    Payload sizes are printed because they are a constraint: the Worker parses
    these on a cold start, and a support table that quietly grows to megabytes
    turns a fast endpoint slow with no code change anywhere.
    """
    from lineupiq.serve.export import export_all

    paths = DataPaths.discover()
    manifest = export_all(paths)
    console.print("[bold]Exported[/]")
    console.print(manifest.describe())
    if manifest.total_bytes > 2_000_000:
        console.print(
            "\n[yellow]Payload exceeds 2 MB.[/] The Worker parses this on a cold start; "
            "consider sharding or moving the support table to D1."
        )


@app.command()
def backtest(
    rule: str = typer.Option("inherit", help="Minutes rule: inherit, historical, conservative."),
    buckets: int = typer.Option(6, help="Cutoff buckets; one RAPM fit per bucket."),
    memory_cap_gb: float = typer.Option(
        DEFAULT_MEMORY_CAP_GB, help="Hard ceiling on process memory. 0 disables the cap."
    ),
    allow_uncapped: bool = typer.Option(
        False, "--allow-uncapped", help="Proceed even if the memory cap cannot be applied."
    ),
) -> None:
    """Backtest the trade projection against moves that actually happened.

    The power analysis is printed **first**, before any result. At the sample
    size three seasons of mid-season moves provide, the expected honest answer
    is that no accuracy claim is supported -- and that is a pre-commitment, not
    a retreat after seeing the number.
    """
    _guard_memory(memory_cap_gb, allow_uncapped=allow_uncapped)

    import json

    from lineupiq.eval.backtest_trade import run_trade_backtest
    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.trade import rule_by_name

    paths = DataPaths.discover()
    possessions = load_all_gold(paths, "possession_facts")
    chosen = rule_by_name(rule)

    console.print(f"[dim]minutes rule: [bold]{chosen.name}[/] -- {chosen.detail}[/]")
    result = run_trade_backtest(
        possessions,
        rule=chosen,
        n_buckets=buckets,
        progress=lambda message: console.print(f"[dim]{message}[/]"),
    )

    power = Table(title="Power analysis -- computed before any result", header_style="bold")
    power.add_column("")
    power.add_column("Value", justify="right")
    power.add_row("Evaluable mid-season moves", f"{result.n_moves}")
    power.add_row("Team net-rating noise (sd)", f"{result.residual_sd:.2f} per 100")
    power.add_row("Minimum detectable effect", f"{float(result.power['mde']):.2f} per 100")
    power.add_row(
        "Sign-accuracy 95% half-width",
        f"+/-{float(result.power['sign_accuracy_ci_half_width']):.1%}",
    )
    verdict = str(result.power["verdict"])
    power.add_row(
        "Verdict", f"[red]{verdict}[/]" if verdict == "UNDERPOWERED" else f"[green]{verdict}[/]"
    )
    console.print(power)

    if verdict == "UNDERPOWERED":
        console.print(
            "[bold yellow]No accuracy claim is made at this sample size.[/] The numbers "
            "below are reported for completeness and for the placebo comparison, not as "
            "evidence that the projection works."
        )

    for label, scores in (("Real moves", result.real), ("Placebo (non-movers)", result.placebo)):
        if not scores or scores.get("n", 0) < 3:
            console.print(f"[dim]{label}: n = {scores.get('n', 0)}, too few to score[/]")
            continue
        out = Table(title=label, header_style="bold")
        out.add_column("")
        out.add_column("Value", justify="right")
        out.add_row("n", f"{int(scores['n'])}")
        out.add_row("Mean projected delta", f"{float(scores['mean_projected']):+.3f}")
        out.add_row("Mean observed delta", f"{float(scores['mean_observed']):+.3f}")
        out.add_row("Mean DiD delta", f"{float(scores['mean_did']):+.3f}")
        out.add_row("Corr(projected, DiD)", f"{float(scores['correlation_projected_did']):+.3f}")
        low, high = scores["sign_agreement_ci"]
        out.add_row(
            "Sign agreement vs DiD",
            f"{float(scores['sign_agreement_vs_did']):.1%} [{float(low):.0%}, {float(high):.0%}]",
        )
        out.add_row("MAE vs DiD", f"{float(scores['mean_abs_error_vs_did']):.3f}")
        console.print(out)

    if result.variance:
        console.print(
            f"\n[bold]Variance decomposition:[/] the minutes rule carries "
            f"{result.variance['mean_minutes_variance_share']:.0%} of a projection's "
            f"variance on average, and dominates it in "
            f"{result.variance['share_where_minutes_dominates']:.0%} of cases. "
            f"{result.variance['share_interval_includes_zero']:.0%} of 80% intervals "
            "contain zero."
        )

    for note in result.notes:
        console.print(f"[yellow]note:[/] {note}")

    directory = paths.runs / "trade"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "minutes_rule": result.minutes_rule,
        "n_moves": result.n_moves,
        "n_placebo": result.n_placebo,
        "residual_sd": result.residual_sd,
        "power": result.power,
        "real": result.real,
        "placebo": result.placebo,
        "variance": result.variance,
        "notes": result.notes,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (directory / f"{result.minutes_rule}.json").write_text(
        f"{text}\n", encoding="utf-8", newline="\n"
    )
    console.print(f"\nrun log written to [cyan]{directory / f'{result.minutes_rule}.json'}[/]")


@app.command()
def parity() -> None:
    """Write the Python/TypeScript parity fixtures.

    The Worker re-implements three things: the lineup hash, the support tier,
    and the whole served selection model. None of the three raises on
    disagreement -- a hash mismatch returns zero rows and looks like missing
    data, a tier mismatch serves a confident number where Python would have
    refused, and a scorer mismatch serves a plausible shot mix that is wrong.

    So Python writes its answers for a fixed sample and a vitest suite inside
    workerd asserts TypeScript reproduces them, to 1e-9. Neither implementation
    is the reference; the fixture is the contract.
    """
    from lineupiq.serve.parity import write_parity_fixture, write_selection_parity_fixture

    paths = DataPaths.discover()
    path = write_parity_fixture(paths)
    import json as _json

    fixture = _json.loads(path.read_text(encoding="utf-8"))
    table = Table(title="Parity fixture", header_style="bold")
    table.add_column("")
    table.add_column("Value", justify="right")
    table.add_row("Cases", f"{fixture['n_cases']:,}")
    for tier, count in sorted(fixture["tier_counts"].items()):
        table.add_row(f"  tier: {tier}", f"{count:,}")
    console.print(table)
    console.print(f"wrote [cyan]{path}[/]")
    if min(fixture["tier_counts"].values()) == 0:
        console.print(
            "[yellow]At least one tier has no cases.[/] A parity fixture that never "
            "exercises a branch cannot prove the branch agrees."
        )

    # The scorer's own fixture. Separate file, separate sample, because it
    # exercises different branches: an unseen shooter, a two-man lineup, an
    # empty defence, a team/season that never existed.
    selection_path = write_selection_parity_fixture(paths)
    selection = _json.loads(selection_path.read_text(encoding="utf-8"))
    scorer = Table(title="Selection scorer parity", header_style="bold")
    scorer.add_column("")
    scorer.add_column("Value", justify="right")
    scorer.add_row("Cases", f"{selection['n_cases']:,}")
    scorer.add_row("Unseen shooters", f"{selection['n_unknown_shooters']:,}")
    scorer.add_row("Terms", f"{len(selection['term_names'])}")
    console.print(scorer)
    console.print(f"wrote [cyan]{selection_path}[/]")
    if selection["n_unknown_shooters"] == 0:
        console.print(
            "[yellow]No unseen shooter in the sample.[/] The league-mix fallback is "
            "the branch most likely to differ between the two languages, and an "
            "all-known sample cannot prove it agrees."
        )


snowflake_app = typer.Typer(help="The optional Snowflake adapter. Off the demo path.")
app.add_typer(snowflake_app, name="snowflake")


@snowflake_app.command("ddl")
def snowflake_ddl() -> None:
    """Generate the Snowflake DDL from the committed data contracts.

    The contracts are derived from the parquet on disk, so generating the SQL
    from them makes schema drift impossible rather than merely discouraged.
    Hand-maintaining a parallel .sql file is how a repository ends up with
    pretty SQL that does not match its data.
    """
    from lineupiq.io.schema import generate_snowflake_ddl, load_specs, write_snowflake_ddl

    paths = DataPaths.discover()
    specs = load_specs(paths)
    if not specs:
        console.print("[yellow]No committed contracts. Run `lineupiq contracts` first.[/]")
        raise typer.Exit(code=2)

    files = generate_snowflake_ddl(specs)
    written = write_snowflake_ddl(paths, files)

    table = Table(title="Generated DDL", header_style="bold")
    table.add_column("File")
    table.add_column("Bytes", justify="right")
    for path in written:
        table.add_row(path.name, f"{path.stat().st_size:,}")
    console.print(table)

    spec_table = Table(title="Tables described", header_style="bold")
    spec_table.add_column("Schema")
    spec_table.add_column("Table")
    spec_table.add_column("Columns", justify="right")
    spec_table.add_column("Rows", justify="right")
    for spec in sorted(specs, key=lambda s: (s.schema, s.table)):
        spec_table.add_row(spec.schema, spec.table, str(len(spec.columns)), f"{spec.rows:,}")
    console.print(spec_table)


@snowflake_app.command("check")
def snowflake_check() -> None:
    """Assert the committed DDL still matches the committed contracts.

    The load-bearing half of the Snowflake story: no account, no cost, and it
    catches the failure that actually happens -- a gold table gaining a column
    while the SQL does not.
    """
    from lineupiq.io.schema import schema_parity_problems

    problems = schema_parity_problems(DataPaths.discover())
    if problems:
        console.print(f"[bold red]{len(problems)} schema parity problem(s)[/]")
        for problem in problems:
            console.print(f"  - {problem}")
        raise typer.Exit(code=1)
    console.print("[green]Snowflake DDL matches every committed contract.[/]")


retrieval_app = typer.Typer(help="Document construction and retrieval evaluation.")
app.add_typer(retrieval_app, name="retrieval")


@retrieval_app.command("ablation")
def retrieval_ablation(
    memory_cap_gb: float = typer.Option(
        DEFAULT_MEMORY_CAP_GB, help="Hard ceiling on process memory. 0 disables the cap."
    ),
    allow_uncapped: bool = typer.Option(
        False, "--allow-uncapped", help="Proceed even if the memory cap cannot be applied."
    ),
) -> None:
    """Compare three corpora on the same queries with the same retrievers.

    The design document asserts that document design drives retrieval quality.
    This measures it: the same facts rendered as an event log, as bare decimals,
    and as the full template with names, archetypes and comparatives.
    """
    _guard_memory(memory_cap_gb, allow_uncapped=allow_uncapped)

    import json

    from lineupiq.io.gold import load_all_gold
    from lineupiq.retrieval.docs import build_documents
    from lineupiq.retrieval.evaluate import run_ablation, summarise

    paths = DataPaths.discover()
    docs = build_documents(
        load_all_gold(paths, "shot_facts"),
        load_all_gold(paths, "possession_facts"),
        load_all_gold(paths, "dim_player"),
    )
    console.print(f"[dim]{len(docs):,} documents at (lineup, team, season) grain[/]")
    if not docs:
        console.print("[yellow]No documents. Run `lineupiq build` first.[/]")
        raise typer.Exit(code=2)

    report = run_ablation(docs)
    console.print(f"[dim]{report.n_queries} generated queries[/]")

    table = Table(title="Corpus ablation -- Recall@10 / MRR / nDCG@10", header_style="bold")
    table.add_column("Corpus")
    for name in ("bm25", "lsa", "rrf"):
        table.add_column(name, justify="right")
    for variant, scores in report.by_corpus.items():
        table.add_row(
            variant,
            *[
                f"{scores[name]['recall']:.3f} / {scores[name]['mrr']:.3f} / "
                f"{scores[name]['ndcg']:.3f}"
                for name in ("bm25", "lsa", "rrf")
            ],
        )
    console.print(table)

    if report.by_kind:
        kinds = Table(title="Recall@10 by query kind, full corpus", header_style="bold")
        kinds.add_column("Query kind")
        for name in ("bm25", "lsa", "rrf"):
            kinds.add_column(name, justify="right")
        for kind, by_retriever in sorted(report.by_kind.items()):
            kinds.add_row(
                kind,
                *[f"{by_retriever.get(name, 0.0):.3f}" for name in ("bm25", "lsa", "rrf")],
            )
        console.print(kinds)

    full = report.by_corpus.get("full", {})
    if full:
        best = max(("bm25", "lsa", "rrf"), key=lambda n: full[n]["recall"])
        console.print(f"\n[bold]Best retriever on the full corpus:[/] {best}")
        if best == "bm25":
            console.print(
                "[yellow]BM25 alone beats the hybrid.[/] Reported because it is the result: "
                "the corpus is built from a closed vocabulary and named entities, which is "
                "exactly what lexical matching is good at. A dense leg earns its place only "
                "when queries are phrased in words the documents do not contain."
            )

    for note in report.notes:
        console.print(f"[dim]note: {note}[/]")

    directory = paths.runs / "retrieval"
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summarise(report), indent=2, sort_keys=True)
    (directory / "ablation.json").write_text(f"{text}\n", encoding="utf-8", newline="\n")
    console.print(f"\nrun log written to [cyan]{directory / 'ablation.json'}[/]")


@app.command()
def groundedness() -> None:
    """Score templated narratives against their evidence. Offline, no model call.

    The narratives are templated, not generated -- no language model has been
    called by this repository. What is being measured is the *checker*, which is
    the part that would still be needed once a model is writing, and whose limits
    are the interesting finding either way.
    """
    import json

    from lineupiq.io.gold import load_all_gold
    from lineupiq.llm.groundedness import CHECKS, check_narrative, score_corpus
    from lineupiq.llm.narratives import TEMPLATES, build_corpus
    from lineupiq.retrieval.docs import build_documents

    paths = DataPaths.discover()
    docs = build_documents(
        load_all_gold(paths, "shot_facts"),
        load_all_gold(paths, "possession_facts"),
        load_all_gold(paths, "dim_player"),
    )
    if not docs:
        console.print("[yellow]No documents. Run `lineupiq build` first.[/]")
        raise typer.Exit(code=2)

    # Both ends of the support distribution. Taking only the best-evidenced
    # groups would let the tier-consistency check pass trivially, because the
    # overclaiming template is only *wrong* below the reporting floor.
    sample = docs[:100] + docs[-100:]
    by_doc = {doc.doc_id: doc for doc in sample}
    below = sum(1 for doc in sample if doc.below_reporting_floor)
    corpus = build_corpus(sample, limit=len(sample))

    console.print(
        f"[dim]{len(sample)} documents ({below} below the reporting floor), "
        f"{len(TEMPLATES)} templates[/]"
    )

    table = Table(title="Groundedness by template", header_style="bold")
    table.add_column("Template")
    table.add_column("n", justify="right")
    table.add_column("Grounded", justify="right")
    table.add_column("Traceability", justify="right")
    table.add_column("Easy control", justify="right")
    table.add_column("Near-miss control", justify="right")

    payload: dict[str, Any] = {"n_documents": len(sample), "n_below_floor": below}
    scores: dict[str, dict[str, Any]] = {}
    for template in TEMPLATES:
        result = score_corpus(corpus[template], by_doc)
        scores[template] = result
        table.add_row(
            template,
            f"{int(result['n'])}",
            f"{float(result['grounded_rate']):.1%}",
            f"{float(result['mean_traceability']):.1%}",
            f"{float(result['control_easy_grounded_rate']):.1%}",
            f"{float(result['control_near_miss_grounded_rate']):.1%}",
        )
    console.print(table)
    payload["by_template"] = scores

    failures = Table(title="Which check caught what", header_style="bold")
    failures.add_column("Template")
    failures.add_column("Check")
    failures.add_column("Narratives flagged", justify="right")
    for template in TEMPLATES:
        counts = scores[template].get("failures_by_check", {})
        if not isinstance(counts, dict) or not counts:
            failures.add_row(template, "[dim]none[/]", "0")
            continue
        for check, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            failures.add_row(template, check, f"{count}")
    console.print(failures)

    detail = dict(CHECKS)
    overclaiming = scores["overclaiming"].get("failures_by_check", {})
    if isinstance(overclaiming, dict) and "tier_consistency" in overclaiming:
        console.print(
            f"[bold]The finding:[/] the overclaiming template quotes only correct numbers -- "
            f"its traceability is "
            f"{float(scores['overclaiming']['mean_traceability']):.1%} -- and "
            f"{overclaiming['tier_consistency']} narratives still fail, on "
            f"[cyan]tier_consistency[/]: {detail['tier_consistency']} "
            "Arithmetic settles provenance and cannot settle meaning."
        )

    directory = paths.runs / "groundedness"
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    (directory / "run.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    console.print(f"\nrun log written to [cyan]{directory / 'run.json'}[/]")

    example = next((d for d in sample if d.below_reporting_floor), sample[0])
    console.print(f"\n[dim]Example -- {example.doc_id}, {example.possessions} possessions:[/]")
    for template in TEMPLATES:
        checked = check_narrative(corpus[template][example.doc_id], example)
        verdict = (
            "[green]grounded[/]"
            if checked.grounded
            else "[red]" + ", ".join(checked.failures) + "[/]"
        )
        console.print(f"  {template:14s} {verdict}")


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
