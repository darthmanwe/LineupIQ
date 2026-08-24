"""Exporting what the Worker serves.

The Worker cannot read parquet, cannot run Python, and has a 10 ms CPU budget
per request. So everything it needs is exported here as compact JSON, committed,
and shipped as a static asset.

What is *not* exported matters as much as what is. The full support table is
about forty thousand five-man groups, and shipping all of it would mean parsing
megabytes on every cold start to answer a question about five players. Only the
groups that clear the directional floor are exported -- a few thousand -- because
a group below that floor is refused anyway, and the refusal needs no row. A
lineup absent from the export is therefore treated as having zero observed
possessions, which is exactly right for a counterfactual combination and exactly
right for a group with two possessions.

Everything here is derived from committed gold, so the export is reproducible
from a clean clone with no network and no account.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lineupiq.models.support import load_thresholds, thresholds_hash
from lineupiq.paths import DataPaths
from lineupiq.transform.zones import ZONES
from lineupiq.util import as_float

__all__ = ["ExportManifest", "export_all", "export_players", "export_support", "export_zones"]

#: Floats are written at this many decimals. Enough for exact parity at 1e-9
#: after a round trip, short enough that the payload does not double in size.
_PRECISION = 10


def _round(value: object) -> float:
    return round(as_float(value), _PRECISION)


def _json_safe(value: Any) -> Any:
    """Replace every non-finite float with null, recursively.

    ``json.dumps`` emits bare ``NaN`` and ``Infinity`` by default. Those are
    Python literals, **not JSON** -- no browser and no Worker can parse them, so
    an export containing one is a 500 in production and a suite that cannot even
    import its fixtures in test.

    A run log legitimately contains NaN: a placebo's sign agreement is undefined,
    and a fit without covariance has no standard error. `null` is what that means
    in JSON, and the writer below passes ``allow_nan=False`` so a future NaN
    fails loudly here instead of shipping.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True)
class ExportManifest:
    """What was written, and how large it turned out."""

    files: dict[str, int]

    @property
    def total_bytes(self) -> int:
        return sum(self.files.values())

    def describe(self) -> str:
        lines = [f"  {name:28s} {size / 1024:8.1f} KB" for name, size in sorted(self.files.items())]
        lines.append(f"  {'total':28s} {self.total_bytes / 1024:8.1f} KB")
        return "\n".join(lines)


def _write(directory: Path, name: str, payload: Any) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    # Separators without spaces, and sorted keys so the bytes are stable and a
    # diff shows a data change rather than a serialisation change.
    # `allow_nan=False` is the guard: if `_json_safe` ever misses a non-finite
    # value, this raises instead of writing a file nothing can parse.
    text = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8", newline="\n")
    return len(text.encode("utf-8"))


def export_zone_surface(paths: DataPaths) -> dict[str, Any]:
    """League-wide points per attempt by zone, and the deviation from the mean.

    This is what the court heatmap fills: **points per attempt minus the league
    average**, which is a *diverging* quantity and is encoded with a diverging
    scale. Filling raw points-per-attempt with a sequential ramp would be the
    obvious mistake -- restricted-area value dwarfs corner-three value, so a
    light-to-dark ramp would just redraw the court and say nothing.

    It is the **league** surface, not a per-lineup one. A per-lineup surface
    needs the served scorer, which is not built; the page says so rather than
    implying the colours are lineup-specific.
    """
    from lineupiq.io.gold import load_all_gold

    shots = load_all_gold(paths, "shot_facts")
    league = as_float((shots["made"] * shots["shot_points"]).mean())

    thresholds = load_thresholds()

    def surface(frame: pl.DataFrame) -> dict[str, Any]:
        grouped = (
            frame.group_by("zone_id")
            .agg(
                pl.len().alias("attempts"),
                pl.col("made").mean().alias("fg"),
                (pl.col("made") * pl.col("shot_points")).mean().alias("points_per_attempt"),
            )
            .sort("zone_id")
        )
        return {
            row["zone_id"]: {
                "attempts": int(row["attempts"]),
                "fg": _round(row["fg"]),
                "points_per_attempt": _round(row["points_per_attempt"]),
                "deviation": _round(as_float(row["points_per_attempt"]) - league),
                # The mark, not a tooltip, carries this. A zone below the floor
                # renders hatched with a dashed edge and no value.
                "below_floor": int(row["attempts"]) < thresholds.min_zone_attempts,
            }
            for row in grouped.iter_rows(named=True)
        }

    # Two worked examples, because at league scale no zone is ever below the
    # floor -- 6,176 attempts in the thinnest zone against a floor of ten. A
    # refusal rendering that never fires on the data shipped beside it is
    # decoration, so a genuinely low-volume shooter is exported alongside a
    # high-volume one and the page shows both courts.
    per_player = shots.group_by("shooter_id").agg(pl.len().alias("attempts")).sort("attempts")
    thin = per_player.filter(pl.col("attempts").is_between(40, 90))
    thick = per_player.sort("attempts", descending=True)

    examples: dict[str, Any] = {}
    players = load_all_gold(paths, "dim_player")
    names = dict(zip(players["player_id"].to_list(), players["player_name"].to_list(), strict=True))
    for label, table in (("high_volume", thick), ("low_volume", thin)):
        if table.is_empty():
            continue
        player_id = int(table["shooter_id"].item(0))
        block = shots.filter(pl.col("shooter_id") == player_id)
        examples[label] = {
            "player_id": player_id,
            "player_name": names.get(player_id, str(player_id)),
            "attempts": block.height,
            "zones": surface(block),
        }

    return {
        "league_points_per_attempt": _round(league),
        "min_zone_attempts": thresholds.min_zone_attempts,
        "zones": surface(shots),
        "examples": examples,
    }


def export_zones() -> dict[str, Any]:
    """The zone vocabulary. One definition, two consumers.

    The court heatmap and the model must agree about what "corner three" means.
    Exporting the taxonomy rather than restating it in TypeScript is what makes
    that structural instead of a convention someone has to remember.
    """
    from lineupiq.transform.zone_geometry import COURT_VIEWBOX, ZONE_OUTLINES, zone_svg_path

    # The SVG path ships with the zone. This is the whole point of the file: the
    # court heatmap draws what the model scores, because both come from the same
    # geometry constants. A test walks a dense grid asserting that every point
    # inside an outline is a point `derive_zone` puts in that zone.
    return {
        "viewBox": COURT_VIEWBOX,
        "zones": [
            {
                "id": zone_id,
                "label": label,
                "path": zone_svg_path(ZONE_OUTLINES[zone_id]),
                "labelAt": _zone_label_anchor(zone_id),
            }
            for zone_id, label in ZONES
        ],
        "count": len(ZONES),
    }


#: Where each zone's own count is drawn, in SVG coordinates.
#:
#: Hand-placed rather than computed from a centroid: a polygon centroid for the
#: lane-minus-restricted-area ring lands inside the hole, and for the top-three
#: region it lands off the arc. Every zone labels its own n, so the anchor has to
#: be somewhere a reader will actually look.
_ZONE_LABEL_ANCHORS: dict[str, tuple[float, float]] = {
    "restricted_area": (0.0, 375.0),
    "paint_non_ra": (0.0, 250.0),
    "mid_baseline": (-150.0, 350.0),
    "mid_wing": (-165.0, 230.0),
    "mid_top": (0.0, 185.0),
    "corner_three_left": (-235.0, 350.0),
    "corner_three_right": (235.0, 350.0),
    "wing_three": (-225.0, 245.0),
    "top_three": (0.0, 95.0),
}


def _zone_label_anchor(zone_id: str) -> dict[str, float]:
    x, y = _ZONE_LABEL_ANCHORS.get(zone_id, (0.0, 200.0))
    return {"x": x, "y": y}


def export_support(paths: DataPaths) -> dict[str, Any]:
    """Observed evidence per five-man group, above the directional floor."""
    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.support import build_lineup_support

    thresholds = load_thresholds()
    table = build_lineup_support(load_all_gold(paths, "stints"), load_all_gold(paths, "shot_facts"))
    total = table.height
    kept = table.filter(pl.col("possessions") >= thresholds.directional_possessions)

    return {
        "thresholds": {
            "reportable_possessions": thresholds.reportable_possessions,
            "reportable_attempts": thresholds.reportable_attempts,
            "directional_possessions": thresholds.directional_possessions,
            "directional_attempts": thresholds.directional_attempts,
        },
        "thresholds_sha256": thresholds_hash(),
        "n_observed_lineups": total,
        "n_exported": kept.height,
        # A lineup absent from this map has fewer possessions than the
        # directional floor, which is the same as having none for every decision
        # the API makes.
        "lineups": {
            row["lineup_hash"]: [int(row["possessions"]), int(row["min_player_attempts"])]
            for row in kept.iter_rows(named=True)
        },
    }


def export_players(paths: DataPaths) -> dict[str, Any]:
    """Names, shot volume, and RAPM with its standard errors.

    ``off_includes_zero`` and ``def_includes_zero`` are exported rather than
    recomputed in TypeScript, so the refusal contract's arithmetic lives in one
    place. A defensive coefficient whose interval contains zero is not evidence
    of defence, and the surface that displays it needs to know that without
    doing statistics.
    """
    from lineupiq.io.gold import load_all_gold, load_pooled_gold

    players = load_all_gold(paths, "dim_player")
    attempts = (
        load_all_gold(paths, "shot_facts")
        .group_by("shooter_id")
        .agg(pl.len().alias("attempts"))
        .rename({"shooter_id": "player_id"})
    )
    frame = players.join(attempts, on="player_id", how="left").with_columns(
        pl.col("attempts").fill_null(0)
    )

    try:
        rapm = load_pooled_gold(paths, "player_rapm")
        frame = frame.join(rapm, on="player_id", how="left")
    except FileNotFoundError:
        # RAPM has not been fitted yet. Exporting the players without it is
        # correct; inventing zeroes would not be.
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("off_rapm"),
            pl.lit(None, dtype=pl.Float64).alias("def_rapm"),
            pl.lit(None, dtype=pl.Float64).alias("off_se"),
            pl.lit(None, dtype=pl.Float64).alias("def_se"),
            pl.lit(None, dtype=pl.Boolean).alias("off_includes_zero"),
            pl.lit(None, dtype=pl.Boolean).alias("def_includes_zero"),
            pl.lit(0, dtype=pl.Int64).alias("possessions"),
        )

    out: dict[str, Any] = {}
    for row in frame.iter_rows(named=True):
        entry: dict[str, Any] = {
            "name": row["player_name"],
            "attempts": int(row["attempts"]),
        }
        if row.get("off_rapm") is not None:
            entry.update(
                {
                    "off_rapm": _round(row["off_rapm"]),
                    "def_rapm": _round(row["def_rapm"]),
                    "off_se": _round(row["off_se"]),
                    "def_se": _round(row["def_se"]),
                    "off_includes_zero": bool(row["off_includes_zero"]),
                    "def_includes_zero": bool(row["def_includes_zero"]),
                    "possessions": int(row.get("possessions") or 0),
                }
            )
        out[str(row["player_id"])] = entry
    return {"players": out, "count": len(out)}


def export_player_zones(paths: DataPaths) -> dict[str, Any]:
    """Per-player, per-zone make rates, each shipping its own shrinkage weight.

    These are the conversion model's profiles, fitted on the full corpus. The
    weight beside every rate is the share carried by that player's own evidence
    rather than by the league prior, and it is exported rather than left in a
    model card because it is the difference between "he shoots 68% at the rim"
    and "we have eleven attempts and the league shoots 65%".

    Only players with at least one recorded attempt in a zone get a row for it.
    A shrunk rate exists for every (player, zone) pair the fit saw, but serving a
    rate for a zone a player has never shot from would be serving the prior
    while looking like a measurement.
    """
    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.shot_model import fit_profiles
    from lineupiq.transform.zones import ZONE_IDS

    shots = load_all_gold(paths, "shot_facts")
    profiles = fit_profiles(shots)

    observed = (
        shots.group_by(["shooter_id", "zone_id"])
        .agg(pl.len().alias("attempts"), pl.col("made").sum().alias("makes"))
        .filter(pl.col("attempts") > 0)
    )

    players: dict[str, list[dict[str, Any]]] = {}
    for row in observed.iter_rows(named=True):
        pid, zone = int(row["shooter_id"]), str(row["zone_id"])
        attempts, makes = int(row["attempts"]), int(row["makes"])
        players.setdefault(str(pid), []).append(
            {
                "zone_id": zone,
                "attempts": attempts,
                # The raw rate alongside the shrunk one, so a reader can see how
                # far the prior moved it. Publishing only the shrunk value hides
                # the mechanism doing the most work on small samples.
                "raw_rate": _round(makes / attempts),
                "shrunk_rate": _round(profiles.player_zone_rate.get((pid, zone), 0.0)),
                "shrinkage_weight": _round(profiles.player_zone_weight.get((pid, zone), 0.0)),
            }
        )

    for rows in players.values():
        rows.sort(key=lambda r: ZONE_IDS.index(r["zone_id"]))

    return {
        "zones": list(ZONE_IDS),
        "league_zone_rate": {z: _round(r) for z, r in sorted(profiles.zone_rate.items())},
        "players": players,
        "count": len(players),
    }


def export_selection_model(paths: DataPaths) -> dict[str, Any]:
    """The served selection model: coefficients plus the per-player mixes.

    This is the whole closed form the Worker evaluates. Nine dot products and a
    softmax, which is microseconds against a 10 ms budget -- the reason the model
    is a conditional logit rather than the gradient-boosted reference it is
    benchmarked against.
    """
    from lineupiq.models.train import latest_run

    run = latest_run(paths, kind="selection")
    if run is None:
        return {"available": False, "reason": "no selection run log committed"}

    model = run.get("model", {})
    return {
        "available": True,
        "git_sha": run.get("git_sha"),
        "term_names": model.get("term_names", []),
        "coefficients": [_round(c) for c in model.get("coefficients", [])],
        "observed_mix": {k: _round(v) for k, v in (model.get("observed_mix") or {}).items()},
        "sign_audit": model.get("sign_audit", {}),
        "n_shots": run.get("n_shots"),
        "seasons": run.get("seasons", []),
    }


def export_selection_profiles(paths: DataPaths) -> dict[str, Any]:
    """Everything the Worker needs to evaluate the selection model itself.

    ``export_selection_model`` ships the coefficients; this ships the profiles
    they multiply. Together they are the whole closed form, which is the point of
    choosing a conditional logit over the gradient-boosted reference: scoring a
    counterfactual lineup is nine dot products and a softmax, so it fits inside a
    10 ms CPU budget with room to spare.

    The profiles are refitted here on the **full** corpus, which is the served
    model -- the same thing ``train_selection`` fits after cross-validation. It
    is not the fold-wise fit, and it must not be: a served model that had never
    seen a third of the league would be a worse model for no benefit, because
    serving is not an evaluation.

    Only players who actually appear are exported. An unseen shooter's log-ratio
    is exactly zero by construction, so the Worker falls back to the
    alternative-specific constants rather than to some arbitrary player -- and
    that fallback needs no data to express.
    """
    from lineupiq.features.shot_context import attach_possession_context
    from lineupiq.io.gold import load_all_gold
    from lineupiq.models.selection import fit_selection_profiles, zone_attribute
    from lineupiq.transform.zones import ZONE_IDS

    # The possession context has to be attached the same way `selection` attaches
    # it, or the fitted `seconds_mean`/`seconds_std` would not be the ones the
    # coefficients were fitted against.
    shots = attach_possession_context(
        load_all_gold(paths, "shot_facts"), load_all_gold(paths, "possession_facts")
    ).filter(pl.col("lineup_for_hash").is_not_null() & pl.col("lineup_against_hash").is_not_null())
    profiles = fit_selection_profiles(shots)

    league = np.log(np.maximum(profiles.league_mix, 1e-12))

    def log_ratio(mix: np.ndarray) -> list[float]:
        return [_round(v) for v in (np.log(np.maximum(mix, 1e-12)) - league)]

    # Stored as the log ratio rather than the raw mix. The Worker needs only the
    # ratio, computing it there would duplicate the epsilon clamp in a second
    # language, and a clamp that disagrees across the boundary is exactly the
    # class of bug the parity fixture exists to catch.
    return {
        "zones": list(ZONE_IDS),
        "rim": [float(v) for v in zone_attribute("rim")],
        "three": [float(v) for v in zone_attribute("three")],
        "league_mix": [_round(v) for v in profiles.league_mix],
        "shooter_log_ratio": {
            str(pid): log_ratio(mix) for pid, mix in sorted(profiles.shooter_mix.items())
        },
        "shooter_weight": {
            str(pid): _round(w) for pid, w in sorted(profiles.shooter_weight.items())
        },
        "team_log_ratio": {
            f"{team}:{season}": log_ratio(mix)
            for (team, season), mix in sorted(profiles.team_mix.items())
        },
        "player_three_rate": {
            str(pid): _round(v) for pid, v in sorted(profiles.player_three_rate.items())
        },
        "player_rim_rate": {
            str(pid): _round(v) for pid, v in sorted(profiles.player_rim_rate.items())
        },
        "opp_three_allowed": {
            str(pid): _round(v) for pid, v in sorted(profiles.opp_three_allowed.items())
        },
        "opp_rim_allowed": {
            str(pid): _round(v) for pid, v in sorted(profiles.opp_rim_allowed.items())
        },
        "league_three_rate": _round(profiles.league_three_rate),
        "league_rim_rate": _round(profiles.league_rim_rate),
        "seconds_mean": _round(profiles.seconds_mean),
        "seconds_std": _round(profiles.seconds_std),
        "shooter_prior_strength": _round(profiles.shooter_prior_strength),
        "team_prior_strength": _round(profiles.team_prior_strength),
        "n_shots": shots.height,
    }


def export_evaluation(paths: DataPaths) -> dict[str, Any]:
    """Published evaluation results, so /api/eval/* serves measurements.

    Read from the run logs rather than recomputed. An endpoint that recomputed
    its own metrics would be reporting a different number than the README, which
    is the exact failure the report generator exists to prevent.
    """
    from lineupiq.models.train import latest_run

    out: dict[str, Any] = {}

    conversion = latest_run(paths)
    if conversion:
        out["conversion"] = {
            "git_sha": conversion.get("git_sha"),
            "n_shots": conversion.get("n_shots"),
            "metrics": conversion.get("metrics", {}),
            "controls": conversion.get("controls", {}),
        }

    selection = latest_run(paths, kind="selection")
    if selection:
        out["selection"] = {
            "git_sha": selection.get("git_sha"),
            "n_shots": selection.get("n_shots"),
            "metrics": selection.get("metrics", {}),
            "controls": selection.get("controls", {}),
            "sign_audit": selection.get("model", {}).get("sign_audit", {}),
        }

    for name, relative in (
        ("rapm", "rapm/run.json"),
        ("retrieval", "retrieval/ablation.json"),
        # Groundedness ships the two distractor controls alongside every rate,
        # because a grounded rate of 1.00 proves nothing on its own -- a checker
        # that accepts everything also scores 1.00. The near-miss control (same
        # lineup, one player swapped) is the one that separates them.
        ("groundedness", "groundedness/run.json"),
    ):
        path = paths.runs / relative
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            # The lambda trace is a hundred rows of grid search and nothing a
            # client needs; it stays in the committed run log.
            payload.pop("lambda_trace", None)
            out[name] = payload

    trade_directory = paths.runs / "trade"
    if trade_directory.exists():
        out["trade"] = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(trade_directory.glob("*.json"))
        }

    return {"available": sorted(out), **out}


def export_coverage(paths: DataPaths) -> dict[str, Any]:
    """Every data-quality gate with its threshold, measurement and verdict.

    Re-derived from committed gold rather than read from a cached report, so an
    endpoint cannot serve a gate result that no longer holds. Four of these
    gates exist because of bugs that were silent when found; the detail string
    on each says which.
    """
    from lineupiq.features.shot_context import attach_possession_context
    from lineupiq.io.gold import load_all_gold
    from lineupiq.validate.checks import run_gates

    shot_facts = load_all_gold(paths, "shot_facts")
    possession_facts = load_all_gold(paths, "possession_facts")
    tables = {
        "shot_facts": shot_facts,
        "stints": load_all_gold(paths, "stints"),
        "possession_facts": possession_facts,
        "shots_with_context": attach_possession_context(shot_facts, possession_facts),
    }
    results = run_gates(tables)

    return {
        "gates": [
            {
                "name": result.name,
                "measured": _round(result.measured),
                "threshold": _round(result.threshold),
                "comparison": result.comparison,
                "verdict": result.verdict,
                "severity": result.severity,
                "detail": result.detail,
            }
            for result in results
        ],
        "n_gates": len(results),
        "n_passing": sum(1 for result in results if result.passed),
        "coverage": {
            "shots": shot_facts.height,
            "possessions": possession_facts.height,
            "seasons": sorted({int(s) for s in shot_facts["season"].unique().to_list()}),
        },
    }


def export_snapshot(paths: DataPaths) -> dict[str, Any]:
    """Which committed build is being served.

    Read from the contract sidecars, so the snapshot identity and the data
    fingerprints cannot disagree.
    """
    contracts: dict[str, Any] = {}
    if paths.contracts.exists():
        for path in sorted(paths.contracts.glob("*.json")):
            stored = json.loads(path.read_text(encoding="utf-8"))
            contracts[path.stem] = {
                "rows": stored["rows"],
                "content_sha256": stored["content_sha256"][:16],
            }
    return {
        "contracts": contracts,
        "n_contracts": len(contracts),
        "thresholds_sha256": thresholds_hash(),
    }


def export_all(paths: DataPaths, directory: Path | None = None) -> ExportManifest:
    """Write every served artefact and report the payload sizes.

    Sizes are reported because they are a constraint, not a curiosity: the
    Worker parses these on a cold start, and a support table that quietly grows
    to ten megabytes turns a fast endpoint into a slow one with no code change
    anywhere.
    """
    target = directory or (paths.root / "apps" / "web" / "public" / "data")
    files = {
        "zones.json": _write(target, "zones.json", export_zones()),
        "zone_surface.json": _write(target, "zone_surface.json", export_zone_surface(paths)),
        "support.json": _write(target, "support.json", export_support(paths)),
        "players.json": _write(target, "players.json", export_players(paths)),
        "selection_model.json": _write(
            target, "selection_model.json", export_selection_model(paths)
        ),
        "snapshot.json": _write(target, "snapshot.json", export_snapshot(paths)),
        "evaluation.json": _write(target, "evaluation.json", export_evaluation(paths)),
        "coverage.json": _write(target, "coverage.json", export_coverage(paths)),
        "player_zones.json": _write(target, "player_zones.json", export_player_zones(paths)),
        "selection_profiles.json": _write(
            target, "selection_profiles.json", export_selection_profiles(paths)
        ),
    }
    return ExportManifest(files=files)
