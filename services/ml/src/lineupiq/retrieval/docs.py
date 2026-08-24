"""Building the documents retrieval searches over.

The original design document contains its own best insight and then does not
follow it: it warns that a stint is too short to carry stable statistical
content, and then proposes indexing per-stint documents. A stint is about ninety
seconds and four possessions. Its embedding encodes noise.

So documents are built at ``(lineup_hash, team, season)`` grain, and only for
groups with enough possessions to say anything. Four properties are deliberate,
and the ablation below measures whether they matter rather than asserting it:

**Names and vocabulary are literal.** A query for "stretch five and a slashing
guard" can only match if those words are present in the text. Retrieval cannot
infer a role from a number.

**Comparatives, not bare numbers.** "Top quintile from the corners" retrieves;
"0.412" does not. Embeddings encode relations between words, and a decimal has
no relations.

**Style tags come from a closed vocabulary.** That is the half of hybrid
retrieval BM25 handles perfectly, and it is why the lexical leg is not
decoration.

**Caveats travel with the number.** A document that says "+2.1 per 100, not
distinguishable from zero at this sample" produces a narrative that inherits the
hedge. One that says "+2.1" produces a narrative that invents confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from lineupiq.transform.zones import ZONE_IDS
from lineupiq.util import as_float

__all__ = [
    "CORPUS_VARIANTS",
    "MIN_DOC_POSSESSIONS",
    "LineupDoc",
    "build_documents",
    "render_document",
]

#: A group needs at least this many possessions to get a document. Below it
#: there is nothing stable to describe, and an indexed document that describes
#: noise is worse than an absent one -- it retrieves.
MIN_DOC_POSSESSIONS = 50

#: The three corpora the ablation compares.
#:
#: ``events`` is the design document's own proposal, reproduced faithfully so
#: the comparison is fair rather than a straw man.
CORPUS_VARIANTS: tuple[str, ...] = ("events", "numbers", "full")

#: Closed style vocabulary. Assigned from measured shot mix, not from opinion,
#: and every tag has a stated threshold so a reader can check the label.
STYLE_TAGS: tuple[tuple[str, str], ...] = (
    ("three-heavy", "at least 42% of attempts from behind the arc"),
    ("rim-pressuring", "at least 34% of attempts at the rim"),
    ("mid-range-reliant", "at least 22% of attempts from mid-range"),
    ("balanced", "no zone group above its league share by more than five points"),
)

#: Archetype vocabulary, assigned per player from his own shot mix. Present in
#: the prose because a query for a role has to have words to match.
ARCHETYPES: tuple[tuple[str, str], ...] = (
    ("stretch big", "a big who takes most of his shots from outside"),
    ("interior big", "a big whose attempts are concentrated at the rim"),
    ("perimeter scorer", "a wing or guard leaning heavily on threes"),
    ("slashing guard", "a guard whose attempts concentrate at the rim"),
    ("balanced wing", "no zone group dominant"),
)


@dataclass(frozen=True)
class LineupDoc:
    """One indexed document, plus the facts a narrative may cite."""

    doc_id: str
    lineup_hash: str
    team_id: int
    season: int
    player_ids: tuple[int, ...]
    player_names: tuple[str, ...]
    possessions: int
    points_per_100: float
    three_rate: float
    rim_rate: float
    mid_rate: float
    style_tags: tuple[str, ...]
    archetypes: tuple[str, ...]
    #: Zone shares, for the numbers-only corpus and for groundedness checking.
    zone_shares: dict[str, float] = field(default_factory=dict)
    #: Raw event lines, for the `events` corpus variant only.
    event_lines: tuple[str, ...] = ()
    #: True when the possession count is too low for a point estimate. The
    #: caveat that travels with every number in the rendered text.
    below_reporting_floor: bool = False
    #: Where this group's rating sits in the league distribution, as words.
    #: Computed once at build time -- a comparative is only meaningful against
    #: the whole corpus, so it cannot be produced by a renderer looking at one
    #: document.
    ppp_context: str = "around the league average"

    @property
    def facts(self) -> dict[str, float | int | str]:
        """Every citable quantity, keyed by the id a narrative must reference."""
        out: dict[str, float | int | str] = {
            "possessions": self.possessions,
            "points_per_100": round(self.points_per_100, 2),
            "three_rate": round(self.three_rate, 4),
            "rim_rate": round(self.rim_rate, 4),
            "mid_rate": round(self.mid_rate, 4),
        }
        for zone, share in self.zone_shares.items():
            out[f"share_{zone}"] = round(share, 4)
        return out


def _quintile_word(value: float, distribution: np.ndarray) -> str:
    """Describe a value by where it sits, not by what it is."""
    if not distribution.size:
        return "around the league average"
    rank = float((distribution < value).mean())
    if rank >= 0.8:
        return "in the top quintile"
    if rank >= 0.6:
        return "above the league median"
    if rank >= 0.4:
        return "around the league median"
    if rank >= 0.2:
        return "below the league median"
    return "in the bottom quintile"


def _archetype_for(three: float, rim: float) -> str:
    if three >= 0.45 and rim <= 0.25:
        return "perimeter scorer"
    if rim >= 0.45 and three <= 0.15:
        return "interior big"
    if three >= 0.35 and rim >= 0.30:
        return "stretch big"
    if rim >= 0.38:
        return "slashing guard"
    return "balanced wing"


def _style_tags_for(three: float, rim: float, mid: float) -> tuple[str, ...]:
    tags: list[str] = []
    if three >= 0.42:
        tags.append("three-heavy")
    if rim >= 0.34:
        tags.append("rim-pressuring")
    if mid >= 0.22:
        tags.append("mid-range-reliant")
    if not tags:
        tags.append("balanced")
    return tuple(tags)


def build_documents(
    shots: pl.DataFrame,
    possessions: pl.DataFrame,
    players: pl.DataFrame,
    *,
    min_possessions: int = MIN_DOC_POSSESSIONS,
    reporting_floor: int = 200,
) -> list[LineupDoc]:
    """Assemble one document per ``(lineup, team, season)`` with enough evidence."""
    names = dict(zip(players["player_id"].to_list(), players["player_name"].to_list(), strict=True))

    # Per-player shot mix, for the archetype vocabulary.
    per_player = (
        shots.group_by("shooter_id")
        .agg(
            pl.len().alias("attempts"),
            pl.col("is_three").mean().alias("three_rate"),
            pl.col("zone_id").is_in(["restricted_area", "paint_non_ra"]).mean().alias("rim_rate"),
        )
        .filter(pl.col("attempts") >= 50)
    )
    player_archetype = {
        int(row["shooter_id"]): _archetype_for(
            as_float(row["three_rate"]), as_float(row["rim_rate"])
        )
        for row in per_player.iter_rows(named=True)
    }

    grouped = (
        possessions.filter(
            pl.col("off_lineup_hash").is_not_null() & (pl.col("stint_quality") == "VALID")
        )
        .group_by(["off_lineup_hash", "offense_team_id", "season"])
        .agg(
            pl.len().alias("possessions"),
            pl.col("points").sum().alias("points"),
            pl.col("off_lineup").first().alias("lineup"),
        )
        .filter(pl.col("possessions") >= min_possessions)
        .sort("possessions", descending=True)
    )

    # Shot mix per lineup, from the shot table rather than the possession table:
    # possessions know points, shots know zones.
    mix = (
        shots.filter(pl.col("lineup_for_hash").is_not_null())
        .group_by("lineup_for_hash")
        .agg(
            pl.len().alias("attempts"),
            *[(pl.col("zone_id") == zone).mean().alias(f"share_{zone}") for zone in ZONE_IDS],
        )
    )
    mix_lookup = {row["lineup_for_hash"]: row for row in mix.iter_rows(named=True)}

    ppp_distribution = np.array(
        [
            100.0 * as_float(r["points"]) / max(int(r["possessions"]), 1)
            for r in grouped.iter_rows(named=True)
        ]
    )

    docs: list[LineupDoc] = []
    for row in grouped.iter_rows(named=True):
        key = row["off_lineup_hash"]
        shot_row = mix_lookup.get(key)
        if shot_row is None:
            continue
        shares = {zone: as_float(shot_row[f"share_{zone}"]) for zone in ZONE_IDS}
        three = sum(
            v for z, v in shares.items() if z.endswith("_three") or z.startswith("corner_three")
        )
        rim = shares["restricted_area"] + shares["paint_non_ra"]
        mid = shares["mid_baseline"] + shares["mid_wing"] + shares["mid_top"]

        ids = tuple(int(p) for p in row["lineup"])
        possessions_count = int(row["possessions"])
        docs.append(
            LineupDoc(
                doc_id=f"{key}:{int(row['offense_team_id'])}:{int(row['season'])}",
                lineup_hash=key,
                team_id=int(row["offense_team_id"]),
                season=int(row["season"]),
                player_ids=ids,
                player_names=tuple(names.get(p, str(p)) for p in ids),
                possessions=possessions_count,
                points_per_100=100.0 * as_float(row["points"]) / max(possessions_count, 1),
                three_rate=three,
                rim_rate=rim,
                mid_rate=mid,
                style_tags=_style_tags_for(three, rim, mid),
                archetypes=tuple(
                    dict.fromkeys(player_archetype.get(p, "balanced wing") for p in ids)
                ),
                zone_shares=shares,
                event_lines=tuple(f"possession by {names.get(p, p)}" for p in ids),
                below_reporting_floor=possessions_count < reporting_floor,
            )
        )

    # The comparative needs the whole corpus, so it is resolved after every
    # document exists rather than inside the loop.
    return [
        replace(doc, ppp_context=_quintile_word(doc.points_per_100, ppp_distribution))
        for doc in docs
    ]


def render_document(doc: LineupDoc, variant: str) -> str:
    """Render one document in one of the three corpus variants.

    ``events``  -- the design document's own proposal: a per-possession event
                   log. Reproduced faithfully so the ablation is a fair test and
                   not a straw man.
    ``numbers`` -- the facts as bare decimals, no vocabulary and no comparatives.
    ``full``    -- names, archetypes, style tags, comparatives, and the caveat.
    """
    if variant == "events":
        return " ".join(doc.event_lines) + " " + " ".join(doc.player_names)

    if variant == "numbers":
        pairs = " ".join(f"{k}={v}" for k, v in doc.facts.items())
        return f"lineup {doc.lineup_hash} team {doc.team_id} season {doc.season} {pairs}"

    if variant != "full":
        raise KeyError(f"unknown corpus variant {variant!r}; known: {CORPUS_VARIANTS}")

    caveat = (
        f"This group played {doc.possessions} possessions, below the 200-possession "
        "reporting floor, so the rating is directional and not a point estimate."
        if doc.below_reporting_floor
        else f"This group played {doc.possessions} possessions, enough to support a point estimate."
    )
    return (
        f"Lineup of {', '.join(doc.player_names)} for team {doc.team_id} "
        f"in season {doc.season}. "
        f"Roles on the floor: {', '.join(doc.archetypes)}. "
        f"Style: {', '.join(doc.style_tags)}. "
        f"Scores {doc.ppp_context} at {doc.points_per_100:.1f} points per 100 possessions. "
        f"Shot profile: {_share_phrase(doc.three_rate)} of attempts from three, "
        f"{_share_phrase(doc.rim_rate)} at the rim, "
        f"{_share_phrase(doc.mid_rate)} from mid-range. "
        f"{caveat}"
    )


def _share_phrase(value: float) -> str:
    """A comparative plus the number, never the number alone."""
    if value >= 0.45:
        return f"a heavy share ({value:.0%})"
    if value >= 0.35:
        return f"an above-average share ({value:.0%})"
    if value >= 0.22:
        return f"a modest share ({value:.0%})"
    return f"a small share ({value:.0%})"
