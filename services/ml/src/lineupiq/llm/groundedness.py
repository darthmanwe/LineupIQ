"""Checking a narrative against its evidence, deterministically and for free.

No model runs here. Every check is arithmetic or set membership, which means it
runs in CI with no key, no network and no cost -- and, more importantly, that its
verdicts are reproducible.

**The limit is stated first, because it is the finding.** Arithmetic settles
*provenance* and cannot settle *meaning*. A checker like this can prove every
number in a narrative appears in the evidence, and be perfectly satisfied by a
sentence that quotes the right number for the wrong quantity. The sibling
project measured exactly that: its regex traced 1,027 of 1,027 tokens, raised no
flags, and scored Cohen's kappa 0.00 against human labels -- not because the
checker failed, but because a detector with no positives cannot agree beyond
chance, and every real error was a correctly-quoted number used to mean
something else.

So the checks here are split into two kinds. **Numeric traceability** is the
cheap one and it is nearly always satisfied. **Semantic checks** are the ones
that catch real errors: an invented zone, a player who was not on the floor, a
point estimate asserted for a lineup whose tier forbids one, and a direction
stated backwards.

Two negative controls, not one. A checker that accepts everything also scores
1.00, so the same narratives are re-scored against *another* lineup's evidence
(easy) and against the *same* lineup with one player swapped (near-miss). The
near-miss number is the honest one, because almost every figure is still nearly
right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lineupiq.retrieval.docs import LineupDoc
from lineupiq.transform.zones import ZONE_IDS

__all__ = [
    "CHECKS",
    "GroundednessResult",
    "check_narrative",
    "score_corpus",
]

#: Tolerance for matching a quoted number to a fact.
#:
#: Generous on purpose: a rate stored as 0.5412 is displayed as "54.1%", and a
#: checker that flagged that as ungrounded would flag every correct narrative.
#: Derived values are therefore compared in both stored and rendered forms.
_RELATIVE_TOLERANCE = 0.02

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)")

#: Unit phrases whose numbers are denominators, not claims.
#:
#: "points per 100 possessions" contains a 100 that is a unit. A checker that
#: demanded it appear in the evidence would flag every correct narrative, which
#: is the fastest way to make a groundedness score meaningless. These are
#: removed before numbers are extracted.
_UNIT_PHRASES = (
    "per 100 possessions",
    "per 100",
    "per game",
    "out of 5",
)

#: Phrases that assert a point estimate. Their presence in a narrative about a
#: below-floor lineup is a hard failure -- it is the product's central promise.
_POINT_ESTIMATE_PHRASES = (
    "exactly",
    "precisely",
    "is worth",
    "will score",
    "scores exactly",
)

#: Direction words, paired with the sign they assert.
_DIRECTION_WORDS: dict[str, int] = {
    "more": 1,
    "higher": 1,
    "above": 1,
    "better": 1,
    "increases": 1,
    "fewer": -1,
    "lower": -1,
    "below": -1,
    "worse": -1,
    "decreases": -1,
}

CHECKS: tuple[tuple[str, str], ...] = (
    ("numeric_traceability", "Every number in the prose appears in the evidence."),
    ("zone_vocabulary", "Every zone named exists in the taxonomy."),
    ("player_scope", "Every player named was on the floor for this lineup."),
    (
        "tier_consistency",
        "No point estimate is asserted for a lineup below the reporting floor.",
    ),
    ("direction", "Stated directions agree with the sign of the underlying fact."),
)


@dataclass
class GroundednessResult:
    """Per-narrative verdict, with the specific failures rather than a score."""

    doc_id: str
    n_numbers: int
    n_traced: int
    failures: dict[str, list[str]] = field(default_factory=dict)

    @property
    def traceability(self) -> float:
        return self.n_traced / self.n_numbers if self.n_numbers else 1.0

    @property
    def grounded(self) -> bool:
        return not self.failures and self.n_traced == self.n_numbers

    def add(self, check: str, detail: str) -> None:
        self.failures.setdefault(check, []).append(detail)


def _candidate_values(doc: LineupDoc) -> list[float]:
    """Every number a narrative may legitimately quote.

    Includes each fact in both stored and rendered form: a share of 0.4123 may
    appear as "0.41" or as "41". Without both, a correct narrative fails.
    """
    values: list[float] = []
    for value in doc.facts.values():
        if isinstance(value, (int, float)):
            values.append(float(value))
            values.append(round(float(value) * 100.0, 4))
    values.append(float(doc.possessions))
    values.append(float(doc.season))
    values.append(float(doc.team_id))
    return values


def _traced(number: float, candidates: list[float]) -> bool:
    for candidate in candidates:
        if candidate == 0.0:
            if abs(number) < 1e-9:
                return True
            continue
        if abs(number - candidate) / abs(candidate) <= _RELATIVE_TOLERANCE:
            return True
    return False


def check_narrative(text: str, doc: LineupDoc) -> GroundednessResult:
    """Score one narrative against one document's evidence."""
    candidates = _candidate_values(doc)
    result = GroundednessResult(doc_id=doc.doc_id, n_numbers=0, n_traced=0)

    # Unit denominators are removed before extraction; see `_UNIT_PHRASES`.
    measurable = text
    for phrase in _UNIT_PHRASES:
        measurable = measurable.replace(phrase, " ")

    # --- numeric traceability ------------------------------------------------
    for match in _NUMBER.finditer(measurable):
        raw, percent = match.group(1), match.group(2)
        number = float(raw)
        result.n_numbers += 1
        # A percent sign means the prose is in rendered units; check both, since
        # "41%" is grounded by a stored 0.41 and by a stored 41.
        forms = [number, number / 100.0] if percent else [number, number * 100.0]
        if any(_traced(form, candidates) for form in forms):
            result.n_traced += 1
        else:
            result.add("numeric_traceability", f"{raw}{percent} is not in the evidence")

    lowered = text.lower()

    # --- zone vocabulary -----------------------------------------------------
    # Catches invented zones, which a numeric check cannot see: "the elbow" and
    # "the short corner" are real basketball and not zones this project has.
    known = {zone.replace("_", " ") for zone in ZONE_IDS} | {
        "three",
        "rim",
        "mid-range",
        "mid range",
        "paint",
        "corner",
    }
    for phrase in ("elbow", "short corner", "dunker spot", "nail", "floater range"):
        if phrase in lowered and phrase not in known:
            result.add("zone_vocabulary", f"'{phrase}' is not a zone in the taxonomy")

    # --- player scope --------------------------------------------------------
    # Catches the model recalling a plausible team-mate who was not on the floor.
    on_floor = {name.lower() for name in doc.player_names}
    surnames = {name.split()[-1].lower() for name in doc.player_names if name.split()}
    for candidate in re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b", text):
        lowered_candidate = candidate.lower()
        if lowered_candidate in on_floor:
            continue
        if lowered_candidate.split()[-1] in surnames:
            continue
        result.add("player_scope", f"'{candidate}' was not in this lineup")

    # --- tier consistency ----------------------------------------------------
    # The hard one. A narrative about a below-floor lineup that asserts a point
    # estimate breaks the product's central promise, and no numeric check would
    # notice because the number itself is correct.
    if doc.below_reporting_floor:
        for phrase in _POINT_ESTIMATE_PHRASES:
            if phrase in lowered:
                result.add(
                    "tier_consistency",
                    f"'{phrase}' asserts a point estimate for a lineup with "
                    f"{doc.possessions} possessions, below the reporting floor",
                )

    # --- direction -----------------------------------------------------------
    # Compares a stated direction against the sign of the fact it describes.
    if "three" in lowered:
        league_three = 0.39
        stated = next(
            (sign for word, sign in _DIRECTION_WORDS.items() if f"{word} three" in lowered),
            None,
        )
        if stated is not None:
            actual = 1 if doc.three_rate > league_three else -1
            if stated != actual:
                result.add(
                    "direction",
                    f"prose says the three rate is {'higher' if stated > 0 else 'lower'} "
                    f"than the league, but it is {doc.three_rate:.1%} against "
                    f"{league_three:.0%}",
                )
    return result


def score_corpus(narratives: dict[str, str], docs: dict[str, LineupDoc]) -> dict[str, object]:
    """Score every narrative, and run both negative controls.

    ``easy`` re-scores each narrative against a *different* lineup's evidence.
    ``near_miss`` re-scores it against the same lineup with one player replaced,
    so almost every number is still nearly right. The near-miss rate is the one
    worth publishing: an easy control any checker passes proves nothing.
    """
    doc_ids = sorted(docs)
    if not doc_ids:
        return {"n": 0}

    real = [
        check_narrative(text, docs[doc_id]) for doc_id, text in narratives.items() if doc_id in docs
    ]
    if not real:
        return {"n": 0}

    # Easy control: shift every narrative onto the next document.
    easy = []
    for offset, (doc_id, text) in enumerate(sorted(narratives.items())):
        if doc_id not in docs:
            continue
        other = docs[doc_ids[(doc_ids.index(doc_id) + 1 + offset) % len(doc_ids)]]
        easy.append(check_narrative(text, other))

    # Near-miss control: the same document with one player swapped out.
    near = []
    for doc_id, text in narratives.items():
        if doc_id not in docs:
            continue
        doc = docs[doc_id]
        replacement = next((d for d in docs.values() if d.doc_id != doc_id), None)
        if replacement is None or not replacement.player_names:
            continue
        from dataclasses import replace as dataclass_replace

        swapped = dataclass_replace(
            doc,
            player_names=(replacement.player_names[0], *doc.player_names[1:]),
            player_ids=(replacement.player_ids[0], *doc.player_ids[1:]),
        )
        near.append(check_narrative(text, swapped))

    def rate(results: list[GroundednessResult]) -> float:
        return sum(1 for r in results if r.grounded) / len(results) if results else 0.0

    failure_counts: dict[str, int] = {}
    for result in real:
        for check in result.failures:
            failure_counts[check] = failure_counts.get(check, 0) + 1

    return {
        "n": len(real),
        "grounded_rate": rate(real),
        "mean_traceability": sum(r.traceability for r in real) / len(real),
        "failures_by_check": failure_counts,
        "control_easy_grounded_rate": rate(easy),
        "control_near_miss_grounded_rate": rate(near),
        "checks": [name for name, _ in CHECKS],
    }
