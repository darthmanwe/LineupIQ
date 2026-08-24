"""Narratives, and an honest label on where they come from.

**These are templated, not generated.** No language model has been called by this
repository. The writer/judge pair, the content-addressed cache and the
human-labelled judge agreement are outstanding, and
``/api/eval/judge`` returns ``501`` naming them.

What templated narratives *are* good for is exercising the groundedness checker
against something with the shape of real prose, so its verdicts and its two
negative controls are measured rather than asserted. That is worth having on its
own: the checker is the part that would still be needed once a model is writing,
and its limits are the interesting finding either way.

Three templates on purpose. The **faithful** one should pass every check. The
**overclaiming** one quotes only correct numbers and still fails, because it
asserts a point estimate for a lineup whose tier forbids one -- the single most
important failure a groundedness checker has to catch, and the one arithmetic
alone cannot. The **hallucinating** one names a player who was not on the floor.

A checker that cannot separate those three is not measuring anything, so the run
reports the rate for each template rather than one aggregate.
"""

from __future__ import annotations

from lineupiq.retrieval.docs import LineupDoc

__all__ = ["TEMPLATES", "render_narrative"]

TEMPLATES: tuple[str, ...] = ("faithful", "overclaiming", "hallucinating")


def render_narrative(doc: LineupDoc, template: str, *, intruder: str | None = None) -> str:
    """Render one narrative about one lineup.

    ``intruder`` is the name the hallucinating template inserts. Passing it in
    rather than inventing one keeps the fixture reproducible.
    """
    names = list(doc.player_names)
    first = names[0] if names else "the lineup"
    second = names[1] if len(names) > 1 else first

    if template == "faithful":
        # Every number traceable, the tier respected, the caveat carried.
        hedge = (
            "so the rating is directional rather than a point estimate"
            if doc.below_reporting_floor
            else "enough to support a point estimate"
        )
        return (
            f"{first} and {second} shared the floor for {doc.possessions} possessions, "
            f"{hedge}. The group took {doc.three_rate:.0%} of its attempts from three and "
            f"{doc.rim_rate:.0%} at the rim, scoring {doc.ppp_context} among lineups in "
            f"this snapshot."
        )

    if template == "overclaiming":
        # Every number here is correct. It still fails `tier_consistency` when
        # the lineup is below the floor, which is exactly the point: the error
        # is in what the sentence claims, not in what it quotes.
        return (
            f"{first} and {second} are worth exactly {doc.points_per_100:.1f} points per 100 "
            f"possessions together, over {doc.possessions} possessions. They will score "
            f"at that rate."
        )

    if template == "hallucinating":
        name = intruder or "Marcus Fairweather"
        return (
            f"{first} and {name} anchored this group across {doc.possessions} possessions, "
            f"taking {doc.three_rate:.0%} of their attempts from three."
        )

    raise KeyError(f"unknown template {template!r}; known: {TEMPLATES}")


def build_corpus(docs: list[LineupDoc], *, limit: int = 200) -> dict[str, dict[str, str]]:
    """One narrative per template for the first ``limit`` documents.

    Documents are taken in the order they arrive -- sorted by possessions, so the
    sample is the best-evidenced groups. That biases the ``faithful`` template
    toward passing ``tier_consistency`` trivially, so the run deliberately also
    includes below-floor documents; see the CLI, which mixes both ends.
    """
    out: dict[str, dict[str, str]] = {template: {} for template in TEMPLATES}
    for doc in docs[:limit]:
        # The intruder is drawn from another document so it is a real name that
        # was really not on this floor -- a harder case than a made-up string.
        other = next((d for d in docs if d.doc_id != doc.doc_id and d.player_names), None)
        intruder = other.player_names[0] if other else None
        for template in TEMPLATES:
            out[template][doc.doc_id] = render_narrative(doc, template, intruder=intruder)
    return out
