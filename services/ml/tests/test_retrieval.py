"""Tests for document construction, retrieval and groundedness.

The two that matter most are negative ones.

``test_numbers_only_corpus_is_nearly_useless`` is the ablation's whole claim in
miniature: the same facts rendered as bare decimals must retrieve badly. If it
ever passes with a high score, the queries have stopped testing what they were
built to test.

``test_near_miss_control_is_harder_than_the_easy_one`` guards the groundedness
checker against the failure mode that a checker accepting everything also scores
1.00. Swapping one player leaves almost every number nearly right, so a checker
with no discrimination will not notice.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lineupiq.llm.groundedness import check_narrative, score_corpus
from lineupiq.retrieval.docs import (
    CORPUS_VARIANTS,
    LineupDoc,
    build_documents,
    render_document,
)
from lineupiq.retrieval.evaluate import build_queries, evaluate_corpus, run_ablation
from lineupiq.retrieval.index import BM25, LsaIndex, reciprocal_rank_fusion, tokenise


def _doc(
    doc_id: str = "abc:10:2023",
    names: tuple[str, ...] = ("Alpha One", "Beta Two", "Gamma Three", "Delta Four", "Echo Five"),
    possessions: int = 400,
    three_rate: float = 0.44,
    rim_rate: float = 0.30,
    below_floor: bool = False,
) -> LineupDoc:
    return LineupDoc(
        doc_id=doc_id,
        lineup_hash=doc_id.split(":")[0],
        team_id=10,
        season=2023,
        player_ids=(1, 2, 3, 4, 5),
        player_names=names,
        possessions=possessions,
        points_per_100=114.3,
        three_rate=three_rate,
        rim_rate=rim_rate,
        mid_rate=0.26,
        style_tags=("three-heavy",),
        archetypes=("stretch big", "perimeter scorer"),
        zone_shares={"restricted_area": 0.20, "wing_three": 0.22},
        event_lines=("possession by Alpha One",),
        below_reporting_floor=below_floor,
        ppp_context="in the top quintile",
    )


# --- documents --------------------------------------------------------------


def test_full_variant_carries_names_vocabulary_and_a_caveat() -> None:
    text = render_document(_doc(), "full")
    assert "Alpha One" in text
    assert "stretch big" in text
    assert "three-heavy" in text
    assert "top quintile" in text
    # The comparative appears alongside the number, never instead of a caveat.
    assert "possessions" in text


def test_a_below_floor_document_says_so_in_its_own_text() -> None:
    """The caveat has to travel with the number, not sit in a sidecar."""
    text = render_document(_doc(possessions=90, below_floor=True), "full")
    assert "below the 200-possession reporting floor" in text
    assert "directional" in text

    above = render_document(_doc(possessions=400), "full")
    assert "enough to support a point estimate" in above


def test_numbers_variant_has_no_vocabulary_to_match() -> None:
    text = render_document(_doc(), "numbers")
    assert "Alpha One" not in text
    assert "stretch big" not in text
    assert "points_per_100" in text


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown corpus variant"):
        render_document(_doc(), "whatever-scores-best")


def test_every_declared_variant_renders() -> None:
    for variant in CORPUS_VARIANTS:
        assert render_document(_doc(), variant)


# --- retrievers -------------------------------------------------------------


def test_bm25_ranks_the_document_that_contains_the_query_terms() -> None:
    texts = [
        "lineup of Alpha One and Beta Two, three-heavy, stretch big",
        "lineup of Zulu Nine and Yankee Eight, rim-pressuring, interior big",
        "lineup of Xray Seven and Whisky Six, balanced",
    ]
    scores = BM25.fit(texts).scores("Alpha One three-heavy")
    assert int(np.argmax(scores)) == 0
    assert scores[0] > scores[1]


def test_bm25_idf_is_never_negative() -> None:
    """A term in most of the corpus must not actively push documents down."""
    texts = ["common term here"] * 10 + ["common term rare"]
    index = BM25.fit(texts)
    assert index._idf("common") >= 0.0
    assert index._idf("rare") > index._idf("common")


def test_tokeniser_is_shared_and_simple() -> None:
    assert tokenise("Alpha-One, 42% THREE") == ["alpha", "one", "42", "three"]


def test_lsa_returns_a_score_per_document() -> None:
    texts = [f"lineup number {i} with a stretch big and three-heavy style" for i in range(12)]
    scores = LsaIndex.fit(texts).scores("stretch big three-heavy")
    assert scores.shape == (12,)
    assert np.isfinite(scores).all()


def test_rrf_uses_ranks_not_scores() -> None:
    """A leg with huge raw scores must not dominate purely by scale."""
    tiny = np.array([0.3, 0.2, 0.1])
    huge = np.array([100.0, 200.0, 300.0])
    fused = reciprocal_rank_fusion([tiny, huge])
    # Document 0 is first for `tiny` and last for `huge`; document 2 the
    # reverse. Rank fusion must tie them.
    assert fused[0] == pytest.approx(fused[2])
    assert fused.shape == (3,)


def test_rrf_of_one_leg_preserves_its_order() -> None:
    scores = np.array([0.1, 0.9, 0.5])
    fused = reciprocal_rank_fusion([scores])
    assert list(np.argsort(-fused)) == list(np.argsort(-scores))


# --- ablation ---------------------------------------------------------------


def _corpus(n: int = 60) -> list[LineupDoc]:
    rng = np.random.default_rng(4)
    # Alphabetic names on purpose: the checker's name extractor recognises
    # capitalised alphabetic words, and a fixture full of digits would make the
    # player-scope check silently untestable.
    first = ["Avery", "Bryce", "Caleb", "Damon", "Elias", "Finn", "Grady", "Hollis"]
    last = ["Ashwood", "Brightly", "Calloway", "Dunmore", "Ellery", "Fairbank"]
    names = [f"{f} {last[i % len(last)]}" for i, f in enumerate(first * 5)]
    docs: list[LineupDoc] = []
    for i in range(n):
        chosen = tuple(str(names[int(j)]) for j in rng.choice(len(names), 5, replace=False))
        three = float(rng.uniform(0.25, 0.55))
        docs.append(
            LineupDoc(
                doc_id=f"h{i:03d}:10:2023",
                lineup_hash=f"h{i:03d}",
                team_id=1610612700 + (i % 5),
                season=2023,
                player_ids=tuple(range(i, i + 5)),
                player_names=chosen,
                possessions=int(rng.integers(60, 900)),
                points_per_100=float(rng.uniform(100, 125)),
                three_rate=three,
                rim_rate=float(rng.uniform(0.2, 0.4)),
                mid_rate=0.2,
                style_tags=("three-heavy",) if three > 0.42 else ("balanced",),
                archetypes=("stretch big",) if i % 2 else ("interior big",),
                zone_shares={"restricted_area": 0.2},
                event_lines=(f"possession by {chosen[0]}",),
                below_reporting_floor=False,
                ppp_context="in the top quintile" if i % 3 else "below the league median",
            )
        )
    return docs


def test_queries_have_decidable_relevance() -> None:
    docs = _corpus()
    queries = build_queries(docs)
    assert queries
    for query in queries:
        assert query.n_relevant >= 1
        assert query.kind in {"players", "style", "composite"}


def test_numbers_only_corpus_is_nearly_useless() -> None:
    """The ablation's claim, in miniature.

    The same facts as bare decimals must retrieve badly, because the queries are
    made of words and a decimal has no words. If this ever scores well, the
    queries have stopped testing what they were built for.
    """
    docs = _corpus()
    queries = build_queries(docs)
    full, _ = evaluate_corpus(docs, queries, "full")
    numbers, _ = evaluate_corpus(docs, queries, "numbers")

    assert full["bm25"]["recall"] > 0.5
    assert numbers["bm25"]["recall"] < 0.3
    assert full["bm25"]["recall"] > 2 * numbers["bm25"]["recall"]


def test_ablation_reports_every_variant_and_its_caveats() -> None:
    report = run_ablation(_corpus())
    assert set(report.by_corpus) == set(CORPUS_VARIANTS)
    assert report.n_queries > 0
    # The honesty notes are part of the output, not decoration.
    assert any("not hand-graded" in note for note in report.notes)
    assert any("LSA" in note for note in report.notes)


def test_ablation_on_a_tiny_corpus_refuses_rather_than_inventing() -> None:
    report = run_ablation(_corpus(5))
    assert report.n_queries == 0
    assert any("too few documents" in note for note in report.notes)


def test_build_documents_respects_the_possession_floor() -> None:
    shots = pl.DataFrame(
        {
            "shooter_id": [1] * 60,
            "lineup_for_hash": ["h1"] * 60,
            "is_three": [True] * 30 + [False] * 30,
            "zone_id": ["wing_three"] * 30 + ["restricted_area"] * 30,
        }
    )
    possessions = pl.DataFrame(
        {
            "off_lineup_hash": ["h1"] * 10,
            "offense_team_id": [10] * 10,
            "season": [2023] * 10,
            "points": [2] * 10,
            "off_lineup": [[1, 2, 3, 4, 5]] * 10,
            "stint_quality": ["VALID"] * 10,
        }
    )
    players = pl.DataFrame({"player_id": [1, 2, 3, 4, 5], "player_name": list("abcde")})
    # Ten possessions is below the fifty-possession floor.
    assert build_documents(shots, possessions, players) == []


# --- groundedness -----------------------------------------------------------


def test_a_faithful_narrative_is_grounded() -> None:
    doc = _doc()
    text = (
        f"Alpha One and Beta Two played {doc.possessions} possessions together, "
        f"taking {doc.three_rate:.0%} of their attempts from three."
    )
    result = check_narrative(text, doc)
    assert result.grounded
    assert result.traceability == 1.0


def test_an_invented_number_is_caught() -> None:
    result = check_narrative("They took 87% of their shots from three.", _doc())
    assert not result.grounded
    assert "numeric_traceability" in result.failures


def test_a_player_who_was_not_on_the_floor_is_caught() -> None:
    """The check a numeric one cannot make: a plausible recalled team-mate."""
    result = check_narrative("Alpha One and Victor Nine spaced the floor.", _doc())
    assert "player_scope" in result.failures
    assert any("Victor Nine" in detail for detail in result.failures["player_scope"])


def test_an_invented_zone_is_caught() -> None:
    result = check_narrative("They lived at the elbow.", _doc())
    assert "zone_vocabulary" in result.failures


def test_a_point_estimate_on_a_below_floor_lineup_is_a_hard_failure() -> None:
    """The product's central promise, checked.

    Every number in this sentence is correct. It still fails, because the lineup
    has 90 possessions and the sentence asserts a point estimate.
    """
    doc = _doc(possessions=90, below_floor=True)
    text = f"This group is worth exactly {doc.points_per_100:.1f} points per 100."
    result = check_narrative(text, doc)
    assert "tier_consistency" in result.failures
    # And the numbers themselves are all traceable, which is the point.
    assert result.traceability == 1.0


def test_a_backwards_direction_is_caught() -> None:
    doc = _doc(three_rate=0.44)
    result = check_narrative("They shoot fewer three attempts than the league.", doc)
    assert "direction" in result.failures


def test_near_miss_control_is_harder_than_the_easy_one() -> None:
    """A checker that accepts everything also scores 1.00.

    The easy control re-scores against a different lineup, where names and
    numbers are both wrong. The near-miss swaps one player, so almost everything
    is still right -- and a checker with no discrimination will not notice.
    """
    docs = {d.doc_id: d for d in _corpus(12)}
    narratives = {
        doc_id: (
            f"{doc.player_names[0]} and {doc.player_names[1]} played {doc.possessions} possessions."
        )
        for doc_id, doc in docs.items()
    }
    scored = score_corpus(narratives, docs)

    assert scored["n"] == len(narratives)
    assert scored["grounded_rate"] == 1.0
    # Both controls must fall below the real rate; the near-miss is the number
    # worth publishing because it is the harder of the two.
    assert scored["control_easy_grounded_rate"] < scored["grounded_rate"]
    assert scored["control_near_miss_grounded_rate"] < scored["grounded_rate"]


def test_score_corpus_handles_an_empty_input() -> None:
    assert score_corpus({}, {})["n"] == 0
