"""Retrieval evaluation, and the ablation that turns an assertion into a number.

The design document asserts that document *design* drives retrieval quality.
This measures it: three corpora built from the same underlying facts, scored
with the same queries and the same retrievers.

**About the relevance judgements.** They are derived programmatically, not
hand-graded. A query is constructed from a document's own attributes -- two of
its players, its style tags, its archetypes -- and every document sharing those
attributes is relevant. That is a real and checkable relevance definition, and
it is a **weaker** claim than human grading: it measures whether a retriever can
find documents that state facts the query names. It does not measure semantic
relevance, and no number here should be read as if it did. The plan called for
40 hand-graded queries; that requires a human, and it is listed as outstanding
rather than quietly substituted.

The consequence is worth stating plainly: this ablation can show that a corpus
carrying names and vocabulary beats one carrying bare decimals, because the
queries name things. It cannot show that the full template beats a different
good template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lineupiq.config import SEED
from lineupiq.retrieval.docs import CORPUS_VARIANTS, LineupDoc, render_document
from lineupiq.retrieval.index import BM25, RETRIEVERS, LsaIndex, reciprocal_rank_fusion

__all__ = [
    "Query",
    "RetrievalReport",
    "build_queries",
    "evaluate_corpus",
    "run_ablation",
]

#: How many queries of each kind to generate.
N_PER_KIND = 15

#: Cutoff for the headline metrics.
K = 10


@dataclass(frozen=True)
class Query:
    """One query and the documents that satisfy it, by construction."""

    text: str
    kind: str
    relevant: frozenset[str]

    @property
    def n_relevant(self) -> int:
        return len(self.relevant)


def build_queries(docs: list[LineupDoc], *, seed: int = SEED) -> list[Query]:
    """Generate queries whose relevance is decidable from the documents.

    Three kinds, chosen because they test different retrieval mechanisms:

    - ``players``   -- two named players. Pure lexical matching; BM25 should be
      excellent and a dense space has nothing to add.
    - ``style``     -- a style tag plus an archetype. Closed vocabulary, so still
      lexical, but the tokens are shared across many documents.
    - ``composite`` -- a role plus a performance band. The only kind where a
      dense space has a reason to help, because the band is words rather than a
      single token.
    """
    rng = np.random.default_rng(seed)
    queries: list[Query] = []
    if len(docs) < 20:
        return queries

    by_style: dict[str, set[str]] = {}
    by_archetype: dict[str, set[str]] = {}
    by_context: dict[str, set[str]] = {}
    for doc in docs:
        for tag in doc.style_tags:
            by_style.setdefault(tag, set()).add(doc.doc_id)
        for archetype in doc.archetypes:
            by_archetype.setdefault(archetype, set()).add(doc.doc_id)
        by_context.setdefault(doc.ppp_context, set()).add(doc.doc_id)

    # --- players: two names from one document -------------------------------
    picks = rng.choice(len(docs), size=min(N_PER_KIND, len(docs)), replace=False)
    for i in picks:
        doc = docs[int(i)]
        names = list(doc.player_names)[:2]
        if len(names) < 2:
            continue
        wanted = {
            other.doc_id for other in docs if all(name in other.player_names for name in names)
        }
        queries.append(
            Query(
                text=f"lineups with {names[0]} and {names[1]}",
                kind="players",
                relevant=frozenset(wanted),
            )
        )

    # --- style: a tag plus an archetype -------------------------------------
    tags = sorted(by_style)
    archetypes = sorted(by_archetype)
    for _ in range(N_PER_KIND):
        tag = str(rng.choice(tags))
        archetype = str(rng.choice(archetypes))
        wanted = by_style[tag] & by_archetype[archetype]
        if not wanted:
            continue
        queries.append(
            Query(
                text=f"a {tag} lineup with a {archetype}",
                kind="style",
                relevant=frozenset(wanted),
            )
        )

    # --- composite: role plus performance band ------------------------------
    contexts = sorted(by_context)
    for _ in range(N_PER_KIND):
        archetype = str(rng.choice(archetypes))
        context = str(rng.choice(contexts))
        wanted = by_archetype[archetype] & by_context[context]
        if not wanted:
            continue
        queries.append(
            Query(
                text=f"a lineup with a {archetype} that scores {context}",
                kind="composite",
                relevant=frozenset(wanted),
            )
        )

    return queries


def _metrics(ranked: list[str], relevant: frozenset[str], k: int) -> dict[str, float]:
    """Recall@k, MRR and nDCG@k for one query."""
    top = ranked[:k]
    hits = [doc_id in relevant for doc_id in top]

    recall = sum(hits) / min(len(relevant), k) if relevant else 0.0

    reciprocal = 0.0
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            reciprocal = 1.0 / position
            break

    gains = [1.0 if hit else 0.0 for hit in hits]
    dcg = sum(gain / np.log2(position + 1) for position, gain in enumerate(gains, start=1))
    ideal = sum(1.0 / np.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return {
        "recall": float(recall),
        "mrr": float(reciprocal),
        "ndcg": float(dcg / ideal) if ideal else 0.0,
    }


@dataclass
class RetrievalReport:
    """Scores for every retriever on every corpus, plus the query breakdown."""

    n_documents: int
    n_queries: int
    by_corpus: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def evaluate_corpus(
    docs: list[LineupDoc], queries: list[Query], variant: str, *, k: int = K
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Score every retriever on one corpus variant."""
    texts = [render_document(doc, variant) for doc in docs]
    doc_ids = [doc.doc_id for doc in docs]

    bm25 = BM25.fit(texts)
    lsa = LsaIndex.fit(texts)

    accumulated: dict[str, list[dict[str, float]]] = {name: [] for name in RETRIEVERS}
    per_kind: dict[str, dict[str, list[float]]] = {}

    for query in queries:
        bm25_scores = bm25.scores(query.text)
        lsa_scores = lsa.scores(query.text)
        fused = reciprocal_rank_fusion([bm25_scores, lsa_scores])

        for name, scores in (("bm25", bm25_scores), ("lsa", lsa_scores), ("rrf", fused)):
            order = np.argsort(-scores)
            ranked = [doc_ids[int(i)] for i in order]
            result = _metrics(ranked, query.relevant, k)
            accumulated[name].append(result)
            per_kind.setdefault(query.kind, {}).setdefault(name, []).append(result["recall"])

    scores_by_retriever = {
        name: {
            metric: float(np.mean([row[metric] for row in rows])) if rows else 0.0
            for metric in ("recall", "mrr", "ndcg")
        }
        | {"n": float(len(rows))}
        for name, rows in accumulated.items()
    }
    kinds = {
        kind: {name: float(np.mean(values)) for name, values in retrievers.items()}
        for kind, retrievers in per_kind.items()
    }
    return scores_by_retriever, kinds


def run_ablation(docs: list[LineupDoc], *, seed: int = SEED, k: int = K) -> RetrievalReport:
    """Score every retriever on every corpus variant."""
    queries = build_queries(docs, seed=seed)
    report = RetrievalReport(n_documents=len(docs), n_queries=len(queries))
    if not queries:
        report.notes.append("too few documents to generate queries")
        return report

    for variant in CORPUS_VARIANTS:
        scores, kinds = evaluate_corpus(docs, queries, variant, k=k)
        report.by_corpus[variant] = scores
        if variant == "full":
            report.by_kind = kinds

    report.notes.append(
        "Relevance judgements are derived programmatically from document "
        "attributes, not hand-graded. They measure whether a retriever finds "
        "documents stating facts the query names -- a weaker claim than semantic "
        "relevance. Hand-graded queries remain outstanding."
    )
    report.notes.append(
        "The dense leg is TF-IDF plus truncated SVD (LSA), not a neural embedding "
        "model. It runs offline from a clean clone, which is why it is used here; "
        "the deployed retriever would use Workers AI at 384 dimensions and would "
        "need its own measurement."
    )
    return report


def summarise(report: RetrievalReport) -> dict[str, Any]:
    return {
        "n_documents": report.n_documents,
        "n_queries": report.n_queries,
        "by_corpus": report.by_corpus,
        "by_kind": report.by_kind,
        "notes": report.notes,
    }
