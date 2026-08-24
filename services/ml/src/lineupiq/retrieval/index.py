"""Retrievers, and an honest account of which one is which.

Three legs, all reproducible from a clean clone with no network and no account:

**BM25** -- Okapi BM25, implemented here rather than pulled in. It is forty
lines, the parameters matter to the result, and a dependency would hide them.

**LSA** -- TF-IDF followed by truncated SVD. This is a genuine dense retriever
over a learned latent space, and it is **not** a neural embedding model. It is
labelled LSA everywhere for that reason. It answers the question the ablation
needs answered -- does a dense space beat lexical matching on *this* corpus --
without requiring a model download, which would make the evaluation
unreproducible offline and therefore worth less than the thing it measures.

**RRF** -- reciprocal rank fusion over the two. Ranks, not scores, because BM25
scores and cosine similarities are not on a comparable scale and normalising
them into one is a choice that quietly decides the outcome.

The deployed dense leg would be Workers AI at 384 dimensions, which is a
different model and would need its own measurement. That is stated rather than
elided: nothing here licenses a claim about the deployed retriever's recall.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BM25",
    "RETRIEVERS",
    "RRF_K",
    "LsaIndex",
    "reciprocal_rank_fusion",
    "tokenise",
]

#: Okapi BM25 defaults. `k1` controls term-frequency saturation and `b` controls
#: length normalisation; both are stated here because they change the ranking
#: and a reader should not have to guess them.
BM25_K1 = 1.5
BM25_B = 0.75

#: RRF constant. 60 is the value from the original paper and is what makes the
#: fusion insensitive to the raw score scales it is deliberately ignoring.
RRF_K = 60

RETRIEVERS: tuple[str, ...] = ("bm25", "lsa", "rrf")

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Deliberately simple and shared by every leg, so a difference between
    retrievers is a difference in the retriever and not in its tokeniser.
    """
    return _TOKEN.findall(text.lower())


@dataclass
class BM25:
    """Okapi BM25 over a fixed corpus."""

    documents: list[list[str]] = field(default_factory=list)
    k1: float = BM25_K1
    b: float = BM25_B
    _document_frequency: Counter[str] = field(default_factory=Counter)
    _term_frequency: list[Counter[str]] = field(default_factory=list)
    _lengths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _average_length: float = 0.0

    @classmethod
    def fit(cls, texts: list[str], **kwargs: float) -> BM25:
        tokenised = [tokenise(text) for text in texts]
        index = cls(documents=tokenised, **kwargs)  # type: ignore[arg-type]
        index._term_frequency = [Counter(doc) for doc in tokenised]
        index._document_frequency = Counter()
        for counts in index._term_frequency:
            index._document_frequency.update(counts.keys())
        index._lengths = np.array([len(doc) for doc in tokenised], dtype=float)
        index._average_length = float(index._lengths.mean()) if index._lengths.size else 0.0
        return index

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        frequency = self._document_frequency.get(term, 0)
        # The +0.5 smoothing is Robertson-Sparck-Jones. Without the max(), a
        # term appearing in more than half the corpus gets a negative idf and
        # actively pushes documents down, which is a well-known BM25 foot-gun.
        return max(math.log(1 + (n - frequency + 0.5) / (frequency + 0.5)), 0.0)

    def scores(self, query: str) -> np.ndarray:
        terms = tokenise(query)
        out = np.zeros(len(self.documents))
        if not self._average_length:
            return out
        for term in terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i, counts in enumerate(self._term_frequency):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / self._average_length)
                out[i] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return out


@dataclass
class LsaIndex:
    """TF-IDF plus truncated SVD. A dense retriever, not a neural one."""

    n_components: int = 128
    _vectoriser: object | None = None
    _svd: object | None = None
    _matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @classmethod
    def fit(cls, texts: list[str], *, n_components: int = 128, seed: int = 0) -> LsaIndex:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        index = cls(n_components=n_components)
        vectoriser = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+", min_df=2)
        sparse = vectoriser.fit_transform(texts)

        # SVD cannot produce more components than the smaller matrix dimension.
        components = min(n_components, min(sparse.shape) - 1)
        svd = TruncatedSVD(n_components=max(components, 2), random_state=seed)
        dense = svd.fit_transform(sparse)

        index._vectoriser = vectoriser
        index._svd = svd
        # L2-normalised, so a dot product is a cosine similarity.
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        index._matrix = dense / np.maximum(norms, 1e-12)
        return index

    def scores(self, query: str) -> np.ndarray:
        if self._vectoriser is None or self._svd is None:
            return np.zeros(self._matrix.shape[0])
        vector = self._svd.transform(self._vectoriser.transform([query]))  # type: ignore[attr-defined]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros(self._matrix.shape[0])
        return self._matrix @ (vector[0] / norm)


def reciprocal_rank_fusion(score_sets: list[np.ndarray], *, k: int = RRF_K) -> np.ndarray:
    """Fuse rankings by reciprocal rank, ignoring the scores themselves.

    BM25 scores are unbounded sums and cosine similarities live in [-1, 1].
    Normalising them onto one scale is a choice that quietly decides which leg
    wins, so only the ranks are used.
    """
    if not score_sets:
        return np.zeros(0)
    fused = np.zeros_like(score_sets[0], dtype=float)
    for scores in score_sets:
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))
        fused += 1.0 / (k + ranks + 1)
    return fused
