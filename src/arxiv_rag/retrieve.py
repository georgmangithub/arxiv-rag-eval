"""Brute-force dense retrieval over an in-memory list of items.

Design goals:
- Generic: works on anything with a `.text` attribute (chunks, sentences, ...).
- Exact: cosine over normalized vectors, no ANN index; corpora here are small.
- Deterministic: ties break by item index.
"""
from typing import Protocol, Sequence, TypeVar

import numpy as np


class HasText(Protocol):
    text: str


class _Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


T = TypeVar("T", bound=HasText)


class DenseRetriever:
    def __init__(self, embedder: _Embedder, items: Sequence[T]):
        self.embedder = embedder
        self.items = list(items)
        self.matrix = embedder.embed_passages([it.text for it in self.items])

    def search(self, query: str, k: int) -> list[tuple[T, float]]:
        """Top-k (item, cosine) pairs, best first."""
        n = len(self.items)
        k = min(k, n)
        if k <= 0:
            return []
        scores = self.matrix @ self.embedder.embed_query(query)
        if k < n:
            # partition at the k-th best, then keep everything tied with it so the
            # final index tie-break is exact rather than argpartition's arbitrary pick
            cutoff = scores[np.argpartition(-scores, k - 1)[k - 1]]
            cand = np.flatnonzero(scores >= cutoff)
        else:
            cand = np.arange(n)
        order = cand[np.lexsort((cand, -scores[cand]))][:k]
        return [(self.items[i], float(scores[i])) for i in order]
