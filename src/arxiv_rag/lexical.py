"""Okapi BM25 over an in-memory list of items, same interface as DenseRetriever.

Design goals:
- Generic: works on anything with a `.text` attribute.
- Fast: inverted index (term -> doc ids + precomputed BM25 weights as numpy arrays); a
  query touches only its own terms' postings.
- Deterministic: ties break by item index.
- Plain: no stemming (keeps exact technical terms matchable, and avoids a dependency);
  a small English stopword list is dropped.

Defaults k1=0.9, b=0.4 are Pyserini's BM25 defaults. idf = ln(1 + (N - df + 0.5) / (df + 0.5)),
always > 0, so every matching doc scores > 0.
"""
import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from arxiv_rag.retrieve import HasText, T

WORD = re.compile(r"[^\W_]+")  # runs of letters/digits (unicode aware, underscore is a separator)
COMPOUND = re.compile(r"[^\W_]+(?:[-._][^\W_]+)+")  # "gpt-4", "3.5", "bert_base", "e.g"
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the this to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase terms: every alphanumeric run, plus each hyphen/dot/underscore-joined compound.

    "GPT-4" -> ["gpt", "4", "gpt-4"]; "3.5" -> ["3", "5", "3.5"]. The compound keeps the
    exact technical term matchable while the parts keep partial matches working; the same
    rule runs on documents and queries so they stay consistent. Stopwords are dropped from
    the parts only (a compound like "state-of-the-art" survives whole). Order is not meaningful.
    """
    text = text.lower()
    parts = [w for w in WORD.findall(text) if w not in STOPWORDS]
    return parts + COMPOUND.findall(text)


class BM25Retriever:
    def __init__(self, items: Sequence[T], k1: float = 0.9, b: float = 0.4):
        self.items = list(items)
        self.k1, self.b = k1, b
        n = len(self.items)
        tfs = [Counter(tokenize(it.text)) for it in self.items]
        lengths = np.array([sum(c.values()) for c in tfs], dtype=np.float64)
        avgdl = lengths.mean() if n and lengths.sum() > 0 else 1.0
        norm = k1 * (1.0 - b + b * lengths / avgdl)  # per-doc length normalisation
        post: dict[str, tuple[list[int], list[int]]] = {}
        for d, c in enumerate(tfs):
            for term, tf in c.items():
                ids, f = post.setdefault(term, ([], []))
                ids.append(d)
                f.append(tf)
        # term -> (doc ids ascending, idf * tf * (k1 + 1) / (tf + norm)); weights fixed at build time
        self.index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, (ids, f) in post.items():
            ids_a, f_a = np.array(ids, dtype=np.int64), np.array(f, dtype=np.float64)
            idf = math.log(1.0 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            self.index[term] = (ids_a, idf * f_a * (k1 + 1.0) / (f_a + norm[ids_a]))

    def search(self, query: str, k: int) -> list[tuple[T, float]]:
        """Top-k (item, BM25) pairs, best first.

        Only items sharing at least one term with the query are returned, so the list can be
        shorter than k and is [] when nothing matches. Repeated query terms count once.
        """
        k = min(k, len(self.items))
        if k <= 0:
            return []
        scores = np.zeros(len(self.items))
        for term in sorted(set(tokenize(query))):  # fixed order: float sums must not depend on the hash seed
            hit = self.index.get(term)
            if hit is not None:
                scores[hit[0]] += hit[1]
        cand = np.flatnonzero(scores > 0)
        if len(cand) == 0:
            return []
        if len(cand) > k:
            # keep everything tied with the k-th best so the index tie-break is exact
            cutoff = scores[cand[np.argpartition(-scores[cand], k - 1)[k - 1]]]
            cand = cand[scores[cand] >= cutoff]
        order = cand[np.lexsort((cand, -scores[cand]))][:k]
        return [(self.items[i], float(scores[i])) for i in order]
