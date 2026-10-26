"""Fuse several retrievers' ranked lists (RRF or convex combination).

Design goals:
- Same interface as DenseRetriever: search(query, k) -> list[(item, score)].
- Each sub-retriever is asked for `depth` results; fusion sees only those. A fused list is
  therefore at most the union of the sub-lists, even if k > depth.
- Items are identified by `chunk_id` when present, else object identity (override with `key`).
- Deterministic: ties break by best (lowest) rank in any list, then first appearance
  (list order, then rank).
- "rrf": sum of w / (k_rrf + rank) over lists containing the item, rank 1-based. Scale free.
- "convex": per list, min-max normalise the returned scores to [0, 1] (all-equal list -> all 1),
  items missing from a list get 0, then a weighted sum. The normalisation depends on `depth`
  (the min is the depth-th score), so results change when depth changes.
- `weights` default to equal (all 1.0) and apply to both methods.
"""
import math
from typing import Callable, Sequence

METHODS = ("rrf", "convex")


def default_key(item) -> object:
    cid = getattr(item, "chunk_id", None)
    return cid if cid is not None else id(item)


class HybridRetriever:
    def __init__(self, retrievers: Sequence, method: str = "rrf", k_rrf: int = 60,
                 weights: Sequence[float] | None = None, depth: int = 100,
                 key: Callable = default_key):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        if not retrievers:
            raise ValueError("need at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must match retrievers in length")
        if depth < 1 or k_rrf <= 0:
            raise ValueError("depth must be >= 1 and k_rrf > 0")
        self.retrievers = list(retrievers)
        self.method, self.k_rrf, self.depth, self.key = method, k_rrf, depth, key
        self.weights = [1.0] * len(self.retrievers) if weights is None else [float(w) for w in weights]

    def search(self, query: str, k: int) -> list[tuple[object, float]]:
        """Top-k (item, fused score) pairs, best first."""
        if k <= 0:
            return []
        items: dict[object, object] = {}  # insertion order = first appearance
        parts: dict[object, list[float]] = {}  # contributions per item, in list order
        best_rank: dict[object, int] = {}
        for ret, w in zip(self.retrievers, self.weights):
            hits = ret.search(query, self.depth)
            scores = [s for _, s in hits]
            lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
            for rank, (item, s) in enumerate(hits, 1):
                kk = self.key(item)
                items.setdefault(kk, item)
                best_rank[kk] = min(best_rank.get(kk, rank), rank)
                if self.method == "rrf":
                    c = w / (self.k_rrf + rank)
                else:
                    c = w * ((s - lo) / (hi - lo) if hi > lo else 1.0)
                parts.setdefault(kk, []).append(c)
        order = {kk: i for i, kk in enumerate(items)}
        fused = {kk: math.fsum(p) for kk, p in parts.items()}  # exact sum: order-independent ties
        ranked = sorted(fused, key=lambda kk: (-fused[kk], best_rank[kk], order[kk]))[:k]
        return [(items[kk], fused[kk]) for kk in ranked]
