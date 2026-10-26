from dataclasses import dataclass

import numpy as np

from arxiv_rag.rerank import Reranker, RerankedRetriever
from fakes import FakeRerankBackend


@dataclass
class Item:
    text: str


class ListRetriever:
    """Returns items in fixed order with descending fake scores."""

    def __init__(self, items):
        self.items = items
        self.asked: list[int] = []

    def search(self, query, k):
        self.asked.append(k)
        return [(it, 1.0 - i * 0.01) for i, it in enumerate(self.items[:k])]


def make(tmp_path):
    be = FakeRerankBackend()
    return Reranker(backend=be, cache_dir=tmp_path), be


def test_score_values_and_order(tmp_path):
    rr, _ = make(tmp_path)
    s = rr.score("cat sat", ["dog", "cat sat down", "cat"])
    assert s.dtype == np.float32
    assert list(s) == [0, 2, 1]
    assert rr.score("cat", []).shape == (0,)


def test_reorders_by_reranker(tmp_path):
    rr, _ = make(tmp_path)
    items = [Item("nothing"), Item("cat sat"), Item("cat")]
    out = RerankedRetriever(ListRetriever(items), rr).search("cat sat", 3)
    assert [it.text for it, _ in out] == ["cat sat", "cat", "nothing"]
    assert [s for _, s in out] == [2.0, 1.0, 0.0]


def test_tie_break_by_base_rank(tmp_path):
    rr, _ = make(tmp_path)
    items = [Item("x a"), Item("y a"), Item("z a"), Item("b")]
    out = RerankedRetriever(ListRetriever(items), rr).search("a", 3)
    assert [it.text for it, _ in out] == ["x a", "y a", "z a"]


def test_depth_smaller_than_k_still_returns_k(tmp_path):
    rr, _ = make(tmp_path)
    base = ListRetriever([Item(f"t{i}") for i in range(10)])
    out = RerankedRetriever(base, rr, depth=3).search("q", 5)
    assert len(out) == 5 and base.asked == [5]


def test_k_smaller_than_depth_truncates(tmp_path):
    rr, _ = make(tmp_path)
    base = ListRetriever([Item("a"), Item("b"), Item("a b"), Item("c")])
    out = RerankedRetriever(base, rr, depth=4).search("a b", 2)
    assert base.asked == [4]
    assert [it.text for it, _ in out] == ["a b", "a"]


def test_few_candidates_and_k_zero(tmp_path):
    rr, _ = make(tmp_path)
    r = RerankedRetriever(ListRetriever([Item("a")]), rr)
    assert len(r.search("a", 5)) == 1
    assert r.search("a", 0) == []
    assert RerankedRetriever(ListRetriever([]), rr).search("a", 5) == []


def test_cache_hits_avoid_rescoring(tmp_path):
    rr, be = make(tmp_path)
    rr.score("q a", ["a", "b"])
    rr.score("q a", ["b", "a", "c"])
    assert [t for _, t in be.calls] == [["a", "b"], ["c"]]
    rr.score("q a", ["a", "b", "c"])
    assert len(be.calls) == 2
    rr.score("other", ["a"])  # same text, different query: not a hit
    assert len(be.calls) == 3


def test_duplicate_texts_scored_once(tmp_path):
    rr, be = make(tmp_path)
    rr.score("a", ["a", "a", "b"])
    assert be.calls[0][1] == ["a", "b"]


def test_cache_survives_reload(tmp_path):
    rr, _ = make(tmp_path)
    first = rr.score("a b", ["a", "b", "a b"])
    rr2, be2 = make(tmp_path)
    assert list(rr2.score("a b", ["a", "b", "a b"])) == list(first)
    assert be2.calls == []


def test_corrupt_cache_is_cold(tmp_path):
    rr, _ = make(tmp_path)
    rr.score("a", ["a"])
    rr._cache_path().write_bytes(b"not a zip")
    rr2, be2 = make(tmp_path)
    assert list(rr2.score("a", ["a"])) == [1.0]
    assert len(be2.calls) == 1


def test_model_name_isolates_cache(tmp_path):
    be = FakeRerankBackend()
    Reranker("m1", backend=be, cache_dir=tmp_path).score("a", ["a"])
    Reranker("m2", backend=be, cache_dir=tmp_path).score("a", ["a"])
    assert len(be.calls) == 2


def test_lazy_backend_not_loaded_on_construct(tmp_path):
    assert Reranker(cache_dir=tmp_path)._backend is None
