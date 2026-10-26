from dataclasses import dataclass

import pytest

from arxiv_rag.hybrid import HybridRetriever


@dataclass(frozen=True)
class It:
    chunk_id: str
    text: str = ""


class Fixed:
    """Retriever returning a canned ranked list (truncated to the requested k)."""
    def __init__(self, hits):
        self.hits, self.asked = hits, []

    def search(self, query, k):
        self.asked.append(k)
        return self.hits[:k]


A, B, C, D = (It(c) for c in "ABCD")


def ids(hits):
    return [h.chunk_id for h, _ in hits]


def test_rrf_exact_and_missing():
    r1 = Fixed([(A, 9.0), (B, 5.0), (C, 1.0)])
    r2 = Fixed([(B, 0.9), (D, 0.5)])
    h = HybridRetriever([r1, r2], "rrf", k_rrf=60).search("q", 10)
    exp = {"A": 1 / 61, "B": 1 / 62 + 1 / 61, "C": 1 / 63, "D": 1 / 62}
    assert ids(h) == ["B", "A", "D", "C"]
    for item, s in h:
        assert abs(s - exp[item.chunk_id]) < 1e-12


def test_convex_exact_and_missing():
    r1 = Fixed([(A, 10.0), (B, 6.0), (C, 2.0)])  # normalised 1, .5, 0
    r2 = Fixed([(B, 4.0), (D, 2.0)])  # 1, 0
    h = HybridRetriever([r1, r2], "convex").search("q", 10)
    exp = {"A": 1.0, "B": 1.5, "C": 0.0, "D": 0.0}
    assert ids(h)[:2] == ["B", "A"]
    for item, s in h:
        assert abs(s - exp[item.chunk_id]) < 1e-12
    # C and D tie at 0: C has better rank (3 vs 2)? D rank 2 < C rank 3 -> D first
    assert ids(h)[2:] == ["D", "C"]


def test_convex_weights_and_constant_list():
    r1 = Fixed([(A, 3.0), (B, 1.0)])
    r2 = Fixed([(B, 7.0), (C, 7.0)])  # all-equal -> both 1
    h = HybridRetriever([r1, r2], "convex", weights=[2.0, 1.0]).search("q", 10)
    exp = {"A": 2.0, "B": 1.0, "C": 1.0}
    assert {i.chunk_id: s for i, s in h} == pytest.approx(exp)
    assert ids(h) == ["A", "B", "C"]  # B/C tie: B has best rank 1 (list 2), C rank 2


def test_deterministic_ties_first_appearance():
    r1 = Fixed([(A, 1.0), (B, 0.5)])
    r2 = Fixed([(B, 1.0), (A, 0.5)])
    hy = HybridRetriever([r1, r2], "rrf")
    assert ids(hy.search("q", 2)) == ["A", "B"] and hy.search("q", 2) == hy.search("q", 2)
    r3 = Fixed([(C, 1.0)])
    r4 = Fixed([(D, 1.0)])
    assert ids(HybridRetriever([r3, r4], "rrf").search("q", 2)) == ["C", "D"]


def test_depth_and_k():
    r1 = Fixed([(A, 3.0), (B, 2.0), (C, 1.0)])
    hy = HybridRetriever([r1], "rrf", depth=2)
    assert ids(hy.search("q", 10)) == ["A", "B"] and r1.asked == [2]  # fused list limited by depth
    assert hy.search("q", 0) == [] and len(hy.search("q", 1)) == 1


def test_convex_depends_on_depth():
    r = Fixed([(A, 10.0), (B, 5.0), (C, 0.0)])
    s_full = dict((i.chunk_id, s) for i, s in HybridRetriever([r], "convex", depth=3).search("q", 3))
    s_cut = dict((i.chunk_id, s) for i, s in HybridRetriever([r], "convex", depth=2).search("q", 3))
    assert s_full["B"] == 0.5 and s_cut["B"] == 0.0


def test_empty_and_short_lists():
    hy = HybridRetriever([Fixed([]), Fixed([(A, 1.0)])], "convex")
    assert ids(hy.search("q", 5)) == ["A"]
    assert HybridRetriever([Fixed([])], "rrf").search("q", 5) == []


def test_identity_key_without_chunk_id():
    class Plain:
        text = "x"
    p, q = Plain(), Plain()
    h = HybridRetriever([Fixed([(p, 1.0), (q, 0.5)]), Fixed([(p, 1.0)])], "rrf").search("q", 5)
    assert [i for i, _ in h] == [p, q]


def test_validation():
    with pytest.raises(ValueError):
        HybridRetriever([Fixed([])], "bogus")
    with pytest.raises(ValueError):
        HybridRetriever([Fixed([])], "rrf", weights=[1, 2])
    with pytest.raises(ValueError):
        HybridRetriever([], "rrf")


def test_with_real_bm25_and_dense_like():
    from arxiv_rag.lexical import BM25Retriever
    items = [It("1", "cats purr"), It("2", "dogs bark"), It("3", "cats and dogs")]
    b = BM25Retriever(items)
    h = HybridRetriever([b, b], "rrf").search("cats", 5)
    assert ids(h) == ["1", "3"]
