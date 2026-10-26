from dataclasses import dataclass

from arxiv_rag.embed import Embedder
from arxiv_rag.retrieve import DenseRetriever
from fakes import FakeBackend


@dataclass
class Item:
    text: str


def retriever(tmp_path, texts):
    return DenseRetriever(Embedder(cache_dir=tmp_path, backend=FakeBackend()), [Item(t) for t in texts])


TEXTS = ["cats purr softly", "dogs bark loudly", "stock markets fell", "cats and dogs", "quantum field theory"]


def test_ranking_and_order(tmp_path):
    hits = retriever(tmp_path, TEXTS).search("cats purr", 3)
    assert hits[0][0].text == "cats purr softly"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and len(hits) == 3
    assert hits[0][1] > 0.7


def test_exact_text_scores_one(tmp_path):
    (item, score), = retriever(tmp_path, TEXTS).search("quantum field theory", 1)
    assert item.text == "quantum field theory" and abs(score - 1.0) < 1e-5


def test_k_larger_than_n(tmp_path):
    assert len(retriever(tmp_path, TEXTS).search("cats", 100)) == len(TEXTS)


def test_k_zero_and_empty(tmp_path):
    assert retriever(tmp_path, TEXTS).search("cats", 0) == []
    assert retriever(tmp_path, []).search("cats", 5) == []


def test_ties_break_by_index(tmp_path):
    r = retriever(tmp_path, ["same", "same", "same", "same", "other"])
    got = [it for it, _ in r.search("same", 2)]
    assert got == [r.items[0], r.items[1]]
    assert [it for it, _ in r.search("same", 4)] == r.items[:4]
