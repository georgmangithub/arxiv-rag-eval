import math
import random
import time
from dataclasses import dataclass

from arxiv_rag.lexical import BM25Retriever, tokenize


@dataclass
class Item:
    text: str


def bm25(items, **kw):
    return BM25Retriever([Item(t) for t in items], **kw)


def test_tokenize_basic_and_stopwords():
    assert tokenize("The Cat, and the HAT!") == ["cat", "hat"]
    assert tokenize("") == [] and tokenize("... --") == []


def test_tokenize_compounds():
    assert sorted(tokenize("GPT-4")) == ["4", "gpt", "gpt-4"]
    assert sorted(tokenize("3.5")) == ["3", "3.5", "5"]
    assert sorted(tokenize("bert_base")) == ["base", "bert", "bert_base"]
    # separators with spaces around do not join
    assert tokenize("a - b") == ["b"] and tokenize("x . y") == ["x", "y"]
    assert "state-of-the-art" in tokenize("state-of-the-art")  # stopword part dropped, compound kept
    assert "the" not in tokenize("state-of-the-art")


def test_no_stemming():
    assert tokenize("models modeling") == ["models", "modeling"]


def test_exact_scores_three_docs():
    docs = ["apple banana apple", "banana cherry", "cherry date elder fig"]
    k1, b = 0.9, 0.4
    r = bm25(docs, k1=k1, b=b)
    n, avgdl = 3, (3 + 2 + 4) / 3
    idf = lambda df: math.log(1 + (n - df + 0.5) / (df + 0.5))
    tfw = lambda tf, dl: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    hits = r.search("apple banana", 3)
    exp = {
        0: idf(1) * tfw(2, 3) + idf(2) * tfw(1, 3),
        1: idf(2) * tfw(1, 2),
    }
    assert [h[0].text for h in hits] == [docs[0], docs[1]]
    for (item, s), i in zip(hits, (0, 1)):
        assert abs(s - exp[i]) < 1e-9
    # literal sanity check of the idf constants: df=1 -> ln(1+2.5/1.5), df=2 -> ln(1+1.5/2.5)
    assert abs(idf(1) - math.log(1 + 2.5 / 1.5)) < 1e-12 and abs(idf(2) - math.log(1.6)) < 1e-12


def test_idf_rare_term_outranks_common():
    r = bm25(["common rare", "common x", "common y", "common z"])
    hits = r.search("common rare", 4)
    assert hits[0][0].text == "common rare"
    s = dict((h.text, sc) for h, sc in r.search("rare", 4))
    c = dict((h.text, sc) for h, sc in r.search("common", 4))
    assert s["common rare"] > c["common rare"]


def test_k_edge_cases_and_no_match():
    r = bm25(["alpha beta", "beta gamma", "delta"])
    assert r.search("beta", 0) == [] and r.search("beta", -1) == []
    assert len(r.search("beta", 10)) == 2  # k > n: only matching docs, no padding
    assert r.search("zzz", 3) == [] and r.search("", 3) == [] and r.search("the of", 3) == []
    assert bm25([]).search("x", 3) == []


def test_ordering_and_tie_break_by_index():
    r = bm25(["same words", "same words", "other", "same words"])
    hits = r.search("same words", 2)
    assert [h[0] for h in hits] == [r.items[0], r.items[1]]
    assert hits[0][1] == hits[1][1]
    full = r.search("same words", 10)
    assert [h[0] for h in full] == [r.items[0], r.items[1], r.items[3]]
    assert r.search("same words", 2) == r.search("same words", 2)
    scores = [s for _, s in full]
    assert scores == sorted(scores, reverse=True)


def test_compound_term_matches_exactly():
    r = bm25(["we use gpt-4 here", "gpt models and 4 things"])
    assert r.search("GPT-4", 2)[0][0].text == "we use gpt-4 here"


def test_speed_6k_docs_200_queries():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(5000)]
    docs = [" ".join(rng.choices(vocab, k=120)) for _ in range(6000)]
    t = time.perf_counter()
    r = bm25(docs)
    queries = [" ".join(rng.choices(vocab, k=12)) for _ in range(200)]
    for q in queries:
        assert len(r.search(q, 10)) == 10
    assert time.perf_counter() - t < 5


def test_scores_do_not_depend_on_hash_seed():
    import os
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent("""
        from arxiv_rag.lexical import BM25Retriever
        class D:
            def __init__(self, t): self.text = t
        docs = [D(f"alpha beta{i} gamma delta{i % 3} epsilon zeta eta theta alpha") for i in range(30)]
        print(repr([s for _, s in BM25Retriever(docs).search("alpha gamma delta1 eta theta beta4 zeta epsilon", 30)]))
    """)
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}).stdout
            for seed in ("1", "2", "3", "4")}
    assert len(outs) == 1
