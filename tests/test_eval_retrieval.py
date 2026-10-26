import json
import sys

import numpy as np
import pytest

from arxiv_rag import eval_chunking as ev
from arxiv_rag import eval_retrieval as er
from arxiv_rag.chunk import Chunk
from arxiv_rag.embed import Embedder
from arxiv_rag.hybrid import HybridRetriever
from arxiv_rag.lexical import BM25Retriever
from arxiv_rag.rerank import Reranker, RerankedRetriever
from arxiv_rag.retrieve import DenseRetriever
from fakes import FakeBackend, FakeRerankBackend
from test_eval_chunking import make_docs, row, write_gold

Q1 = "ZetaNet retrieval long text"
Q2 = "Beta results baseline model improve"
Q3 = "quantum gardening soil"


def make_chunks(n=4):
    return [Chunk(f"p1:structure:{i}", "p1", "structure", f"word{i} shared text number {i}", i * 10, i * 10 + 10,
                  None, 1, 1) for i in range(n)]


def gold_rows(docs):
    d1, d2 = docs["p1"], docs["p2"]
    return [
        row(d1, "Alpha uses ZetaNet for retrieval. It works well on long text.", status="auto_ok", question=Q1,
            overlap=0.1, span_type="paragraph"),
        row(d1, "Beta results improve by 12.5% over the baseline model.", status="auto_ok", question=Q2,
            overlap=0.5, span_type="sentence", judge={"copy": True}),
        row(d2, "Gamma is a separate topic about quantum gardening and soil.", status="auto_ok", paper="p2",
            question=Q3, overlap=0.3, span_type="sentence"),
    ]


@pytest.fixture
def env(tmp_path):
    docs = make_docs()
    path = write_gold(tmp_path / "g.jsonl", gold_rows(docs))
    gold = ev.load_gold(path, docs).items
    emb = Embedder(backend=FakeBackend(), cache_dir=tmp_path / "emb")
    rr = Reranker(backend=FakeRerankBackend(), cache_dir=tmp_path / "rr")
    return docs, gold, emb, rr


def test_gold_overlap_is_loaded(env):
    _, gold, _, _ = env
    assert [g.overlap for g in gold] == [0.1, 0.5, 0.3]
    assert er.lexical_split(gold) == pytest.approx(0.3)


def test_build_configs_names_order_and_types(tmp_path):
    chunks = make_chunks()
    emb = Embedder(backend=FakeBackend(), cache_dir=tmp_path)
    rr = Reranker(backend=FakeRerankBackend(), cache_dir=tmp_path)
    cfgs = er.build_configs(chunks, emb, rr, list(er.BASE_RETRIEVERS), True, 7)
    assert list(cfgs) == ["dense", "bm25", "hybrid-rrf", "hybrid-convex", "dense+rerank", "bm25+rerank",
                          "hybrid-rrf+rerank", "hybrid-convex+rerank"]
    assert isinstance(cfgs["dense"], DenseRetriever) and isinstance(cfgs["bm25"], BM25Retriever)
    h = cfgs["hybrid-convex"]
    assert isinstance(h, HybridRetriever) and h.method == "convex" and h.depth == 100
    assert cfgs["hybrid-rrf"].method == "rrf" and cfgs["hybrid-rrf"].k_rrf == 60
    assert h.weights == [1.0, 1.0] and h.retrievers[0] is cfgs["dense"] and h.retrievers[1] is cfgs["bm25"]
    w = cfgs["bm25+rerank"]
    assert isinstance(w, RerankedRetriever) and w.base is cfgs["bm25"] and w.depth == 7 and w.reranker is rr


def test_build_configs_no_rerank_and_unknown(tmp_path):
    emb = Embedder(backend=FakeBackend(), cache_dir=tmp_path)
    assert list(er.build_configs(make_chunks(), emb, None, ["bm25", "dense"], False, 5)) == ["bm25", "dense"]
    with pytest.raises(ValueError):
        er.build_configs(make_chunks(), emb, None, ["splade"], False, 5)


def test_metrics_on_hand_built_ranking():
    m = ev.score_ranking([False, False, True] + [False] * 17, [True] * 3 + [False] * 17, [0] * 20)
    assert [m[f"recall@{k}"] for k in er.RECALL_KS[:3]] == [0, 1, 1]
    assert m["mrr@20"] == pytest.approx(1 / 3)
    assert not any(k.endswith("tok") for k in er.METRICS)
    # shorter than K_MAX (BM25 may return few hits) must not break scoring
    m = ev.score_ranking([True], [True], [0])
    assert m["recall@20"] == 1 and m["mrr@20"] == 1


def test_paired_diff_direction():
    boot = ev.PaperBootstrap(["a", "a", "b", "c"], 200, 0)
    better, base = np.array([1.0, 1, 1, 0]), np.array([1.0, 0, 0, 0])
    d = ev.paired_diff(better, base, boot)
    assert d["diff"] == pytest.approx(0.5) and d["frac_le_0"] < 0.5


def test_truncation_counter(tmp_path):
    backend = FakeRerankBackend()
    backend.max_tokens = 10
    rr = Reranker(backend=backend, cache_dir=tmp_path)
    c = er.CountingReranker(rr)
    long, short = " ".join(["w"] * 20), "tiny"
    c.score("q", [short, long, long])
    c.score("q", [long])  # repeated pair is not counted twice
    assert (c.pairs, c.over_limit) == (2, 1) and c.longest == 1 + 20 + 3
    assert c.max_tokens == 10


def run_eval(env, **kw):
    docs, gold, emb, rr = env
    a = dict(chunking="structure", names=list(er.BASE_RETRIEVERS), rerank=True, depth=5, max_tokens=64,
             overlap=8, n_boot=50, seed=0)
    a.update(kw)
    return er.evaluate(list(docs.values()), gold, emb, rr, a["chunking"], a["names"], a["rerank"], a["depth"],
                       a["max_tokens"], a["overlap"], a["n_boot"], a["seed"])


def test_evaluate_structure(env):
    res = run_eval(env)
    assert len(res["configs"]) == 8
    for c in res["configs"].values():
        assert set(c["metrics"]) == set(er.METRICS)
        assert all(0 <= v["lo"] <= v["mean"] <= v["hi"] <= 1 for v in c["metrics"].values())
    assert "dense" not in res["paired_vs_dense"] and "bm25" in res["paired_vs_dense"]
    assert set(res["rerank_vs_base"]) == {n + "+rerank" for n in er.BASE_RETRIEVERS}
    t = res["rerank_truncation"]
    assert t["over_limit"] == 0 and t["distinct_pairs"] > 0
    assert sum(v["n"] for v in res["by_lexical_overlap"].values()) == 3
    assert {v["n"] for v in res["by_copy"].values()} == {1, 2}
    assert res["lexical_split"] == pytest.approx(0.3)
    assert res["configs"]["bm25"]["metrics"]["recall@20"]["mean"] == 1.0  # exact words, tiny docs


def test_each_pair_scored_once(env):
    _, _, _, rr = env
    run_eval(env)
    seen = [(q, t) for q, texts in rr.backend.calls for t in texts]
    assert seen and len(seen) == len(set(seen))


def test_no_rerank_has_no_truncation_section(env):
    res = run_eval(env, rerank=False, names=["dense", "bm25"])
    assert list(res["configs"]) == ["dense", "bm25"] and not res["rerank_vs_base"]
    assert "rerank_truncation" not in res


def test_main_limit_and_outputs(env, tmp_path, monkeypatch):
    docs, _, emb, rr = env
    monkeypatch.setattr(er, "read_ids_file", lambda p: ["x"])
    monkeypatch.setattr(er, "ids_to_stems", lambda ids: ids)
    monkeypatch.setattr(er, "load_documents", lambda d, s: list(docs.values()))
    monkeypatch.setattr(er, "Embedder", lambda: emb)
    monkeypatch.setattr(er, "Reranker", lambda: rr)
    monkeypatch.setattr(er, "git_sha", lambda: "abc123")
    monkeypatch.setattr(er, "git_dirty", lambda: True)
    gold = tmp_path / "g.jsonl"
    write_gold(gold, gold_rows(docs))
    out = tmp_path / "res" / "p3"
    monkeypatch.setattr(sys, "argv", ["x", "--gold", str(gold), "--out", str(out), "--limit", "2", "--max-tokens",
                                      "64", "--bootstrap", "20", "--rerank-depth", "5"])
    er.main()
    rep = json.loads((tmp_path / "res" / "p3.json").read_text(encoding="utf-8"))
    assert rep["n_questions"] == 2 and rep["git_commit"] == "abc123"
    md_bytes = (tmp_path / "res" / "p3.md").read_bytes()
    assert b"\r" not in md_bytes
    md = md_bytes.decode("utf-8")
    for s in ("# Phase 3", "Lexical-anchor bias", "machine-verified", "dense+rerank", "reranked vs its base",
              "By lexical overlap", "No parameters were tuned", "(uncommitted changes)", "exceed the model"):
        assert s in md


# --- setup recorded from the retriever objects ---------------------------------------------

def test_setup_is_read_from_retrievers(tmp_path):
    emb = Embedder(backend=FakeBackend(), cache_dir=tmp_path)
    rr = Reranker(backend=FakeRerankBackend(), cache_dir=tmp_path)
    cfgs = er.build_configs(make_chunks(), emb, rr, list(er.BASE_RETRIEVERS), True, 7)
    cfgs["bm25"].k1, cfgs["bm25"].b = 1.2, 0.75
    cfgs["hybrid-rrf"].k_rrf, cfgs["hybrid-convex"].weights = 33, [2.0, 1.0]
    s = er.describe_setup(cfgs, rr, 7)
    assert s["bm25"] == {"k1": 1.2, "b": 0.75, "tokenizer": er.BM25_TOKENIZER}
    assert s["fusion"]["hybrid-rrf"] == {"method": "rrf", "weights": [1.0, 1.0], "k_rrf": 33, "depth": 100}
    assert s["fusion"]["hybrid-convex"]["weights"] == [2.0, 1.0]
    assert s["reranker"] == {"model": rr.model_name, "tag": "fake", "depth": 7}
    assert er.describe_setup({"dense": cfgs["dense"]}, None, 7) == {}


def test_lexical_split_skips_missing_or_zero_overlap(env):
    _, gold, _, _ = env
    assert er.lexical_split(gold) == pytest.approx(0.3)
    gold[0].overlap = None
    assert er.lexical_split(gold) is None
    for g in gold:
        g.overlap = 0.0
    assert er.lexical_split(gold) is None


def test_evaluate_skips_overlap_breakdown(env, capsys):
    _, gold, _, _ = env
    gold[1].overlap = None
    res = run_eval(env, rerank=False, names=["dense", "bm25"])
    assert res["by_lexical_overlap"] == {} and res["lexical_split"] is None
    assert "no `overlap` field" in res["lexical_overlap_skipped"]
    assert "breakdown skipped" in capsys.readouterr().out
    report = {"config": {"gold": "g", "chunking": "structure", "max_tokens": 64, "overlap": 8, "bootstrap": 50,
                         "seed": 0, "rerank_depth": 5},
              "draft": False, "gold": {}, "n_questions": 3, "n_papers": 2, "embedding": {"model": "m", "backend": "b"},
              "git_commit": None, "results": res}
    md = er.render_markdown(report)
    assert "## By lexical overlap" in md and "Skipped: some gold rows" in md


# --- rerank ceiling, hybrid agreement, anchors, ablation -------------------------------------

def test_rerank_ceiling_and_hybrid_agreement(env):
    res = run_eval(env)
    rc = res["rerank_ceiling"]
    assert rc["candidates"] == er.K_MAX and set(rc["configs"]) == set(er.BASE_RETRIEVERS)
    assert rc["configs"]["bm25"] == 1.0  # tiny corpus: the top 20 holds every chunk the query matches
    ag = res["hybrid_rerank_agreement"]
    assert ag["configs"] == ["hybrid-rrf+rerank", "hybrid-convex+rerank"] and ag["n"] == 3
    assert 0 <= ag["same_top1"] <= 1 and 0 <= ag["same_first_hit_rank"] <= 1
    assert "rerank_ceiling" not in run_eval(env, rerank=False)
    assert "hybrid_rerank_agreement" not in run_eval(env, rerank=False)


def test_ceiling_is_below_one_when_candidates_miss_the_evidence(env):
    _, gold, _, _ = env
    gold[0].paper_id = "p2"  # no chunk of p2 covers this evidence span, so no candidate can hit
    res = run_eval(env, names=["bm25"])
    assert res["rerank_ceiling"]["configs"]["bm25"] < 1.0


def test_anchor_helpers():
    assert er.anchor_bucket(None) == er.ANCHOR_BUCKETS[3]
    assert [er.anchor_bucket(d) for d in (0, 1, 2, 5, 6)] == [er.ANCHOR_BUCKETS[i] for i in (0, 0, 1, 1, 2)]
    assert er.remove_anchor_words("How does ZetaNet-based retrieval work on 12.5% of cases?", ["zetanet", "12.5%"]) \
        == "How does retrieval work on of cases?"
    # short all-letter acronyms match whole words only: MAP != mapping
    assert er.remove_anchor_words("MAP and mapping", ["map"]) == "and mapping"
    assert er.remove_anchor_words("nothing here", []) == "nothing here"


def test_paper_df_counts_papers_not_chunks():
    chunks = [Chunk(f"p{i}:s:{j}", f"p{i}", "s", t, 0, 1, None, 1, 1)
              for i, texts in enumerate([["BERT is used", "BERT again"], ["we use bert-large"], ["unrelated"]])
              for j, t in enumerate(texts)]
    df = er.paper_df(["bert", "unrelated", "missing"], er._paper_norms(chunks))
    assert df == {"bert": 2, "unrelated": 1, "missing": 0}


def test_anchor_terms_and_evaluate_anchor_sections(env):
    _, gold, _, _ = env
    assert gold[0].quote.startswith("Alpha uses ZetaNet")
    assert er.anchor_terms(gold[0]) == ["zetanet"] and er.anchor_terms(gold[1]) == []
    res = run_eval(env)
    r = res["by_anchor_rarity"]
    assert r[er.ANCHOR_BUCKETS[0]]["n"] == 1 and r[er.ANCHOR_BUCKETS[3]]["n"] == 2
    assert res["anchor_common"]["n"] == 0
    ab = res["ablation"]
    assert (ab["n"], ab["n_changed"], ab["n_empty"]) == (3, 1, 0)
    assert set(ab["configs"]) == set(er.BASE_RETRIEVERS)  # no reranked config in the ablation
    for c in ab["configs"].values():
        for k in er.ABLATION_KS:
            v = c[f"recall@{k}"]
            assert v["change"] == pytest.approx(v["ablated"] - v["normal"])
    assert ab["configs"]["bm25"]["recall@10"]["normal"] == res["configs"]["bm25"]["metrics"]["recall@10"]["mean"]


def test_ablation_that_removes_every_word_scores_a_miss(env):
    _, gold, _, _ = env
    gold[0].question = "ZetaNet"
    res = run_eval(env, rerank=False, names=["dense", "bm25"])
    assert res["ablation"]["n_empty"] == 1
    assert res["ablation"]["configs"]["bm25"]["recall@10"]["ablated"] <= 2 / 3 + 1e-9


def test_anchor_note_is_computed():
    def cfg(r5):
        return {"metrics": {"recall@5": {"mean": r5}}}
    res = {"configs": {"bm25": cfg(0.8), "dense": cfg(0.6)},
           "anchor_common": {"n": 7, "recall@5": {"bm25": 0.7, "dense": 0.4}},
           "ablation": {"configs": {"bm25": {"recall@5": {"ablated": 0.5}}, "dense": {"recall@5": {"ablated": 0.3}}}}}
    note = er.anchor_note(res)
    assert "+0.200 on all questions" in note and "+0.300 on the 7 questions" in note
    assert "+0.200 after anchor removal" in note
    assert "persists for common anchors and after anchor removal" in note and "broader lexical-match advantage" in note
    res["ablation"]["configs"]["bm25"]["recall@5"]["ablated"] = 0.2
    assert "does not persist" in er.anchor_note(res)
    assert er.anchor_note({"configs": {"dense": cfg(0.5)}, "ablation": {"configs": {}}}) == ""


def run_main(env, tmp_path, monkeypatch, *extra):
    docs, _, emb, rr = env
    monkeypatch.setattr(er, "read_ids_file", lambda p: ["x"])
    monkeypatch.setattr(er, "ids_to_stems", lambda ids: ids)
    monkeypatch.setattr(er, "load_documents", lambda d, s: list(docs.values()))
    monkeypatch.setattr(er, "Embedder", lambda: emb)
    monkeypatch.setattr(er, "Reranker", lambda: rr)
    gold = tmp_path / "g.jsonl"
    write_gold(gold, gold_rows(docs))
    out = tmp_path / "res" / "p3"
    monkeypatch.setattr(sys, "argv", ["x", "--gold", str(gold), "--out", str(out), "--max-tokens", "64",
                                      "--bootstrap", "20", "--rerank-depth", "5", *extra])
    er.main()
    return (json.loads((tmp_path / "res" / "p3.json").read_text(encoding="utf-8")),
            (tmp_path / "res" / "p3.md").read_text(encoding="utf-8"))


def test_main_partial_banner_config_and_new_sections(env, tmp_path, monkeypatch):
    rep, md = run_main(env, tmp_path, monkeypatch, "--limit", "2")
    assert rep["partial"] is True and "PARTIAL: first 2 questions (file order)" in md
    c = rep["config"]
    assert c["bm25"]["k1"] == 0.9 and c["bm25"]["tokenizer"] == er.BM25_TOKENIZER
    assert c["fusion"]["hybrid-rrf"]["k_rrf"] == 60 and c["fusion"]["hybrid-convex"]["weights"] == [1.0, 1.0]
    assert c["reranker"] == {"model": rep["reranker"]["model"], "tag": "fake", "depth": 5}
    for s in ("## By anchor rarity", "## Anchor-removed ablation", "## Rerank ceiling", "## Reranked hybrids compared",
              "identical top-1 chunk", "overfit 2 questions", "BM25 minus dense recall@5",
              "Lexical: BM25 (k1=0.9, b=0.4)"):
        assert s in md
    assert "192" not in md


def test_main_full_run_not_partial_and_lexical_lines_only_when_configured(env, tmp_path, monkeypatch):
    rep, md = run_main(env, tmp_path, monkeypatch, "--retrievers", "dense", "--no-rerank")
    assert rep["partial"] is False and "PARTIAL" not in md
    assert "bm25" not in rep["config"] and "fusion" not in rep["config"] and rep["reranker"] is None
    assert "- Lexical:" not in md and "- Hybrid" not in md and "## Rerank ceiling" not in md
