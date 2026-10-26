import json

import numpy as np
import pytest

from arxiv_rag import eval_chunking as ev
from arxiv_rag.chunk import Chunk
from arxiv_rag.document import build_document
from arxiv_rag.embed import Embedder
from fakes import FakeBackend


def chunk(paper="p1", start=0, end=10):
    return Chunk("c", paper, "fixed", "x", start, end, None, 1, 1)


def item(start=100, end=200, paper="p1", **kw):
    d = dict(qid=f"{paper}:{start}-{end}", paper_id=paper, question="q", char_start=start, char_end=end,
             span_type="paragraph", status="ok", title_overlap=0.0)
    d.update(kw)
    return ev.GoldItem(**d)


def test_hit_rule_boundary():
    gold = item(100, 200)  # 100 chars
    assert not ev.is_hit(chunk(start=51, end=149), gold)  # covers 49 of 100
    assert ev.is_hit(chunk(start=0, end=150), gold)  # exactly 50% -> hit
    assert not ev.is_hit(chunk(start=0, end=149), gold)  # 49%
    assert ev.is_hit(chunk(start=150, end=400), gold)  # 50% from the other side
    assert not ev.is_hit(chunk(start=151, end=400), gold)  # 49%
    assert ev.is_hit(chunk(start=90, end=300), gold)  # full cover
    assert not ev.is_hit(chunk(paper="p2", start=0, end=300), gold)  # wrong paper
    assert not ev.is_hit(chunk(start=300, end=400), gold)  # disjoint


def test_hit_rule_uses_span_not_text():
    # A header-prefixed chunk is judged by its span only: the header adds no coverage.
    c = Chunk("c", "p1", "structure+header", "Title > Sec\n\n" + "x" * 49, 100, 149, None, 1, 1)
    assert not ev.is_hit(c, item(100, 200))
    c.char_end = 150
    assert ev.is_hit(c, item(100, 200))


def test_score_ranking_hand_computed():
    hits = [False, False, True, False, False] + [False] * 15
    paper = [False, True, True] + [False] * 17
    tokens = [300, 300, 300, 300, 300] + [100] * 15
    m = ev.score_ranking(hits, paper, tokens)
    assert [m[f"recall@{k}"] for k in (1, 3, 5, 10, 20)] == [0, 1, 1, 1, 1]
    assert m["mrr@20"] == pytest.approx(1 / 3)
    assert [m[f"paper_recall@{k}"] for k in (1, 5, 10)] == [0, 1, 1]
    # 512: only the first chunk fits (300), second would reach 600 -> stop, no hit
    # 1024: three chunks (900) fit, fourth would be 1200 -> hit at rank 3
    assert [m[f"recall@{b}tok"] for b in (512, 1024, 2048)] == [0, 1, 1]


def test_budget_stops_at_first_overflow_and_mrr_zero_on_miss():
    m = ev.score_ranking([False, True], [True, True], [600, 10])
    assert m["recall@512tok"] == 0 and m["recall@1024tok"] == 1  # 600+10 fits 1024
    m = ev.score_ranking([False, True, True], [True] * 3, [100, 500, 10])
    assert m["recall@512tok"] == 0  # 600 > 512 stops the walk; later small chunks are not skipped to
    miss = ev.score_ranking([False] * 20, [False] * 20, [10] * 20)
    assert miss["mrr@20"] == 0 and miss["recall@20"] == 0


# --- gold loading ------------------------------------------------------------------------

def make_docs():
    a = build_document("p1", "Alpha Paper", [
        {"text": "Alpha uses ZetaNet for retrieval. It works well on long text.", "section": "1 Intro", "page": 1},
        {"text": "Beta results improve by 12.5% over the baseline model.", "section": "2 Results", "page": 2},
    ])
    b = build_document("p2", "Beta Paper", [
        {"text": "Gamma is a separate topic about quantum gardening and soil.", "section": "1 Intro", "page": 1},
    ])
    return {"p1": a, "p2": b}


def row(doc, quote, status="ok", paper="p1", sha=None, **kw):
    s = doc.text.index(quote)
    r = {"qid": f"{paper}:{s}-{s + len(quote)}", "paper_id": paper, "question": "What?",
         "evidence_quote": quote, "char_start": s, "char_end": s + len(quote),
         "doc_sha256": sha or ev.sha256_text(doc.text), "span_type": "sentence", "status": status,
         "overlap": 0.1, "title_overlap": 0.0}
    r.update(kw)
    return r


def write_gold(path, rows, extra=()):
    lines = [json.dumps(r) for r in rows] + list(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_gold_filters_stale_dupes_and_junk(tmp_path, capsys):
    docs = make_docs()
    d1, d2 = docs["p1"], docs["p2"]
    good = row(d1, "Alpha uses ZetaNet for retrieval.")
    edited = row(d1, "Beta results improve by 12.5% over the baseline model.", status="edit")
    draft = row(d2, "Gamma is a separate topic about quantum gardening and soil.", status="draft", paper="p2")
    bad = row(d1, "It works well on long text.", status="bad")
    stale_sha = row(d1, "It works well on long text.", sha="0" * 64)
    stale_quote = row(d1, "Alpha uses ZetaNet for retrieval.", evidence_quote="Alpha uses ZetaNet for retrievals.",
                      qid="p1:other")
    dup = dict(good, question="newer wording")
    off = row(d1, "Alpha uses ZetaNet for retrieval.", paper="p9", qid="p9:1-2")
    path = write_gold(tmp_path / "g.jsonl", [good, edited, draft, bad, stale_sha, stale_quote, dup, off],
                      extra=["{not json", json.dumps({"qid": "x"})])
    g = ev.load_gold(path, docs)
    assert [i.qid for i in g.items] == [good["qid"], edited["qid"]]
    assert g.items[0].question == "newer wording"  # last wins
    c = g.counts
    assert (c["unparseable"], c["status_filtered"], c["stale"], c["duplicates"], c["off_corpus"]) == (2, 2, 2, 1, 1)
    assert "stale gold rows dropped" in capsys.readouterr().out
    g2 = ev.load_gold(path, docs, allow_draft=True)
    assert {i.status for i in g2.items} == {"ok", "edit", "draft"}


def test_load_gold_accepts_auto_ok_and_reports_status_and_copy(tmp_path):
    docs = make_docs()
    d1 = docs["p1"]
    a = row(d1, "Alpha uses ZetaNet for retrieval.", status="auto_ok",
            judge={"answerable": True, "specific": True, "copy": True})
    b = row(d1, "Beta results improve by 12.5% over the baseline model.", status="auto_ok")  # no verdict
    c = row(d1, "It works well on long text.", status="auto_bad")
    g = ev.load_gold(write_gold(tmp_path / "g.jsonl", [a, b, c]), docs)
    assert [i.status for i in g.items] == ["auto_ok", "auto_ok"] and g.by_status == {"auto_ok": 2}
    assert [i.copy for i in g.items] == [True, False] and g.counts["status_filtered"] == 1


def test_load_gold_dedupes_before_status_filter(tmp_path):
    docs = make_docs()
    d1 = docs["p1"]
    ok = row(d1, "Alpha uses ZetaNet for retrieval.")
    later_bad = dict(ok, status="bad")
    auto = row(d1, "Beta results improve by 12.5% over the baseline model.", status="auto_ok")
    other = row(d1, "It works well on long text.")
    path = write_gold(tmp_path / "g.jsonl", [ok, auto, other, later_bad, dict(auto, status="draft")])
    g = ev.load_gold(path, docs)
    assert [i.qid for i in g.items] == [other["qid"]]  # ok->bad and auto_ok->draft: last row wins
    assert g.counts["duplicates"] == 2 and g.counts["status_filtered"] == 2


def test_load_gold_errors_when_nothing_remains(tmp_path):
    docs = make_docs()
    path = write_gold(tmp_path / "g.jsonl", [row(docs["p1"], "Alpha uses ZetaNet for retrieval.", status="draft")])
    with pytest.raises(ValueError, match="no usable gold rows"):
        ev.load_gold(path, docs)


# --- bootstrap ---------------------------------------------------------------------------

def pids(n, per=5):
    return [f"p{i // per}" for i in range(n)]


def test_bootstrap_deterministic_and_ci_contains_mean():
    v = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1] * 5, dtype=float)
    a = ev.mean_ci(v, ev.bootstrap_indices(pids(len(v), per=3), 500, seed=3))
    b = ev.mean_ci(v, ev.bootstrap_indices(pids(len(v), per=3), 500, seed=3))
    assert a == b
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert ev.mean_ci(v, ev.bootstrap_indices(pids(len(v), per=3), 500, seed=4)) != a


def test_bootstrap_resamples_whole_papers():
    # paper A: 3 questions all 1; paper B: 1 question, 0 -> a resample mean is one of 1, 3/4, 0
    boot = ev.bootstrap_indices(["A", "A", "A", "B"], 400, seed=1)
    means = boot.means(np.array([1.0, 1.0, 1.0, 0.0]))
    assert set(np.round(means, 6)) <= {1.0, 0.75, 0.0} and len(set(np.round(means, 6))) == 3
    ci = ev.mean_ci(np.array([1.0, 1.0, 1.0, 0.0]), boot)
    assert ci["lo"] == 0.0 and ci["hi"] == 1.0  # per-question resampling would give a narrower CI


def test_bootstrap_is_paired_across_strategies():
    boot = ev.bootstrap_indices(pids(40), 200, seed=0)
    x = np.arange(40) % 3 / 2
    assert ev.paired_diff(x, x, boot)["diff"] == 0
    assert np.allclose(boot.means(x) - boot.means(x * 0.5), boot.means(x * 0.5))  # same papers drawn


def test_paired_diff_sign():
    x = np.array([1, 1, 1, 0, 1, 1, 0, 1] * 5, dtype=float)
    y = np.array([0, 1, 0, 0, 1, 0, 0, 1] * 5, dtype=float)
    idx = ev.bootstrap_indices(pids(len(x), per=8), 500, seed=0)
    d = ev.paired_diff(x, y, idx)
    assert d["diff"] > 0 and d["lo"] > 0 and d["frac_le_0"] == 0.0
    r = ev.paired_diff(y, x, idx)
    assert r["diff"] < 0 and r["hi"] < 0 and r["frac_le_0"] == 1.0
    assert ev.paired_diff(x, x, idx)["frac_le_0"] == 1.0  # zero difference counts as <= 0


# --- end to end with a fake embedder -----------------------------------------------------

def paper(pid, topic):
    recs = [{"text": f"{topic} is the central subject of this section and {topic} appears often. " * 3,
             "section": "1 Intro", "page": 1},
            {"text": f"Unrelated filler words about nothing in particular for {pid}. " * 4,
             "section": "2 Other", "page": 2}]
    return build_document(pid, f"Paper on {topic}", recs)


def test_evaluate_and_outputs(tmp_path):
    docs = [paper("p1", "zebras"), paper("p2", "volcanoes"), paper("p3", "glaciers")]
    gold = []
    for d, topic in zip(docs, ("zebras", "volcanoes", "glaciers")):
        q = d.paragraphs[0]
        gold.append(ev.GoldItem(f"{d.paper_id}:{q.char_start}-{q.char_end}", d.paper_id, f"{topic} central subject",
                                q.char_start, q.char_end, "paragraph", "ok", 0.0 if topic != "zebras" else 0.5))
    emb = Embedder("fake-model", cache_dir=tmp_path / "cache", backend=FakeBackend())
    res = ev.evaluate(docs, gold, emb, ["fixed", "structure"], max_tokens=60, overlap=8, n_boot=50, seed=0)
    for s in ("fixed", "structure"):
        m = res["strategies"][s]["metrics"]
        assert m["recall@20"]["mean"] == 1.0 and m["paper_recall@1"]["mean"] == 1.0
        assert m["recall@1"]["lo"] <= m["recall@1"]["mean"] <= m["recall@1"]["hi"]
        assert res["strategies"][s]["chunks"]["count"] > 0
    assert set(res["paired_vs_fixed"]) == {"structure"}
    assert set(res["paired_vs_fixed"]["structure"]) == set(ev.PAIRED_METRICS)
    assert res["by_span_type"]["paragraph"]["n"] == 3
    assert {k: v["n"] for k, v in res["by_title_overlap"].items()} == {"title_overlap == 0": 2, "title_overlap > 0": 1}
    again = ev.evaluate(docs, gold, emb, ["fixed", "structure"], max_tokens=60, overlap=8, n_boot=50, seed=0)
    assert again == res

    report = {"config": {"gold": "g.jsonl", "max_tokens": 60, "overlap": 8, "bootstrap": 50, "seed": 0},
              "draft": True, "gold": {"rows": 3}, "n_questions": 3, "n_papers": 3, "n_papers_without_sections": 1,
              "embedding": {"model": "fake-model", "backend": "fake"}, "git_commit": None, "results": res}
    jp, mp = ev.write_outputs(report, tmp_path / "out" / "phase2")
    assert json.loads(jp.read_text(encoding="utf-8"))["results"]["strategies"]["fixed"]["chunks"]["count"] > 0
    md = mp.read_bytes()
    assert b"\r" not in md and b"\r" not in jp.read_bytes()
    text = md.decode("utf-8")
    assert "DRAFT GOLD" in text and "recall@10" in text and "Paired difference" in text and "quantized" in text


def test_is_dirty_ignores_gold_draft_and_results():
    assert not ev.is_dirty("")
    assert not ev.is_dirty("?? eval/gold_draft.jsonl\n?? eval/gold_draft.jsonl.rejected\n"
                           " M docs/results/phase2.md\n?? docs/results/phase2.json\n")
    assert ev.is_dirty(" M src/arxiv_rag/chunk.py\n")
    assert ev.is_dirty("?? eval/gold_draft.jsonl\n?? src/new.py\n")


def test_report_provenance_and_sensitivity(tmp_path):
    docs = [paper("p1", "zebras"), paper("p2", "volcanoes")]
    gold = []
    for k, d in enumerate(docs):
        q = d.paragraphs[0]
        gold.append(ev.GoldItem(f"{d.paper_id}:{q.char_start}-{q.char_end}", d.paper_id, "central subject",
                                q.char_start, q.char_end, "paragraph", "auto_ok", 0.0, copy=bool(k)))
    emb = Embedder("fake-model", cache_dir=tmp_path / "cache", backend=FakeBackend())
    res = ev.evaluate(docs, gold, emb, ["fixed", "structure"], max_tokens=60, overlap=8, n_boot=20, seed=0)
    sens = res["sensitivity"]
    assert [v["n"] for v in sens.values()] == [2, 1]
    assert set(sens["all"]["strategies"]["fixed"]) == set(ev.SENSITIVITY_METRICS)
    report = {"config": {"gold": "g.jsonl", "max_tokens": 60, "overlap": 8, "bootstrap": 20, "seed": 0},
              "draft": False, "gold": {"rows": 2}, "gold_status_counts": {"auto_ok": 2}, "n_gold_papers": 2,
              "n_questions": 2, "n_papers": 2, "n_papers_without_sections": 0,
              "embedding": {"model": "fake-model", "backend": "fake"}, "git_commit": "abc123", "git_dirty": True,
              "results": res}
    text = ev.render_markdown(report)
    assert "Gold set: machine-verified (LLM judge + rule filters), not human-reviewed" in text
    assert "auto_ok 2" in text and "abc123 (uncommitted changes)" in text
    assert "Sensitivity" in text and "non-copy" in text and "multiple comparisons" in text
    assert "two chances" in text and "[CLS]/[SEP]" in text and "K_MAX" in text and "cluster bootstrap" in text
    clean = ev.render_markdown(dict(report, git_dirty=False, gold_status_counts={"ok": 2}))
    assert "uncommitted" not in clean and "machine-verified (LLM judge" not in clean


def test_max_tokens_over_model_limit_raises(tmp_path):
    emb = Embedder("fake-model", cache_dir=tmp_path, backend=FakeBackend())
    with pytest.raises(ValueError, match="exceeds"):
        ev.evaluate([paper("p1", "zebras")], [item()], emb, ["fixed"], max_tokens=1000, overlap=8, n_boot=5, seed=0)


def test_is_dirty_checks_both_sides_of_a_rename():
    from arxiv_rag.eval_chunking import is_dirty
    assert is_dirty("R  eval/gold_draft.jsonl -> eval/gold_v1.jsonl\n")
    assert not is_dirty("R  eval/gold_draft.jsonl -> eval/gold_draft_old.jsonl\n")


def test_load_gold_keeps_quote_and_marks_missing_overlap(tmp_path):
    docs = make_docs()
    d1 = docs["p1"]
    a = row(d1, "Alpha uses ZetaNet for retrieval.")
    b = row(d1, "It works well on long text.")
    del b["overlap"]
    g = ev.load_gold(write_gold(tmp_path / "g.jsonl", [a, b]), docs)
    assert [i.overlap for i in g.items] == [0.1, None]
    assert [i.quote for i in g.items] == ["Alpha uses ZetaNet for retrieval.", "It works well on long text."]
