import json
from argparse import Namespace

import pytest

from arxiv_rag import gate
from arxiv_rag.embed import Embedder
from fakes import FakeBackend
from test_eval_chunking import make_docs, row, write_gold

Q1 = "ZetaNet retrieval long text"
Q2 = "Beta results baseline model improve"
Q3 = "quantum gardening soil"


@pytest.fixture
def env(tmp_path):
    """Tiny data dir (2 parsed papers), ids file, gold file, and a fake-backed embedder."""
    docs = make_docs()
    data = tmp_path / "data"
    (data / "parsed").mkdir(parents=True)
    for pid, d in docs.items():
        recs = [{"text": p.text, "section": p.section, "page": p.page} for p in d.paragraphs]
        (data / "parsed" / f"{pid}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    ids = tmp_path / "ids.txt"
    ids.write_text("p1\np2\n", encoding="utf-8")
    d1, d2 = docs["p1"], docs["p2"]
    rows = [row(d1, "Alpha uses ZetaNet for retrieval. It works well on long text.", status="auto_ok", question=Q1),
            row(d1, "Beta results improve by 12.5% over the baseline model.", status="auto_ok", question=Q2),
            row(d2, "Gamma is a separate topic about quantum gardening and soil.", status="auto_ok", paper="p2",
                question=Q3)]
    gold = write_gold(tmp_path / "gold.jsonl", rows)
    emb = Embedder(backend=FakeBackend(), cache_dir=tmp_path / "emb")
    argv = ["--gold", str(gold), "--ids-file", str(ids), "--data-dir", str(data),
            "--baseline", str(tmp_path / "base.json"), "--max-tokens", "64", "--overlap", "8"]
    return Namespace(argv=argv, emb=emb, tmp=tmp_path, gold=gold, data=data, rows=rows)


def make_baseline(env, names=("bm25", "dense")):
    assert gate.main(env.argv + ["--update-baseline", "--retrievers", *names], env.emb) == 0
    return json.loads((env.tmp / "base.json").read_text())


def edit_baseline(env, fn):
    p = env.tmp / "base.json"
    b = json.loads(p.read_text())
    fn(b["chunkings"]["structure"])
    p.write_text(json.dumps(b))


def test_judge_rules():
    assert gate.judge(0.9, 0.9, 0.0) == "OK"
    assert gate.judge(0.9, 0.9 - 1e-7, 0.0) == "OK"  # rounding noise
    assert gate.judge(0.9, 0.85, 0.0) == "FAIL"  # deterministic: any drop fails
    assert gate.judge(0.9, 0.95, 0.0) == "IMPROVED"
    assert gate.judge(0.7, 0.695, 0.01) == "WITHIN TOLERANCE"
    assert gate.judge(0.7, 0.685, 0.01) == "FAIL"
    assert gate.judge(0.7, 0.705, 0.01) == "WITHIN TOLERANCE"  # a rise inside the tolerance is noise
    assert gate.judge(0.7, 0.72, 0.01) == "IMPROVED"


def test_round_trip_then_pass(env, capsys):
    b = make_baseline(env)
    s = b["chunkings"]["structure"]
    assert set(s["retrievers"]) == {"bm25", "dense"}
    assert s["retrievers"]["bm25"]["tolerance"] == 0.0 and s["retrievers"]["dense"]["tolerance"] == 0.01
    assert s["gold_questions"] == 3 and s["chunks"] > 0
    assert set(b["meta"]) == {"git_commit", "date", "gold", "corpus_sha256"}
    assert b["meta"]["corpus_sha256"] == gate.sha256_file(env.tmp / "ids.txt")
    summary = env.tmp / "summary.md"
    assert gate.main(env.argv + ["--retrievers", "bm25", "dense", "--summary", str(summary)], env.emb) == 0
    text = summary.read_text()
    assert "passed" in text and "| bm25 recall@5 |" in text and "| dense mrr@20 |" in text
    assert "FAIL" not in text and "stale" not in text
    assert "| metric | baseline | current | delta | status |" in capsys.readouterr().out


def test_update_baseline_merges_and_keeps_tolerance(env):
    make_baseline(env, ("bm25",))
    edit_baseline(env, lambda s: s["retrievers"]["bm25"].update(tolerance=0.05))
    b = make_baseline(env, ("dense",))
    r = b["chunkings"]["structure"]["retrievers"]
    assert set(r) == {"bm25", "dense"} and r["bm25"]["tolerance"] == 0.05
    b = make_baseline(env, ("bm25",))  # a re-measured retriever keeps its tolerance
    assert b["chunkings"]["structure"]["retrievers"]["bm25"]["tolerance"] == 0.05


def test_update_baseline_drops_entries_for_a_changed_corpus(env):
    make_baseline(env, ("bm25", "dense"))
    edit_baseline(env, lambda s: s.update(chunks=s["chunks"] + 1))
    b = make_baseline(env, ("bm25",))
    assert set(b["chunkings"]["structure"]["retrievers"]) == {"bm25"}


def test_drop_fails_with_exit_1(env):
    make_baseline(env, ("bm25",))
    edit_baseline(env, lambda s: s["retrievers"]["bm25"]["metrics"].update({"recall@1": 1.5}))
    summary = env.tmp / "s.md"
    assert gate.main(env.argv + ["--summary", str(summary)], env.emb) == 1
    text = summary.read_text()
    assert "FAILED" in text and "| bm25 recall@1 | 1.5000 |" in text


def test_rise_passes_with_stale_warning(env):
    make_baseline(env, ("bm25",))
    edit_baseline(env, lambda s: s["retrievers"]["bm25"]["metrics"].update({"recall@1": 0.0, "mrr@20": 0.0}))
    summary = env.tmp / "s.md"
    assert gate.main(env.argv + ["--summary", str(summary)], env.emb) == 0
    text = summary.read_text()
    assert "IMPROVED" in text and "baseline is stale; run with --update-baseline and commit" in text


def test_tolerance_absorbs_small_drop_for_dense(env):
    make_baseline(env, ("dense",))
    edit_baseline(env, lambda s: s["retrievers"]["dense"]["metrics"].update({"recall@5": 1.005}))
    assert gate.main(env.argv + ["--retrievers", "dense"], env.emb) == 0
    edit_baseline(env, lambda s: s["retrievers"]["dense"]["metrics"].update({"recall@5": 1.02}))
    assert gate.main(env.argv + ["--retrievers", "dense"], env.emb) == 1


@pytest.mark.parametrize("key", ["chunks", "gold_questions"])
def test_count_mismatch_fails(env, key):
    make_baseline(env, ("bm25",))
    edit_baseline(env, lambda s: s.update({key: s[key] + 1}))
    summary = env.tmp / "s.md"
    assert gate.main(env.argv + ["--summary", str(summary)], env.emb) == 1
    assert "FAILED" in summary.read_text()


def test_stale_gold_exits_2_with_instructions(env, capsys):
    make_baseline(env, ("bm25",))
    stale = dict(env.rows[0], doc_sha256="0" * 64)
    write_gold(env.gold, [stale] + env.rows[1:])
    assert gate.main(env.argv, env.emb) == 2
    err = capsys.readouterr().err
    assert "STALE" in err and "parse.py" in err and "regenerate the gold set" in err and "revert" in err
    assert gate.main(env.argv + ["--update-baseline"], env.emb) == 2  # never rewrites the baseline from stale gold


def test_all_gold_stale_exits_2(env):
    write_gold(env.gold, [dict(r, doc_sha256="0" * 64) for r in env.rows])
    assert gate.main(env.argv, env.emb) == 2


def test_missing_baseline_entry_and_setup_mismatch_exit_2(env, capsys):
    assert gate.main(env.argv, env.emb) == 2  # no baseline file
    make_baseline(env, ("bm25",))
    assert gate.main(env.argv + ["--retrievers", "dense"], env.emb) == 2  # no dense entry
    assert "--update-baseline" in capsys.readouterr().err
    args = [a if a != "64" else "32" for a in env.argv]
    assert gate.main(args, env.emb) == 2  # different --max-tokens


def test_missing_parsed_data_exits_2(env):
    (env.data / "parsed" / "p2.jsonl").unlink()
    assert gate.main(env.argv, env.emb) == 2


def test_summary_is_appended(env):
    make_baseline(env, ("bm25",))
    summary = env.tmp / "s.md"
    summary.write_text("earlier step\n")
    assert gate.main(env.argv + ["--summary", str(summary)], env.emb) == 0
    text = summary.read_text()
    assert text.startswith("earlier step\n") and "Retrieval regression gate" in text


def test_render_summary_columns():
    rows = [gate.Row("chunks", 10, 10, "OK", True), gate.Row("dense recall@5", 0.7, 0.695, "WITHIN TOLERANCE"),
            gate.Row("bm25 recall@5", 0.9, 0.8, "FAIL")]
    text = gate.render_summary(rows, {"bm25": 0.0, "dense": 0.01}, "structure", "FAILED")
    assert "| chunks | 10 | 10 | +0 | OK |" in text
    assert "| dense recall@5 | 0.7000 | 0.6950 | -0.0050 | WITHIN TOLERANCE (tol 0.01) |" in text
    assert "| bm25 recall@5 | 0.9000 | 0.8000 | -0.1000 | FAIL |" in text
