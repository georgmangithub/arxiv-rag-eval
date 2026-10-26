import io
import json

import pytest

from arxiv_rag import review
from arxiv_rag.document import PARAGRAPH_SEP


def item(n, status="draft", span_type="paragraph", **extra):
    return {"qid": f"p{n}:0-10", "paper_id": f"p{n}", "question": f"Question {n}?",
            "evidence_quote": f"Evidence {n}.", "char_start": 0, "char_end": 10,
            "doc_sha256": "a" * 64, "span_type": span_type, "status": status, **extra}


def write(path, rows, raw=()):
    lines = [json.dumps(r, ensure_ascii=False) for r in rows] + list(raw)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def read(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run(monkeypatch, path, keys, **kw):
    monkeypatch.setattr("sys.stdin", io.StringIO(keys))
    df = review.DraftFile(path)
    return df, review.review(df, {"p1": "Title One"}, **kw)


def test_decisions_persist_after_each_step(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2), item(3)])
    run(monkeypatch, path, "o\nb\nq\n")
    assert [r["status"] for r in read(path)] == ["ok", "bad", "draft"]


def test_edit_keeps_offsets_and_stores_original(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])
    run(monkeypatch, path, "e\n  A  better\nquestion?\nq\n")
    r = read(path)[0]
    assert r["status"] == "edit" and r["question"] == "A better"  # one line per prompt
    assert r["question_original"] == "Question 1?"
    assert (r["char_start"], r["char_end"], r["evidence_quote"]) == (0, 10, "Evidence 1.")


def test_empty_edit_cancels_and_note_saved(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])
    run(monkeypatch, path, "e\n\nn\nlooks odd\nq\n")
    r = read(path)[0]
    assert r["status"] == "draft" and r["review_note"] == "looks odd" and "question_original" not in r


def test_undo_restores_row_and_revisits_item(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2)])
    run(monkeypatch, path, "o\nu\nb\nq\n")
    assert [r["status"] for r in read(path)] == ["bad", "draft"]
    write(path, [item(1), item(2)])
    run(monkeypatch, path, "e\nnew q\nu\nq\n")
    r = read(path)[0]
    assert r["status"] == "draft" and r["question"] == "Question 1?" and "question_original" not in r


def test_skip_stays_draft_and_resume_skips_reviewed(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2), item(3)])
    run(monkeypatch, path, "s\no\nq\n")
    assert [r["status"] for r in read(path)] == ["draft", "ok", "draft"]
    df, _ = run(monkeypatch, path, "o\no\n")
    assert [r["status"] for r in read(path)] == ["ok", "ok", "ok"]
    assert review.select_items(df, False, False) == []
    assert review.select_items(df, True, False) == [0, 1, 2]


def test_flagged_filter(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2, auto_flag="no_anchor")])
    run(monkeypatch, path, "b\n", flagged=True)
    assert [r["status"] for r in read(path)] == ["draft", "bad"]


def test_unparseable_lines_and_bytes_preserved(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    junk = b'{"half written'
    path.write_bytes(json.dumps(item(1)).encode() + b"\r\n" + junk + b"\n\xff\xfe bad\n"
                     + json.dumps(item(2)).encode() + b"\n")
    run(monkeypatch, path, "o\no\n")
    lines = path.read_bytes().split(b"\n")
    assert lines[0].endswith(b"\r") and json.loads(lines[0])["status"] == "ok"
    assert lines[1] == junk and lines[2] == b"\xff\xfe bad"
    assert json.loads(lines[3])["status"] == "ok"


def test_unreviewed_rows_untouched_byte_for_byte(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    raw = b'{ "qid":"x",  "status": "ok"  }\n'
    path.write_bytes(raw + json.dumps(item(2)).encode() + b"\n")
    run(monkeypatch, path, "o\n")
    assert path.read_bytes().startswith(raw)


def test_lock_blocks_second_session_and_is_removed(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])
    lock = tmp_path / "d.jsonl.lock"
    with review.Lock(path):
        assert lock.exists()
        with pytest.raises(SystemExit):
            with review.Lock(path):
                pass
        assert lock.exists()  # the failed attempt must not remove the holder's lock
    assert not lock.exists()


def test_main_removes_lock_on_keyboard_interrupt(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(review, "review", boom)
    monkeypatch.setattr("sys.argv", ["review", "--draft", str(path), "--data-dir", str(tmp_path)])
    review.main()
    assert not (tmp_path / "d.jsonl.lock").exists()


def test_main_refuses_when_lock_exists(tmp_path, monkeypatch):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])
    (tmp_path / "d.jsonl.lock").write_text("")
    monkeypatch.setattr("sys.argv", ["review", "--draft", str(path)])
    with pytest.raises(SystemExit):
        review.main()
    assert (tmp_path / "d.jsonl.lock").exists()


def test_cross_paragraph_boundary_shown(tmp_path, capsys):
    row = item(1, span_type="cross_paragraph", auto_flag="one_half_only",
               evidence_quote=f"First one.{PARAGRAPH_SEP}Second one.")
    review.show_item(row, 1, 1, review.Counter(), "Some Title")
    out = capsys.readouterr().out
    assert "paragraph boundary" in out and "Some Title" in out and "BOTH sentences" in out


def test_export_filters_dedupes_lf(tmp_path, capsys):
    draft = tmp_path / "d.jsonl"
    out = tmp_path / "gold.jsonl"
    a = item(1, status="ok")
    a2 = dict(a, question="newer")
    write(draft, [a, item(2, status="draft"), item(3, status="bad"),
                  item(4, status="edit", span_type="sentence"), a2], raw=["garbage"])
    review.export(draft, out)
    data = out.read_bytes()
    assert b"\r" not in data
    rows = [json.loads(ln) for ln in data.decode("utf-8").splitlines()]
    assert [r["qid"] for r in rows] == ["p1:0-10", "p4:0-10"]
    assert rows[0]["question"] == "newer"
    text = capsys.readouterr().out
    assert "WARNING" in text and "sentence" in text


def test_export_no_warning_at_150(tmp_path, capsys):
    draft = tmp_path / "d.jsonl"
    write(draft, [item(n, status="ok") for n in range(150)])
    review.export(draft, tmp_path / "g.jsonl")
    assert "WARNING" not in capsys.readouterr().out


def test_stats(tmp_path, capsys):
    draft = tmp_path / "d.jsonl"
    write(draft, [item(1), item(2, status="ok", span_type="sentence", auto_flag="too_long"),
                  item(3, auto_flag="too_long")])
    review.stats(draft)
    out = capsys.readouterr().out
    assert "too_long" in out and "2" in out
    ok_line = next(ln for ln in out.splitlines() if ln.startswith("ok"))
    assert ok_line.split()[-1] == "1"
