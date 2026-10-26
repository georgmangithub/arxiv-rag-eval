import json

import pytest

from arxiv_rag import judge, review
from arxiv_rag.document import PARAGRAPH_SEP
from arxiv_rag.goldgen import MalformedReply, QuotaExhausted

GOOD = {"answerable": True, "specific": True, "copy": False, "reason": "ok"}


def item(n, status="draft", span_type="paragraph", **extra):
    quote = f"First {n}.{PARAGRAPH_SEP}Second {n}." if span_type == "cross_paragraph" else f"Evidence {n}."
    return {"qid": f"p{n}:0-10", "paper_id": f"p{n}", "question": f"Question {n}?",
            "evidence_quote": quote, "char_start": 0, "char_end": 10,
            "doc_sha256": "a" * 64, "span_type": span_type, "status": status, **extra}


def write(path, rows, raw=()):
    lines = [json.dumps(r, ensure_ascii=False) for r in rows] + list(raw)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def read(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run(path, replies, **kw):
    """Judge `path` with canned replies (a list, or a callable of (system, user)); returns (result, calls)."""
    calls = []

    def ask(system, user):
        calls.append((system, user))
        r = replies(system, user) if callable(replies) else replies[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)

    res = judge.judge_items(review.DraftFile(path), ask, "judge-m", **kw)
    return res, calls


def test_copy_is_recorded_not_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])
    res, _ = run(path, [{**GOOD, "copy": True}])
    row = read(path)[0]
    assert row["status"] == "auto_ok" and row["judge"]["copy"] is True and not res["auto_bad_by_reason"]


@pytest.mark.parametrize("verdict,status,reason", [
    (GOOD, "auto_ok", None),
    ({**GOOD, "answerable": False}, "auto_bad", "not_answerable"),
    ({**GOOD, "specific": False}, "auto_bad", "not_specific"),
])
def test_verdict_to_status(tmp_path, verdict, status, reason):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])
    res, _ = run(path, [verdict])
    row = read(path)[0]
    assert row["status"] == status and res[status] == 1
    assert row["judge"] == {"model": "judge-m", **verdict}
    assert (reason is None) or res["auto_bad_by_reason"][reason] == 1


@pytest.mark.parametrize("flag,status", [
    ("no_anchor", "auto_bad"), ("no_specific_term", "auto_bad"), ("too_long", "auto_bad"),
    ("one_half_only", "auto_ok"),
])
def test_auto_flag_override(tmp_path, flag, status):
    path = tmp_path / "d.jsonl"
    write(path, [item(1, auto_flag=flag)])
    run(path, [GOOD])
    assert read(path)[0]["status"] == status


def test_cross_needs_both_recorded_only(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1, span_type="cross_paragraph"), item(2)])
    _, calls = run(path, [{**GOOD, "needs_both": False}, GOOD])
    rows = read(path)
    assert rows[0]["status"] == "auto_ok" and rows[0]["judge"]["needs_both"] is False
    assert "needs_both" not in rows[1]["judge"]
    assert "needs_both" in calls[0][0] and "needs_both" not in calls[1][0]
    assert "[1] First 1." in calls[0][1] and "[2] Second 1." in calls[0][1]


def test_cross_without_needs_both_is_malformed(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1, span_type="cross_paragraph")])
    res, _ = run(path, [GOOD])
    assert res["malformed"] == 1 and read(path)[0]["status"] == "draft"


def test_prompt_is_blind(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1, overlap=0.77, title_overlap=0.123, auto_flag="one_half_only")])
    _, calls = run(path, [GOOD])
    user = calls[0][1]
    assert "Question 1?" in user and "Evidence 1." in user
    assert "p1" not in user and "0.77" not in user and "0.123" not in user and "one_half_only" not in user


def test_resumable(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2), item(3, status="ok"), item(4, status="bad")])
    res, calls = run(path, [GOOD, {**GOOD, "specific": False}])
    assert len(calls) == 2 and res["judged"] == 2
    again, calls2 = run(path, [])
    assert calls2 == [] and again["judged"] == 0
    assert [r["status"] for r in read(path)] == ["auto_ok", "auto_bad", "ok", "bad"]


def test_limit(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2), item(3)])
    _, calls = run(path, [GOOD, GOOD, GOOD], limit=2)
    assert len(calls) == 2 and [r["status"] for r in read(path)] == ["auto_ok", "auto_ok", "draft"]


def test_preserves_rows_fields_and_unparseable_lines(tmp_path):
    path = tmp_path / "d.jsonl"
    a, b = item(1, overlap=0.5, custom="x"), item(2)
    write(path, [a, b], raw=["garbage {", '{ "qid":"x",  "status": "ok"  }'])
    before = path.read_bytes().split(b"\n")
    run(path, [GOOD, GOOD])
    after = path.read_bytes().split(b"\n")
    assert after[2:] == before[2:]  # unparseable / unjudged raw lines byte-for-byte
    rows = [json.loads(ln) for ln in after[:2]]
    assert [r["qid"] for r in rows] == [a["qid"], b["qid"]]
    for new, old in zip(rows, (a, b)):
        assert {k: v for k, v in new.items() if k not in ("status", "judge")} == \
               {k: v for k, v in old.items() if k != "status"}
    assert not (tmp_path / "d.jsonl.tmp").exists()


def test_saved_after_every_item(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2)])
    seen = []

    def replies(system, user):
        seen.append([r["status"] for r in read(path)])
        return GOOD

    run(path, replies)
    assert seen == [["draft", "draft"], ["auto_ok", "draft"]]


def test_malformed_counted_and_retried_next_run(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2), item(3), item(4)])
    res, _ = run(path, ["not json", {"answerable": "yes", "specific": True, "copy": False}, "", GOOD])
    assert res["malformed"] == 3 and res["judged"] == 1
    assert [r["status"] for r in read(path)] == ["draft", "draft", "draft", "auto_ok"]
    res2, calls = run(path, [GOOD, GOOD, GOOD])
    assert res2["judged"] == 3 and len(calls) == 3


def test_malformed_reply_exception_from_client_counts(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1)])
    res, _ = run(path, [MalformedReply("unexpected 200 response shape")])
    assert res["malformed"] == 1


def test_parse_strips_think_and_fences():
    v = json.dumps(GOOD)
    assert judge.parse_verdict('<think>{"answerable": false} hmm</think>\n' + v)["answerable"] is True
    assert judge.parse_verdict("reasoning...\n</think>\n```json\n" + v + "\n```")["specific"] is True
    with pytest.raises(MalformedReply):
        judge.parse_verdict("<think>never closed " + v)
    with pytest.raises(MalformedReply):
        judge.parse_verdict("<think>only thoughts</think>")


def test_reason_is_trimmed():
    v = judge.parse_verdict(json.dumps({**GOOD, "reason": "x " * 500}))
    assert len(v["reason"]) <= judge.MAX_REASON_CHARS


def test_quota_keeps_progress_and_counts(tmp_path):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2)])
    res = judge.new_result()
    replies = [GOOD, QuotaExhausted("quota")]

    def ask(system, user):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return json.dumps(r)

    with pytest.raises(QuotaExhausted):
        judge.judge_items(review.DraftFile(path), ask, "m", res=res)
    assert res["judged"] == 1 and [r["status"] for r in read(path)] == ["auto_ok", "draft"]


def test_main_lock_quota_and_interrupt(tmp_path, monkeypatch, capsys):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2)])
    monkeypatch.setattr("sys.argv", ["judge", "--draft", str(path)])
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_API_KEY", "k")
    lock = tmp_path / "d.jsonl.lock"
    lock.write_bytes(b"")
    with pytest.raises(SystemExit):
        judge.main()
    assert lock.exists() and read(path)[0]["status"] == "draft"  # foreign lock kept, file untouched
    lock.unlink()

    calls = []

    def chat(self, system, user, json_mode=True, temperature=0.3):
        calls.append(json_mode)
        if len(calls) == 2:
            raise QuotaExhausted("LLM quota exhausted; rerun later")
        return json.dumps(GOOD)

    monkeypatch.setattr(judge.LLMClient, "chat", chat)
    judge.main()  # quota: returns cleanly, progress kept, lock removed
    assert not lock.exists()
    assert [r["status"] for r in read(path)] == ["auto_ok", "draft"]
    out = capsys.readouterr().out
    assert "quota exhausted" in out and "judged=1" in out

    def boom(self, system, user, json_mode=True, temperature=0.3):
        raise KeyboardInterrupt

    monkeypatch.setattr(judge.LLMClient, "chat", boom)
    judge.main()
    assert not lock.exists()


def test_dry_run_makes_no_calls_or_writes(tmp_path, monkeypatch, capsys):
    path = tmp_path / "d.jsonl"
    write(path, [item(1), item(2, span_type="cross_paragraph")])
    before = path.read_bytes()
    monkeypatch.setattr("sys.argv", ["judge", "--draft", str(path), "--dry-run"])
    judge.main()
    assert path.read_bytes() == before and "2 items to judge" in capsys.readouterr().out
    assert not (tmp_path / "d.jsonl.lock").exists()


def test_export_and_stats_include_auto_ok(tmp_path, capsys):
    path = tmp_path / "d.jsonl"
    write(path, [item(1, status="auto_ok"), item(2, status="auto_bad"), item(3, status="ok"), item(4)])
    out = tmp_path / "gold.jsonl"
    review.export(path, out)
    assert [r["status"] for r in read(out)] == ["auto_ok", "ok"]
    review.stats(path)
    lines = capsys.readouterr().out.splitlines()
    assert any(ln.startswith("auto_ok") for ln in lines) and any(ln.startswith("auto_bad") for ln in lines)


# --- --restatus ----------------------------------------------------------------------------

def jrow(n, status, flag=None, **verdict):
    extra = {"auto_flag": flag} if flag else {}
    return item(n, status=status, judge={"model": "m", **GOOD, **verdict}, **extra)


def test_restatus_recomputes_and_spares_others(tmp_path, capsys):
    path = tmp_path / "d.jsonl"
    rows = [jrow(1, "auto_bad", copy=True),                      # old copy rule -> now auto_ok
            jrow(2, "auto_ok", flag="too_long"),                  # flag veto -> auto_bad
            jrow(3, "auto_ok", flag="one_half_only"),             # unchanged
            jrow(4, "auto_ok", specific=False),                   # -> auto_bad
            jrow(5, "ok", specific=False),                        # human: untouched
            item(6), item(7, status="auto_bad"),                  # no verdict: untouched
            jrow(8, "edit"), jrow(9, "bad")]
    write(path, rows, raw=["garbage {"])
    before = path.read_bytes().split(b"\n")
    res = judge.restatus(review.DraftFile(path))
    got = [json.loads(ln)["status"] for ln in path.read_text(encoding="utf-8").splitlines()[:9]]
    assert got == ["auto_ok", "auto_bad", "auto_ok", "auto_bad", "ok", "draft", "auto_bad", "edit", "bad"]
    assert res["changed"] == 3 and res["checked"] == 4 and dict(res["by_status"]) == {"auto_ok": 2, "auto_bad": 2}
    after = path.read_bytes().split(b"\n")
    assert after[9:] == before[9:] and after[4] == before[4]
    judge.print_restatus(res)
    assert "changed=3" in capsys.readouterr().out
    assert judge.restatus(review.DraftFile(path))["changed"] == 0  # idempotent


def test_restatus_main_no_llm_and_lock(tmp_path, monkeypatch, capsys):
    path = tmp_path / "d.jsonl"
    write(path, [jrow(1, "auto_bad", copy=True)])
    monkeypatch.setattr("sys.argv", ["judge", "--draft", str(path), "--restatus"])
    monkeypatch.setattr(judge.LLMClient, "from_env", staticmethod(lambda *a, **k: pytest.fail("LLM used")))
    lock = tmp_path / "d.jsonl.lock"
    lock.write_bytes(b"")
    with pytest.raises(SystemExit):
        judge.main()
    assert read(path)[0]["status"] == "auto_bad"
    lock.unlink()
    judge.main()
    assert read(path)[0]["status"] == "auto_ok" and not lock.exists()
    assert "changed=1" in capsys.readouterr().out
