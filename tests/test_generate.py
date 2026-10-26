import json

import pytest
import requests

from arxiv_rag import generate as gen
from arxiv_rag.chunk import Chunk
from arxiv_rag.eval_chunking import load_gold, sha256_text
from arxiv_rag.goldgen import MalformedReply, QuotaExhausted
from test_eval_chunking import make_docs, row, write_gold


def chunk(i, paper="p1", start=0, end=10, text=None):
    return Chunk(f"{paper}:structure:{i}", paper, "structure", text or f"passage text {i}", start, end, None, 1, 1)


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, system, user):
        self.calls.append((system, user))
        r = self.replies(system, user) if callable(self.replies) else self.replies[len(self.calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r, {"prompt_tokens": 100, "completion_tokens": 20}


@pytest.fixture
def env(tmp_path):
    docs = make_docs()
    d1 = docs["p1"]
    path = write_gold(tmp_path / "g.jsonl", [
        row(d1, "Alpha uses ZetaNet for retrieval.", question="What does Alpha use?"),
        row(d1, "Beta results improve by 12.5% over the baseline model.", question="How much does Beta improve?"),
    ])
    items = load_gold(path, docs).items
    shas = {p: sha256_text(d.text) for p, d in docs.items()}
    searchers = {"dense": lambda q, k: [chunk(i, text=f"TITLE-FREE {i}") for i in range(1, 9)][:k],
                 "bm25": lambda q, k: [chunk(1, start=0, end=20)][:k]}
    return items, shas, searchers, tmp_path / "answers.jsonl"


CONFIGS = [gen.parse_config(c) for c in gen.DEFAULT_CONFIGS]


def read(p):
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_parse_config():
    assert gen.parse_config("closed-book") == ("closed-book", "closed-book", 0)
    assert gen.parse_config("dense-k5") == ("dense-k5", "dense", 5)
    assert gen.parse_config("bm25", 3) == ("bm25-k3", "bm25", 3)
    for bad in ("dense-k0", "hybrid", "dense-k", "bm25-3"):
        with pytest.raises(ValueError):
            gen.parse_config(bad)


def test_prompt_has_numbered_passages_and_no_titles():
    u = gen.build_user_prompt("Why?", ["first text", "second text"])
    assert u == "Passages:\n[1] first text\n\n[2] second text\n\nQuestion: Why?"
    assert gen.build_user_prompt("Why?", []) == "Question: Why?"
    assert "INSUFFICIENT" in gen.RAG_SYSTEM and "INSUFFICIENT" in gen.CLOSED_SYSTEM


@pytest.mark.parametrize("text,n,want", [
    ("A [1] and B [3].", 5, ([1, 3], 0)),
    ("A [1, 2] B [2][4].", 5, ([1, 2, 4], 0)),
    ("A [2-4].", 5, ([2, 3, 4], 0)),
    ("A [2–3] and [1, 4-5]", 5, ([1, 2, 3, 4, 5], 0)),
    ("A [0] [6] [9, 2].", 5, ([], 3)),  # a group with any bad number is rejected whole
    ("A [0, 1] and [1, 5].", 5, ([1, 5], 1)),
    ("A [1, 6] [2].", 5, ([2], 1)),
    ("A [4-7].", 5, ([], 1)),
    ("A [Passage 2] and [passage 3], [Passages 1, 4].", 5, ([1, 2, 3, 4], 0)),
    ("A [Passage 9].", 5, ([], 1)),
    ("A【2】【1】 and 【3，4】 and 【9】 and [2‑3].", 5, ([1, 2, 3, 4], 1)),  # full-width brackets, U+2011 dash
    ("A [5-3].", 5, ([], 1)),
    ("no cites, array[i] and [see 1].", 5, ([], 0)),
    ("A [1].", 0, ([], 0)),  # closed-book: nothing to cite
])
def test_parse_citations(text, n, want):
    assert gen.parse_citations(text, n) == want


@pytest.mark.parametrize("text,want", [
    ("INSUFFICIENT", True), ("insufficient.", True), ('"INSUFFICIENT"', True), ("**INSUFFICIENT**", True),
    ("  Insufficient!\n", True), ("_INSUFFICIENT_", True),
    ("INSUFFICIENT: the passages do not say", False), ("INSUFFICIENT [1]", False),
    ("It is insufficient evidence [1].", False), ("ZetaNet [1].", False), ("", False),
])
def test_is_abstention(text, want):
    assert gen.is_abstention(text) is want


@pytest.mark.parametrize("text,want", [
    ("The passages do not mention the learning rate.", True),
    ("The provided passage does not provide this.", True),
    ("This is not mentioned in the passages.", True),
    ("Not specified in the given passages.", True),
    ("That cannot be determined from the text.", True),
    ("There is no information about the dataset size.", True),
    ("I cannot answer that.", True), ("I don't know.", True), ("I can't answer this question.", True),
    ("The passages do not mention X [1].", False),  # cites: treated as an attempt
    ("INSUFFICIENT", False),  # explicit abstention is a separate signal
    ("Alpha uses ZetaNet.", False), ("The model does not use dropout.", False),
    ("Information about the dataset is in Section 3.", False),
])
def test_is_prose_refusal(text, want):
    assert gen.is_prose_refusal(text) is want


def test_clean_answer_strips_think_and_rejects_empty():
    assert gen.clean_answer("<think>hmm</think>\n  Answer   here [1]. ") == "Answer here [1]."
    with pytest.raises(MalformedReply):
        gen.clean_answer("<think>only thoughts</think>")


def test_generate_rows_interleaved_per_question(env):
    items, shas, searchers, out = env
    llm = FakeLLM(lambda s, u: "INSUFFICIENT" if s == gen.CLOSED_SYSTEM else "It uses ZetaNet [1][2][7].")
    counts = gen.generate_answers(items, shas, CONFIGS, searchers, llm, "m", out)
    assert counts["written"] == 6 and counts["abstained"] == 2
    rows = read(out)
    assert [(r["qid"], r["config"]) for r in rows[:3]] == [(items[0].qid, c[0]) for c in CONFIGS]
    assert rows[3]["qid"] == items[1].qid
    cb, dense, bm25 = rows[:3]
    assert cb["abstained"] and cb["retrieved"] == [] and cb["citations"] == [] and cb["model"] == "m"
    assert cb["prompt_tokens"] == 100 and cb["completion_tokens"] == 20
    assert dense["citations"] == [1, 2] and dense["invalid_citations"] == 1 and not dense["abstained"]
    assert [r["rank"] for r in dense["retrieved"]] == [1, 2, 3, 4, 5]
    assert dense["retrieved"][0] == {"rank": 1, "chunk_id": "p1:structure:1", "paper_id": "p1",
                                     "char_start": 0, "char_end": 10}
    assert bm25["citations"] == [1] and bm25["invalid_citations"] == 2  # only 1 passage came back
    assert dense["doc_sha256"] == shas["p1"] and dense["question"] == "What does Alpha use?"
    # prompts: no titles or ids, passages numbered, closed-book has none
    sys_cb, user_cb = llm.calls[0]
    assert sys_cb == gen.CLOSED_SYSTEM and user_cb == "Question: What does Alpha use?"
    sys_d, user_d = llm.calls[1]
    assert sys_d == gen.RAG_SYSTEM and user_d.startswith("Passages:\n[1] TITLE-FREE 1\n\n[2] TITLE-FREE 2")
    assert "Alpha Paper" not in user_d and "p1" not in user_d and user_d.count("[") == 5


def test_resume_skips_done_and_survives_truncated_last_line(env, capsys):
    items, shas, searchers, out = env
    llm = FakeLLM(lambda s, u: "ZetaNet [1].")
    gen.generate_answers(items[:1], shas, CONFIGS, searchers, llm, "m", out)
    with out.open("a", encoding="utf-8") as f:
        f.write('{"qid": "p1:0-5", "doc_sha')  # crash mid-write
    llm2 = FakeLLM(lambda s, u: "ZetaNet [1].")
    counts = gen.generate_answers(items, shas, CONFIGS, searchers, llm2, "m", out)
    assert counts["skipped_existing"] == 3 and counts["written"] == 3 and len(llm2.calls) == 3
    assert "not a valid answer row" in capsys.readouterr().out
    assert len(gen.load_answers(out)) == 6  # the new rows start on a fresh line
    assert gen.generate_answers(items, shas, CONFIGS, searchers, llm2, "m", out)["written"] == 0
    assert len(llm2.calls) == 3


def test_changed_doc_sha_is_not_done(env):
    items, shas, searchers, out = env
    gen.generate_answers(items[:1], shas, CONFIGS[:1], searchers, FakeLLM(["a"]), "m", out)
    new = dict(shas, p1="f" * 64)
    llm = FakeLLM(["b"])
    assert gen.generate_answers(items[:1], new, CONFIGS[:1], searchers, llm, "m", out)["written"] == 1


def test_quota_stops_cleanly_and_keeps_progress(env):
    items, shas, searchers, out = env
    llm = FakeLLM(["one [1].", "two [1].", QuotaExhausted("daily")])
    with pytest.raises(QuotaExhausted):
        gen.generate_answers(items, shas, CONFIGS, searchers, llm, "m", out)
    assert len(read(out)) == 2
    llm2 = FakeLLM(lambda s, u: "x [1].")
    assert gen.generate_answers(items, shas, CONFIGS, searchers, llm2, "m", out)["written"] == 4


def test_malformed_reply_not_written_and_retried(env):
    items, shas, searchers, out = env
    llm = FakeLLM(["fine [1].", "<think>x</think>"])
    counts = gen.generate_answers(items[:1], shas, CONFIGS[:2], searchers, llm, "m", out)
    assert counts["malformed"] == 1 and counts["written"] == 1
    assert [r["config"] for r in read(out)] == ["closed-book"]
    assert len(gen.pending(items[:1], shas, CONFIGS[:2], gen.done_keys(out))) == 1


def test_main_rejects_unknown_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate", "--configs", "hybrid"])
    with pytest.raises(SystemExit, match="unknown config"):
        gen.main()


class Resp:
    status_code = 200
    headers = {}
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_usage_client_reads_last_usage_without_patching_requests(monkeypatch):
    sent = {}
    payload = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    monkeypatch.setattr(requests, "post", lambda url, **kw: sent.update(body=kw["json"]) or Resp(payload))
    stub = requests.post
    client = gen.UsageClient("k", "https://example.com/v1", "m", rpm=0)
    assert client.last_usage is None
    text, usage = client.ask("sys", "usr")
    assert text == "hi" and usage == {"prompt_tokens": 7, "completion_tokens": 3}
    assert client.last_usage == usage
    assert sent["body"]["temperature"] == 0.0 and "response_format" not in sent["body"]
    assert requests.post is stub  # never touched
    payload.pop("usage")
    assert client.ask("sys", "usr") == ("hi", {}) and client.last_usage is None


def test_failed_chat_resets_last_usage(monkeypatch):
    ok = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    monkeypatch.setattr(requests, "post", lambda url, **kw: Resp(ok))
    client = gen.UsageClient("k", "https://example.com/v1", "m", rpm=0)
    client.ask("s", "u")
    monkeypatch.setattr(requests, "post", lambda url, **kw: Resp({"unexpected": 1}))
    with pytest.raises(MalformedReply):
        client.ask("s", "u")
    assert client.last_usage is None


def test_runtime_error_fails_one_item_and_is_retried_once(env):
    items, shas, searchers, out = env
    boom = RuntimeError("LLM request failed: 500")
    llm = FakeLLM(lambda s, u: "ZetaNet [1].")
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        if calls["n"] == 2:  # second call fails once, the in-run retry succeeds
            raise boom
        return llm(system, user)

    counts = gen.generate_answers(items[:1], shas, CONFIGS, searchers, flaky, "m", out)
    assert counts["written"] == 3 and counts["failed"] == 0 and calls["n"] == 4
    out.unlink()
    always = lambda system, user: (_ for _ in ()).throw(boom) if "Passages" in user else llm(system, user)  # noqa: E731
    counts = gen.generate_answers(items[:1], shas, CONFIGS, searchers, always, "m", out)
    assert counts["written"] == 1 and counts["failed"] == 2  # both RAG configs failed twice; run continued


def test_aborts_after_five_consecutive_failures(env):
    items, shas, searchers, out = env

    def dead(system, user):
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="5 consecutive failures"):
        gen.generate_answers(items * 3, shas, CONFIGS, searchers, dead, "m", out)
    assert not out.exists() or read(out) == []


def test_quota_is_not_swallowed_by_runtime_error_handling(env):
    items, shas, searchers, out = env
    with pytest.raises(QuotaExhausted):
        gen.generate_answers(items, shas, CONFIGS, searchers, FakeLLM([QuotaExhausted("daily")]), "m", out)


def test_main_holds_lock_on_the_output_file(env, monkeypatch, tmp_path):
    items, shas, _, out = env
    docs = make_docs()
    ids = tmp_path / "ids.txt"
    ids.write_text("p1\np2\n", encoding="utf-8")
    goldf = write_gold(tmp_path / "g2.jsonl", [row(docs["p1"], "Alpha uses ZetaNet for retrieval.", question="Q?")])

    class Fake:
        model = "m"

        @classmethod
        def from_env(cls, *a, **k):
            return cls()

        def ask(self, system, user, json_mode=False):
            assert (tmp_path / "answers.jsonl.lock").exists()  # held while generating
            return "INSUFFICIENT", {}

    monkeypatch.setattr(gen, "load_documents", lambda *a, **k: list(docs.values()))
    monkeypatch.setattr(gen, "ids_to_stems", lambda ids: ids)
    monkeypatch.setattr(gen, "UsageClient", Fake)
    monkeypatch.setattr(gen, "load_dotenv", lambda: None)
    argv = ["generate", "--configs", "closed-book", "--out", str(out), "--ids-file", str(ids), "--gold", str(goldf)]
    lock = tmp_path / "answers.jsonl.lock"
    lock.write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="exists"):
        gen.main()
    assert not out.exists()
    lock.unlink()
    gen.main()
    assert len(read(out)) == 1 and not lock.exists()
