import hashlib
import json

import pytest

from arxiv_rag import goldgen
from arxiv_rag.document import PARAGRAPH_SEP, build_document
from arxiv_rag.goldgen import (
    MalformedReply,
    generate,
    has_banned_phrase,
    load_dotenv,
    looks_like_boilerplate,
    overlap_ratio,
    parse_question,
    sample_spans,
    title_overlap,
)
from arxiv_rag.chunk import sentence_spans

SENT = "The DPR-2 retrieval model encodes every passage into a dense vector using a shared encoder network."


def _para(i: int, section: str = "Method") -> dict:
    # Three sentences, ~45 words, distinct per index.
    return {
        "text": f"{SENT} Variant {i} of DPR-2 improves recall over the baseline on the benchmark by a clear margin overall. "
        f"We also report ablation results for DPR-2 configuration {i} across all evaluated datasets in detail.",
        "section": section,
        "page": 1,
    }


def _doc(pid: str, n_paras: int = 30):
    return build_document(pid, f"Title {pid}", [_para(i, "Method" if i % 10 < 5 else "Results") for i in range(n_paras)])


def test_boilerplate_filter():
    assert looks_like_boilerplate("Code is available at https://github.com/foo/bar for all experiments here.")
    assert looks_like_boilerplate("0.91 0.88 0.76 12.5 33.1 acc 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0")
    assert looks_like_boilerplate("= + - * / ( ) [ ] { } < > = + - * / ( ) a b")
    assert not looks_like_boilerplate(SENT)


def test_sampling_is_deterministic_and_seed_sensitive():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    one = sample_spans(docs, 40, seed=1)
    assert one == sample_spans(docs, 40, seed=1)
    assert [s.qid for s in one] != [s.qid for s in sample_spans(docs, 40, seed=2)]


def test_sampling_mix_quotes_and_even_papers():
    docs = [_doc("a"), _doc("b"), _doc("c"), _doc("d")]
    spans = sample_spans(docs, 40, seed=0)
    assert len(spans) == 40
    types = [s.span_type for s in spans]
    assert (types.count("paragraph"), types.count("cross_paragraph"), types.count("sentence")) == (20, 10, 10)
    by_id = {d.paper_id: d for d in docs}
    for s in spans:
        assert s.quote in by_id[s.paper_id].text
        assert by_id[s.paper_id].text[s.start : s.end] == s.quote
    per_paper = [sum(s.paper_id == d.paper_id for s in spans) for d in docs]
    assert max(per_paper) <= 10  # cap = ceil(40 / 4)


def test_cross_paragraph_span_stays_in_one_section_and_crosses_boundary():
    docs = [_doc("a")]
    cross = [s for s in sample_spans(docs, 60, seed=0) if s.span_type == "cross_paragraph"]
    assert cross
    for s in cross:
        assert "\n\n" in s.quote
        first, second = s.quote.split("\n\n")
        assert sentence_spans(first) and len(sentence_spans(second)) >= 1


def test_short_and_boilerplate_spans_are_skipped():
    records = [
        {"text": "Too short to use.", "section": "Intro", "page": 1},
        {"text": "See https://example.com/code for the code release of this project and all of the data we used.", "section": "Intro", "page": 1},
        {"text": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25", "section": "Intro", "page": 1},
        {"text": SENT + " Another sentence that has enough words to count as a real one.", "section": "References", "page": 2},
    ]
    doc = build_document("x", "T", records)
    assert sample_spans([doc], 20, seed=0) == []


def test_over_long_paragraph_is_skipped():
    long_text = " ".join(["word"] * 250) + "."
    doc = build_document("x", "T", [{"text": long_text, "section": "Intro", "page": 1}])
    assert sample_spans([doc], 10, seed=0) == []


def test_parse_question_variants():
    assert parse_question('{"question": "What does X do?"}') == "What does X do?"
    assert parse_question('```json\n{"question": "What  does\nX do?"}\n```') == "What does X do?"
    assert parse_question('Sure! {"question": "Why Y?"} hope that helps') == "Why Y?"
    assert parse_question('{"question": "SKIP"}') is None
    assert parse_question("SKIP") is None
    assert parse_question('{"skip": true}') is None
    for bad in ["", "no json here", "{broken", "[1, 2]", '{"answer": "x"}', '{"question": 5}']:
        with pytest.raises(MalformedReply):
            parse_question(bad)


def test_banned_phrases():
    assert has_banned_phrase("What does the authors' method do?".replace("authors'", "the authors"))
    assert has_banned_phrase("How does This Paper handle retrieval?")
    assert has_banned_phrase("What is the proposed approach for X?")
    assert not has_banned_phrase("How does ColBERT score late interaction?")


def test_overlap_ratio():
    span = "The retrieval model encodes every passage into a dense vector"
    assert overlap_ratio("What color is the sky?", span) < 0.3
    assert overlap_ratio("encodes every passage into a dense vector", span) == 1.0
    # 4 of 8 question words form the longest shared run
    assert overlap_ratio("How does it encodes every passage quickly now", span) == pytest.approx(3 / 8, abs=0.01)
    assert overlap_ratio("", span) == 0.0


def test_generate_writes_filters_and_resumes(tmp_path):
    docs = [_doc("a"), _doc("b")]
    spans = sample_spans(docs, 8, seed=0)
    titles = {d.paper_id: d.title for d in docs}
    replies = iter(
        ['{"question": "How does the DPR-2 retriever encode passages?"}', '{"question": "SKIP"}',
         '{"question": "What does this paper do?"}', "garbage"]
        + ['{"question": "How is recall of DPR-2 measured?"}'] * 20
    )
    calls = []

    def ask(span, span_type=""):
        calls.append(span)
        return next(replies)

    out = tmp_path / "gold.jsonl"
    counts = generate(spans, titles, ask, out)
    assert counts["rejected"] == 2 and counts["malformed"] == 1
    assert counts["written"] == len(spans) - 3
    items = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert items[0]["status"] == "draft" and isinstance(items[0]["overlap"], float)
    assert set(items[0]) == {
        "qid", "paper_id", "question", "evidence_quote", "char_start", "char_end", "doc_sha256",
        "span_type", "status", "overlap", "title_overlap",
    }
    by_id = {d.paper_id: d for d in docs}
    for it in items:
        doc = by_id[it["paper_id"]]
        assert doc.text[it["char_start"] : it["char_end"]] == it["evidence_quote"]
        assert it["doc_sha256"] == hashlib.sha256(doc.text.encode("utf-8")).hexdigest()

    # Rerun: everything written or rejected is skipped; only the malformed span is retried.
    calls.clear()
    counts2 = generate(spans, titles, ask, out)
    assert len(calls) == 1 and counts2["written"] == 1
    assert counts2["skipped_existing"] == len(spans) - 1


def test_load_dotenv_real_env_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# c\nLLM_MODEL="m1"\nLLM_API_KEY=filekey\n\nLLM_BASE_URL=https://x/v1\n', encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "realkey")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    load_dotenv(env)
    assert goldgen.os.environ["LLM_API_KEY"] == "realkey"
    assert goldgen.os.environ["LLM_MODEL"] == "m1"
    assert goldgen.os.environ["LLM_BASE_URL"] == "https://x/v1"
    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.delenv("LLM_BASE_URL")


def test_llm_client_retries_on_429_then_succeeds(monkeypatch):
    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._body, self.headers, self.text = code, body, {}, ""

        def json(self):
            return self._body

    seq = [Resp(429), Resp(503), Resp(200, {"choices": [{"message": {"content": "hi"}}]})]
    monkeypatch.setattr(goldgen.requests, "post", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(goldgen.time, "sleep", lambda s: None)
    client = goldgen.LLMClient("k", "https://x/v1/", "m", rpm=0)
    assert client("span") == "hi"
    assert client.url == "https://x/v1/chat/completions"


# --- review-fix tests --------------------------------------------------------

def _resp(code, body=None, text=""):
    class R:
        status_code, headers = code, {}

        def json(self):
            if isinstance(body, Exception):
                raise body
            return body

    r = R()
    r.text = text
    return r


def test_no_duplicate_qids_across_span_types():
    # One-sentence paragraphs are both a "paragraph" and a "sentence" candidate with equal offsets.
    one = "The retrieval model encodes every passage into a dense vector using a shared encoder network today."
    doc = build_document("d", "T", [{"text": one, "section": "M", "page": 1} for _ in range(6)])
    for seed in range(20):
        spans = sample_spans([doc], 12, seed=seed)
        qids = [s.qid for s in spans]
        assert len(qids) == len(set(qids))


def test_generate_dedupes_qids_within_run(tmp_path):
    docs = [_doc("a")]
    sp = sample_spans(docs, 4, seed=0)[0]
    out = tmp_path / "g.jsonl"
    calls = []
    counts = generate([sp, sp], {}, lambda s, t: calls.append(s) or '{"question": "What does DPR-2 use to encode passages?"}', out)
    assert len(calls) == 1 and counts["written"] == 1 and counts["skipped_existing"] == 1


def test_span_quality_requires_clean_sentence_edges():
    records = [
        {"text": "and then the retrieval model encodes every passage into a dense vector using a shared encoder network.", "section": "M", "page": 1},
        {"text": "The retrieval model encodes every passage into a dense vector using a shared encoder network, see Out-", "section": "M", "page": 1},
        {"text": "line of the method in the next section where we describe every single component in much detail.", "section": "M", "page": 2},
        {"text": SENT + " " + SENT, "section": "M", "page": 2},
    ]
    doc = build_document("q", "T", records)
    for seed in range(5):
        spans = sample_spans([doc], 40, seed=seed)
        for s in spans:
            for part in s.quote.split("\n\n"):
                assert part[0].isupper() or part[0].isdigit()
                assert part.rstrip()[-1] in ".!?"
            assert "Out-" not in s.quote and not s.quote.startswith("and then")


def test_title_overlap():
    assert title_overlap("How does ColBERT rank passages?", "ColBERT: Late Interaction") == pytest.approx(1 / 3, abs=0.01)
    assert title_overlap("What is the?", "Anything") == 0.0
    assert title_overlap("dense retrieval", "") == 0.0


def test_prompt_has_no_title():
    assert "title" not in goldgen.build_user_prompt("some span").lower()
    assert "title" not in goldgen.SYSTEM_PROMPT.lower()


def _env(tmp_path, monkeypatch, text, raw=None):
    env = tmp_path / ".env"
    if raw is not None:
        env.write_bytes(raw)
    else:
        env.write_text(text, encoding="utf-8")
    monkeypatch.delenv("K1", raising=False)
    goldgen.load_dotenv(env)
    return goldgen.os.environ.pop("K1", None)


def test_dotenv_bom_export_quotes_and_comments(tmp_path, monkeypatch):
    assert _env(tmp_path, monkeypatch, "", raw=b"\xef\xbb\xbfK1=abc\n") == "abc"  # BOM
    assert _env(tmp_path, monkeypatch, "export K1=abc\n") == "abc"
    assert _env(tmp_path, monkeypatch, 'K1="a b # not comment"  # real\n') == "a b # not comment"
    assert _env(tmp_path, monkeypatch, "K1='x y'\n") == "x y"
    assert _env(tmp_path, monkeypatch, "K1=abc   # trailing\n") == "abc"
    assert _env(tmp_path, monkeypatch, "K1=ab#cd\n") == "ab#cd"  # no space: not a comment


def test_base_url_validation():
    with pytest.raises(ValueError):
        goldgen.LLMClient("k", "http://example.com/v1", "m")
    goldgen.LLMClient("k", "http://localhost:11434/v1", "m")
    goldgen.LLMClient("k", "http://127.0.0.1:11434/v1", "m")
    goldgen.LLMClient("k", "https://api.example.com/v1", "m")


def test_api_key_redacted_in_error(monkeypatch):
    monkeypatch.setattr(goldgen.requests, "post", lambda *a, **k: _resp(401, text="bad key SECRET123 given"))
    client = goldgen.LLMClient("SECRET123", "https://x/v1", "m", rpm=0)
    with pytest.raises(RuntimeError) as ei:
        client("span")
    assert "SECRET123" not in str(ei.value) and "bad key" not in str(ei.value)
    assert "401" in str(ei.value) and "LLM_API_KEY" in str(ei.value)


def test_non_auth_error_body_is_redacted(monkeypatch):
    monkeypatch.setattr(goldgen.requests, "post", lambda *a, **k: _resp(400, text="bad SECRET123 given"))
    client = goldgen.LLMClient("SECRET123", "https://x/v1", "m", rpm=0)
    with pytest.raises(RuntimeError) as ei:
        client("span")
    assert "SECRET123" not in str(ei.value) and "***" in str(ei.value)


def test_only_connection_errors_and_timeouts_are_retried(monkeypatch):
    monkeypatch.setattr(goldgen.time, "sleep", lambda s: None)
    client = goldgen.LLMClient("SECRET123", "https://x/v1", "m", rpm=0, max_retries=2)
    calls = []

    def boom(exc):
        def post(*a, **k):
            calls.append(1)
            raise exc
        return post

    monkeypatch.setattr(goldgen.requests, "post", boom(goldgen.requests.exceptions.InvalidHeader("bad SECRET123")))
    with pytest.raises(RuntimeError) as ei:
        client("span")
    assert len(calls) == 1 and "SECRET123" not in str(ei.value) and "InvalidHeader" in str(ei.value)
    for exc in (goldgen.requests.ConnectionError("x"), goldgen.requests.Timeout("x")):
        calls.clear()
        monkeypatch.setattr(goldgen.requests, "post", boom(exc))
        with pytest.raises(RuntimeError):
            client("span")
        assert len(calls) == 3


@pytest.mark.parametrize("body", [ValueError("bad json"), {}, {"choices": []}, {"choices": [{}]}, None])
def test_malformed_200_maps_to_malformed_reply(monkeypatch, body):
    monkeypatch.setattr(goldgen.requests, "post", lambda *a, **k: _resp(200, body))
    client = goldgen.LLMClient("k", "https://x/v1", "m", rpm=0)
    with pytest.raises(MalformedReply):
        client("span")


def test_load_done_ids_tolerates_truncated_last_line(tmp_path, capsys):
    p = tmp_path / "g.jsonl"
    p.write_text('{"qid": "a:1-2"}\n{"qid": "b:3-', encoding="utf-8")
    assert goldgen.load_done_ids(p) == {("a:1-2", "")}
    assert "WARNING" in capsys.readouterr().out


def test_append_starts_on_fresh_line_and_uses_lf(tmp_path):
    docs = [_doc("a")]
    sp = sample_spans(docs, 4, seed=0)[0]
    out = tmp_path / "g.jsonl"
    out.write_bytes(b'{"qid": "zzz:1-')  # truncated, no newline
    generate([sp], {}, lambda s, t: '{"question": "What does DPR-2 use to encode passages?"}', out)
    raw = out.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines[0] == '{"qid": "zzz:1-' and json.loads(lines[1])["qid"] == sp.qid


# --- second review round ------------------------------------------------------

def _doc_from(texts, pid="z"):
    return build_document(pid, "T", [{"text": t, "section": "M", "page": 1} for t in texts])


def test_caption_and_replacement_char_spans_are_rejected():
    cap = "Table 3 shows the retrieval model results for every passage encoder in the shared evaluation suite."
    bad = "The retrieval model encodes every passage into a dense vector using a shared " + chr(0xFFFD) + " encoder network."
    cross_cap = "Figure 2 depicts how the retrieval model encodes every passage into a dense vector for scoring."
    doc = _doc_from([SENT + " " + SENT, cap, SENT, bad, SENT, cross_cap])
    rejects = goldgen.Counter()
    cands = goldgen.candidate_spans(doc, rejects)
    quotes = [s.quote for v in cands.values() for s in v]
    assert quotes and not any(q.startswith(("Table", "Figure")) or chr(0xFFFD) in q for q in quotes)
    assert not any(s.quote.split(PARAGRAPH_SEP)[-1].startswith("Figure") for s in cands["cross_paragraph"])
    assert rejects["caption"] >= 2 and rejects["replacement_char"] >= 1


def test_cross_paragraph_prompt_note_and_no_title():
    cross = goldgen.build_user_prompt("A." + PARAGRAPH_SEP + "B.", "cross_paragraph")
    assert "BOTH" in cross and "BOTH" not in goldgen.build_user_prompt("A.", "paragraph")
    assert "title" not in cross.lower()


def test_generate_passes_span_type_to_ask(tmp_path):
    sp = sample_spans([_doc("a")], 20, seed=0)
    seen = []
    generate(sp, {}, lambda q, t: seen.append(t) or '{"question": "What does DPR-2 use to encode passages?"}', tmp_path / "g.jsonl")
    assert "cross_paragraph" in seen


def test_resume_keys_on_sha_and_reports_stale(tmp_path):
    sp = sample_spans([_doc("a")], 4, seed=0)[0]
    out = tmp_path / "g.jsonl"
    old = {"qid": sp.qid, "doc_sha256": "0" * 64, "question": "old?"}
    out.write_text(json.dumps(old) + chr(10), encoding="utf-8")
    calls = []
    counts = generate([sp], {}, lambda q, t: calls.append(q) or '{"question": "What does DPR-2 use to encode passages?"}', out)
    assert len(calls) == 1 and counts["written"] == 1 and counts["stale"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2  # stale record kept
    calls.clear()
    counts = generate([sp], {}, lambda q, t: calls.append(q) or "x", out)
    assert not calls and counts["skipped_existing"] == 1 and counts["stale"] == 1


def test_rejected_file_stores_sha(tmp_path):
    sp = sample_spans([_doc("a")], 4, seed=0)[0]
    out = tmp_path / "g.jsonl"
    generate([sp], {}, lambda q, t: '{"question": "SKIP"}', out)
    rec = json.loads((tmp_path / "g.jsonl.rejected").read_text(encoding="utf-8").splitlines()[0])
    assert rec["doc_sha256"] == sp.doc_sha256 and rec["reason"] == "skip"


@pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 13, 250])
def test_quotas_sum_to_n(n):
    docs = [_doc(str(i), 40) for i in range(10)]
    assert len(sample_spans(docs, n, seed=0)) == n


def test_default_ids_file_is_repo_anchored():
    assert goldgen.DEFAULT_IDS_FILE.is_absolute()
    assert goldgen.DEFAULT_IDS_FILE == goldgen.Path(goldgen.__file__).resolve().parents[2] / "eval" / "corpus_v1.txt"


def test_failfast_error_chains_nothing(monkeypatch):
    def post(*a, **k):
        raise goldgen.requests.exceptions.InvalidHeader("Bearer SECRET123")
    monkeypatch.setattr(goldgen.requests, "post", post)
    client = goldgen.LLMClient("SECRET123", "https://x/v1", "m", rpm=0)
    with pytest.raises(RuntimeError) as ei:
        client("span")
    assert ei.value.__context__ is None and ei.value.__cause__ is None


def test_key_is_stripped_so_trimmed_echo_is_redacted(monkeypatch):
    monkeypatch.setattr(goldgen.requests, "post", lambda *a, **k: _resp(400, text="got SECRET123"))
    client = goldgen.LLMClient("  SECRET123\n", "https://x/v1", "m", rpm=0)
    assert client.api_key == "SECRET123"
    with pytest.raises(RuntimeError) as ei:
        client("span")
    assert "SECRET123" not in str(ei.value)


@pytest.mark.parametrize("bad", ["SECRET’123", "SECRET\x00123", "SEC RET\t123"])
def test_non_ascii_or_unprintable_key_rejected_without_echo(bad):
    with pytest.raises(ValueError) as ei:
        goldgen.LLMClient(bad, "https://x/v1", "m")
    assert "SECRET" not in str(ei.value)


def test_specific_terms():
    assert goldgen.specific_terms("GLatM beats BERT and GPT-4 by 12.5% on FDBench v3.") >= {
        "glatm", "bert", "gpt-4", "12.5%", "fdbench"}
    assert goldgen.specific_terms("The model is used to realize these records in natural language.") == set()


def test_math_and_roadmap_filters():
    assert goldgen.is_math_heavy("Let \U0001d449 be the norm and \U0001d708 = \U0001d449∕\U0001d440 ≥ 1.")
    assert not goldgen.is_math_heavy(SENT)
    assert goldgen.is_roadmap("We introduce the setup (§2.1), then describe retrieval (§2.2).")
    assert goldgen.is_roadmap("Section 3 covers data and Section 4 covers results.")
    assert not goldgen.is_roadmap("Results are in Section 4.")


def test_spans_without_specific_terms_are_rejected():
    plain = "The model is used to realize these fixed records in natural language for every example."
    doc = build_document("p", "T", [{"text": plain, "section": "Method", "page": 1}])
    rejects = goldgen.Counter()
    assert all(not v for v in goldgen.candidate_spans(doc, rejects).values())
    assert rejects["no_specific_term"] >= 1


def test_question_without_anchor_term_is_rejected(tmp_path):
    sp = sample_spans([_doc("a")], 1, seed=0)
    out = tmp_path / "g.jsonl"
    counts = generate(sp, {}, lambda q, t: '{"question": "What is used to encode passages?"}', out)
    assert counts["rejected"] == 1 and not out.read_text(encoding="utf-8").strip()
    assert "no_anchor" in (tmp_path / "g.jsonl.rejected").read_text(encoding="utf-8")


def test_generic_acronyms_are_not_specific():
    assert goldgen.specific_terms("The LLM uses RAG with an MLLM.") == set()
    assert goldgen.specific_terms("ThreadRAG improves the LLM.") == {"threadrag"}


def test_overlong_question_is_rejected(tmp_path):
    sp = sample_spans([_doc("a")], 1, seed=0)
    long_q = "What does DPR-2 " + "and also " * 20 + "do?"
    counts = generate(sp, {}, lambda q, t: json.dumps({"question": long_q}), tmp_path / "g.jsonl")
    assert counts["rejected"] == 1
    assert "too_long" in (tmp_path / "g.jsonl.rejected").read_text(encoding="utf-8")


# --- term quality, anchoring, retry, recheck -----------------------------------

@pytest.mark.parametrize("text, expected", [
    ("iPhone PyTorch GPT-4o Llama-2-7B BM25 COVID-19 ThreadRAG 12.5%",
     {"iphone", "pytorch", "gpt-4o", "llama-2-7b", "bm25", "covid-19", "threadrag", "12.5%"}),
    ("Wei et al. 2026b, see 1990s and the 3rd run in IV, Fig 3a and S1.", set()),
    ("IN AND THE OF BY", set()),  # all-caps English words
    ("Wei (Wei et al., 2026b) uses BM25.", {"bm25"}),
    ("v3 A4 UK", set()),  # short figure refs, short all-caps
])
def test_specific_terms_junk_and_keepers(text, expected):
    assert goldgen.specific_terms(text) == expected


def test_specific_terms_is_linear_on_long_tokens():
    assert goldgen.specific_terms("a1" * 20000) == {"a1" * 20000}


@pytest.mark.parametrize("question, quote, expected", [
    ("What recall does BM25-based retrieval reach?", "BM25 retrieves candidates.", True),
    ("How does Llama-2 behave?", "We fine-tune Llama-2-7B on the corpus.", True),
    ("How do BERTs encode text?", "BERT encodes text.", True),
    ("What does GPT 4 score?", "GPT-4 scores highly.", True),
    ("What is the F1 score of the tagger?", "The tagger reaches an F1-score of 0.9.", True),
    ("Did it gain 12.5 percent?", "Accuracy improved by 12.5%.", True),
    ("Did it gain 12.5%?", "Accuracy improved by 12.5%.", True),
    ("What does the ThreadRAG router do?", "ThreadRAG routes queries.", True),
    ("What is used to encode passages?", "The DPR-2 model encodes passages.", False),
    ("What was reported in 2026b?", "As shown by Wei (Wei et al., 2026b) and Lee 2026b.", False),
    ("What does the LLM do?", "The LLM improves BM25.", False),
    ("What about subBM25 retrieval?", "BM25 retrieves candidates.", False),  # not at a word start
    ("What about 112.5 percent?", "Accuracy improved by 12.5%.", False),
])
def test_is_anchored(question, quote, expected):
    assert goldgen.is_anchored(question, quote) is expected


def test_cross_paragraph_needs_a_term_somewhere_and_anchor_to_the_span():
    a = "The DPR-2 model encodes every passage into a dense vector using a shared encoder network."
    b = "The second sentence describes only generic things about encoders and their training data overall."
    c = "Then the second model keeps one vector per token for every passage in the collection of documents."
    doc = build_document("p", "T", [
        {"text": a, "section": "S", "page": 1}, {"text": b, "section": "S", "page": 1},
        {"text": c, "section": "S", "page": 1}])
    cands = goldgen.candidate_spans(doc, rejects := goldgen.Counter())
    assert any(s.quote == a + PARAGRAPH_SEP + b for s in cands["cross_paragraph"])  # term in one half is enough
    assert all(s.quote != b + PARAGRAPH_SEP + c for s in cands["cross_paragraph"])  # no term in either half
    assert rejects["no_specific_term"] >= 1

    quote = a + PARAGRAPH_SEP + b
    assert goldgen.question_reject_reason("How does DPR-2 encode passages?", "cross_paragraph", quote) == ""
    assert goldgen.question_reject_reason("How are passages encoded?", "cross_paragraph", quote) == "no_anchor"


def test_prompt_describes_the_filter():
    p = goldgen.SYSTEM_PROMPT
    assert "inner capital" in p and "digit" in p and "percentage" in p
    assert "LLM" in p and "citation year" in p


def test_math_heavy_ignores_minus_and_comparisons():
    assert not goldgen.is_math_heavy("Scores of −2, −3 ≤ x ≥ 1, ≈ 2, ∼ 3, ∈ S, ≠ 4, ± 5 and −5.")
    assert goldgen.is_math_heavy("∑ ∀ ∃ ∂ over the set")  # remaining operators still count


# retry / recheck

_Q_OK = '{"question": "How does DPR-2 encode passages?"}'


def _rej(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _setup_retry(tmp_path, n=4, reason="no_anchor"):
    docs = [_doc("a")]
    spans = sample_spans(docs, n, seed=0)
    out = tmp_path / "g.jsonl"
    rej = tmp_path / "g.jsonl.rejected"
    _rej(rej, [{"qid": s.qid, "doc_sha256": s.doc_sha256, "reason": reason} for s in spans])
    return docs, spans, out, rej


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_retry_rejected_writes_successes_and_clears_rejected(tmp_path):
    docs, spans, out, rej = _setup_retry(tmp_path)
    asked = []
    res = goldgen.retry_rejected(docs, lambda q, t: asked.append((q, t)) or _Q_OK, out, {"no_anchor"})
    assert res["written"] == len(spans) == len(asked) and res["targeted"] == len(spans)
    assert {i["qid"] for i in _lines(out)} == {s.qid for s in spans}
    assert {i["qid"]: i["span_type"] for i in _lines(out)} == {s.qid: s.span_type for s in spans}
    assert _lines(rej) == []
    assert [t for _, t in asked] == [i["span_type"] for i in _lines(out)]


def test_retry_rejected_rerejects_with_new_reason_and_keeps_other_rows(tmp_path):
    docs, spans, out, rej = _setup_retry(tmp_path)
    rows = _lines(rej) + [{"qid": "a:1-2", "doc_sha256": "x", "reason": "skip"}]
    _rej(rej, rows)
    replies = iter([_Q_OK, '{"question": "What is used to encode passages?"}', '{"question": "SKIP"}', _Q_OK])
    res = goldgen.retry_rejected(docs, lambda q, t: next(replies), out, {"no_anchor"})
    assert res["written"] == 2 and res["rejected"] == 2
    left = {r["qid"]: r["reason"] for r in _lines(rej)}
    assert left[spans[1].qid] == "no_anchor" and left[spans[2].qid] == "skip" and left["a:1-2"] == "skip"
    assert spans[0].qid not in left and spans[3].qid not in left


def test_retry_rejected_stale_sha_and_unknown_paper_untouched(tmp_path):
    docs, spans, out, rej = _setup_retry(tmp_path, n=2)
    rows = [dict(r, doc_sha256="old") for r in _lines(rej)] + [{"qid": "zzz:1-9", "doc_sha256": "x", "reason": "no_anchor"}]
    _rej(rej, rows)
    res = goldgen.retry_rejected(docs, lambda q, t: pytest.fail("no LLM call expected"), out, {"no_anchor"})
    assert res["stale"] == 3 and res["written"] == 0
    assert _lines(rej) == rows and not out.read_text(encoding="utf-8").strip()


def test_retry_rejected_span_failing_current_filters_gets_new_reason(tmp_path):
    plain = "The model is used to realize these fixed records in natural language for every example."
    doc = build_document("p", "T", [{"text": plain, "section": "Method", "page": 1}])
    sha = goldgen.doc_sha256(doc)
    out, rej = tmp_path / "g.jsonl", tmp_path / "g.jsonl.rejected"
    _rej(rej, [{"qid": f"p:0-{len(plain)}", "doc_sha256": sha, "reason": "too_long"}])
    res = goldgen.retry_rejected([doc], lambda q, t: pytest.fail("no LLM call expected"), out, {"too_long"})
    assert res["failed_filters"] == 1
    assert _lines(rej)[0]["reason"] == "no_specific_term"


def test_retry_rejected_dry_run_changes_nothing(tmp_path):
    docs, spans, out, rej = _setup_retry(tmp_path)
    before = rej.read_bytes()
    res = goldgen.retry_rejected(docs, None, out, {"no_anchor", "too_long"}, dry_run=True)
    assert res["targeted"] == len(spans) and res["by_reason"] == {"no_anchor": len(spans)}
    assert rej.read_bytes() == before and not out.exists()


def test_retry_rejected_interrupted_loses_nothing_and_rerun_does_not_duplicate(tmp_path):
    docs, spans, out, rej = _setup_retry(tmp_path)
    calls = []

    def flaky(q, t):
        calls.append(q)
        if len(calls) == 3:
            raise RuntimeError("boom")
        return _Q_OK

    with pytest.raises(RuntimeError):
        goldgen.retry_rejected(docs, flaky, out, {"no_anchor"})
    assert len(_lines(out)) == 2 and len(_lines(rej)) == len(spans) - 2  # progress saved, nothing lost
    res = goldgen.retry_rejected(docs, lambda q, t: _Q_OK, out, {"no_anchor"})
    assert res["written"] == len(spans) - 2
    assert sorted(i["qid"] for i in _lines(out)) == sorted(s.qid for s in spans)  # no duplicates
    assert _lines(rej) == []


def test_retry_rejected_crash_before_rewrite_does_not_duplicate(tmp_path):
    docs, spans, out, rej = _setup_retry(tmp_path)
    goldgen.retry_rejected(docs, lambda q, t: _Q_OK, out, {"no_anchor"})
    _rej(rej, [{"qid": s.qid, "doc_sha256": s.doc_sha256, "reason": "no_anchor"} for s in spans])  # as if rewrite never happened
    res = goldgen.retry_rejected(docs, lambda q, t: pytest.fail("no LLM call expected"), out, {"no_anchor"})
    assert res["already_written"] == len(spans) and len(_lines(out)) == len(spans) and _lines(rej) == []


def test_retry_rejected_malformed_row_kept(tmp_path):
    docs, spans, out, rej = _setup_retry(tmp_path, n=1)
    res = goldgen.retry_rejected(docs, lambda q, t: "garbage", out, {"no_anchor"})
    assert res["malformed"] == 1 and len(_lines(rej)) == 1


def test_recheck_sets_and_clears_auto_flag_preserving_everything_else(tmp_path):
    sp = sample_spans([_doc("a")], 3, seed=0)
    good = goldgen.make_item(sp[0], "How does DPR-2 encode passages?")
    bad = goldgen.make_item(sp[1], "What encodes passages?")
    stale = dict(goldgen.make_item(sp[2], "How does DPR-2 encode passages?"), auto_flag="no_anchor", status="ok")
    long_q = dict(goldgen.make_item(sp[0], "DPR-2 " + "word " * 40), qid="x:1-2")
    out = tmp_path / "g.jsonl"
    out.write_text("\n".join([json.dumps(good), "not json {", json.dumps(bad), json.dumps(stale), json.dumps(long_q)]) + "\n",
                   encoding="utf-8")
    counts = goldgen.recheck(out)
    assert counts == {"clean": 2, "no_anchor": 1, "too_long": 1}
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and lines[0] == json.dumps(good) and lines[1] == "not json {"
    rows = [json.loads(line) for line in (lines[0], lines[2], lines[3], lines[4])]
    assert [r.get("auto_flag") for r in rows] == [None, "no_anchor", None, "too_long"]
    assert rows[2]["status"] == "ok" and {k: v for k, v in rows[1].items() if k != "auto_flag"} == bad
    assert not (tmp_path / "g.jsonl.tmp").exists()


def test_recheck_flags_cross_paragraph_half(tmp_path):
    a = "The DPR-2 model encodes every passage into a dense vector using a shared encoder network."
    c = "Later ColBERTv2 stores one vector per token for every passage in the collection of documents."
    item = {"span_type": "cross_paragraph", "evidence_quote": a + PARAGRAPH_SEP + c, "question": "How does DPR-2 work?"}
    assert goldgen.auto_flag(item) == "one_half_only"  # hint for the reviewer, not a rejection
    assert goldgen.auto_flag(dict(item, question="How do DPR-2 and ColBERTv2 differ?")) == ""
    assert goldgen.auto_flag(dict(item, question="How do the two models differ?")) == "no_anchor"
    assert goldgen.auto_flag(dict(item, evidence_quote="No terms here." + PARAGRAPH_SEP + "None here either today.")) == "no_specific_term"

@pytest.mark.parametrize("span, question, ok", [
    ("MAP score is reported.", "Which mapping heuristic is used?", False),
    ("ANN search is fast.", "How is annotation done?", False),
    ("CLIP embeddings are used.", "Why is clipping applied?", False),
    ("MAP score is reported.", "What MAP does it reach?", True),
    ("CLIP embeddings are used.", "How are CLIPs compared?", True),
    ("BM25 is strong.", "Is BM25-based ranking better?", True),
])
def test_short_acronyms_must_match_a_whole_word(span, question, ok):
    assert goldgen.is_anchored(question, span) is ok


def test_more_generic_acronyms_are_not_specific():
    assert goldgen.specific_terms("We merge the LoRA weights after SFT and GRPO on PDF files.") == set()


def test_infer_span_type_short_single_sentence_paragraph_is_sentence():
    short = "DPR-2 encodes passages into dense vectors quickly."
    doc = build_document("p", "T", [{"text": short, "section": "S", "page": 1}])
    assert goldgen.infer_span_type(doc, 0, len(short)) == "sentence"
    doc2 = build_document("p", "T", [_para(0)])
    assert goldgen.infer_span_type(doc2, 0, len(doc2.text)) == "paragraph"


def test_rejected_rows_store_span_type(tmp_path):
    sp = sample_spans([_doc("a")], 1, seed=0)
    generate(sp, {}, lambda q, t: '{"question": "SKIP"}', tmp_path / "g.jsonl")
    row = json.loads((tmp_path / "g.jsonl.rejected").read_text(encoding="utf-8").splitlines()[0])
    assert row["span_type"] == sp[0].span_type


def test_one_half_only_needs_terms_in_both_halves():
    a = "The DPR-2 model encodes every passage into a dense vector using a shared encoder network."
    b = "The second sentence describes only generic things about encoders and their training data."
    item = {"span_type": "cross_paragraph", "evidence_quote": a + PARAGRAPH_SEP + b, "question": "How does DPR-2 work?"}
    assert goldgen.auto_flag(item) == ""
