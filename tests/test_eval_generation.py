import json

import numpy as np
import pytest

from arxiv_rag import eval_generation as eg
from arxiv_rag.eval_chunking import PaperBootstrap, load_gold, sha256_text
from arxiv_rag.goldgen import MalformedReply, QuotaExhausted
from test_eval_chunking import make_docs, row, write_gold

JM = "judge-m"


@pytest.fixture
def env(tmp_path):
    docs = make_docs()
    d1, d2 = docs["p1"], docs["p2"]
    path = write_gold(tmp_path / "g.jsonl", [
        row(d1, "Alpha uses ZetaNet for retrieval.", question="Q1?", span_type="sentence"),
        row(d1, "Beta results improve by 12.5% over the baseline model.", question="Q2?", span_type="paragraph"),
        row(d2, "Gamma is a separate topic about quantum gardening and soil.", paper="p2", question="Q3?",
            span_type="sentence"),
    ])
    gold = load_gold(path, docs).items
    shas = {p: sha256_text(d.text) for p, d in docs.items()}
    return docs, gold, shas


def ans(item, shas, config, answer=None, cites=(1,), hit_ranks=(), abstained=False, invalid=0, **kw):
    """Answer row with 3 retrieved chunks; the chunks at `hit_ranks` cover the gold evidence.

    Without an explicit `answer` the text cites `cites` as [n] groups plus `invalid` out-of-range [9] groups
    (the eval re-parses citations and abstention from the text).
    """
    if answer is None:
        answer = "Some answer" + ("" if config == "closed-book" else "".join(f" [{c}]" for c in cites) + " [9]" * invalid) + "."
    retrieved = []
    for r in (1, 2, 3):
        if config == "closed-book":
            break
        if r in hit_ranks:
            s, e = item.char_start, item.char_end
        else:
            s, e = 10_000 + r, 10_010 + r  # nothing near the evidence
        retrieved.append({"rank": r, "chunk_id": f"{item.paper_id}:structure:{r}", "paper_id": item.paper_id,
                          "char_start": s, "char_end": e})
    return {"qid": item.qid, "doc_sha256": shas[item.paper_id], "config": config, "model": "gen-m",
            "question": item.question, "answer": answer, "citations": [] if config == "closed-book" else list(cites),
            "invalid_citations": invalid, "retrieved": retrieved, "abstained": abstained,
            "prompt_tokens": 100, "completion_tokens": 10, **kw}


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def verdict(row_, correct="yes", faithful=True, model=JM):
    return {"qid": row_["qid"], "doc_sha256": row_["doc_sha256"], "config": row_["config"], "model": model,
            "correct": correct, "faithful": None if row_["config"] == "closed-book" else faithful,
            "unsupported_claim": "", "prompt_tokens": 200, "completion_tokens": 15}


class Judge:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def __call__(self, system, user):
        self.calls.append((system, user))
        r = self.replies(system, user) if callable(self.replies) else self.replies[len(self.calls) - 1]
        if isinstance(r, Exception):
            raise r
        return (r if isinstance(r, str) else json.dumps(r)), {"prompt_tokens": 200, "completion_tokens": 15}


RAG_OK = {"correct": "yes", "faithful": True, "unsupported_claim": ""}


# --- judge prompt and parsing --------------------------------------------------------------

def test_judge_prompts(env):
    _, gold, _ = env
    s, u = eg.build_judge_prompts(gold[0], "It uses ZetaNet [2].", {2: "text two", 4: "text four"}, True)
    assert "QUESTION: Q1?" in u and "Alpha uses ZetaNet for retrieval." in u
    assert "[2] text two\n\n[4] text four" in u and u.endswith("ANSWER: It uses ZetaNet [2].")
    assert '"faithful"' in s and "not against the reference" in s
    s0, u0 = eg.build_judge_prompts(gold[0], "(none)", {}, True)
    assert "the answer cites no passage" in u0
    sc, uc = eg.build_judge_prompts(gold[0], "It uses X.", {}, False)
    assert "faithful" not in sc and "PASSAGES" not in uc and '"correct"' in sc


def test_parse_judgement():
    v = eg.parse_judgement('```json\n{"correct": "Partial", "faithful": false, "unsupported_claim": " a  b "}\n```', True)
    assert v == {"correct": "partial", "faithful": False, "unsupported_claim": "a b"}
    assert eg.parse_judgement('<think>x</think>{"correct": "no"}', False) == {
        "correct": "no", "faithful": None, "unsupported_claim": ""}
    for raw, retrieval in [("", True), ("nope", True), ('{"correct": "maybe", "faithful": true}', True),
                           ('{"correct": "yes"}', True), ('{"correct": "yes", "faithful": "true"}', True),
                           ("[1]", False)]:
        with pytest.raises(MalformedReply):
            eg.parse_judgement(raw, retrieval)


# --- deterministic metrics -----------------------------------------------------------------

def test_deterministic_metrics(env):
    _, gold, shas = env
    g = gold[0]
    m = eg.deterministic(ans(g, shas, "dense-k5", cites=(1,), hit_ranks=(2,), invalid=2), g)
    assert (m["context_hit"], m["citation_hit"], m["cites_anything"], m["invalid_citations"], m["abstained"]) == (
        1, 0, 1, 2, 0)
    assert eg.deterministic(ans(g, shas, "dense-k5", cites=(2,), hit_ranks=(2,)), g)["citation_hit"] == 1
    m = eg.deterministic(ans(g, shas, "dense-k5", answer="INSUFFICIENT", cites=(), abstained=True), g)
    assert (m["context_hit"], m["citation_hit"], m["cites_anything"], m["abstained"]) == (0, 0, 0, 1)
    assert set(eg.deterministic(ans(g, shas, "closed-book"), g)) == {
        "abstained", "prose_refusal", "answered", "cites_anything", "invalid_citations"}


def test_abstention_and_citations_are_rederived_from_the_text(env):
    _, gold, shas = env
    g = gold[0]
    # stored fields lie: answer is exactly INSUFFICIENT but stored abstained False and stale citations
    lie = ans(g, shas, "dense-k5", answer="insufficient.", cites=(1,), abstained=False)
    d = eg.derive(lie)
    assert d["abstained"] and not d["prose_refusal"] and d["citations"] == [] and eg.refused(d)
    # a reply that merely starts with INSUFFICIENT is not an explicit abstention
    d = eg.derive(ans(g, shas, "dense-k5", answer="INSUFFICIENT: it is not said [1].", abstained=True))
    assert not d["abstained"] and d["citations"] == [1]
    # k is the number of retrieved passages (3): [1, 5] and [0, 1] are rejected whole, [Passage 2] accepted
    d = eg.derive(ans(g, shas, "dense-k5", answer="A [1, 5] B [0, 1] C [Passage 2] D [passage 3]."))
    assert d["citations"] == [2, 3] and d["invalid_citations"] == 2
    d = eg.derive(ans(g, shas, "closed-book", answer="A [1]."))
    assert d["citations"] == [] and d["invalid_citations"] == 0
    assert eg.deterministic(lie, g)["abstained"] == 1.0
    assert eg.derive(eg.derive(lie)) == eg.derive(lie)  # idempotent


def test_prose_refusal_is_derived_and_needs_no_citation(env):
    _, gold, shas = env
    g = gold[0]
    r = ans(g, shas, "dense-k5", answer="The passages do not mention the dataset.")
    d = eg.derive(r)
    assert d["prose_refusal"] and not d["abstained"] and eg.refused(d)
    m = eg.deterministic(r, g)
    assert (m["abstained"], m["prose_refusal"], m["answered"], m["cites_anything"]) == (0, 1, 0, 0)
    assert eg.verdict_of(r, {}) == {"correct": "no", "faithful": None, "faithful_raw": None}
    assert not eg.derive(ans(g, shas, "dense-k5", answer="The passages do not mention X [1].", ))["prose_refusal"]


def test_rel_path_inside_and_outside_repo(tmp_path):
    assert eg.rel_path(eg._ROOT / "eval" / "answers_v1.jsonl") == "eval/answers_v1.jsonl"
    assert eg.rel_path(tmp_path / "x.jsonl") == str(tmp_path / "x.jsonl")


def test_cited_passages_come_from_the_document(env):
    docs, gold, shas = env
    g = gold[0]
    r = ans(g, shas, "dense-k5", cites=(1, 9), hit_ranks=(1,))
    assert eg.cited_passages(r, docs) == {1: docs["p1"].text[g.char_start:g.char_end]}


# --- judging -------------------------------------------------------------------------------

def test_judge_skips_abstentions_caches_and_resumes(env, tmp_path):
    docs, gold, shas = env
    rows = [ans(gold[0], shas, "closed-book", answer="Maybe ZetaNet.", cites=()),
            ans(gold[0], shas, "dense-k5", hit_ranks=(1,)),
            ans(gold[1], shas, "dense-k5", answer="INSUFFICIENT", cites=(), abstained=True)]
    entries = [(r, gold[0] if r["qid"] == gold[0].qid else gold[1]) for r in rows]
    out = tmp_path / "j.jsonl"
    judge = Judge([{"correct": "no"}, RAG_OK])
    done = {}
    c = eg.judge_answers(entries, docs, judge, JM, out, done)
    assert c == {"judged": 2, "malformed": 0, "failed": 0, "sentinel": 0, "cached": 0} and len(judge.calls) == 2
    assert "faithful" not in judge.calls[0][0] and "faithful" in judge.calls[1][0]
    assert "Alpha uses ZetaNet for retrieval." in judge.calls[1][1].split("PASSAGES CITED")[1]  # cited passage text
    stored = eg.load_judgements(out, JM)
    assert len(stored) == 2 and eg.load_judgements(out, "other-model") == {}
    judge2 = Judge([])
    c = eg.judge_answers(entries, docs, judge2, JM, out, stored)
    assert c["judged"] == 0 and c["cached"] == 2 and judge2.calls == []
    assert all(v["answer_sha1"] == eg.answer_sha1(r["answer"]) for v, r in zip(stored.values(), rows))


def test_judge_skips_prose_refusals(env, tmp_path):
    docs, gold, shas = env
    entries = [(ans(gold[0], shas, "dense-k5", answer="I cannot answer this."), gold[0]),
               (ans(gold[1], shas, "dense-k5", answer="  INSUFFICIENT "), gold[1])]
    judge = Judge([])
    c = eg.judge_answers(entries, docs, judge, JM, tmp_path / "j.jsonl", {})
    assert judge.calls == [] and c["judged"] == 0 and c["cached"] == 0


def test_cached_verdict_only_applies_to_the_same_answer_text(env, tmp_path, capsys):
    docs, gold, shas = env
    g = gold[0]
    out = tmp_path / "j.jsonl"
    old = ans(g, shas, "dense-k5", answer="Alpha uses ZetaNet [1].", hit_ranks=(1,))
    done: dict = {}
    eg.judge_answers([(old, g)], docs, Judge([RAG_OK]), JM, out, done)
    new = dict(old, answer="Alpha uses something else [1].")
    judge = Judge([RAG_OK])
    c = eg.judge_answers([(new, g)], docs, judge, JM, out, done)
    assert c["judged"] == 1 and c["cached"] == 0 and len(judge.calls) == 1  # answer changed: re-judged
    assert eg.cached_verdict(eg.derive(old), done) is None
    assert eg.cached_verdict(eg.derive(new), done)["answer_sha1"] == eg.answer_sha1(new["answer"])
    # rows written before answer_sha1 existed are accepted, with one warning
    legacy = {(old["qid"], old["doc_sha256"], "dense-k5"): verdict(old)}
    eg._warned["no_sha"] = False
    capsys.readouterr()
    assert eg.cached_verdict(eg.derive(old), legacy) is not None and eg.cached_verdict(eg.derive(new), legacy) is not None
    assert capsys.readouterr().out.count("without answer_sha1") == 1


def test_one_failing_item_does_not_stop_the_run_and_is_retried_once(env, tmp_path):
    docs, gold, shas = env
    entries = [(ans(g, shas, "dense-k5"), g) for g in gold]
    out = tmp_path / "j.jsonl"
    judge = Judge([RuntimeError("500"), RAG_OK, RAG_OK, RAG_OK])  # first item fails, passes on the retry
    c = eg.judge_answers(entries, docs, judge, JM, out, {})
    assert c["judged"] == 3 and c["failed"] == 0 and len(judge.calls) == 4
    assert eg.load_attempts(out, JM) == {}


def test_failed_runs_are_tracked_then_get_a_sentinel(env, tmp_path):
    docs, gold, shas = env
    g = gold[0]
    r = ans(g, shas, "dense-k5")
    out = tmp_path / "j.jsonl"
    done: dict = {}
    boom = lambda s, u: RuntimeError("secret detail sk-123")  # noqa: E731
    for run in (1, 2):
        c = eg.judge_answers([(r, g)], docs, Judge(boom), JM, out, done)
        assert c["failed"] == 1 and c["sentinel"] == 0 and done == {}
        assert eg.load_attempts(out, JM) == {(r["qid"], r["doc_sha256"], "dense-k5"): run}
    c = eg.judge_answers([(r, g)], docs, Judge(boom), JM, out, done)
    assert c["failed"] == 1 and c["sentinel"] == 1
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    attempts = [x for x in rows if "attempt" in x]
    assert [x["attempt"] for x in attempts] == [1, 2, 3]
    assert set(attempts[0]) == {"qid", "doc_sha256", "config", "judge_model", "error", "attempt"}
    assert attempts[0]["error"] == "RuntimeError" and "secret" not in out.read_text(encoding="utf-8")
    sentinel = eg.load_judgements(out, JM)[(r["qid"], r["doc_sha256"], "dense-k5")]
    assert sentinel["correct"] is None and sentinel["faithful"] is None and sentinel["judge_error"] is True
    judge = Judge([])
    c = eg.judge_answers([(r, g)], docs, judge, JM, out, eg.load_judgements(out, JM))
    assert judge.calls == [] and c["cached"] == 1  # the sentinel is a final verdict


def test_malformed_replies_count_as_failed_runs(env, tmp_path):
    docs, gold, shas = env
    r = ans(gold[0], shas, "dense-k5")
    out = tmp_path / "j.jsonl"
    done: dict = {}
    for _ in range(3):
        eg.judge_answers([(r, gold[0])], docs, Judge(["garbage"]), JM, out, done)
    assert done[(r["qid"], r["doc_sha256"], "dense-k5")]["judge_error"] is True


def test_judge_aborts_after_five_consecutive_failures_but_quota_propagates(env, tmp_path):
    docs, gold, shas = env
    entries = [(ans(g, shas, c), g) for g in gold for c in ("dense-k5", "bm25-k5")]
    judge = Judge(lambda s, u: RuntimeError("down"))
    with pytest.raises(RuntimeError, match="5 consecutive failures"):
        eg.judge_answers(entries, docs, judge, JM, tmp_path / "j.jsonl", {})
    assert len(judge.calls) == 5
    with pytest.raises(QuotaExhausted):
        eg.judge_answers(entries, docs, Judge([RuntimeError("x"), QuotaExhausted("daily")]), JM,
                         tmp_path / "j2.jsonl", {})


def test_judge_malformed_retried_and_quota_keeps_progress(env, tmp_path):
    docs, gold, shas = env
    entries = [(ans(g, shas, "dense-k5"), g) for g in gold]
    out = tmp_path / "j.jsonl"
    c = eg.judge_answers(entries, docs, Judge(["garbage", RAG_OK, RAG_OK]), JM, out, {})
    assert c["malformed"] == 1 and c["judged"] == 2
    done = eg.load_judgements(out, JM)
    assert len(done) == 2
    with out.open("a", encoding="utf-8") as f:
        f.write('{"qid": "trunc')  # truncated write
    assert len(eg.load_judgements(out, JM)) == 2
    out2 = tmp_path / "j2.jsonl"
    with pytest.raises(QuotaExhausted):
        eg.judge_answers(entries, docs, Judge([RAG_OK, QuotaExhausted("daily")]), JM, out2, {})
    assert len(eg.load_judgements(out2, JM)) == 1


# --- population ----------------------------------------------------------------------------

def test_population_keeps_complete_current_questions(env, tmp_path):
    _, gold, shas = env
    rows = [ans(g, shas, c) for g in gold for c in ("closed-book", "dense-k5")]
    rows.remove(rows[-1])  # q3 has no dense answer
    rows.append(dict(ans(gold[0], shas, "dense-k5"), doc_sha256="0" * 64))  # stale sha
    pop = eg.load_population(write_rows(tmp_path / "a.jsonl", rows), gold, shas)
    assert pop["configs"] == ["closed-book", "dense-k5"]
    assert [g.qid for g in pop["items"]] == [gold[0].qid, gold[1].qid]
    assert pop["dropped"] == {"stale": 1, "other_config_rows": 0, "incomplete_questions": 1}
    assert set(pop["rows"]["dense-k5"]) == {gold[0].qid, gold[1].qid}


def test_configs_filter_ignores_stray_configs(env, tmp_path):
    _, gold, shas = env
    rows = [ans(g, shas, c) for g in gold for c in ("closed-book", "dense-k5")]
    rows.append(ans(gold[0], shas, "bm25-k1"))  # stray config that would shrink the paired population
    path = write_rows(tmp_path / "a.jsonl", rows)
    assert len(eg.load_population(path, gold, shas)["items"]) == 1  # unfiltered: only q1 has all three
    pop = eg.load_population(path, gold, shas, ["closed-book", "dense-k5"])
    assert pop["configs"] == ["closed-book", "dense-k5"] and len(pop["items"]) == 3
    assert pop["dropped"]["other_config_rows"] == 1
    assert eg.load_population(path, gold, shas, ["closed-book", "bm25-k5"])["items"] == []


# --- evaluate: hand-computed ---------------------------------------------------------------

def build(env, tmp_path):
    """3 questions x (closed-book, dense-k5, bm25-k5) with known verdicts.

    closed-book: q1 no (abstain), q2 yes, q3 no
    dense-k5:    q1 yes + context hit, q2 partial + miss, q3 abstain + miss
    bm25-k5:     q1 yes + hit, q2 yes + hit (cites nothing -> unfaithful), q3 no + miss
    """
    docs, gold, shas = env
    g1, g2, g3 = gold
    rows = [
        ans(g1, shas, "closed-book", answer="INSUFFICIENT", abstained=True),
        ans(g2, shas, "closed-book"), ans(g3, shas, "closed-book"),
        ans(g1, shas, "dense-k5", hit_ranks=(1,), cites=(1,)),
        ans(g2, shas, "dense-k5", cites=(1, 2)),
        ans(g3, shas, "dense-k5", answer="INSUFFICIENT", abstained=True, cites=()),
        ans(g1, shas, "bm25-k5", hit_ranks=(1,), cites=(1,)),
        ans(g2, shas, "bm25-k5", hit_ranks=(3,), cites=()),
        ans(g3, shas, "bm25-k5", cites=(2,)),
    ]
    pop = eg.load_population(write_rows(tmp_path / "a.jsonl", rows), gold, shas)
    by = {(r["qid"], r["config"]): r for r in rows}
    spec = {(g2.qid, "closed-book"): ("yes", None), (g3.qid, "closed-book"): ("no", None),
            (g1.qid, "dense-k5"): ("yes", True), (g2.qid, "dense-k5"): ("partial", True),
            (g1.qid, "bm25-k5"): ("yes", True), (g2.qid, "bm25-k5"): ("yes", True), (g3.qid, "bm25-k5"): ("no", False)}
    judgements = {}
    for (qid, c), (corr, faith) in spec.items():
        r = by[(qid, c)]
        judgements[(r["qid"], r["doc_sha256"], c)] = verdict(r, corr, faith)
    return pop, judgements


def test_evaluate_hand_computed(env, tmp_path):
    pop, judgements = build(env, tmp_path)
    res = eg.evaluate(pop, judgements, 200, 0)
    assert res["judge_complete"] and res["judge_missing"] == 0
    m = {c: {k: v["mean"] for k, v in d["metrics"].items()} for c, d in res["configs"].items()}
    assert m["closed-book"] == {"correct": pytest.approx(1 / 3), "correct_or_partial": pytest.approx(1 / 3),
                                "abstained": pytest.approx(1 / 3)}
    assert m["dense-k5"]["correct"] == pytest.approx(1 / 3) and m["dense-k5"]["correct_or_partial"] == pytest.approx(2 / 3)
    assert m["dense-k5"]["abstained"] == pytest.approx(1 / 3) and m["dense-k5"]["context_hit"] == pytest.approx(1 / 3)
    assert m["dense-k5"]["faithful"] == 1.0 and res["configs"]["dense-k5"]["metrics"]["faithful"]["n"] == 2
    assert m["bm25-k5"]["correct"] == pytest.approx(2 / 3) and m["bm25-k5"]["context_hit"] == pytest.approx(2 / 3)
    assert m["bm25-k5"]["citation_hit"] == pytest.approx(1 / 3)  # q2 hit at rank 3 but cited nothing
    # q2 cites nothing -> forced unfaithful even though the judge said true; q3 judged false
    assert m["bm25-k5"]["faithful"] == pytest.approx(1 / 3)
    assert res["configs"]["bm25-k5"]["cites_anything"] == pytest.approx(2 / 3)  # over the 3 answered rows
    assert res["configs"]["dense-k5"]["cites_anything"] == 1.0  # the abstention is not in the denominator
    assert res["configs"]["dense-k5"]["n_answered"] == 2
    assert (res["configs"]["dense-k5"]["explicit_abstentions"], res["configs"]["dense-k5"]["prose_refusals"]) == (1, 0)
    # faithful among answers citing >= 1 valid passage: the uncited bm25 q2 drops out instead of counting as false
    assert m["dense-k5"]["faithful_cited"] == 1.0
    assert m["bm25-k5"]["faithful_cited"] == pytest.approx(1 / 2)
    assert res["configs"]["bm25-k5"]["metrics"]["faithful_cited"]["n"] == 2
    assert res["judged"] and res["judge_complete"] and res["excluded_questions"] == 0 and res["n_scored"] == 3
    ci = res["configs"]["bm25-k5"]["metrics"]["correct"]
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    # paired
    d = res["paired_vs_closed_book"]["bm25-k5"]["correct"]
    assert d["diff"] == pytest.approx(1 / 3) and 0 <= d["frac_le_0"] <= 1
    b = res["bm25_vs_dense"]
    assert (b["a"], b["b"]) == ("bm25-k5", "dense-k5")
    assert b["metrics"]["context_hit"]["diff"] == pytest.approx(1 / 3)
    assert b["metrics"]["correct"]["diff"] == pytest.approx(1 / 3)
    # conditional on context hit
    dh = res["by_context_hit"]["dense-k5"]
    assert dh["context_hit"] == {"n": 1, "correct": 1.0, "correct_or_partial": 1.0, "abstained": 0.0}
    assert dh["no context_hit"]["n"] == 2 and dh["no context_hit"]["correct"] == 0.0
    assert dh["no context_hit"]["correct_or_partial"] == 0.5 and dh["no context_hit"]["abstained"] == 0.5
    bh = res["by_context_hit"]["bm25-k5"]
    assert bh["context_hit"]["n"] == 2 and bh["context_hit"]["correct"] == 1.0
    assert bh["no context_hit"] == {"n": 1, "correct": 0.0, "correct_or_partial": 0.0, "abstained": 0.0}
    assert "closed-book" not in res["by_context_hit"]
    # by span type: q1,q3 sentence; q2 paragraph
    st = res["by_span_type"]
    assert st["sentence"]["n"] == 2 and st["paragraph"]["n"] == 1
    assert st["sentence"]["configs"]["bm25-k5"] == {"correct": 0.5, "context_hit": 0.5}
    assert st["paragraph"]["configs"]["closed-book"] == {"correct": 1.0}


def test_no_verdicts_at_all_omits_correctness(env, tmp_path):
    pop, _ = build(env, tmp_path)
    res = eg.evaluate(pop, {}, 50, 0)
    assert not res["judged"] and not res["judge_complete"] and res["n_scored"] == 3
    assert set(res["configs"]["dense-k5"]["metrics"]) == {"abstained", "context_hit", "citation_hit"}
    assert res["paired_vs_closed_book"] == {} and res["by_context_hit"] == {}
    assert set(res["bm25_vs_dense"]["metrics"]) == {"context_hit", "citation_hit"}
    assert "correct" not in res["by_span_type"]["sentence"]["configs"]["dense-k5"]
    assert "JUDGE INCOMPLETE" in eg.render_markdown(report_for(res, pop))


def test_missing_and_sentinel_verdicts_exclude_those_questions_only(env, tmp_path):
    pop, judgements = build(env, tmp_path)
    g1, g2, g3 = pop["items"]
    del judgements[(g1.qid, pop["rows"]["bm25-k5"][g1.qid]["doc_sha256"], "bm25-k5")]  # not judged yet
    d2 = pop["rows"]["dense-k5"][g2.qid]
    judgements[(g2.qid, d2["doc_sha256"], "dense-k5")] = dict(
        verdict(d2), correct=None, faithful=None, judge_error=True)  # sentinel
    res = eg.evaluate(pop, judgements, 50, 0)
    assert res["judged"] and not res["judge_complete"]
    assert (res["judge_missing"], res["judge_errors"], res["excluded_questions"], res["n_scored"]) == (1, 1, 2, 1)
    assert all(c["n"] == 1 for c in res["configs"].values())  # paired: only q3, under every config
    assert res["configs"]["bm25-k5"]["metrics"]["correct"]["mean"] == 0.0  # bm25 q3 was judged "no"
    assert res["configs"]["closed-book"]["metrics"]["correct"]["mean"] == 0.0
    assert set(res["paired_vs_closed_book"]) == {"dense-k5", "bm25-k5"}
    md = eg.render_markdown(report_for(res, pop))
    assert "2 of 3 questions EXCLUDED" in md and "1 judge errors" in md
    assert "JUDGE INCOMPLETE" not in md


def test_prose_refusals_score_as_incorrect_without_a_verdict(env, tmp_path):
    docs, gold, shas = env
    g = gold[0]
    rows = [ans(g, shas, "closed-book", answer="I don't know."),
            ans(g, shas, "dense-k5", answer="The passages do not mention this."),
            ans(g, shas, "bm25-k5", answer="INSUFFICIENT", cites=())]
    pop = eg.load_population(write_rows(tmp_path / "a.jsonl", rows), gold, shas)
    res = eg.evaluate(pop, {}, 20, 0)  # no judge verdicts needed: everything is a refusal
    assert res["judged"] and res["judge_complete"]
    for c in ("closed-book", "dense-k5", "bm25-k5"):
        assert res["configs"][c]["metrics"]["correct"]["mean"] == 0.0
    assert "faithful" not in res["configs"]["dense-k5"]["metrics"]  # nothing to be faithful about
    assert [(res["configs"][c]["explicit_abstentions"], res["configs"][c]["prose_refusals"])
            for c in ("closed-book", "dense-k5", "bm25-k5")] == [(0, 1), (0, 1), (1, 0)]
    assert res["configs"]["dense-k5"]["cites_anything"] is None


def test_gold_span_stats_and_context_hit_caveat(env, tmp_path):
    pop, judgements = build(env, tmp_path)
    gs = eg.evaluate(pop, judgements, 20, 0)["gold_spans"]
    longest = 10  # every fixture chunk covers 10 characters
    lengths = [g.char_end - g.char_start for g in pop["items"]]
    assert gs["n"] == 3 and gs["max_chunk_span"] >= longest
    assert gs["longer_than_chunk"] == sum(n > gs["max_chunk_span"] for n in lengths)
    assert gs["unhittable"] == sum(0.5 * n > gs["max_chunk_span"] for n in lengths)
    rows = {"x": {"q": {"retrieved": [{"char_start": 0, "char_end": 100}]}}}

    class G:
        def __init__(self, n):
            self.char_start, self.char_end = 0, n

    assert eg.gold_span_stats([G(50), G(100), G(101), G(250)], rows) == {
        "n": 4, "max_chunk_span": 100, "longer_than_chunk": 2, "unhittable": 1}
    assert eg.gold_span_stats([G(50)], {"cb": {"q": {"retrieved": []}}}) is None


def test_ratio_ci_matches_plain_mean_when_everything_counts():
    boot = PaperBootstrap(["a", "a", "b", "c"], 100, 0)
    v = np.array([1.0, 0.0, 1.0, 1.0])
    r = eg.ratio_ci(v, np.ones(4, dtype=bool), boot)
    assert r["mean"] == 0.75 and r["n"] == 4
    assert (r["lo"], r["hi"]) == pytest.approx(eg.percentile_ci(boot.means(v)))
    sub = eg.ratio_ci(v, np.array([True, True, False, False]), boot)
    assert sub["mean"] == 0.5 and sub["n"] == 2
    assert eg.ratio_ci(v, np.zeros(4, dtype=bool), boot) is None


# --- output --------------------------------------------------------------------------------

def report_for(res, pop):
    return {"config": {"judge_model": JM, "max_tokens": 256, "overlap": 32, "bootstrap": 50, "seed": 0,
                       "configs": pop["configs"]},
            "partial": False, "dropped": pop["dropped"], "n_gold": 3, "n_questions": len(pop["items"]),
            "n_papers": 2, "n_gold_papers": 2, "generator_models": ["gen-m"], "git_commit": "abc", "git_dirty": False,
            "tokens": {"generation dense-k5": eg.token_stats(list(pop["rows"]["dense-k5"].values())),
                       "judge": eg.token_stats([])},
            "results": res}


def test_token_stats():
    s = eg.token_stats([{"prompt_tokens": 100, "completion_tokens": 10}, {"prompt_tokens": 300, "completion_tokens": None}])
    assert s == {"calls": 2, "mean_prompt_tokens": 200.0, "mean_completion_tokens": 10.0}
    assert eg.token_stats([])["mean_prompt_tokens"] is None


def test_report_written_as_md_and_json(env, tmp_path):
    pop, judgements = build(env, tmp_path)
    res = eg.evaluate(pop, judgements, 50, 0)
    json_path, md_path = eg.write_outputs(report_for(res, pop), tmp_path / "out" / "phase4")
    md = md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["results"]["configs"]["dense-k5"]["n"] == 3
    for part in ("# Phase 4", "machine-verified", "`gen-m`", f"`{JM}`", "Commit: abc", "## Answer quality",
                 "Paired difference vs closed-book", "BM25 vs dense", "Does retrieval failure explain wrong answers",
                 "## By span type", "## Token use", "## Caveats", "evidence-relative"):
        assert part in md
    assert "JUDGE INCOMPLETE" not in md and "PARTIAL" not in md and "EXCLUDED" not in md
    for part in ("## Abstentions and refusals", "explicit abstentions", "prose refusals", "faithful (cited)",
                 "single-chunk 50% rule", "Longest retrieved chunk"):
        assert part in md


def test_main_end_to_end_with_cached_judgements_no_network(env, tmp_path, monkeypatch):
    """--no-judge with a full cache: report written, no LLM client created."""
    docs, _, shas = env
    answers, jfile, ids = tmp_path / "a.jsonl", tmp_path / "j.jsonl", tmp_path / "ids.txt"
    ids.write_text("p1\np2\n", encoding="utf-8")
    goldf = write_gold(tmp_path / "g.jsonl", [row(docs["p1"], "Alpha uses ZetaNet for retrieval.")])
    g = load_gold(goldf, docs).items[0]
    rows = [ans(g, shas, "closed-book"), ans(g, shas, "dense-k5", hit_ranks=(1,))]
    write_rows(answers, rows + [ans(g, shas, 'bm25-k5')])  # stray config: ignored
    jfile.write_text("\n".join(json.dumps(verdict(r, "yes", True)) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(eg, "load_documents", lambda *a, **k: list(docs.values()))
    monkeypatch.setattr(eg, "ids_to_stems", lambda ids: ids)
    monkeypatch.setattr(eg.UsageClient, "from_env", classmethod(lambda *a, **k: pytest.fail("no LLM expected")))
    monkeypatch.setattr("sys.argv", ["eval_generation", "--answers", str(answers), "--judge-out", str(jfile),
                                     "--judge-model", JM, "--ids-file", str(ids), "--gold", str(goldf),
                                     "--configs", "closed-book", "dense",
                                     "--out", str(tmp_path / "res"), "--bootstrap", "20", "--no-judge"])
    eg.main()
    out = json.loads((tmp_path / "res.json").read_text(encoding="utf-8"))
    assert out["n_questions"] == 1 and out["results"]["judge_complete"]
    assert out["results"]["configs"]["dense-k5"]["metrics"]["context_hit"]["mean"] == 1.0
