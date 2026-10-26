"""Phase 4b eval: correctness and faithfulness of the generated answers (see generate.py).

Design goals:
- Two kinds of evidence. Deterministic (no LLM): did the retrieved / cited chunks hit the gold evidence
  (eval_chunking.is_hit), did the answer cite anything, did it abstain. LLM judge: one call per
  non-abstained answer, cached, resumable.
- Same questions for every config: only questions answered under ALL configs in the file are scored, so
  closed-book vs RAG and bm25 vs dense are paired comparisons, and a partial run is still fair.
- Same statistics as Phases 2-3: paper-cluster bootstrap CIs, paired differences on the same resamples.
- Correct is evidence-relative: the judge sees the gold evidence quote as the reference. Faithful is judged
  only against the passages the answer cites, never against the reference.
- Abstentions and citations are re-derived from the answer text (the stored `abstained` / `citations` fields are
  not trusted): an explicit abstention is exactly INSUFFICIENT, a prose refusal is a known refusal phrasing with
  no citation. Neither is sent to the judge: correct = "no", faithful = null. They are reported separately.
- An answer that cites no valid passage cannot be faithful (nothing supports its claims): faithful is forced to
  false for it, whatever the judge says. "faithful (cited)" shows the judge's verdict among cited answers only.
- Safe: judgements are appended one per call (<judge-out>.lock held while judging), a truncated last line is
  ignored, a cached verdict applies only to the answer text it was made for (answer_sha1), QuotaExhausted stops
  cleanly and the report is still written. A failing item does not stop the run (retried once, abort after 5
  consecutive failures); after 3 failed runs it gets a sentinel verdict. Quality metrics are computed on the
  questions judged under every config, and the excluded count is reported.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from arxiv_rag.chunk import Chunk
from arxiv_rag.document import Document, ids_to_stems, load_documents
from arxiv_rag.eval_chunking import (
    DEFAULT_GOLD, DEFAULT_IDS_FILE, HIT_COVERAGE, GoldItem, PaperBootstrap, _cell, _table, git_dirty, git_sha,
    is_hit, load_gold, log, mean_ci, paired_diff, percentile_ci, sha256_text,
)
from arxiv_rag.generate import (
    CLOSED_BOOK, DEFAULT_ANSWERS, DEFAULT_CONFIGS, UsageClient, is_abstention, is_prose_refusal, load_answers,
    parse_citations, parse_config, run_resilient,
)
from arxiv_rag.goldgen import MalformedReply, load_dotenv, open_append
from arxiv_rag.ingest import read_ids_file
from arxiv_rag.judge import DEFAULT_JUDGE_MODEL, strip_think
from arxiv_rag.review import Lock

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JUDGEMENTS = _ROOT / "eval" / "answer_judgements_v1.jsonl"
DEFAULT_OUT = _ROOT / "docs" / "results" / "phase4"
MAX_CLAIM_CHARS = 200
VERDICTS = ("yes", "partial", "no")
MAX_JUDGE_ATTEMPTS = 3  # failed runs before an item gets a sentinel verdict
RAG_METRICS = ("correct", "correct_or_partial", "faithful", "faithful_cited", "abstained", "context_hit",
               "citation_hit")
CB_METRICS = ("correct", "correct_or_partial", "abstained")
JUDGED_METRICS = ("correct", "correct_or_partial", "faithful", "faithful_cited")
PAIRED_VS_CB = ("correct", "correct_or_partial")
PAIRED_BM25_DENSE = ("correct", "correct_or_partial", "context_hit", "citation_hit")

JUDGE_SYSTEM = (
    "You grade an ANSWER to a QUESTION about a scientific paper. The REFERENCE is the passage the question "
    "was written from; it is the ground truth for what the answer should say.\n"
    '- "correct": "yes" if the answer gives the information the reference provides for this question; '
    '"partial" if it gives some of it or contains a minor error; "no" if it is wrong, off-topic or does not '
    "answer.\n"
    "@RULE@"
    "Reply with JSON only: {@KEY@}"
)
RAG_RULE = (
    '- "faithful": true if every factual claim in the ANSWER is supported by the PASSAGES it cites (numbers '
    "like [2] in the answer refer to the numbered passages); judge against those passages only, not against "
    "the reference or your own knowledge. false otherwise.\n"
    '- "unsupported_claim": the first unsupported claim in at most 20 words, or "" if none.\n'
)
RAG_KEY = '"correct": "yes"|"partial"|"no", "faithful": true|false, "unsupported_claim": "..."'
CB_KEY = '"correct": "yes"|"partial"|"no"'


# --- judge ------------------------------------------------------------------------------

def build_judge_prompts(item: GoldItem, answer: str, passages: dict[int, str], retrieval: bool) -> tuple[str, str]:
    """(system, user). `passages` are the cited chunks by their number in the answer."""
    system = JUDGE_SYSTEM.replace("@RULE@", RAG_RULE if retrieval else "").replace(
        "@KEY@", RAG_KEY if retrieval else CB_KEY)
    user = f'QUESTION: {item.question}\n\nREFERENCE:\n"""\n{item.quote}\n"""\n\n'
    if retrieval:
        cited = "\n\n".join(f"[{n}] {t}" for n, t in sorted(passages.items())) or "(the answer cites no passage)"
        user += f"PASSAGES CITED BY THE ANSWER:\n{cited}\n\n"
    return system, user + f"ANSWER: {answer}"


def parse_judgement(raw: str, retrieval: bool) -> dict:
    """{"correct", "faithful" (None for closed-book), "unsupported_claim"}; raises MalformedReply."""
    text = strip_think(raw)
    if not text:
        raise MalformedReply("empty reply")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group() if m else text, strict=False)
    except json.JSONDecodeError as exc:
        raise MalformedReply(f"not JSON: {text[:80]!r}") from exc
    if not isinstance(obj, dict):
        raise MalformedReply(f"not an object: {text[:80]!r}")
    correct = str(obj.get("correct", "")).strip().lower()
    if correct not in VERDICTS:
        raise MalformedReply(f"bad 'correct': {text[:80]!r}")
    out = {"correct": correct, "faithful": None, "unsupported_claim": ""}
    if retrieval:
        if not isinstance(obj.get("faithful"), bool):
            raise MalformedReply(f"missing or non-boolean 'faithful': {text[:80]!r}")
        out["faithful"] = obj["faithful"]
        claim = obj.get("unsupported_claim")
        out["unsupported_claim"] = " ".join(claim.split())[:MAX_CLAIM_CHARS] if isinstance(claim, str) else ""
    return out


def answer_sha1(answer: str) -> str:
    return hashlib.sha1(answer.encode("utf-8")).hexdigest()


def rel_path(path: Path) -> str:
    """Path relative to the repo root when it is inside it (forward slashes), else as given."""
    try:
        return Path(path).resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return str(path)


def derive(row: dict) -> dict:
    """The row with citations and abstention re-derived from the answer text; stored values are ignored.

    Citations are parsed against the number of passages actually retrieved. Idempotent.
    """
    answer = row["answer"]
    cites, invalid = ([], 0) if row["config"] == CLOSED_BOOK else parse_citations(answer, len(row["retrieved"]))
    abstained = is_abstention(answer)
    return {**row, "citations": cites, "invalid_citations": invalid, "abstained": abstained,
            "prose_refusal": not abstained and is_prose_refusal(answer)}


def refused(row: dict) -> bool:
    """Explicit abstention or prose refusal (a derived row): scored as incorrect, never judged."""
    return row["abstained"] or row["prose_refusal"]


def load_judgements(path: Path, model: str) -> dict[tuple[str, str, str], dict]:
    """(qid, sha, config) -> stored verdict row for this judge model (last row wins).

    Failed-attempt rows (key "judge_model", no "model") are not verdicts and are skipped here.
    """
    return {(r["qid"], r["doc_sha256"], r["config"]): r for r in load_answers(path) if r.get("model") == model}


def load_attempts(path: Path, model: str) -> dict[tuple[str, str, str], int]:
    """(qid, sha, config) -> number of runs in which judging this answer failed, for this judge model."""
    out: dict[tuple[str, str, str], int] = {}
    for r in load_answers(path):
        if r.get("judge_model") == model and isinstance(r.get("attempt"), int):
            k = (r["qid"], r["doc_sha256"], r["config"])
            out[k] = max(out.get(k, 0), r["attempt"])
    return out


_warned = {"no_sha": False}


def cached_verdict(row: dict, judgements: dict) -> dict | None:
    """The stored verdict for this answer, or None. Verdicts made for a different answer text do not apply;
    old rows without answer_sha1 are accepted (with a one-time warning)."""
    v = judgements.get((row["qid"], row["doc_sha256"], row["config"]))
    if v is None:
        return None
    h = v.get("answer_sha1")
    if h is None:
        if not _warned["no_sha"]:
            _warned["no_sha"] = True
            log("WARNING: judgement rows without answer_sha1 are accepted without checking the answer text")
        return v
    return v if h == answer_sha1(row["answer"]) else None


Ask = Callable[[str, str], tuple[str, dict]]


def cited_passages(row: dict, docs: dict[str, Document]) -> dict[int, str]:
    """Number in the answer -> text of that chunk (the doc span the chunk covers)."""
    by_rank = {r["rank"]: r for r in row["retrieved"]}
    return {n: docs[by_rank[n]["paper_id"]].text[by_rank[n]["char_start"]:by_rank[n]["char_end"]]
            for n in row["citations"] if n in by_rank}


def needs_judge(entries: list[tuple[dict, GoldItem]], judgements: dict) -> list[tuple[dict, GoldItem]]:
    """(derived row, item) pairs that are answers (not refusals) without a usable verdict."""
    out = []
    for r, it in entries:
        r = derive(r)
        if not refused(r) and cached_verdict(r, judgements) is None:
            out.append((r, it))
    return out


def judge_answers(entries: list[tuple[dict, GoldItem]], docs: dict[str, Document], ask: Ask, model: str,
                  out: Path, done: dict[tuple[str, str, str], dict]) -> dict[str, int]:
    """Judge every answer (not a refusal) without a usable verdict; append each at once.

    QuotaExhausted propagates after the verdicts so far are saved. A RuntimeError fails that item only: it is
    retried once at the end, the run aborts after 5 failures in a row. An item that failed (or got a malformed
    reply) in this run gets an attempt row; at its 3rd failed run it gets a sentinel verdict
    {"correct": None, "faithful": None, "judge_error": True} so it can no longer blank the report.
    """
    derived = [(derive(r), it) for r, it in entries]
    eligible = [(r, it) for r, it in derived if not refused(r)]
    todo = needs_judge(derived, done)
    counts = {"judged": 0, "malformed": 0, "failed": 0, "sentinel": 0, "cached": len(eligible) - len(todo)}
    attempts = load_attempts(out, model) if out.exists() else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open_append(out) as f:
        def write(row: dict) -> None:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

        def record_failure(r: dict, error: str) -> None:
            key = (r["qid"], r["doc_sha256"], r["config"])
            attempts[key] = n = attempts.get(key, 0) + 1
            write({"qid": r["qid"], "doc_sha256": r["doc_sha256"], "config": r["config"], "judge_model": model,
                   "error": error, "attempt": n})
            if n >= MAX_JUDGE_ATTEMPTS:
                v = {"qid": r["qid"], "doc_sha256": r["doc_sha256"], "config": r["config"], "model": model,
                     "correct": None, "faithful": None, "judge_error": True, "unsupported_claim": "",
                     "answer_sha1": answer_sha1(r["answer"]), "prompt_tokens": None, "completion_tokens": None}
                write(v)
                done[key] = v
                counts["sentinel"] += 1
                log(f"judge failed {n} runs for {r['qid']} [{r['config']}]: recorded as judge_error")

        def step(entry) -> None:
            r, it = entry
            retrieval = r["config"] != CLOSED_BOOK
            system, user = build_judge_prompts(it, r["answer"], cited_passages(r, docs) if retrieval else {}, retrieval)
            raw, usage = ask(system, user)
            try:
                v = parse_judgement(raw, retrieval)
            except MalformedReply as exc:
                log(f"malformed {r['qid']} [{r['config']}]: {exc}")
                counts["malformed"] += 1
                record_failure(r, type(exc).__name__)
                return
            row = {"qid": r["qid"], "doc_sha256": r["doc_sha256"], "config": r["config"], "model": model, **v,
                   "answer_sha1": answer_sha1(r["answer"]),
                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
            write(row)
            done[(r["qid"], r["doc_sha256"], r["config"])] = row
            counts["judged"] += 1
            log(f"judged [{counts['judged']}/{len(todo)}] {r['qid']} {r['config']}: correct={v['correct']} "
                f"faithful={v['faithful']} ({time.time() - t0:.0f}s elapsed)")

        stats = run_resilient(todo, step, lambda entry, exc: record_failure(entry[0], type(exc).__name__))
        counts["failed"] = stats["failed"]
    return counts


# --- population and per-answer metrics -------------------------------------------------

def load_population(answers: Path, gold: list[GoldItem], shas: dict[str, str],
                    configs: list[str] | None = None) -> dict:
    """Answer rows of the current gold, restricted to questions answered under every config.

    `configs` limits the population to those configs (rows of other configs are ignored and counted);
    None means every config in the file. Returns {"configs": [names], "items": [GoldItem],
    "rows": {config: {qid: row}}, "dropped": {"stale", "other_config_rows", "incomplete_questions"}}.
    The last row per (qid, sha, config) wins.
    """
    key = {(g.qid, shas[g.paper_id]): g for g in gold}
    rows: dict[str, dict[str, dict]] = {c: {} for c in configs} if configs is not None else {}
    stale = other = 0
    for r in load_answers(answers):
        if (r["qid"], r["doc_sha256"]) not in key:
            stale += 1
            continue
        if configs is not None and r["config"] not in rows:
            other += 1
            continue
        rows.setdefault(r["config"], {})[r["qid"]] = r
    names = list(rows)
    common = set.intersection(*(set(v) for v in rows.values())) if rows else set()
    seen = {q for v in rows.values() for q in v}
    items = [g for g in gold if g.qid in common]
    return {"configs": names, "items": items, "rows": {c: {q: v[q] for q in common} for c, v in rows.items()},
            "dropped": {"stale": stale, "other_config_rows": other, "incomplete_questions": len(seen - common)}}


def chunk_of(r: dict) -> Chunk:
    return Chunk(r["chunk_id"], r["paper_id"], "structure", "", r["char_start"], r["char_end"], None, 0, 0)


def deterministic(row: dict, item: GoldItem) -> dict:
    row = derive(row)
    retrieval = row["config"] != CLOSED_BOOK
    cited = set(row["citations"])
    m = {"abstained": float(row["abstained"]), "prose_refusal": float(row["prose_refusal"]),
         "answered": float(not refused(row)), "cites_anything": float(bool(cited)),
         "invalid_citations": float(row["invalid_citations"])}
    if retrieval:
        m["context_hit"] = float(any(is_hit(chunk_of(r), item) for r in row["retrieved"]))
        m["citation_hit"] = float(any(is_hit(chunk_of(r), item) for r in row["retrieved"] if r["rank"] in cited))
    return m


def verdict_of(row: dict, judgements: dict) -> dict | None:
    """Per-answer verdict: {"correct", "faithful" (forced false when nothing valid is cited), "faithful_raw"},
    {"error": True} for a sentinel, or None if the answer still needs a judge call."""
    row = derive(row)
    if refused(row):
        return {"correct": "no", "faithful": None, "faithful_raw": None}
    v = cached_verdict(row, judgements)
    if v is None:
        return None
    if v.get("judge_error"):
        return {"error": True}
    faithful = v["faithful"]
    if row["config"] != CLOSED_BOOK and not row["citations"]:
        faithful = False  # no cited passage supports anything
    return {"correct": v["correct"], "faithful": faithful, "faithful_raw": v["faithful"]}


def ratio_ci(values: np.ndarray, mask: np.ndarray, boot: PaperBootstrap) -> dict | None:
    """Mean of `values` over `mask` with a cluster-bootstrap CI (the denominator is resampled too)."""
    if not mask.any():
        return None
    v = np.where(mask, values, 0.0)
    num = np.bincount(boot.codes, weights=v, minlength=boot.n_papers)[boot.idx].sum(axis=1)
    den = np.bincount(boot.codes, weights=mask.astype(float), minlength=boot.n_papers)[boot.idx].sum(axis=1)
    ok = den > 0
    lo, hi = percentile_ci(num[ok] / den[ok])
    return {"mean": float(v.sum() / mask.sum()), "lo": lo, "hi": hi, "n": int(mask.sum())}


def gold_span_stats(items: list[GoldItem], rows: dict[str, dict[str, dict]]) -> dict | None:
    """How many gold spans a single chunk can hardly or never hit (context_hit uses a one-chunk 50% rule)."""
    spans = [r["char_end"] - r["char_start"] for per_q in rows.values() for row in per_q.values()
             for r in row["retrieved"]]
    if not spans:
        return None
    longest = max(spans)
    lengths = [g.char_end - g.char_start for g in items]
    return {"n": len(items), "max_chunk_span": longest,
            "longer_than_chunk": sum(n > longest for n in lengths),
            "unhittable": sum(HIT_COVERAGE * n > longest for n in lengths)}


def evaluate(pop: dict, judgements: dict, n_boot: int, seed: int) -> dict:
    """Metrics for the population. Quality metrics use only the questions judged under every config.

    With no such question at all, only the deterministic metrics (on all questions) are computed.
    """
    configs = pop["configs"]
    rows = {c: {q: derive(r) for q, r in pop["rows"][c].items()} for c in configs}
    verdicts = {(c, g.qid): verdict_of(rows[c][g.qid], judgements) for c in configs for g in pop["items"]}
    missing = sum(v is None for v in verdicts.values())  # answers without a verdict yet
    errors = sum(v is not None and bool(v.get("error")) for v in verdicts.values())  # sentinels
    ok = [g for g in pop["items"]
          if all(verdicts[(c, g.qid)] is not None and not verdicts[(c, g.qid)].get("error") for c in configs)]
    judged = bool(ok)
    items = ok if judged else pop["items"]
    boot = PaperBootstrap([g.paper_id for g in items], n_boot, seed)
    per: dict[str, dict[str, np.ndarray]] = {}
    for c in configs:
        cols: dict[str, list[float]] = {}
        for it in items:
            r = rows[c][it.qid]
            vals = deterministic(r, it)
            if judged:
                v = verdicts[(c, it.qid)]
                vals["correct"] = float(v["correct"] == "yes")
                vals["correct_or_partial"] = float(v["correct"] in ("yes", "partial"))
                vals["faithful"] = np.nan if v["faithful"] is None else float(v["faithful"])
                cited = c != CLOSED_BOOK and bool(r["citations"])
                vals["faithful_cited"] = np.nan if (v["faithful_raw"] is None or not cited) else float(v["faithful_raw"])
            for k, x in vals.items():
                cols.setdefault(k, []).append(x)
        per[c] = {k: np.array(x) for k, x in cols.items()}

    res: dict = {"judged": judged, "judge_complete": judged and len(items) == len(pop["items"]),
                 "judge_missing": missing, "judge_errors": errors,
                 "n_scored": len(items), "excluded_questions": len(pop["items"]) - len(items) if judged else 0,
                 "gold_spans": gold_span_stats(items, rows),
                 "configs": {}, "paired_vs_closed_book": {}, "bm25_vs_dense": {}, "by_context_hit": {},
                 "by_span_type": {}}
    for c in configs:
        a, retrieval = per[c], c != CLOSED_BOOK
        metrics: dict = {}
        for m in (RAG_METRICS if retrieval else CB_METRICS):
            if m in JUDGED_METRICS and not judged:
                continue
            if m in ("faithful", "faithful_cited"):
                metrics[m] = ratio_ci(a[m], ~np.isnan(a[m]), boot)
            else:
                metrics[m] = mean_ci(a[m], boot)
        answered = a["answered"] == 1
        res["configs"][c] = {"n": len(items), "metrics": {k: v for k, v in metrics.items() if v is not None},
                             "explicit_abstentions": int(a["abstained"].sum()),
                             "prose_refusals": int(a["prose_refusal"].sum()),
                             "n_answered": int(answered.sum()),
                             "cites_anything": float(a["cites_anything"][answered].mean()) if answered.any() else None,
                             "invalid_citations": int(a["invalid_citations"].sum()),
                             "answers_with_invalid_citations": int((a["invalid_citations"] > 0).sum())}

    if CLOSED_BOOK in per and judged:
        for c in configs:
            if c != CLOSED_BOOK:
                res["paired_vs_closed_book"][c] = {m: paired_diff(per[c][m], per[CLOSED_BOOK][m], boot)
                                                   for m in PAIRED_VS_CB}
    dense = next((c for c in configs if c.startswith("dense-")), None)
    bm25 = next((c for c in configs if c.startswith("bm25-")), None)
    if dense and bm25:
        res["bm25_vs_dense"] = {"a": bm25, "b": dense,
                                "metrics": {m: paired_diff(per[bm25][m], per[dense][m], boot)
                                            for m in PAIRED_BM25_DENSE
                                            if (judged or m not in JUDGED_METRICS) and m in per[bm25] and m in per[dense]}}

    if judged:
        for c in configs:
            if c == CLOSED_BOOK:
                continue
            a = per[c]
            no_answer = np.maximum(a["abstained"], a["prose_refusal"])
            res["by_context_hit"][c] = {
                label: {"n": int(mask.sum()), "correct": float(a["correct"][mask].mean()),
                        "correct_or_partial": float(a["correct_or_partial"][mask].mean()),
                        "abstained": float(no_answer[mask].mean())}
                for label, mask in (("context_hit", a["context_hit"] == 1), ("no context_hit", a["context_hit"] == 0))
                if mask.any()}
    spans = np.array([g.span_type for g in items])
    for t in sorted(set(spans)):
        mask = spans == t
        cell = {}
        for c in configs:
            e = {m: float(per[c][m][mask].mean()) for m in ("correct", "context_hit")
                 if m in per[c] and (judged or m not in JUDGED_METRICS)}
            if e:
                cell[c] = e
        res["by_span_type"][t] = {"n": int(mask.sum()), "configs": cell}
    return res


def token_stats(rows: list[dict]) -> dict:
    p = [r["prompt_tokens"] for r in rows if isinstance(r.get("prompt_tokens"), (int, float))]
    c = [r["completion_tokens"] for r in rows if isinstance(r.get("completion_tokens"), (int, float))]
    return {"calls": len(rows), "mean_prompt_tokens": float(np.mean(p)) if p else None,
            "mean_completion_tokens": float(np.mean(c)) if c else None}


# --- output ------------------------------------------------------------------------------

def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def render_markdown(report: dict) -> str:
    cfg, res = report["config"], report["results"]
    configs = list(res["configs"])
    L = ["# Phase 4: answer generation with citations", ""]
    if report.get("partial"):
        L += [f"> **PARTIAL: {report['n_questions']} questions answered under every config "
              f"(of {report['n_gold']} in the gold set).** Not a full result.", ""]
    if not res["judged"]:
        L += [f"> **JUDGE INCOMPLETE: no question has a verdict under every config "
              f"({res['judge_missing']} answers without a verdict, {res['judge_errors']} judge errors).** "
              "Correctness and faithfulness are omitted; only the deterministic metrics are shown.", ""]
    elif res["excluded_questions"]:
        L += [f"> **{res['excluded_questions']} of {report['n_questions']} questions EXCLUDED from every number "
              f"below: some config's answer has no verdict ({res['judge_missing']} answers still to judge, "
              f"{res['judge_errors']} judge errors).** All metrics are on the {res['n_scored']} questions judged "
              "under every config, so the comparison stays paired.", ""]
    L += ["**Gold set: machine-verified (LLM judge + rule filters), not human-reviewed**", ""]
    d = report["dropped"]
    L += [f"- Questions scored: {res['n_scored']} of {report['n_gold']} gold questions "
          f"({report['n_questions']} answered under every config; answers for {d['incomplete_questions']} "
          f"questions missing under some config, {d['stale']} answer rows stale or off-gold, "
          f"{d.get('other_config_rows', 0)} rows of other configs ignored)",
          f"- Corpus: {report['n_papers']} papers ({report['n_gold_papers']} with scored questions); chunks: "
          f"structure, max_tokens {cfg['max_tokens']}, overlap {cfg['overlap']}",
          f"- Configs: {', '.join(f'`{c}`' for c in configs)}; passages numbered [1]..[k], no titles or ids shown",
          f"- Generator: {', '.join(f'`{m}`' for m in report['generator_models'])}, temperature 0",
          f"- Judge: `{cfg['judge_model']}`, temperature 0, sees the gold evidence quote as the reference",
          f"- Hit rule: same paper and chunk span covers >= {HIT_COVERAGE:.0%} of the evidence span",
          "- Cells: mean [95% percentile cluster bootstrap CI: papers resampled with replacement, all their "
          f"questions kept; {cfg['bootstrap']} resamples, seed {cfg['seed']}]",
          f"- Commit: {report['git_commit'] or 'unknown'}" + (" (uncommitted changes)" if report.get("git_dirty") else ""),
          ""]

    L += ["## Answer quality", "",
          "`correct`: judge says yes; `correct or partial`: yes or partial; `faithful` (headline): every claim "
          "supported by the cited passages, over retrieval answers that are not refusals; an answer that cites no "
          "valid passage is forced to unfaithful, whatever the judge said. `faithful (cited)`: the judge's verdict "
          "among answers that cite at least one valid passage only (no forcing), so it is not penalised for "
          "uncited answers. `abstained`: the whole reply is INSUFFICIENT; prose refusals are counted separately "
          "below. Both count as incorrect. `context hit`: any of the k retrieved chunks hits the gold evidence; "
          "`citation hit`: any cited chunk does.", ""]
    cols = [("correct", "correct"), ("correct_or_partial", "correct or partial"), ("faithful", "faithful"),
            ("faithful_cited", "faithful (cited)"), ("abstained", "abstained"), ("context_hit", "context hit"),
            ("citation_hit", "citation hit")]
    L += _table(["config"] + [t for _, t in cols],
                [[c] + [_cell(res["configs"][c]["metrics"][m]) if m in res["configs"][c]["metrics"] else "-"
                        for m, _ in cols] for c in configs])
    L.append("")
    L += ["## Abstentions and refusals", "",
          "Re-derived from the answer text. Explicit abstention: the reply is exactly INSUFFICIENT. Prose refusal: "
          "a refusal phrasing (\"the passages do not mention ...\", \"I cannot answer\", ...) with no citation. "
          "Neither is judged; both score as incorrect.", ""]
    L += _table(["config", "answers", "explicit abstentions", "prose refusals"],
                [[c, str(v["n"]), str(v["explicit_abstentions"]), str(v["prose_refusals"])]
                 for c, v in res["configs"].items()])
    L.append("")
    L += ["## Citation behaviour", "",
          "Bracket groups are re-parsed from the answer text against the k retrieved passages; a group with any "
          "number that is 0 or above k is one invalid citation and none of its numbers count. `cites anything` "
          "is over non-refused answers only.", ""]
    L += _table(["config", "answered", "cites anything", "invalid citations (total)", "answers with an invalid citation"],
                [[c, str(v["n_answered"]), _f(v["cites_anything"]), str(v["invalid_citations"]),
                  str(v["answers_with_invalid_citations"])]
                 for c, v in res["configs"].items() if c != CLOSED_BOOK])
    L.append("")

    def diffs(title: str, note: str, rows: list[tuple[str, dict]], metrics) -> None:
        if not rows:
            return
        L.extend([f"## {title}", "", note, ""])
        L.extend(_table(["comparison"] + list(metrics),
                        [[name] + [f"{d[m]['diff']:+.3f} [{d[m]['lo']:+.3f}, {d[m]['hi']:+.3f}] ({d[m]['frac_le_0']:.3f})"
                                   for m in metrics] for name, d in rows]))
        L.append("")

    n_cmp = len(res["paired_vs_closed_book"]) + bool(res["bm25_vs_dense"])
    note = ("diff [95% CI] and the fraction of resamples with diff <= 0. Same resampled papers for both configs. "
            f"Not corrected for multiple comparisons ({n_cmp} comparisons).")
    diffs("Paired difference vs closed-book", note,
          [(f"{c} - {CLOSED_BOOK}", d) for c, d in res["paired_vs_closed_book"].items()], PAIRED_VS_CB)
    bd = res["bm25_vs_dense"]
    if bd:
        diffs("Paired difference: BM25 vs dense", note, [(f"{bd['a']} - {bd['b']}", bd["metrics"])],
              [m for m in PAIRED_BM25_DENSE if m in bd["metrics"]])

    if res["by_context_hit"]:
        L += ["## Does retrieval failure explain wrong answers?", "",
              "Correct rate split by whether any retrieved chunk hit the gold evidence. Point estimates; the "
              "`no context_hit` group is small when retrieval is good. A high `correct` there means the model "
              "answered from other passages or its own knowledge. `abstained` here includes prose refusals. "
              "`context_hit` uses the Phase 2/3 single-chunk 50% rule, so a gold span longer than one chunk "
              "(mostly `cross_paragraph`) can never be a context hit: the `no context_hit` group therefore "
              "includes some answerable questions.", ""]
        gs = res.get("gold_spans")
        if gs:
            L += [f"Longest retrieved chunk: {gs['max_chunk_span']} characters. Gold spans longer than that: "
                  f"{gs['longer_than_chunk']} of {gs['n']} scored questions; of these, {gs['unhittable']} are more "
                  "than twice as long and cannot reach the 50% coverage a hit needs.", ""]
        L += _table(["config", "group", "n", "correct", "correct or partial", "abstained"],
                    [[c, g, str(v["n"]), _f(v["correct"]), _f(v["correct_or_partial"]), _f(v["abstained"])]
                     for c, groups in res["by_context_hit"].items() for g, v in groups.items()])
        L.append("")
    L += ["## By span type", "", "Point estimates only; small groups are noisy.", ""]
    L += _table(["span type", "n", "config", "correct", "context hit"],
                [[t, str(v["n"]), c, _f(e.get("correct")), _f(e.get("context_hit"))]
                 for t, v in res["by_span_type"].items() for c, e in v["configs"].items()])
    L.append("")
    tk = report["tokens"]
    L += ["## Token use (per call, for planning)", ""]
    L += _table(["call type", "calls", "mean prompt tokens", "mean completion tokens"],
                [[name, str(s["calls"]), _f(s["mean_prompt_tokens"]), _f(s["mean_completion_tokens"])]
                 for name, s in tk.items()])
    L += ["", "## Caveats", "",
          "- Correctness and faithfulness are judged by one LLM (single judge, not human-checked) with a single "
          "generator model; both can share blind spots. The judge is a different family from the generator.",
          "- The gold set is machine-verified, not human-reviewed (see eval/README.md), and biased towards named, "
          "factual passages; absolute numbers are optimistic.",
          "- `correct` is evidence-relative: the judge sees the gold evidence quote, so an answer that is right "
          "but says something other than the quote provides can be marked partial or no. `faithful` ignores the "
          "reference and checks only the cited passages.",
          "- The k retrieved passages may contain the right information in a chunk other than the gold span "
          "(another paper or a duplicate statement); `context hit` is stricter than 'the answer is available'.",
          "- Abstentions and prose refusals count as incorrect and are excluded from the faithfulness rates; a "
          "config that refuses more can therefore look more faithful. Read the two together. The prose-refusal "
          "list is deliberately conservative, so some refusals phrased differently are still judged as answers.",
          "- The headline `faithful` forces uncited answers to unfaithful; `faithful (cited)` does not. A large gap "
          "between them means the model often answers without citing.",
          "- `context hit` needs one chunk to cover 50% of the gold span; spans longer than a chunk can never "
          "be a hit even when the answer is fully retrievable (see the span counts above).",
          "- Paired comparisons and breakdowns are not corrected for multiple comparisons.", ""]
    return "\n".join(L)


def write_outputs(report: dict, out: Path) -> tuple[Path, Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(out) + ".json"), Path(str(out) + ".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                   help="configs to score; rows of other configs in the answers file are ignored")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--rpm", type=float, default=4, help="max judge requests per minute")
    p.add_argument("--judge-out", type=Path, default=DEFAULT_JUDGEMENTS)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output prefix; writes <out>.json and <out>.md")
    p.add_argument("--no-judge", action="store_true", help="no judge calls; use cached verdicts only")
    p.add_argument("--no-json-mode", action="store_true", help="omit response_format (if the judge model rejects it)")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--ids-file", type=Path, default=DEFAULT_IDS_FILE)
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    args = p.parse_args()
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be >= 1")
    if not args.answers.exists():
        raise SystemExit(f"{args.answers} not found; run arxiv_rag.generate first")

    ids = read_ids_file(args.ids_file)
    if not ids:
        raise SystemExit(f"{args.ids_file} contains no ids")
    try:
        docs = load_documents(args.data_dir, ids_to_stems(ids))
        gold = load_gold(args.gold, {d.paper_id: d for d in docs})
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    by_id = {d.paper_id: d for d in docs}
    shas = {d.paper_id: sha256_text(d.text) for d in docs}
    try:
        wanted = list(dict.fromkeys(parse_config(c)[0] for c in args.configs))
    except ValueError as exc:
        raise SystemExit(str(exc))
    pop = load_population(args.answers, gold.items, shas, wanted)
    if not pop["items"]:
        raise SystemExit("no question has an answer under every config in the answers file")
    log(f"{len(pop['items'])} questions x {len(pop['configs'])} configs {pop['dropped']}")

    judgements = load_judgements(args.judge_out, args.judge_model)
    entries = [(pop["rows"][c][g.qid], g) for c in pop["configs"] for g in pop["items"]]
    if not args.no_judge:
        n_todo = len(needs_judge(entries, judgements))
        if n_todo:
            log(f"judging {n_todo} answers with {args.judge_model} at {args.rpm} rpm")
            load_dotenv()
            client = UsageClient.from_env(args.rpm, model=args.judge_model)
            ask = lambda s, u: client.ask(s, u, json_mode=not args.no_json_mode)  # noqa: E731
            with Lock(args.judge_out):
                judgements = load_judgements(args.judge_out, args.judge_model)  # read after taking the lock
                try:
                    log(str(judge_answers(entries, by_id, ask, client.model, args.judge_out, judgements)))
                except KeyboardInterrupt:
                    log("\ninterrupted (all verdicts so far are saved; rerun to resume)")
                except RuntimeError as exc:  # QuotaExhausted, or too many consecutive failures
                    log(f"{exc} (verdicts so far are saved)")
    scored = {(g.qid, c) for c in pop["configs"] for g in pop["items"]}
    judge_usage = [r for r in judgements.values() if (r["qid"], r["config"]) in scored and not r.get("judge_error")]

    results = evaluate(pop, judgements, args.bootstrap, args.seed)
    gen_rows = [r for c in pop["configs"] for r in pop["rows"][c].values()]
    report = {
        "config": {"answers": rel_path(args.answers), "judge_out": rel_path(args.judge_out), "gold": rel_path(args.gold),
                   "judge_model": args.judge_model, "max_tokens": args.max_tokens, "overlap": args.overlap,
                   "configs": pop["configs"], "bootstrap": args.bootstrap, "seed": args.seed,
                   "hit_coverage": HIT_COVERAGE},
        "partial": len(pop["items"]) < len(gold.items),
        "dropped": pop["dropped"],
        "n_gold": len(gold.items),
        "n_questions": len(pop["items"]),
        "n_papers": len(docs),
        "n_gold_papers": len({g.paper_id for g in pop["items"]}),
        "generator_models": sorted({r["model"] for r in gen_rows if isinstance(r.get("model"), str)}),
        "gold_status_counts": dict(Counter(g.status for g in pop["items"])),
        "git_commit": git_sha(),
        "git_dirty": git_dirty(),
        "tokens": {**{f"generation {c}": token_stats(list(pop["rows"][c].values())) for c in pop["configs"]},
                   "judge": token_stats(judge_usage)},
        "results": results,
    }
    json_path, md_path = write_outputs(report, args.out)
    log(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
