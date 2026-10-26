"""Phase 4a: answer the gold questions with citations, closed-book or from retrieved chunks.

Design goals:
- Same inputs as Phase 3: gold questions, structure chunks (256 tokens), DenseRetriever / BM25Retriever.
- Fair prompts: passages are numbered [1]..[k] and carry NO paper title or id, only chunk text.
- Abstention is explicit: the model must reply exactly "INSUFFICIENT" when it cannot answer; that is the
  only abstention signal (closed-book uses it too). The eval re-derives it from the answer text.
- Budget-aware (Groq free tier: ~1000 requests/day, ~8000 tokens/minute): one append-only JSONL row per
  (qid, doc_sha256, config), resumable, tolerant of a truncated last line, clean stop on QuotaExhausted.
- Partial runs stay useful: configs are interleaved PER QUESTION (all configs of question 1, then of 2, ...),
  so any prefix of the run is a paired comparison on the same questions.
- Testable: the LLM is an injected callable (system, user) -> (text, usage); nothing here needs the network.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

from arxiv_rag import goldgen
from arxiv_rag.chunk import Chunk, chunk_document
from arxiv_rag.document import ids_to_stems, load_documents
from arxiv_rag.embed import Embedder
from arxiv_rag.eval_chunking import (
    DEFAULT_GOLD, DEFAULT_IDS_FILE, GoldItem, _CachedQueries, _warm, load_gold, log, sha256_text,
)
from arxiv_rag.goldgen import LLMClient, MalformedReply, QuotaExhausted, load_dotenv, open_append
from arxiv_rag.ingest import read_ids_file
from arxiv_rag.judge import strip_think
from arxiv_rag.lexical import BM25Retriever
from arxiv_rag.retrieve import DenseRetriever
from arxiv_rag.review import Lock

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANSWERS = _ROOT / "eval" / "answers_v1.jsonl"
DEFAULT_MODEL = goldgen.DEFAULT_MODEL
DEFAULT_CONFIGS = ("closed-book", "dense-k5", "bm25-k5")
CLOSED_BOOK = "closed-book"
ABSTAIN = "INSUFFICIENT"
MAX_RANGE = 50  # a citation range wider than this is garbage, not a range

CLOSED_SYSTEM = (
    "Answer the question briefly, in 1-3 sentences, from your own knowledge. "
    f"If you cannot answer it, reply exactly {ABSTAIN} and nothing else."
)
RAG_SYSTEM = (
    "Answer the question in 1-3 sentences using only the numbered passages. Cite every claim with the "
    "number of its passage in square brackets, e.g. [1] or [2][3]. "
    f"If the passages do not contain the answer, reply exactly {ABSTAIN} and nothing else."
)


# --- configs, prompts, parsing -----------------------------------------------------------

_CFG = re.compile(r"(dense|bm25)(?:-k(\d+))?")


def parse_config(name: str, default_k: int = 5) -> tuple[str, str, int]:
    """(canonical name, retriever kind or "closed-book", k). Bare "dense" means dense-k<default_k>."""
    if name == CLOSED_BOOK:
        return name, CLOSED_BOOK, 0
    m = _CFG.fullmatch(name)
    if not m or (m.group(2) is not None and int(m.group(2)) < 1) or (m.group(2) is None and default_k < 1):
        raise ValueError(f"unknown config {name!r}; expected closed-book, dense[-kN] or bm25[-kN]")
    k = int(m.group(2)) if m.group(2) else default_k
    return f"{m.group(1)}-k{k}", m.group(1), k


def build_user_prompt(question: str, passages: list[str]) -> str:
    """Numbered passages (chunk text only: no title, no id), then the question."""
    if not passages:
        return f"Question: {question}"
    body = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return f"Passages:\n{body}\n\nQuestion: {question}"


_DASH = "–—‐‑-"
_BRACKET = re.compile(rf"[\[【]\s*(?:passages?\s+)?(\d+(?:\s*(?:[,;，、{_DASH}])\s*\d+)*\s*)[\]】]", re.I)
_NUM_OR_RANGE = re.compile(rf"(\d+)(?:\s*[{_DASH}]\s*(\d+))?")


def parse_citations(answer: str, n_passages: int) -> tuple[list[int], int]:
    """([sorted unique cited ints in 1..n_passages], number of invalid bracket groups).

    Understands [n], [n, m], [n-m] (ranges expand), [Passage n] and the full-width 【n】 some models emit. A bracket group is all-or-nothing: if any
    number in it is 0 or above n_passages (or a range is reversed or absurdly wide) the whole group is one
    invalid citation and none of its numbers count. With no passages every citation is ignored.
    """
    valid: set[int] = set()
    invalid = 0
    if n_passages <= 0:
        return [], 0
    for m in _BRACKET.finditer(answer):
        group: set[int] = set()
        ok = True
        for part in re.split(r"[,;，、]", m.group(1)):
            r = _NUM_OR_RANGE.fullmatch(part.strip())
            if not r:
                ok = False
                break
            lo = int(r.group(1))
            hi = int(r.group(2)) if r.group(2) else lo
            if hi < lo or hi - lo > MAX_RANGE or lo < 1 or hi > n_passages:
                ok = False
                break
            group.update(range(lo, hi + 1))
        if ok:
            valid |= group
        else:
            invalid += 1
    return sorted(valid), invalid


_ABSTAIN_RE = re.compile(rf"[\W_]*{ABSTAIN}[\W_]*", re.I)


def is_abstention(answer: str) -> bool:
    """The whole reply is the INSUFFICIENT token, ignoring case, quotes and punctuation (strict)."""
    return _ABSTAIN_RE.fullmatch(answer.strip()) is not None


# Conservative: phrasings that say "the answer is not in the passages" in prose instead of INSUFFICIENT.
_REFUSAL_RES = [re.compile(p, re.I) for p in (
    r"\bpassages?\s+(?:do(?:es)?\s+not|don't|doesn't|did\s+not)\s+(?:say|mention|provide|contain|state|specify)",
    r"\bnot\s+(?:mentioned|stated|specified|provided)\s+in\s+the\s+(?:given\s+|provided\s+)?passages?",
    r"\bcannot\s+be\s+determined\s+from",
    r"\bno\s+information\s+(?:about|on)\b",
    r"\bI\s+(?:cannot|can't|can\s+not|don't|do\s+not)\s+(?:answer|know)\b",
)]


def is_prose_refusal(answer: str) -> bool:
    """A refusal in prose: a known refusal phrasing and no citation (a cited answer is a real attempt)."""
    if _BRACKET.search(answer):
        return False
    return any(r.search(answer) for r in _REFUSAL_RES)


def clean_answer(raw: str) -> str:
    text = " ".join(strip_think(raw).split())
    if not text:
        raise MalformedReply("empty answer")
    return text


# --- LLM ---------------------------------------------------------------------------------

class UsageClient(LLMClient):
    """LLMClient that also returns the token usage of the call (read from LLMClient.last_usage)."""

    def ask(self, system: str, user: str, json_mode: bool = False) -> tuple[str, dict]:
        text = self.chat(system, user, json_mode=json_mode, temperature=0.0)
        return text, dict(self.last_usage or {})


# --- resilient loop ------------------------------------------------------------------------

MAX_CONSECUTIVE_FAILURES = 5


def run_resilient(todo: list, step: Callable, on_final_failure: Callable | None = None) -> dict[str, int]:
    """Run step(entry) for every entry; a RuntimeError fails that entry only.

    QuotaExhausted and KeyboardInterrupt propagate. A failed entry is retried once after the first pass;
    `on_final_failure(entry, exc)` is called when the retry fails too. The run aborts (RuntimeError) after
    MAX_CONSECUTIVE_FAILURES failures in a row. Returns {"failed": entries still failed, "retried": retries made}.
    """
    consecutive = 0
    queue: list = []
    stats = {"failed": 0, "retried": 0}

    def attempt(entry) -> Exception | None:
        nonlocal consecutive
        try:
            step(entry)
        except QuotaExhausted:
            raise
        except RuntimeError as exc:
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"aborting after {consecutive} consecutive failures; last: "
                                   f"{type(exc).__name__}: {exc}") from exc
            return exc
        consecutive = 0
        return None

    for entry in todo:
        exc = attempt(entry)
        if exc is not None:
            log(f"failed ({type(exc).__name__}); will retry once later")
            queue.append(entry)
    for entry in queue:
        stats["retried"] += 1
        exc = attempt(entry)
        if exc is not None:
            stats["failed"] += 1
            log(f"failed again ({type(exc).__name__}); left for the next run")
            if on_final_failure:
                on_final_failure(entry, exc)
    return stats


# --- answers file ------------------------------------------------------------------------

def load_answers(path: Path) -> list[dict]:
    """Valid answer rows in file order; undecodable lines (e.g. a truncated write) are skipped with a warning."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if not all(isinstance(r[k], str) for k in ("qid", "doc_sha256", "config")):
                    raise KeyError("bad key fields")
            except (ValueError, KeyError, TypeError):
                log(f"WARNING: {path}:{n} is not a valid answer row (truncated write?); ignoring it")
                continue
            rows.append(r)
    return rows


def done_keys(path: Path) -> set[tuple[str, str, str]]:
    return {(r["qid"], r["doc_sha256"], r["config"]) for r in load_answers(path)}


Searcher = Callable[[str, int], list[Chunk]]
Ask = Callable[[str, str], tuple[str, dict]]


def pending(items: list[GoldItem], shas: dict[str, str], configs: list[tuple[str, str, int]],
            done: set[tuple[str, str, str]]) -> list[tuple[GoldItem, tuple[str, str, int]]]:
    """(item, config) pairs still to answer, interleaved per question."""
    return [(it, c) for it in items for c in configs if (it.qid, shas[it.paper_id], c[0]) not in done]


def generate_answers(items: list[GoldItem], shas: dict[str, str], configs: list[tuple[str, str, int]],
                     searchers: dict[str, Searcher], ask: Ask, model: str, out: Path) -> dict[str, int]:
    """Append one row per pending (question, config); QuotaExhausted propagates after the rows so far are saved.

    `searchers` maps retriever kind ("dense", "bm25") -> search(query, k) -> chunks, best first.
    Malformed (empty) replies are counted and retried next run.
    """
    todo = pending(items, shas, configs, done_keys(out))
    counts = {"written": 0, "abstained": 0, "malformed": 0, "failed": 0,
              "skipped_existing": len(items) * len(configs) - len(todo)}
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = len(todo)
    with open_append(out) as f:
        def step(entry) -> None:
            it, (name, kind, k) = entry
            chunks = searchers[kind](it.question, k)[:k] if kind != CLOSED_BOOK else []
            system = CLOSED_SYSTEM if kind == CLOSED_BOOK else RAG_SYSTEM
            try:
                raw, usage = ask(system, build_user_prompt(it.question, [c.text for c in chunks]))
                answer = clean_answer(raw)
            except MalformedReply as exc:
                log(f"malformed {it.qid} [{name}]: {exc}")
                counts["malformed"] += 1
                return
            cites, invalid = parse_citations(answer, len(chunks))
            abstained = is_abstention(answer)
            row = {"qid": it.qid, "doc_sha256": shas[it.paper_id], "config": name, "model": model,
                   "question": it.question, "answer": answer, "citations": cites, "invalid_citations": invalid,
                   "retrieved": [{"rank": r, "chunk_id": c.chunk_id, "paper_id": c.paper_id,
                                  "char_start": c.char_start, "char_end": c.char_end}
                                 for r, c in enumerate(chunks, 1)],
                   "abstained": abstained, "prompt_tokens": usage.get("prompt_tokens"),
                   "completion_tokens": usage.get("completion_tokens")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            counts["written"] += 1
            counts["abstained"] += abstained
            log(f"[{counts['written']}/{total}] {it.qid} {name}: {'ABSTAIN' if abstained else 'answered'} "
                f"cites={cites} tokens={usage.get('prompt_tokens')}+{usage.get('completion_tokens')} "
                f"({time.time() - t0:.0f}s elapsed)")

        counts["failed"] = run_resilient(todo, step)["failed"]
    return counts


def build_searchers(docs, embedder, kinds: set[str], max_tokens: int, overlap: int) -> dict[str, Searcher]:
    """Dense / BM25 retrievers over the structure chunks, built as in eval_retrieval."""
    if max_tokens > embedder.max_tokens:
        raise ValueError(f"--max-tokens {max_tokens} exceeds the embedding model's limit {embedder.max_tokens}")
    chunks = [c for d in docs for c in chunk_document(d, "structure", embedder.count_tokens, max_tokens, overlap)]
    log(f"[structure] {len(chunks)} chunks; embedding (cached)")
    retrievers = {}
    if "dense" in kinds:
        _warm(embedder, chunks, "structure")
        retrievers["dense"] = DenseRetriever(_CachedQueries(embedder), chunks)
    if "bm25" in kinds:
        retrievers["bm25"] = BM25Retriever(chunks)
    return {kind: (lambda q, k, r=r: [c for c, _ in r.search(q, k)]) for kind, r in retrievers.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                   help="closed-book, dense[-kN], bm25[-kN]; a bare dense/bm25 uses --k")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--k", type=int, default=5, help="passages for configs given without -kN")
    p.add_argument("--rpm", type=float, default=4, help="max LLM requests per minute (~1.7k tokens/call)")
    p.add_argument("--limit", type=int, default=None, help="answer only the first N gold questions (smoke test)")
    p.add_argument("--out", type=Path, default=DEFAULT_ANSWERS)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--ids-file", type=Path, default=DEFAULT_IDS_FILE)
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    try:
        configs = list(dict.fromkeys(parse_config(c, args.k) for c in args.configs))
    except ValueError as exc:
        raise SystemExit(str(exc))

    ids = read_ids_file(args.ids_file)
    if not ids:
        raise SystemExit(f"{args.ids_file} contains no ids")
    try:
        docs = load_documents(args.data_dir, ids_to_stems(ids))
        gold = load_gold(args.gold, {d.paper_id: d for d in docs})
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    items = gold.items[:args.limit] if args.limit else gold.items
    shas = {d.paper_id: sha256_text(d.text) for d in docs}
    todo = pending(items, shas, configs, done_keys(args.out))
    log(f"{len(items)} questions x {len(configs)} configs = {len(items) * len(configs)} answers; "
        f"{len(todo)} pending, model {args.model}, rpm {args.rpm}")
    if not todo:
        return

    kinds = {kind for _, kind, _ in configs if kind != CLOSED_BOOK}
    searchers = build_searchers(docs, Embedder(), kinds, args.max_tokens, args.overlap) if kinds else {}
    load_dotenv()
    client = UsageClient.from_env(args.rpm, model=args.model)
    with Lock(args.out):  # one writer at a time
        try:
            counts = generate_answers(items, shas, configs, searchers, client.ask, client.model, args.out)
        except KeyboardInterrupt:
            raise SystemExit("\ninterrupted (all answers so far are saved; rerun to resume)")
        except RuntimeError as exc:  # QuotaExhausted, or too many consecutive failures
            raise SystemExit(f"{exc} (answers so far are saved in {args.out})")
    log(f"done: {counts}")


if __name__ == "__main__":
    main()
