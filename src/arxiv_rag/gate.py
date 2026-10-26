"""Phase 5 CI regression gate: re-score retrieval on the pinned gold set and fail on a quality drop.

Design goals:
- No LLM calls, no API keys, no bootstrap: only point estimates of recall@1/@5/@10 and MRR@20, computed with
  the same chunking, retriever construction, hit rule and `score_ranking` as Phases 2 and 3.
- Exit codes tell a quality failure from a broken setup: 0 = pass, 1 = a metric dropped (or the chunk / question
  count changed), 2 = the gate cannot give a verdict (stale gold, missing data or baseline, mismatched setup).
- Stale gold is loud: a gold row whose doc sha or quote no longer matches means parse.py / document.py changed
  the canonical text. Scoring the remaining rows would hide that, so the gate stops instead.
- Tolerance per retriever, stored in the baseline: BM25 is deterministic (0.0, any drop fails and any change is
  reported); dense gets 0.01 for CPU/BLAS nondeterminism. A metric rising above baseline passes but flags the
  baseline as stale so the improvement gets locked in with --update-baseline.
- A silent corpus change cannot pass: chunk count and gold question count must equal the baseline.
- Testable: the embedder is injected (tests use a fake); the data directory and ids file are arguments.
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from arxiv_rag.chunk import STRATEGIES, Chunk, chunk_document
from arxiv_rag.document import Document, ids_to_stems, load_documents
from arxiv_rag.embed import Embedder
from arxiv_rag.eval_chunking import (
    DEFAULT_GOLD, DEFAULT_IDS_FILE, K_MAX, GoldItem, _CachedQueries, _table, _warm, git_sha, is_hit, load_gold,
    log, score_ranking,
)
from arxiv_rag.eval_retrieval import build_configs
from arxiv_rag.ingest import read_ids_file

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = _ROOT / "eval" / "gate_baseline.json"

GATE_RETRIEVERS = ("bm25", "dense")
DEFAULT_TOLERANCE = {"bm25": 0.0, "dense": 0.01}
METRICS = ("recall@1", "recall@5", "recall@10", f"mrr@{K_MAX}")
SCHEMA = 1
EPS = 1e-6  # baseline values are rounded to 6 decimals; far below one question (1/192)
EXIT_OK, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2
STALE_HINT = "baseline is stale; run with --update-baseline and commit"


class GateError(Exception):
    """The gate cannot give a verdict (exit 2): stale gold, missing data or baseline, mismatched setup."""


# --- measuring ---------------------------------------------------------------------------

def check_gold(counts: dict[str, int], gold_path: Path) -> None:
    """Raise GateError if load_gold had to drop stale rows (the canonical document text changed)."""
    if counts.get("stale"):
        raise GateError(
            f"{counts['stale']} gold rows in {gold_path} are STALE (doc_sha256 or quote offsets no longer match the "
            "parsed documents). parse.py or document.py changed the canonical document text. Either revert that "
            "change, or regenerate the gold set (see eval/README.md) and then run --update-baseline.")


def measure(docs: list[Document], gold: list[GoldItem], embedder, chunking: str, names: list[str],
            max_tokens: int, overlap: int) -> dict:
    """Chunk the corpus and score each retriever. Returns {"chunks", "gold_questions", "retrievers"}."""
    unknown = [n for n in names if n not in GATE_RETRIEVERS]
    if unknown:
        raise GateError(f"unknown retrievers {unknown}; expected some of {GATE_RETRIEVERS}")
    if max_tokens > embedder.max_tokens:
        raise GateError(f"--max-tokens {max_tokens} exceeds the embedding model's limit {embedder.max_tokens}")
    chunks: list[Chunk] = [c for d in docs for c in chunk_document(d, chunking, embedder.count_tokens, max_tokens, overlap)]
    log(f"[{chunking}] {len(chunks)} chunks")
    if "dense" in names:
        _warm(embedder, chunks, chunking)
    configs = build_configs(chunks, _CachedQueries(embedder), None, names, False, 0)
    out: dict[str, dict[str, float]] = {}
    for name, retriever in configs.items():
        rows = []
        for item in gold:
            ranked = [c for c, _ in retriever.search(item.question, K_MAX)]
            rows.append(score_ranking([is_hit(c, item) for c in ranked],
                                      [c.paper_id == item.paper_id for c in ranked], [0] * len(ranked)))
        out[name] = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
    return {"chunks": len(chunks), "gold_questions": len(gold), "retrievers": out}


# --- comparing ---------------------------------------------------------------------------

@dataclass
class Row:
    name: str
    baseline: float
    current: float
    status: str  # OK | WITHIN TOLERANCE | IMPROVED | FAIL
    is_count: bool = False

    @property
    def delta(self) -> float:
        return self.current - self.baseline


def judge(baseline: float, current: float, tolerance: float) -> str:
    if current < baseline - tolerance - EPS:
        return "FAIL"
    if current > baseline + tolerance + EPS:
        return "IMPROVED"
    return "OK" if abs(current - baseline) <= EPS else "WITHIN TOLERANCE"


def compare(baseline: dict, measured: dict, chunking: str, names: list[str]) -> list[Row]:
    """Rows for the counts and every (retriever, metric). Raises GateError if the baseline lacks an entry."""
    entry = baseline.get("chunkings", {}).get(chunking)
    if entry is None:
        raise GateError(f"the baseline has no entry for chunking {chunking!r}; run with --update-baseline")
    rows = [Row("gold questions", entry["gold_questions"], measured["gold_questions"],
                "OK" if entry["gold_questions"] == measured["gold_questions"] else "FAIL", True),
            Row("chunks", entry["chunks"], measured["chunks"],
                "OK" if entry["chunks"] == measured["chunks"] else "FAIL", True)]
    for name in names:
        b = entry.get("retrievers", {}).get(name)
        if b is None:
            raise GateError(f"the baseline has no entry for retriever {name!r} on {chunking!r}; "
                            f"run with --update-baseline --retrievers {name}")
        for m in METRICS:
            if m not in b["metrics"]:
                raise GateError(f"the baseline lacks {name} {m}; run with --update-baseline")
            cur = measured["retrievers"][name][m]
            rows.append(Row(f"{name} {m}", b["metrics"][m], cur, judge(b["metrics"][m], cur, b["tolerance"])))
    return rows


def render_summary(rows: list[Row], tolerances: dict[str, float], chunking: str, verdict: str) -> str:
    cells = []
    for r in rows:
        if r.is_count:
            cells.append([r.name, f"{r.baseline:.0f}", f"{r.current:.0f}", f"{r.delta:+.0f}", r.status])
        else:
            tol = tolerances[r.name.split()[0]]
            cells.append([r.name, f"{r.baseline:.4f}", f"{r.current:.4f}", f"{r.delta:+.4f}",
                          r.status + (f" (tol {tol})" if r.status in ("WITHIN TOLERANCE", "FAIL") and tol else "")])
    L = [f"## Retrieval regression gate: {verdict}", "", f"Chunking: `{chunking}`.", ""]
    L += _table(["metric", "baseline", "current", "delta", "status"], cells)
    L.append("")
    if any(r.status == "FAIL" for r in rows):
        L += ["**A metric dropped below its baseline tolerance, or the chunk / question count changed.**", ""]
    if any(r.status == "IMPROVED" for r in rows):
        L += [f"Note: a metric rose above the baseline; {STALE_HINT}.", ""]
    return "\n".join(L)


# --- baseline file -----------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def updated_baseline(old: dict | None, measured: dict, chunking: str, names: list[str], max_tokens: int,
                     overlap: int, meta: dict) -> dict:
    """Merge the measured retrievers into `old`. Retrievers not re-measured are kept only while the chunk and
    question counts still match; otherwise they would describe a different corpus and are dropped."""
    base = {"schema": SCHEMA, "meta": meta, "chunkings": dict((old or {}).get("chunkings", {}))}
    prev = base["chunkings"].get(chunking, {})
    keep = {}
    if prev.get("chunks") == measured["chunks"] and prev.get("gold_questions") == measured["gold_questions"] \
            and prev.get("max_tokens") == max_tokens and prev.get("overlap") == overlap:
        keep = dict(prev.get("retrievers", {}))
    for n in names:
        tol = keep.get(n, {}).get("tolerance", DEFAULT_TOLERANCE[n])
        keep[n] = {"tolerance": tol, "metrics": {m: round(v, 6) for m, v in measured["retrievers"][n].items()}}
    base["chunkings"][chunking] = {"max_tokens": max_tokens, "overlap": overlap, "chunks": measured["chunks"],
                                   "gold_questions": measured["gold_questions"], "retrievers": keep}
    return base


def load_baseline(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateError(f"cannot read baseline {path}: {exc}; create it with --update-baseline")
    except ValueError as exc:
        raise GateError(f"baseline {path} is not valid JSON: {exc}")
    if data.get("schema") != SCHEMA:
        raise GateError(f"baseline {path} has schema {data.get('schema')!r}, expected {SCHEMA}")
    return data


# --- CLI ---------------------------------------------------------------------------------

def run(args: argparse.Namespace, embedder=None) -> int:
    ids = read_ids_file(args.ids_file)
    if not ids:
        raise GateError(f"{args.ids_file} contains no ids")
    try:
        docs = load_documents(args.data_dir, ids_to_stems(ids))
    except FileNotFoundError as exc:
        raise GateError(f"{exc}. Fetch the corpus: python -m arxiv_rag.ingest --ids-file {args.ids_file}, "
                        "then python -m arxiv_rag.parse")
    try:
        loaded = load_gold(args.gold, {d.paper_id: d for d in docs})
    except ValueError as exc:  # no usable rows at all; typically every row went stale
        raise GateError(f"{exc}. If the rows are stale, parse.py or document.py changed the canonical document "
                        "text: revert that change or regenerate the gold set (see eval/README.md).")
    except OSError as exc:
        raise GateError(str(exc))
    check_gold(loaded.counts, args.gold)
    log(f"{len(docs)} papers, {len(loaded.items)} questions {loaded.counts}")

    embedder = embedder or Embedder()
    t0 = time.time()
    measured = measure(docs, loaded.items, embedder, args.chunking, args.retrievers, args.max_tokens, args.overlap)
    log(f"measured in {time.time() - t0:.0f}s")

    if args.update_baseline:
        old = None
        if args.baseline.exists():
            old = load_baseline(args.baseline)
        meta = {"git_commit": git_sha(), "date": date.today().isoformat(), "gold": _rel(args.gold),
                "corpus_sha256": sha256_file(args.ids_file)}
        new = updated_baseline(old, measured, args.chunking, args.retrievers, args.max_tokens, args.overlap, meta)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8", newline="\n")
        log(f"wrote baseline {args.baseline}")
        for n in args.retrievers:
            log(f"  {n}: " + ", ".join(f"{m} {v:.4f}" for m, v in measured["retrievers"][n].items()))
        return EXIT_OK

    baseline = load_baseline(args.baseline)
    entry = baseline.get("chunkings", {}).get(args.chunking, {})
    for key, val in (("max_tokens", args.max_tokens), ("overlap", args.overlap)):
        if entry and entry.get(key) != val:
            raise GateError(f"--{key.replace('_', '-')} {val} differs from the baseline's {entry.get(key)}; "
                            "run with --update-baseline if the change is intended")
    rows = compare(baseline, measured, args.chunking, args.retrievers)
    tolerances = {n: entry["retrievers"][n]["tolerance"] for n in args.retrievers}
    failed = any(r.status == "FAIL" for r in rows)
    text = render_summary(rows, tolerances, args.chunking, "FAILED" if failed else "passed")
    corpus_sha = sha256_file(args.ids_file)
    if baseline.get("meta", {}).get("corpus_sha256") not in (None, corpus_sha):
        text += f"\nNote: {args.ids_file.name} differs from the file the baseline was made with ({STALE_HINT} if intended).\n"
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    return EXIT_REGRESSION if failed else EXIT_OK


def main(argv: list[str] | None = None, embedder=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--retrievers", nargs="+", default=["bm25"], choices=GATE_RETRIEVERS)
    p.add_argument("--chunking", default="structure", choices=STRATEGIES)
    p.add_argument("--update-baseline", action="store_true", help="write the current values to --baseline and exit 0")
    p.add_argument("--summary", type=Path, default=None, help="append a Markdown table here ($GITHUB_STEP_SUMMARY)")
    p.add_argument("--ids-file", type=Path, default=DEFAULT_IDS_FILE)
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    args = p.parse_args(argv)
    try:
        return run(args, embedder)
    except GateError as exc:
        print(f"GATE ERROR (exit {EXIT_ERROR}, no verdict): {exc}", file=sys.stderr, flush=True)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
