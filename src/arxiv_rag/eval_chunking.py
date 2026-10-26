"""Phase 2 eval: dense-retrieval recall per chunking strategy over the pinned corpus.

Design goals:
- Honest relevance: a retrieved chunk is a hit iff it is from the gold paper and its
  character span covers >= HIT_COVERAGE of the evidence span. Spans, not chunk text, are
  compared, so the structure+header prefix can neither help nor hurt the match.
- Pinned gold: rows whose doc_sha256 or quote offsets no longer match the parsed
  document are dropped and counted (eval/README.md), never silently scored.
- Same questions for every strategy, same bootstrap resamples for every strategy, so
  strategy differences are paired and the CIs on them are meaningful. Questions from one paper
  are not independent, so the bootstrap resamples PAPERS (all their questions come along).
- Cheap to re-run: passage vectors come from the embedder's disk cache; the first run
  prints progress while it embeds.
- Testable: the embedder is injected (tests use a fake); nothing here calls an LLM.

Metrics are per-question values in [0, 1]; every CI is a percentile cluster bootstrap of a mean.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from arxiv_rag.chunk import STRATEGIES, Chunk, chunk_document
from arxiv_rag.document import Document, ids_to_stems, load_documents
from arxiv_rag.embed import Embedder
from arxiv_rag.ingest import read_ids_file
from arxiv_rag.retrieve import DenseRetriever
from arxiv_rag.review import EXPORT_STATUSES

_ROOT = Path(__file__).resolve().parents[2]  # repo root, independent of cwd
DEFAULT_GOLD = _ROOT / "eval" / "gold_v1.jsonl"
DEFAULT_IDS_FILE = _ROOT / "eval" / "corpus_v1.txt"
DEFAULT_OUT = _ROOT / "docs" / "results" / "phase2"

K_MAX = 20
RECALL_KS = (1, 3, 5, 10, 20)
PAPER_KS = (1, 5, 10)
BUDGETS = (512, 1024, 2048)  # tokens of retrieved context
HIT_COVERAGE = 0.5  # fraction of the evidence span's characters a chunk must cover
PAIRED_METRICS = ("recall@5", "recall@10", "recall@1024tok")
SENSITIVITY_METRICS = PAIRED_METRICS
NON_COPY = "non-copy (judge.copy == false)"
BASELINE = "fixed"
KEY_METRICS = ("recall@1", "recall@5", "recall@10", "mrr@20", "recall@1024tok")  # breakdown tables
REQUIRED_FIELDS = ("qid", "paper_id", "question", "evidence_quote", "char_start", "char_end",
                   "doc_sha256", "span_type", "status")


def log(msg: str) -> None:
    print(msg, flush=True)


# --- gold loading ------------------------------------------------------------------------

@dataclass
class GoldItem:
    qid: str
    paper_id: str
    question: str
    char_start: int
    char_end: int
    span_type: str
    status: str
    title_overlap: float
    copy: bool = False  # judge.copy; rows without a judge verdict count as non-copy


@dataclass
class GoldLoad:
    items: list[GoldItem]
    counts: dict[str, int] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)  # status of the kept items


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_gold(path: Path, docs: dict[str, Document], allow_draft: bool = False) -> GoldLoad:
    """Usable gold rows; counts say what was dropped and why.

    Order: parse, dedupe by (qid, doc_sha256) with the last row winning (whatever its status, so
    an `ok` row followed by a `bad` one for the same key is gone), then status filter, off-corpus,
    stale (sha or quote mismatch). Raises ValueError if nothing remains.
    """
    wanted = set(EXPORT_STATUSES) | ({"draft"} if allow_draft else set())
    shas = {pid: sha256_text(d.text) for pid, d in docs.items()}
    c: Counter = Counter()
    latest: dict[tuple[str, str], tuple[GoldItem, str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if any(k not in r for k in REQUIRED_FIELDS):
                    raise KeyError("missing field")
                start, end = int(r["char_start"]), int(r["char_end"])
                judge = r.get("judge")
                item = GoldItem(r["qid"], r["paper_id"], r["question"], start, end, r["span_type"],
                                r["status"], float(r.get("title_overlap", 0.0)),
                                isinstance(judge, dict) and judge.get("copy") is True)
                quote, sha = r["evidence_quote"], r["doc_sha256"]
                key = (item.qid, sha)
                hash(key)
            except (ValueError, KeyError, TypeError, AttributeError):
                log(f"WARNING: {path}:{n} is not a valid gold row; skipping it")
                c["unparseable"] += 1
                continue
            c["rows"] += 1
            if key in latest:
                c["duplicates"] += 1
            latest[key] = (item, quote, sha)  # last wins (position of the first occurrence is kept)
    kept: list[GoldItem] = []
    for item, quote, sha in latest.values():
        if item.status not in wanted:
            c["status_filtered"] += 1
            continue
        doc = docs.get(item.paper_id)
        if doc is None:
            c["off_corpus"] += 1
            continue
        start, end = item.char_start, item.char_end
        if sha != shas[item.paper_id] or not 0 <= start < end <= len(doc.text) or doc.text[start:end] != quote:
            c["stale"] += 1
            continue
        kept.append(item)
    c["kept"] = len(kept)
    if c["stale"]:
        log(f"WARNING: {c['stale']} stale gold rows dropped (doc_sha256 or quote offsets no longer match); "
            "regenerate them instead of editing offsets")
    if not kept:
        raise ValueError(f"no usable gold rows in {path} ({dict(c)}); "
                         "review items to 'ok' or pass --allow-draft for a dry run")
    return GoldLoad(kept, dict(c), dict(Counter(i.status for i in kept)))


# --- relevance and per-question metrics --------------------------------------------------

def is_hit(chunk: Chunk, item: GoldItem) -> bool:
    """Same paper and the chunk's span covers >= HIT_COVERAGE of the evidence span's characters."""
    if chunk.paper_id != item.paper_id:
        return False
    length = item.char_end - item.char_start
    covered = min(chunk.char_end, item.char_end) - max(chunk.char_start, item.char_start)
    return covered > 0 and covered >= HIT_COVERAGE * length


def score_ranking(hits: list[bool], same_paper: list[bool], tokens: list[int]) -> dict[str, float]:
    """Per-question metric values for one ranked list (best first, up to K_MAX long)."""
    m: dict[str, float] = {}
    for k in RECALL_KS:
        m[f"recall@{k}"] = float(any(hits[:k]))
    m[f"mrr@{K_MAX}"] = next((1.0 / (i + 1) for i, h in enumerate(hits[:K_MAX]) if h), 0.0)
    for k in PAPER_KS:
        m[f"paper_recall@{k}"] = float(any(same_paper[:k]))
    for budget in BUDGETS:
        used, got = 0, False
        for h, t in zip(hits, tokens):
            used += t
            if used > budget:
                break  # strict prefix: stop at the first chunk that does not fit
            got = got or h
        m[f"recall@{budget}tok"] = float(got)
    return m


METRIC_NAMES = ([f"recall@{k}" for k in RECALL_KS] + [f"mrr@{K_MAX}"] + [f"paper_recall@{k}" for k in PAPER_KS]
                + [f"recall@{b}tok" for b in BUDGETS])


# --- bootstrap ---------------------------------------------------------------------------

class PaperBootstrap:
    """Cluster bootstrap: each resample draws papers with replacement, keeping all their questions.

    Shared by all strategies, so differences between strategies are paired on the same papers.
    """

    def __init__(self, paper_ids: list[str], n_boot: int, seed: int):
        uniq = sorted(set(paper_ids))
        code = {p: i for i, p in enumerate(uniq)}
        self.codes = np.array([code[p] for p in paper_ids])
        self.n_papers = len(uniq)
        self.sizes = np.bincount(self.codes, minlength=self.n_papers)
        self.idx = np.random.default_rng(seed).integers(0, self.n_papers, size=(n_boot, self.n_papers))
        self.denom = self.sizes[self.idx].sum(axis=1)

    def means(self, values: np.ndarray) -> np.ndarray:
        """Mean over the questions of each resample, shape (n_boot,)."""
        sums = np.bincount(self.codes, weights=values, minlength=self.n_papers)
        return sums[self.idx].sum(axis=1) / self.denom


def bootstrap_indices(paper_ids: list[str], n_boot: int, seed: int) -> PaperBootstrap:
    return PaperBootstrap(paper_ids, n_boot, seed)


def percentile_ci(means: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mean_ci(values: np.ndarray, boot: PaperBootstrap) -> dict[str, float]:
    lo, hi = percentile_ci(boot.means(values))
    return {"mean": float(values.mean()), "lo": lo, "hi": hi}


def paired_diff(a: np.ndarray, b: np.ndarray, boot: PaperBootstrap) -> dict[str, float]:
    """a - b: point difference, bootstrap CI, and the fraction of resamples with diff <= 0."""
    d = a - b
    means = boot.means(d)
    lo, hi = percentile_ci(means)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "frac_le_0": float((means <= 0).mean())}


# --- evaluation --------------------------------------------------------------------------

class _CachedQueries:
    """Embedder wrapper that embeds each query once (queries are not cached on disk)."""

    def __init__(self, embedder):
        self._e, self._q = embedder, {}

    def embed_passages(self, texts):
        return self._e.embed_passages(texts)

    def embed_query(self, text):
        if text not in self._q:
            self._q[text] = self._e.embed_query(text)
        return self._q[text]


def _warm(embedder, chunks: list[Chunk], label: str, block: int = 1024) -> None:
    """Embed in blocks first (cache fills, progress shows); the retriever then only looks up."""
    t0, texts = time.time(), [c.text for c in chunks]
    for i in range(0, len(texts), block):
        embedder.embed_passages(texts[i:i + block])
        done = min(i + block, len(texts))
        rate = done / max(time.time() - t0, 1e-9)
        log(f"  [{label}] embedded {done}/{len(texts)} chunks ({rate:.0f}/s, "
            f"eta {(len(texts) - done) / rate:.0f}s)")


def chunk_stats(token_counts: list[int]) -> dict[str, float]:
    s = sorted(token_counts)
    return {"count": len(s), "mean_tokens": statistics.fmean(s), "median_tokens": float(statistics.median(s)),
            "p95_tokens": float(s[min(len(s) - 1, int(0.95 * len(s)))])}


def evaluate(docs: list[Document], gold: list[GoldItem], embedder, strategies: list[str],
             max_tokens: int, overlap: int, n_boot: int, seed: int) -> dict:
    """Per-strategy metrics with CIs, paired diffs vs BASELINE, and breakdowns."""
    if max_tokens > embedder.max_tokens:
        raise ValueError(f"--max-tokens {max_tokens} exceeds the embedding model's limit {embedder.max_tokens}")
    q_embedder = _CachedQueries(embedder)
    idx = bootstrap_indices([g.paper_id for g in gold], n_boot, seed)
    per_q: dict[str, dict[str, np.ndarray]] = {}  # strategy -> metric -> (n_questions,)
    stats: dict[str, dict] = {}
    for strat in strategies:
        t0 = time.time()
        chunks = [c for d in docs for c in chunk_document(d, strat, embedder.count_tokens, max_tokens, overlap)]
        tok = {id(c): embedder.count_tokens(c.text) for c in chunks}
        stats[strat] = chunk_stats(list(tok.values()))
        log(f"[{strat}] {len(chunks)} chunks in {time.time() - t0:.0f}s; embedding")
        _warm(embedder, chunks, strat)
        retriever = DenseRetriever(q_embedder, chunks)
        rows = []
        for i, item in enumerate(gold):
            ranked = [c for c, _ in retriever.search(item.question, K_MAX)]
            rows.append(score_ranking([is_hit(c, item) for c in ranked],
                                      [c.paper_id == item.paper_id for c in ranked],
                                      [tok[id(c)] for c in ranked]))
            if (i + 1) % 100 == 0:
                log(f"  [{strat}] searched {i + 1}/{len(gold)} questions")
        per_q[strat] = {m: np.array([r[m] for r in rows]) for m in METRIC_NAMES}
        log(f"[{strat}] done in {time.time() - t0:.0f}s")

    results: dict = {"strategies": {}, "paired_vs_fixed": {}, "by_span_type": {}, "by_title_overlap": {},
                     "sensitivity": {}}
    for strat in strategies:
        results["strategies"][strat] = {"metrics": {m: mean_ci(per_q[strat][m], idx) for m in METRIC_NAMES},
                                        "chunks": stats[strat]}
    if BASELINE in strategies:
        for strat in strategies:
            if strat != BASELINE:
                results["paired_vs_fixed"][strat] = {
                    m: paired_diff(per_q[strat][m], per_q[BASELINE][m], idx) for m in PAIRED_METRICS}

    def breakdown(groups: dict[str, np.ndarray], metrics=KEY_METRICS) -> dict:
        return {g: {"n": int(mask.sum()),
                    "strategies": {s: {m: float(per_q[s][m][mask].mean()) for m in metrics} for s in strategies}}
                for g, mask in groups.items() if mask.any()}

    span_types = np.array([g.span_type for g in gold])
    results["by_span_type"] = breakdown({t: span_types == t for t in sorted(set(span_types))})
    title = np.array([g.title_overlap for g in gold])
    results["by_title_overlap"] = breakdown({"title_overlap == 0": title == 0, "title_overlap > 0": title > 0})
    copy = np.array([g.copy for g in gold])
    results["sensitivity"] = breakdown({"all": np.ones(len(gold), dtype=bool), NON_COPY: ~copy},
                                       SENSITIVITY_METRICS)
    return results


# --- output ------------------------------------------------------------------------------

# paths that change during the workflow itself and do not make the code "dirty"
_GIT_IGNORED_PREFIXES = ("eval/gold_draft", "docs/results/")


def is_dirty(porcelain: str) -> bool:
    """True if `git status --porcelain` lists any change outside gold-draft files and result outputs."""
    for line in porcelain.splitlines():
        # A rename is listed as "old -> new"; either side outside the ignored paths counts.
        for part in line[3:].split(" -> "):
            path = part.strip().strip('"').replace("\\", "/")
            if path and not path.startswith(_GIT_IGNORED_PREFIXES):
                return True
    return False


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_sha() -> str | None:
    out = _git("rev-parse", "HEAD")
    return (out or "").strip() or None


def git_dirty() -> bool | None:
    out = _git("status", "--porcelain", "--untracked-files=all")
    return None if out is None else is_dirty(out)


def _cell(v: dict) -> str:
    return f"{v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
            + ["| " + " | ".join(r) + " |" for r in rows])


def render_markdown(report: dict) -> str:
    cfg, res, strategies = report["config"], report["results"], list(report["results"]["strategies"])
    L = ["# Phase 2: chunking strategies, dense retrieval", ""]
    if report["draft"]:
        L += ["> **DRAFT GOLD, NOT REVIEWED.** Questions include unreviewed `draft` rows (--allow-draft); "
              "this is a pipeline dry run, not a result.", ""]
    g = report["gold"]
    L += [f"- Questions: {report['n_questions']} (rows read {g.get('rows', 0)}, stale {g.get('stale', 0)}, "
          f"status-filtered {g.get('status_filtered', 0)}, off-corpus {g.get('off_corpus', 0)}, "
          f"duplicates {g.get('duplicates', 0)}, unparseable {g.get('unparseable', 0)})",
          f"- Corpus: {report['n_papers']} papers ({report.get('n_gold_papers', '?')} with questions); "
          f"gold: `{cfg['gold']}`",
          "- Gold status (kept rows): "
          + (", ".join(f"{k} {v}" for k, v in sorted(report.get("gold_status_counts", {}).items())) or "unknown"),
          f"- Chunking: max_tokens {cfg['max_tokens']}, overlap {cfg['overlap']}",
          f"- Embedding: {report['embedding']['model']} ({report['embedding']['backend']}), cosine, exact search",
          f"- Hit rule: same paper and chunk span covers >= {HIT_COVERAGE:.0%} of the evidence span",
          "- Cells: mean [95% percentile cluster bootstrap CI: papers resampled with replacement, all their "
          f"questions kept; {cfg['bootstrap']} resamples, seed {cfg['seed']}]",
          f"- Commit: {report['git_commit'] or 'unknown'}" + (" (uncommitted changes)" if report.get("git_dirty") else ""),
          ""]
    if report.get("gold_status_counts", {}).get("auto_ok"):
        L[2:2] = ["**Gold set: machine-verified (LLM judge + rule filters), not human-reviewed**", ""]

    def metric_table(title: str, names: list[str]) -> None:
        L.extend([f"## {title}", ""])
        L.extend(_table(["strategy"] + names,
                        [[s] + [_cell(res["strategies"][s]["metrics"][m]) for m in names] for s in strategies]))
        L.append("")

    metric_table("Chunk recall", [f"recall@{k}" for k in RECALL_KS] + [f"mrr@{K_MAX}"])
    metric_table("Paper recall (any chunk from the right paper)", [f"paper_recall@{k}" for k in PAPER_KS])
    metric_table("Recall at a context budget (tokens of retrieved chunks)", [f"recall@{b}tok" for b in BUDGETS])
    L += ["## Chunk statistics (tokens per chunk, as embedded)", ""]
    L += _table(["strategy", "chunks", "mean", "median", "p95"],
                [[s, str(c["count"]), f"{c['mean_tokens']:.1f}", f"{c['median_tokens']:.0f}", f"{c['p95_tokens']:.0f}"]
                 for s in strategies for c in [res["strategies"][s]["chunks"]]])
    L.append("")
    if res["paired_vs_fixed"]:
        L += [f"## Paired difference vs `{BASELINE}`", "",
              "diff [95% CI] and the fraction of resamples with diff <= 0 (a one-sided bootstrap p-like number). "
              "Same resampled papers for both strategies. Paired comparisons are not corrected for multiple "
              "comparisons (3 strategies x 3 metrics).", ""]
        L += _table(["strategy"] + list(PAIRED_METRICS),
                    [[s] + [f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] ({d['frac_le_0']:.3f})"
                            for d in (res["paired_vs_fixed"][s][m] for m in PAIRED_METRICS)]
                     for s in res["paired_vs_fixed"]])
        L.append("")

    def breakdown(title: str, data: dict, note: str, metrics=KEY_METRICS) -> None:
        L.extend([f"## {title}", "", note, ""])
        L.extend(_table(["group", "n", "strategy"] + list(metrics),
                        [[grp, str(v["n"]), s] + [f"{v['strategies'][s][m]:.3f}" for m in metrics]
                         for grp, v in data.items() for s in strategies]))
        L.append("")

    breakdown("By span type", res["by_span_type"], "Point estimates only; small groups are noisy.")
    breakdown("By title overlap", res["by_title_overlap"],
              "title_overlap > 0 means the question shares words with the paper title. If structure+header "
              "gains mainly there, the gain is title leakage, not better chunking.")
    if res.get("sensitivity"):
        breakdown("Sensitivity: copy-like questions", res["sensitivity"],
                  "`non-copy` drops items the judge marked `copy` (questions that mostly repeat the evidence "
                  "wording, which favour lexical overlap); items without a judge verdict count as non-copy. "
                  "Point estimates only.", SENSITIVITY_METRICS)
    L += ["## Caveats", "",
          "- The embedding model is the quantized `Qdrant/bge-small-en-v1.5-onnx-Q` (fastembed default), not fp32.",
          f"- {report['n_papers_without_sections']} of {report['n_papers']} papers have no detected sections, so "
          "the structure strategies treat each as one section.",
          "- Gold questions are biased towards named, factual passages (see eval/README.md); absolute recall is optimistic.",
          "- Questions are LLM-drafted. Rows count if human-reviewed (`ok`/`edit`) or machine-verified "
          "(`auto_ok`: LLM judge + rule filters, no human review); see the gold status line above.",
          "- Chunks are only comparable by span; the header prefix of structure+header is not scored as evidence.",
          "- `fixed` uses overlapping windows, so two of its top-k chunks can cover the same evidence "
          "(two chances at one hit); the other strategies do not overlap.",
          "- The token-budget metric walks the ranking and stops at the first chunk that would overflow "
          "the budget (no skipping to smaller chunks). Chunk tokens include [CLS]/[SEP] and, for "
          "structure+header, the header prefix: header tokens are a real cost of that strategy.",
          f"- The ranking walk is capped at K_MAX = {K_MAX}; budget metrics cannot see beyond rank {K_MAX}.", ""]
    if report["draft"]:
        L += ["- **DRAFT GOLD**: unreviewed questions were included; do not cite these numbers.", ""]
    return "\n".join(L)


def write_outputs(report: dict, out: Path) -> tuple[Path, Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(out) + ".json"), Path(str(out) + ".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--allow-draft", action="store_true", help="also score unreviewed draft rows (dry run only)")
    p.add_argument("--ids-file", type=Path, default=DEFAULT_IDS_FILE)
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output prefix; writes <out>.json and <out>.md")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be >= 1")

    ids = read_ids_file(args.ids_file)
    if not ids:
        raise SystemExit(f"{args.ids_file} contains no ids")
    try:
        docs = load_documents(args.data_dir, ids_to_stems(ids))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    try:
        gold = load_gold(args.gold, {d.paper_id: d for d in docs}, args.allow_draft)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    draft = any(g.status == "draft" for g in gold.items)
    log(f"{len(docs)} papers, {len(gold.items)} questions {gold.counts} by status {gold.by_status}" + ("  ** DRAFT GOLD **" if draft else ""))

    embedder = Embedder()
    t0 = time.time()
    results = evaluate(docs, gold.items, embedder, args.strategies, args.max_tokens, args.overlap,
                       args.bootstrap, args.seed)
    report = {
        "config": {"gold": str(args.gold), "max_tokens": args.max_tokens, "overlap": args.overlap,
                   "strategies": args.strategies, "bootstrap": args.bootstrap, "seed": args.seed,
                   "hit_coverage": HIT_COVERAGE, "k_max": K_MAX, "budgets": list(BUDGETS)},
        "draft": draft,
        "gold": gold.counts,
        "n_questions": len(gold.items),
        "n_papers": len(docs),
        "n_papers_without_sections": sum(all(p.section is None for p in d.paragraphs) for d in docs),
        "embedding": {"model": embedder.model_name, "backend": embedder.backend.tag},
        "git_commit": git_sha(),
        "git_dirty": git_dirty(),
        "gold_status_counts": gold.by_status,
        "n_gold_papers": len({g.paper_id for g in gold.items}),
        "runtime_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    json_path, md_path = write_outputs(report, args.out)
    log(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
