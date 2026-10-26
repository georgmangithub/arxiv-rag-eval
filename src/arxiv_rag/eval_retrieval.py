"""Phase 3 eval: compare retrievers (dense, BM25, hybrid, +cross-encoder rerank) on ONE chunking.

Design goals:
- Same questions, gold rules, hit rule, metrics and paper bootstrap as Phase 2 (eval_chunking);
  only the retriever changes, so differences are paired on the same questions and resampled papers.
- No tuning on the gold set: BM25 k1/b are Pyserini defaults, hybrids fuse dense + BM25 at depth 100
  (RRF k=60; convex with equal weights), the reranker re-orders the base top `--rerank-depth`.
- Every (query, chunk) pair is scored by the cross-encoder at most once: Reranker caches scores on disk and
  in memory, and base configs run before the reranked ones so candidates shared between them are reused.
- Truncation is checked, not assumed: every distinct pair sent to the reranker is length-counted and the
  number above the reranker's max_tokens is reported.
- The token-budget metric of Phase 2 is dropped: chunks are identical across configs here, so it adds little.
- Testable: embedder and reranker are injected (tests use fakes); nothing here calls an LLM.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from arxiv_rag import goldgen
from arxiv_rag.chunk import STRATEGIES, Chunk, chunk_document
from arxiv_rag.document import Document, ids_to_stems, load_documents
from arxiv_rag.embed import Embedder
from arxiv_rag.eval_chunking import (
    DEFAULT_GOLD, DEFAULT_IDS_FILE, HIT_COVERAGE, K_MAX, METRIC_NAMES, PAPER_KS, RECALL_KS, GoldItem,
    PaperBootstrap, _CachedQueries, _cell, _table, _warm, chunk_stats, git_dirty, git_sha, is_hit,
    load_gold, log, mean_ci, paired_diff, score_ranking,
)
from arxiv_rag.hybrid import HybridRetriever
from arxiv_rag.ingest import read_ids_file
from arxiv_rag.lexical import BM25Retriever
from arxiv_rag.rerank import Reranker, RerankedRetriever
from arxiv_rag.retrieve import DenseRetriever

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _ROOT / "docs" / "results" / "phase3"

BASE_RETRIEVERS = ("dense", "bm25", "hybrid-rrf", "hybrid-convex")
BASELINE = "dense"
HYBRID_DEPTH = 100
RRF_K = 60
METRICS = [m for m in METRIC_NAMES if not m.endswith("tok")]  # budget metrics dropped (see module doc)
PAIRED_METRICS = ("recall@1", "recall@5", "recall@10", "mrr@20")
KEY_METRICS = PAIRED_METRICS
RERANK_SUFFIX = "+rerank"
BM25_TOKENIZER = "lowercase alnum + joined compounds, small stopword list, no stemming"
ANCHOR_BUCKETS = ("rarest anchor in 1 paper", "rarest anchor in 2-5 papers", "rarest anchor in >5 papers",
                  "no shared anchor term")
ABLATION_KS = (5, 10)


# --- reranker truncation accounting ------------------------------------------------------

class CountingReranker:
    """Reranker wrapper that length-checks each distinct (query, text) pair once.

    `over_limit` counts pairs whose untruncated token length exceeds the reranker's max_tokens, i.e.
    pairs the cross-encoder would have silently truncated.
    """

    def __init__(self, reranker: Reranker):
        self._r = reranker
        self._seen: set[tuple[str, str]] = set()
        self.pairs = 0
        self.over_limit = 0
        self.longest = 0

    @property
    def model_name(self) -> str:
        return self._r.model_name

    @property
    def max_tokens(self) -> int:
        return self._r.max_tokens

    def score(self, query, texts):
        texts = list(texts)
        for t in texts:
            if (query, t) in self._seen:
                continue
            self._seen.add((query, t))
            n = self._r.count_pair_tokens(query, t)
            self.pairs += 1
            self.longest = max(self.longest, n)
            if n > self._r.max_tokens:
                self.over_limit += 1
        return self._r.score(query, texts)


# --- configs -----------------------------------------------------------------------------

def build_configs(chunks: list[Chunk], embedder, reranker, names: list[str], rerank: bool,
                  depth: int) -> dict[str, object]:
    """name -> retriever, in run order: all base configs first, then their reranked versions."""
    unknown = [n for n in names if n not in BASE_RETRIEVERS]
    if unknown:
        raise ValueError(f"unknown retrievers {unknown}; expected some of {BASE_RETRIEVERS}")
    dense = DenseRetriever(embedder, chunks)
    bm25 = BM25Retriever(chunks) if any(n != "dense" for n in names) else None
    base: dict[str, object] = {}
    for n in names:
        if n == "dense":
            base[n] = dense
        elif n == "bm25":
            base[n] = bm25
        else:
            base[n] = HybridRetriever([dense, bm25], method=n.split("-", 1)[1], k_rrf=RRF_K, depth=HYBRID_DEPTH)
    configs = dict(base)
    if rerank:
        for n, r in base.items():
            configs[n + RERANK_SUFFIX] = RerankedRetriever(r, reranker, depth)
    return configs


# --- evaluation --------------------------------------------------------------------------

def lexical_split(gold: list[GoldItem]) -> float | None:
    """Median question/quote word-overlap: the threshold of the low/high lexical-overlap breakdown.

    None when the split would be degenerate: some item lacks `overlap` or the median is 0.
    """
    if any(g.overlap is None for g in gold):
        return None
    median = float(np.median([g.overlap for g in gold]))
    return median if median > 0 else None


def describe_setup(configs: dict[str, object], reranker, depth: int) -> dict:
    """Retrieval parameters read back from the built retriever objects (not from module constants)."""
    hybrids = {n: r for n, r in configs.items() if isinstance(r, HybridRetriever)}
    bm25 = configs.get("bm25") or next((sub for h in hybrids.values() for sub in h.retrievers
                                        if isinstance(sub, BM25Retriever)), None)
    setup: dict = {}
    if bm25 is not None:
        setup["bm25"] = {"k1": bm25.k1, "b": bm25.b, "tokenizer": BM25_TOKENIZER}
    if hybrids:
        setup["fusion"] = {n: {"method": h.method, "weights": list(h.weights), "k_rrf": h.k_rrf, "depth": h.depth}
                           for n, h in hybrids.items()}
    reranked = [r for r in configs.values() if isinstance(r, RerankedRetriever)]
    if reranked:
        setup["reranker"] = {"model": reranker.model_name, "tag": getattr(reranker, "_tag", None),
                             "depth": reranked[0].depth}
    return setup


# --- anchor analysis ---------------------------------------------------------------------

def _paper_norms(chunks: list[Chunk]) -> dict[str, list[tuple[str, set[int], set[int]]]]:
    """paper -> normalised chunk texts (goldgen normalisation) with word-start/word-end index sets."""
    out: dict[str, list] = {}
    for c in chunks:
        q, starts = goldgen._normalize(c.text)
        out.setdefault(c.paper_id, []).append((q, starts, starts | {len(q)}))
    return out


def _term_in(term: str, q: str, starts: set[int], ends: set[int]) -> bool:
    """goldgen.is_anchored's matching rule for one term against one normalised text."""
    norm = goldgen._normalize(term)[0].strip(".")
    for cand in {norm, norm.removesuffix("s")}:
        whole = cand.isalpha() and len(cand) <= 4
        i = q.find(cand) if cand else -1
        while i >= 0:
            j = i + len(cand)
            if i in starts and (not whole or j in ends or (q[j : j + 1] == "s" and j + 1 in ends)):
                return True
            i = q.find(cand, i + 1)
    return False


def _matches(term: str, text: str) -> bool:
    q, starts = goldgen._normalize(text)
    return _term_in(term, q, starts, starts | {len(q)})


def anchor_terms(item: GoldItem) -> list[str]:
    """Specific terms of the evidence quote that the question reuses.

    goldgen.is_anchored cannot be called per term (it re-extracts specific terms, and the lowercased terms
    no longer look specific), so this applies its matching rule (`_term_in`) to each term.
    """
    return sorted(t for t in goldgen._terms(item.quote, heads=True) if _matches(t, item.question))


def paper_df(terms, norms: dict[str, list]) -> dict[str, int]:
    """term -> number of papers with a chunk containing it (the gold paper itself included)."""
    return {t: sum(any(_term_in(t, q, st, en) for q, st, en in cs) for cs in norms.values()) for t in set(terms)}


def anchor_bucket(df: int | None) -> str:
    if df is None:
        return ANCHOR_BUCKETS[3]
    return ANCHOR_BUCKETS[0] if df <= 1 else ANCHOR_BUCKETS[1] if df <= 5 else ANCHOR_BUCKETS[2]


def remove_anchor_words(question: str, terms: list[str]) -> str:
    """Ablated query: drop every whitespace-separated word of the question that matches (the is_anchored
    rule: normalised, at a word start, short acronyms whole-word) at least one shared anchor term."""
    return " ".join(w for w in question.split() if not any(_matches(t, w) for t in terms))


def evaluate(docs: list[Document], gold: list[GoldItem], embedder, reranker, chunking: str, names: list[str],
             rerank: bool, depth: int, max_tokens: int, overlap: int, n_boot: int, seed: int) -> dict:
    if max_tokens > embedder.max_tokens:
        raise ValueError(f"--max-tokens {max_tokens} exceeds the embedding model's limit {embedder.max_tokens}")
    t0 = time.time()
    chunks = [c for d in docs for c in chunk_document(d, chunking, embedder.count_tokens, max_tokens, overlap)]
    stats = chunk_stats([embedder.count_tokens(c.text) for c in chunks])
    log(f"[{chunking}] {len(chunks)} chunks in {time.time() - t0:.0f}s; embedding")
    _warm(embedder, chunks, chunking)
    counting = CountingReranker(reranker) if rerank else None
    configs = build_configs(chunks, _CachedQueries(embedder), counting, names, rerank, depth)
    boot = PaperBootstrap([g.paper_id for g in gold], n_boot, seed)

    per_q: dict[str, dict[str, np.ndarray]] = {}
    seconds: dict[str, float] = {}
    cand_n = max(depth, K_MAX)  # candidates a reranker sees
    ceiling_hits: dict[str, list[bool]] = {}  # base retriever -> per question: a hit in the top cand_n
    top1: dict[str, list] = {}  # config -> per question: chunk_id ranked first
    first_hit: dict[str, list] = {}  # config -> per question: rank of the first hit (None = none in K_MAX)

    def score_one(ranked: list[Chunk], item: GoldItem) -> dict[str, float]:
        ranked = ranked[:K_MAX]
        return score_ranking([is_hit(c, item) for c in ranked],
                             [c.paper_id == item.paper_id for c in ranked], [0] * len(ranked))

    for name, retriever in configs.items():
        t1 = time.time()
        log(f"[{name}] searching {len(gold)} questions")
        rows = []
        is_base = not name.endswith(RERANK_SUFFIX)
        top1[name], first_hit[name] = [], []
        for i, item in enumerate(gold):
            ranked = [c for c, _ in retriever.search(item.question, cand_n if rerank and is_base else K_MAX)]
            if rerank and is_base:
                ceiling_hits.setdefault(name, []).append(any(is_hit(c, item) for c in ranked[:cand_n]))
            rows.append(score_one(ranked, item))
            top1[name].append(ranked[0].chunk_id if ranked else None)
            first_hit[name].append(next((r for r, c in enumerate(ranked[:K_MAX], 1) if is_hit(c, item)), None))
            if (i + 1) % 10 == 0 or i + 1 == len(gold):
                extra = f", {counting.pairs} reranker pairs checked" if counting and name.endswith(RERANK_SUFFIX) else ""
                log(f"  [{name}] {i + 1}/{len(gold)} questions ({time.time() - t1:.0f}s{extra})")
        per_q[name] = {m: np.array([r[m] for r in rows]) for m in METRICS}
        seconds[name] = round(time.time() - t1, 1)
        log(f"[{name}] done in {seconds[name]:.0f}s")

    res: dict = {"configs": {n: {"metrics": {m: mean_ci(per_q[n][m], boot) for m in METRICS}, "seconds": seconds[n]}
                             for n in configs},
                 "paired_vs_dense": {}, "rerank_vs_base": {}, "by_span_type": {}, "by_copy": {},
                 "by_lexical_overlap": {}, "chunks": stats, "setup": describe_setup(configs, reranker, depth)}
    if BASELINE in configs:
        for n in configs:
            if n != BASELINE:
                res["paired_vs_dense"][n] = {m: paired_diff(per_q[n][m], per_q[BASELINE][m], boot)
                                             for m in PAIRED_METRICS}
    for n in configs:
        if n.endswith(RERANK_SUFFIX):
            b = n[:-len(RERANK_SUFFIX)]
            res["rerank_vs_base"][n] = {m: paired_diff(per_q[n][m], per_q[b][m], boot) for m in PAIRED_METRICS}

    def breakdown(groups: dict[str, np.ndarray]) -> dict:
        return {g: {"n": int(mask.sum()),
                    "configs": {c: {m: float(per_q[c][m][mask].mean()) for m in KEY_METRICS} for c in configs}}
                for g, mask in groups.items() if mask.any()}

    spans = np.array([g.span_type for g in gold])
    res["by_span_type"] = breakdown({t: spans == t for t in sorted(set(spans))})
    copy = np.array([g.copy for g in gold])
    res["by_copy"] = breakdown({"copy (judge.copy == true)": copy, "non-copy (judge.copy == false)": ~copy})
    split = lexical_split(gold)
    res["lexical_split"] = split
    if split is None:
        res["lexical_overlap_skipped"] = ("some gold rows have no `overlap` field"
                                          if any(g.overlap is None for g in gold) else "the median overlap is 0")
        log(f"NOTE: lexical-overlap breakdown skipped: {res['lexical_overlap_skipped']}")
    else:
        ov = np.array([g.overlap for g in gold])
        res["by_lexical_overlap"] = breakdown({f"overlap < {split:.3f} (low)": ov < split,
                                               f"overlap >= {split:.3f} (high)": ov >= split})

    if ceiling_hits:
        res["rerank_ceiling"] = {"candidates": cand_n,
                                 "configs": {n: float(np.mean(h)) for n, h in ceiling_hits.items()}}
    hy = ("hybrid-rrf" + RERANK_SUFFIX, "hybrid-convex" + RERANK_SUFFIX)
    if all(h in top1 for h in hy):
        res["hybrid_rerank_agreement"] = {
            "configs": list(hy), "n": len(gold),
            "same_top1": float(np.mean([a == b for a, b in zip(top1[hy[0]], top1[hy[1]])])),
            "same_first_hit_rank": float(np.mean([a == b for a, b in zip(first_hit[hy[0]], first_hit[hy[1]])]))}

    # anchor rarity: paper document frequency of each question's rarest shared evidence term
    norms = _paper_norms(chunks)
    shared = [anchor_terms(g) for g in gold]
    df = paper_df([t for ts in shared for t in ts], norms)
    rarest = [min(df[t] for t in ts) if ts else None for ts in shared]
    buckets = np.array([anchor_bucket(d) for d in rarest])
    res["by_anchor_rarity"] = breakdown({b: buckets == b for b in ANCHOR_BUCKETS})
    common = (buckets == ANCHOR_BUCKETS[1]) | (buckets == ANCHOR_BUCKETS[2])
    res["anchor_common"] = {"n": int(common.sum()),
                            "recall@5": {c: float(per_q[c]["recall@5"][common].mean()) for c in configs}
                            if common.any() else {}}

    # anchor-removed ablation: base (non-reranked) retrievers only; cross-encoder scoring of new queries is skipped
    ablated = [remove_anchor_words(g.question, ts) if ts else g.question for g, ts in zip(gold, shared)]
    base_names = [n for n in configs if not n.endswith(RERANK_SUFFIX)]
    abl_cfg: dict[str, dict] = {}
    for name in base_names:
        log(f"[{name}] ablation: {len(gold)} anchor-removed questions")
        rows = [score_one([c for c, _ in configs[name].search(q, K_MAX)] if q.strip() else [], g)
                for g, q in zip(gold, ablated)]
        abl_cfg[name] = {}
        for k in ABLATION_KS:
            m = f"recall@{k}"
            new, old = np.array([r[m] for r in rows]), per_q[name][m]
            abl_cfg[name][m] = {"normal": float(old.mean()), "ablated": float(new.mean()),
                                "change": float(new.mean() - old.mean())}
    res["ablation"] = {"n": len(gold), "n_changed": sum(a != g.question for a, g in zip(ablated, gold)),
                       "n_empty": sum(not a.strip() for a in ablated), "configs": abl_cfg}
    if counting is not None:
        res["rerank_truncation"] = {"model": counting.model_name, "max_tokens": counting.max_tokens,
                                    "distinct_pairs": counting.pairs, "over_limit": counting.over_limit,
                                    "longest_pair_tokens": counting.longest}
        if counting.over_limit:
            log(f"WARNING: {counting.over_limit} reranker pairs exceed {counting.max_tokens} tokens (truncated)")
    return res


# --- output ------------------------------------------------------------------------------

def anchor_tables(L: list[str], res: dict, names: list[str]) -> None:
    """Anchor-rarity breakdown and anchor-removed ablation sections."""
    if res.get("by_anchor_rarity"):
        L.extend(["## By anchor rarity", "",
                  "For each question: the evidence-quote terms (`goldgen._terms(quote, heads=True)`) that the "
                  "question reuses (`goldgen.is_anchored` matching rule, applied per term), the rarest by paper document frequency over the corpus "
                  "chunks (papers with a chunk containing the term, same normalised matching). The df includes "
                  "the gold paper itself, so df=1 means the term occurs only in the gold paper and that bucket "
                  "dominates. Point estimates only.", ""])
        L.extend(_table(["group", "n", "retriever", "recall@1", "recall@5", "recall@10"],
                        [[g, str(v["n"]), n] + [f"{v['configs'][n][m]:.3f}" for m in ("recall@1", "recall@5", "recall@10")]
                         for g, v in res["by_anchor_rarity"].items() for n in names]))
        L.append("")
    ab = res.get("ablation")
    if ab:
        L.extend(["## Anchor-removed ablation", "",
                  "Each question is rewritten by deleting every whitespace-separated word for which "
                  "a shared anchor term matches it under the `goldgen.is_anchored` rule (see above); "
                  f"{ab['n_changed']} of {ab['n']} questions changed, {ab['n_empty']} became empty (scored as a miss). "
                  "Only the non-reranked base retrievers are rerun: reranked configs are skipped to avoid hours of new "
                  "cross-encoder scoring. Point estimates; change = ablated - normal.", ""])
        ms = [f"recall@{k}" for k in ABLATION_KS]
        L.extend(_table(["retriever"] + [f"{m} normal -> ablated (change)" for m in ms],
                        [[n] + [f"{c[m]['normal']:.3f} -> {c[m]['ablated']:.3f} ({c[m]['change']:+.3f})" for m in ms]
                         for n, c in ab["configs"].items()]))
        L.append("")


def anchor_note(res: dict) -> str:
    """Computed sentence about whether the BM25 lead over dense survives common anchors and anchor removal."""
    c, ab = res.get("anchor_common", {}), res.get("ablation", {}).get("configs", {})
    if "bm25" not in res["configs"] or "dense" not in res["configs"] or not ab:
        return ""
    m = res["configs"]
    all_lead = m["bm25"]["metrics"]["recall@5"]["mean"] - m["dense"]["metrics"]["recall@5"]["mean"]
    abl_lead = ab["bm25"]["recall@5"]["ablated"] - ab["dense"]["recall@5"]["ablated"]
    parts = [f"BM25 minus dense recall@5 is {all_lead:+.3f} on all questions"]
    common_lead = None
    if c.get("recall@5"):
        common_lead = c["recall@5"]["bm25"] - c["recall@5"]["dense"]
        parts.append(f"{common_lead:+.3f} on the {c['n']} questions whose rarest anchor occurs in more than one paper")
    parts.append(f"{abl_lead:+.3f} after anchor removal")
    text = ", ".join(parts) + ". "
    persists = abl_lead > 0 and (common_lead is None or common_lead > 0)
    if persists:
        text += ("The BM25 lead persists for common anchors and after anchor removal, so it points to a broader "
                 "lexical-match advantage of BM25 over the quantized bge-small on LLM-paraphrased factual questions; "
                 "it cannot be fully separated from the gold-set construction.")
    else:
        text += ("The BM25 lead does not persist in every one of these checks, which is consistent with part of it "
                 "being driven by the required anchor term; it cannot be fully separated from the gold-set construction.")
    return text


def render_markdown(report: dict) -> str:
    cfg, res = report["config"], report["results"]
    names = list(res["configs"])
    setup = res.get("setup", {})
    L = ["# Phase 3: retrievers on one chunking", ""]
    if report.get("partial"):
        L += [f"> **PARTIAL: first {report['n_questions']} questions (file order).** Smoke test, not a result.", ""]
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
          f"- Chunking: {cfg['chunking']}, max_tokens {cfg['max_tokens']}, overlap {cfg['overlap']} "
          f"({res['chunks']['count']} chunks, mean {res['chunks']['mean_tokens']:.1f} tokens)",
          f"- Embedding: {report['embedding']['model']} ({report['embedding']['backend']}), cosine, exact search",
          ]
    if "bm25" in setup:
        b = setup["bm25"]
        L.append(f"- Lexical: BM25 (k1={b['k1']}, b={b['b']}), {b['tokenizer']}")
    for n, f in setup.get("fusion", {}).items():
        L.append(f"- Hybrid `{n}`: dense + BM25, each to depth {f['depth']}; "
                 + (f"RRF k={f['k_rrf']}" if f["method"] == "rrf" else "convex = min-max normalised")
                 + f"; weights {f['weights']}")
    L += [("- Rerank: " + (f"{res['rerank_truncation']['model']}, top {cfg['rerank_depth']} of each base retriever"
                          if "rerank_truncation" in res else "off")),
          "- No parameters were tuned on the gold set",
          f"- Hit rule: same paper and chunk span covers >= {HIT_COVERAGE:.0%} of the evidence span",
          "- Cells: mean [95% percentile cluster bootstrap CI: papers resampled with replacement, all their "
          f"questions kept; {cfg['bootstrap']} resamples, seed {cfg['seed']}]",
          f"- Commit: {report['git_commit'] or 'unknown'}" + (" (uncommitted changes)" if report.get("git_dirty") else ""),
          ""]
    if report.get("gold_status_counts", {}).get("auto_ok"):
        L[2:2] = ["**Gold set: machine-verified (LLM judge + rule filters), not human-reviewed**", ""]
    L[2:2] = ["> **Lexical-anchor bias:** every gold question was required to reuse a specific term from its "
              "evidence passage, which favours BM25 and hybrids over what real users would ask. Read lexical vs "
              "dense gaps with that in mind (see the breakdown by lexical overlap).", ""]

    def metric_table(title: str, ms: list[str]) -> None:
        L.extend([f"## {title}", ""])
        L.extend(_table(["retriever"] + ms, [[n] + [_cell(res["configs"][n]["metrics"][m]) for m in ms] for n in names]))
        L.append("")

    metric_table("Chunk recall", [f"recall@{k}" for k in RECALL_KS] + [f"mrr@{K_MAX}"])
    metric_table("Paper recall (any chunk from the right paper)", [f"paper_recall@{k}" for k in PAPER_KS])

    def diff_table(title: str, note: str, data: dict) -> None:
        if not data:
            return
        L.extend([f"## {title}", "", note, ""])
        L.extend(_table(["retriever"] + list(PAIRED_METRICS),
                        [[n] + [f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] ({d['frac_le_0']:.3f})"
                                for d in (data[n][m] for m in PAIRED_METRICS)] for n in data]))
        L.append("")

    n_cmp = len(res["paired_vs_dense"]) + len(res["rerank_vs_base"])
    note = ("diff [95% CI] and the fraction of resamples with diff <= 0 (a one-sided bootstrap p-like number). "
            f"Same resampled papers for both retrievers. Not corrected for multiple comparisons "
            f"({n_cmp} comparisons x {len(PAIRED_METRICS)} metrics).")
    diff_table(f"Paired difference vs `{BASELINE}`", note, res["paired_vs_dense"])
    diff_table("Paired difference: reranked vs its base retriever", note, res["rerank_vs_base"])

    def breakdown(title: str, data: dict, text: str) -> None:
        L.extend([f"## {title}", "", text, ""])
        L.extend(_table(["group", "n", "retriever"] + list(KEY_METRICS),
                        [[grp, str(v["n"]), n] + [f"{v['configs'][n][m]:.3f}" for m in KEY_METRICS]
                         for grp, v in data.items() for n in names]))
        L.append("")

    breakdown("By span type", res["by_span_type"], "Point estimates only; small groups are noisy.")
    breakdown("By copy-like questions", res["by_copy"],
              "`copy` = the judge marked the question as mostly repeating the evidence wording (favours lexical "
              "retrieval); items without a judge verdict count as non-copy. Point estimates only.")
    if res["by_lexical_overlap"]:
        breakdown("By lexical overlap", res["by_lexical_overlap"],
                  f"Split at the median `overlap` of the scored questions ({res['lexical_split']:.3f}); `overlap` is the "
                  "longest common word n-gram between question and evidence as a fraction of the question's words. "
                  "A retriever that wins mainly in the high group is exploiting wording reuse. Point estimates only.")
    else:
        L += ["## By lexical overlap", "", f"Skipped: {res.get('lexical_overlap_skipped', 'no overlap data')}.", ""]
    if "rerank_ceiling" in res:
        rc = res["rerank_ceiling"]
        L.extend([f"## Rerank ceiling", "",
                  f"Fraction of questions whose top-{rc['candidates']} base candidates contain a hit: the best "
                  "recall@k any reranker of that base retriever could reach.", ""])
        L.extend(_table(["base retriever", f"ceiling (top {rc['candidates']})", "base recall@10", "reranked recall@10"],
                        [[n, f"{c:.3f}", _cell(res["configs"][n]["metrics"]["recall@10"]),
                          _cell(res["configs"][n + RERANK_SUFFIX]["metrics"]["recall@10"])]
                         for n, c in rc["configs"].items()]))
        L.append("")
        c = rc["configs"]
        if "dense" in c and len(c) > 1 and c["dense"] < max(c.values()):
            L += [f"Dense has the lowest-ceiling candidate pool ({c['dense']:.3f} vs best {max(c.values()):.3f}), so "
                  "reranking dense cannot reach what reranking a lexical or hybrid pool can.", ""]
    agree = res.get("hybrid_rerank_agreement")
    if agree:
        L += ["## Reranked hybrids compared", "",
              f"`{agree['configs'][0]}` vs `{agree['configs'][1]}` on the same {agree['n']} questions: identical "
              f"top-1 chunk for {agree['same_top1']:.1%} of questions, identical first-hit rank for "
              f"{agree['same_first_hit_rank']:.1%}. "
              + ("With reranking the two hybrids are indistinguishable in this trial."
                 if agree["same_top1"] == 1 and agree["same_first_hit_rank"] == 1 else
                 "They differ on the remainder, so they are not fully indistinguishable."), ""]
    anchor_tables(L, res, names)
    L += ["## Caveats", "",
          "- Lexical-anchor bias: every gold question was required to reuse a specific term from its evidence, "
          "which favours BM25/hybrid; absolute BM25 and hybrid numbers are optimistic. " + anchor_note(res),
          "- The embedding model is the quantized `Qdrant/bge-small-en-v1.5-onnx-Q` (fastembed default), not fp32.",
          ]
    if "rerank_truncation" in res:
        t = res["rerank_truncation"]
        L += [f"- Reranker: `{t['model']}`, candidates = base top {cfg['rerank_depth']}. Truncation check: "
              f"{t['over_limit']} of {t['distinct_pairs']} distinct (query, chunk) pairs exceed the model's "
              f"{t['max_tokens']}-token limit (longest pair {t['longest_pair_tokens']} tokens)."]
    L += [f"- No tuning on the gold set: BM25 k1/b, RRF k, hybrid depth, fusion weights "
          f"and rerank depth are fixed defaults. Tuning them here would overfit {report['n_questions']} questions.",
          "- BM25 tokenizer: lowercase alphanumeric runs plus hyphen/dot-joined compounds, no stemming.",
          "- Questions are LLM-drafted and machine-verified (`auto_ok`: LLM judge + rule filters, no human review) "
          "unless the gold status line says otherwise.",
          "- Paired comparisons and breakdowns are not corrected for multiple comparisons; breakdown cells are "
          "point estimates on small groups.",
          f"- The token-budget metric of Phase 2 is omitted: chunks are identical across retrievers. Ranking "
          f"walks are capped at K_MAX = {K_MAX}.", ""]
    if report["draft"]:
        L += ["- **DRAFT GOLD**: unreviewed questions were included; do not cite these numbers.", ""]
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--allow-draft", action="store_true", help="also score unreviewed draft rows (dry run only)")
    p.add_argument("--ids-file", type=Path, default=DEFAULT_IDS_FILE)
    p.add_argument("--data-dir", type=Path, default=_ROOT / "data")
    p.add_argument("--chunking", default="structure", choices=STRATEGIES)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--retrievers", nargs="+", default=list(BASE_RETRIEVERS), choices=BASE_RETRIEVERS)
    p.add_argument("--rerank-depth", type=int, default=50)
    p.add_argument("--no-rerank", action="store_true", help="skip the +rerank configs")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output prefix; writes <out>.json and <out>.md")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="score only the first N questions (smoke test)")
    args = p.parse_args()
    if args.bootstrap < 1 or args.rerank_depth < 1 or (args.limit is not None and args.limit < 1):
        raise SystemExit("--bootstrap, --rerank-depth and --limit must be >= 1")

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
    items = gold.items[:args.limit] if args.limit else gold.items
    partial = len(items) < len(gold.items)
    draft = any(g.status == "draft" for g in items)
    log(f"{len(docs)} papers, {len(items)} questions {gold.counts} by status {gold.by_status}"
        + ("  ** DRAFT GOLD **" if draft else ""))

    embedder, reranker = Embedder(), Reranker()
    t0 = time.time()
    results = evaluate(docs, items, embedder, reranker, args.chunking, args.retrievers, not args.no_rerank,
                       args.rerank_depth, args.max_tokens, args.overlap, args.bootstrap, args.seed)
    report = {
        "config": {"gold": str(args.gold), "chunking": args.chunking, "max_tokens": args.max_tokens,
                   "overlap": args.overlap, "retrievers": args.retrievers, "rerank": not args.no_rerank,
                   "rerank_depth": args.rerank_depth, **results["setup"],
                   "bootstrap": args.bootstrap, "seed": args.seed, "limit": args.limit,
                   "hit_coverage": HIT_COVERAGE, "k_max": K_MAX},
        "draft": draft,
        "partial": partial,
        "gold": gold.counts,
        "n_questions": len(items),
        "n_papers": len(docs),
        "embedding": {"model": embedder.model_name, "backend": embedder.backend.tag},
        "reranker": results["setup"].get("reranker"),
        "git_commit": git_sha(),
        "git_dirty": git_dirty(),
        "gold_status_counts": dict(Counter(i.status for i in items)),
        "n_gold_papers": len({g.paper_id for g in items}),
        "runtime_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    json_path, md_path = write_outputs(report, args.out)
    log(f"wrote {json_path} and {md_path}")


def write_outputs(report: dict, out: Path) -> tuple[Path, Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(out) + ".json"), Path(str(out) + ".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return json_path, md_path


if __name__ == "__main__":
    main()
