# arxiv-rag-eval

A RAG system over arXiv AI papers, built to answer one question honestly: **which retrieval choices
actually improve answer quality, and by how much?** Every design decision is measured, not assumed.

Everything is free to build and run: local embeddings, a free-tier LLM API with a local Ollama
fallback, and open-source tooling. It runs on an 8 GB laptop with no GPU.

> Status: Phases 0-3 done. Phase 2: fixed, recursive and structure-aware chunking are statistically tied; a title/section header hurts. Phase 3: on these entity-named factual questions BM25 beats dense by ~23 points at recall@5, and reranking mainly helps dense. See [docs/results/](docs/results/).

## Plan

| Phase | What | What it measures |
|---|---|---|
| 1 | Ingest arXiv PDFs and parse them (layout, tables, equations) | Parse quality on messy PDFs |
| 2 | Chunking strategies (fixed, recursive, structure-aware) | Retrieval recall@k per strategy |
| 3 | Retrieval: dense vs BM25 vs hybrid, plus reranking | recall@k, MRR |
| 4 | Generation with citations | Faithfulness, answer correctness |
| 5 | Eval harness in CI (regression gate) | Quality over time |
| 6 | Results write-up: what worked, what failed | |

## Layout

```
src/arxiv_rag/   library code
tests/           pytest suite
docs/            design notes and results
data/            downloaded papers and indexes (gitignored)
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
```

## Running the eval

Phase 2 compares the chunking strategies by dense-retrieval recall over the pinned corpus:

```bash
PYTHONPATH=src .venv/Scripts/python -u -m arxiv_rag.eval_chunking --gold eval/gold_v1.jsonl --out docs/results/phase2
```

Writes `<out>.json` and `<out>.md` (recall@k, MRR, paper recall, recall at 512/1024/2048 context tokens, chunk stats,
bootstrap 95% CIs, paired differences vs `fixed`, breakdowns by span type and title overlap). The first run embeds
all chunks (~25 min on CPU); later runs hit the embedding cache. `--allow-draft` scores unreviewed draft questions
for a pipeline dry run; the outputs are then marked DRAFT. Stale gold rows (re-parsed documents) are dropped and counted.

Phase 3 compares retrievers (dense, BM25, hybrid RRF/convex, each also with cross-encoder reranking) on one chunking (`structure`, 256 tokens), with the same gold set, hit rule and paper bootstrap:

```bash
PYTHONPATH=src .venv/Scripts/python -u -m arxiv_rag.eval_retrieval --gold eval/gold_v1.jsonl --out docs/results/phase3
```

Writes `<out>.json` and `<out>.md` (recall, paper recall, paired differences vs `dense` and of each `+rerank` vs its base, breakdowns by span type, copy and lexical overlap, reranker truncation check). Nothing is tuned on the gold set. Reranking runs at ~12 pairs/s on CPU, so the first run can take up to ~1 h; scores are cached, so reruns are fast. `--no-rerank` skips the reranked configs and `--limit N` scores only the first N questions.
