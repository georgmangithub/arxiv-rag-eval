# arxiv-rag-eval

A RAG system over arXiv AI papers, built to answer one question honestly: **which retrieval choices
actually improve answer quality, and by how much?** Every design decision is measured, not assumed.

Everything is free to build and run: local embeddings, a free-tier LLM API with a local Ollama
fallback, and open-source tooling. It runs on an 8 GB laptop with no GPU.

> Status: Phases 0-4 done (Phase 4 partial: 31 questions, Groq quota). Phase 2: chunking strategies are statistically tied; a title header hurts. Phase 3: BM25 beats dense by ~23 points at recall@5 on these entity-named questions. Phase 4: retrieval lifts correct answers from 13% (closed-book) to 84% (dense) and 97% (BM25). See [docs/results/](docs/results/).

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

Phase 4 answers the gold questions with citations (closed-book, dense top-5, BM25 top-5 over the `structure` chunks), then scores correctness and faithfulness. It needs `LLM_API_KEY` in `.env` and is built for the Groq free tier (~1000 requests/day per model, ~8000 tokens/minute):

```bash
PYTHONPATH=src .venv/Scripts/python -u -m arxiv_rag.generate --out eval/answers_v1.jsonl            # 3 x 192 calls, --rpm 4
PYTHONPATH=src .venv/Scripts/python -u -m arxiv_rag.eval_generation --answers eval/answers_v1.jsonl --out docs/results/phase4
```

`generate` appends one row per (question, config) and runs the configs interleaved per question, so a stopped run is still a paired comparison on the first N questions. A quota stop is clean: rerun the same command later to resume. `--limit N` answers only the first N questions. `eval_generation` judges each non-abstained answer once (verdicts cached in `eval/answer_judgements_v1.jsonl`, same resume behaviour; `--no-judge` uses the cache only) and writes `<out>.json` and `<out>.md`: correct / faithful / abstention rates, retrieval-hit rates, paired differences vs closed-book and BM25 vs dense, correct rate split by whether retrieval hit the evidence, and breakdowns by span type. Only questions answered under every config in `--configs` (default `closed-book dense-k5 bm25-k5`) are scored; quality metrics use the questions judged under every config and the excluded count is reported (an item whose judge call fails is retried once, and after 3 failed runs gets a `judge_error` verdict). Abstentions (reply is exactly `INSUFFICIENT`), prose refusals and citations are re-derived from the answer text, not read from the stored fields. Correctness is judged against the gold evidence quote, so it is evidence-relative.
