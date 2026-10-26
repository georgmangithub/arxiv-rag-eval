# Findings: what actually helps retrieval and answers over arXiv papers

This project measured each choice in a small RAG pipeline instead of assuming it helps. It runs on a laptop with no GPU, using free tooling. The sections below say what held up, what didn't, and how far the numbers can be trusted.

Detailed reports: [Phase 2](results/phase2.md) (chunking), [Phase 3](results/phase3.md) (retrievers), [Phase 4](results/phase4.md) (answers).

## Setup in one paragraph

The corpus is 100 recent cs.CL papers about retrieval, pinned in `eval/corpus_v1.txt`, parsed into paragraphs with section and page metadata. The gold set has 192 questions, each tied to an exact evidence span in a paper. A retrieved chunk is a hit if it covers at least 50% of that span. Every number below is a mean with a 95% confidence interval from a paper-level cluster bootstrap: whole papers are resampled, because questions from the same paper are not independent.

## Results

### 1. Chunking strategy barely matters, and adding a title header hurts (Phase 2)

| chunking (dense bge-small, 256 tokens) | recall@5 | recall@10 |
|---|---|---|
| fixed windows, 32-token overlap | 0.72 | 0.78 |
| recursive (paragraph → sentence → word) | 0.71 | 0.78 |
| structure-aware (within sections) | 0.72 | 0.82 |
| structure-aware + "title > section" prefix | **0.63** | 0.73 |

- **Three-way tie.** Fixed, recursive and structure-aware chunking are statistically tied. Every paired difference against fixed has a CI that includes zero.
- **The header hurts.** Prefixing each chunk with "title > section" lowers recall@5 by 9 points (CI −16 to −3). It is not a bug: the spans are identical and the embeddings were re-checked. Almost every title in this corpus is about retrieval, so the prefix adds the same generic words to every chunk. That moves chunks of different papers closer together and dilutes the detail a question asks about.
- **Header verdict.** On a topically narrow corpus, don't embed titles into chunks.
- **Overlap helps boundary-crossing evidence.** On questions whose evidence crosses a paragraph boundary (n = 43, noisy), only fixed's overlapping windows can span it, and fixed scores 0.86 recall@5 against about 0.72 for the others.

### 2. BM25 beats dense retrieval on this kind of question (Phase 3)

| retriever (structure chunks) | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| dense (bge-small) | 0.48 | 0.72 | 0.82 |
| **BM25** | **0.81** | **0.95** | 0.95 |
| hybrid, RRF | 0.64 | 0.87 | 0.93 |
| hybrid, convex | 0.73 | 0.94 | 0.95 |
| dense + MiniLM reranker (top 50) | 0.77 | 0.87 | 0.90 |
| BM25 / hybrids + reranker | 0.80 to 0.81 | 0.92 | 0.96 to 0.97 |

- **BM25 leads by a wide margin.** It beats dense by 23 points at recall@5 (CI +17 to +29).
- **The lead isn't only rare names.** The gap is similar whether a question's named term appears in 1 paper or in more than 5. With the named terms removed from the questions, BM25 still leads 0.84 to 0.56.
- **The likely cause.** The questions paraphrase a specific passage, and the small quantized embedder is weak at matching exact entities and numbers.
- **Reranking fixes dense's ranking, not its recall.** It lifts dense's recall@1 from 0.48 to 0.77. It cannot pass dense's ceiling: dense's top 50 contains the evidence only 91% of the time, against 98% for BM25.
- **The fusion method stops mattering after reranking.** The two hybrids rank their candidates differently, yet the cross-encoder brings both to the same top-1 for every question.
- **Recommendation for this kind of corpus and question:** plain BM25. It is the most accurate setup here, and it needs no model.

### 3. Retrieval is what makes answers correct (Phase 4, partial)

| answering setup (gpt-oss-120b) | correct | abstained | evidence in the top 5 |
|---|---|---|---|
| closed-book | 0.13 | 0.32 | – |
| dense, top 5 | 0.84 | 0.10 | 0.74 |
| BM25, top 5 | **0.97** | 0.03 | 0.90 |

- **Retrieval adds 71 to 84 points.** That is the gain in correct answers over closed-book.
- **BM25's edge carries through to answers.** It is 13 points better than dense (CI +3 to +26).
- **The edge is explained by retrieval.** Whenever the evidence was among the retrieved chunks, both setups answered correctly every time. The answer gap is the retrieval gap.
- **Faithfulness was 100%.** Every answer's claims were supported by the passages it cited. A hand check of 10 judge verdicts agreed with all 10.

**This phase is partial: only 31 questions.** Groq's daily quota for the generator ran out, so the intervals are wide.

## What we got wrong along the way

These were all caught before they reached a result, and each one would have biased it.

- **Query instruction missing.** The bge query instruction was not being applied, because fastembed doesn't add it for this model. Every dense number would have been understated.
- **Title shown to the question writer.** The LLM writing gold questions saw the paper title. That would have favoured the header strategy, which embeds the title.
- **Fragments as evidence.** 35% of the "cross-paragraph" evidence spans were PDF column-break fragments, not real sentence pairs.
- **Strict copy check.** The judge's "copy" check would have dropped most single-sentence questions and skewed the gold set towards paragraphs. Copy-like questions are kept instead, and every phase reports results without them as a sensitivity check.
- **Over-narrow confidence intervals.** Resampling questions instead of papers made the intervals too narrow.
- **Unrecognised citation format.** The generator often writes citations with full-width brackets, like `【2】`. The parser missed them, which made a third of the answers look uncited.

## Limitations: how far to trust this

- **The gold set is machine-verified, not human-reviewed.** gpt-oss-120b wrote the questions, and qwen3.8-27b judged them, alongside rule filters. A spot check of 20 accepted questions found about 17 good.
- **The gold set favours named, factual questions.** Each question had to reuse a specific term from its evidence, so that it could be found among 100 papers. That inflates absolute recall and favours BM25. The comparisons between chunking strategies are less affected than the dense-vs-BM25 gap. For conceptual questions with no names, a hybrid is the safer default.
- **One embedding model,** and a quantized one: bge-small, as packaged by fastembed. A stronger embedder could narrow the BM25 gap.
- **One narrow corpus.** It is 100 papers on a single topic. The header result especially depends on that.
- **Phase 4 covers 31 questions.** Its effects are large enough to be clear, but the intervals are wide.
- **No correction for multiple comparisons.** The paired tests are not corrected for running many of them.

## Reproducing

Every result file records the commit it ran on. The gold set, the answers and the judge verdicts are committed. CI re-checks BM25 recall against a committed baseline on every push. The [README](../README.md) has the commands.
