# Phase 3: retrievers on one chunking

> **Lexical-anchor bias:** every gold question was required to reuse a specific term from its evidence passage, which favours BM25 and hybrids over what real users would ask. Read lexical vs dense gaps with that in mind (see the breakdown by lexical overlap).

**Gold set: machine-verified (LLM judge + rule filters), not human-reviewed**

- Questions: 192 (rows read 192, stale 0, status-filtered 0, off-corpus 0, duplicates 0, unparseable 0)
- Corpus: 100 papers (93 with questions); gold: `eval\gold_v1.jsonl`
- Gold status (kept rows): auto_ok 192
- Chunking: structure, max_tokens 256, overlap 32 (5685 chunks, mean 174.4 tokens)
- Embedding: BAAI/bge-small-en-v1.5 (fastembed-0.9.1), cosine, exact search
- Lexical: BM25 (k1=0.9, b=0.4), lowercase alnum + joined compounds, small stopword list, no stemming
- Hybrid `hybrid-rrf`: dense + BM25, each to depth 100; RRF k=60; weights [1.0, 1.0]
- Hybrid `hybrid-convex`: dense + BM25, each to depth 100; convex = min-max normalised; weights [1.0, 1.0]
- Rerank: Xenova/ms-marco-MiniLM-L-6-v2, top 50 of each base retriever
- No parameters were tuned on the gold set
- Hit rule: same paper and chunk span covers >= 50% of the evidence span
- Cells: mean [95% percentile cluster bootstrap CI: papers resampled with replacement, all their questions kept; 1000 resamples, seed 0]
- Commit: 33d1299afcfcb3be22188fecf9c768f214faa6ab

## Chunk recall

| retriever | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 | mrr@20 |
|---|---|---|---|---|---|---|
| dense | 0.484 [0.416, 0.563] | 0.656 [0.589, 0.723] | 0.719 [0.657, 0.784] | 0.818 [0.764, 0.870] | 0.865 [0.817, 0.909] | 0.595 [0.538, 0.657] |
| bm25 | 0.812 [0.759, 0.867] | 0.922 [0.886, 0.957] | 0.948 [0.915, 0.974] | 0.953 [0.922, 0.980] | 0.969 [0.944, 0.990] | 0.874 [0.835, 0.914] |
| hybrid-rrf | 0.641 [0.569, 0.713] | 0.818 [0.764, 0.876] | 0.870 [0.820, 0.914] | 0.927 [0.887, 0.959] | 0.964 [0.937, 0.989] | 0.741 [0.686, 0.797] |
| hybrid-convex | 0.729 [0.665, 0.793] | 0.922 [0.881, 0.957] | 0.938 [0.897, 0.971] | 0.953 [0.921, 0.980] | 0.974 [0.950, 0.995] | 0.824 [0.777, 0.866] |
| dense+rerank | 0.766 [0.709, 0.822] | 0.844 [0.792, 0.889] | 0.865 [0.813, 0.909] | 0.901 [0.857, 0.940] | 0.906 [0.862, 0.946] | 0.812 [0.761, 0.857] |
| bm25+rerank | 0.802 [0.749, 0.855] | 0.896 [0.851, 0.936] | 0.922 [0.881, 0.957] | 0.969 [0.945, 0.990] | 0.984 [0.966, 1.000] | 0.860 [0.818, 0.900] |
| hybrid-rrf+rerank | 0.807 [0.755, 0.859] | 0.901 [0.856, 0.941] | 0.922 [0.879, 0.958] | 0.964 [0.937, 0.989] | 0.974 [0.952, 0.995] | 0.862 [0.821, 0.902] |
| hybrid-convex+rerank | 0.807 [0.755, 0.859] | 0.901 [0.856, 0.941] | 0.922 [0.879, 0.958] | 0.964 [0.937, 0.989] | 0.974 [0.949, 0.995] | 0.862 [0.822, 0.902] |

## Paper recall (any chunk from the right paper)

| retriever | paper_recall@1 | paper_recall@5 | paper_recall@10 |
|---|---|---|---|
| dense | 0.854 [0.795, 0.907] | 0.932 [0.890, 0.965] | 0.948 [0.911, 0.978] |
| bm25 | 0.974 [0.949, 0.995] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| hybrid-rrf | 0.932 [0.891, 0.966] | 0.969 [0.941, 0.990] | 0.984 [0.963, 1.000] |
| hybrid-convex | 0.948 [0.909, 0.977] | 0.995 [0.983, 1.000] | 1.000 [1.000, 1.000] |
| dense+rerank | 0.958 [0.923, 0.985] | 0.964 [0.931, 0.990] | 0.964 [0.931, 0.990] |
| bm25+rerank | 0.995 [0.984, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| hybrid-rrf+rerank | 0.995 [0.984, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| hybrid-convex+rerank | 0.995 [0.984, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

## Paired difference vs `dense`

diff [95% CI] and the fraction of resamples with diff <= 0 (a one-sided bootstrap p-like number). Same resampled papers for both retrievers. Not corrected for multiple comparisons (11 comparisons x 4 metrics).

| retriever | recall@1 | recall@5 | recall@10 | mrr@20 |
|---|---|---|---|---|
| bm25 | +0.328 [+0.250, +0.406] (0.000) | +0.229 [+0.172, +0.288] (0.000) | +0.135 [+0.087, +0.188] (0.000) | +0.279 [+0.214, +0.341] (0.000) |
| hybrid-rrf | +0.156 [+0.106, +0.209] (0.000) | +0.151 [+0.104, +0.200] (0.000) | +0.109 [+0.069, +0.155] (0.000) | +0.145 [+0.111, +0.181] (0.000) |
| hybrid-convex | +0.245 [+0.177, +0.311] (0.000) | +0.219 [+0.165, +0.274] (0.000) | +0.135 [+0.091, +0.185] (0.000) | +0.228 [+0.180, +0.272] (0.000) |
| dense+rerank | +0.281 [+0.208, +0.353] (0.000) | +0.146 [+0.090, +0.204] (0.000) | +0.083 [+0.042, +0.126] (0.000) | +0.217 [+0.163, +0.272] (0.000) |
| bm25+rerank | +0.318 [+0.248, +0.386] (0.000) | +0.203 [+0.139, +0.264] (0.000) | +0.151 [+0.100, +0.203] (0.000) | +0.264 [+0.207, +0.319] (0.000) |
| hybrid-rrf+rerank | +0.323 [+0.252, +0.393] (0.000) | +0.203 [+0.135, +0.267] (0.000) | +0.146 [+0.092, +0.202] (0.000) | +0.267 [+0.210, +0.320] (0.000) |
| hybrid-convex+rerank | +0.323 [+0.252, +0.393] (0.000) | +0.203 [+0.135, +0.267] (0.000) | +0.146 [+0.092, +0.202] (0.000) | +0.267 [+0.210, +0.320] (0.000) |

## Paired difference: reranked vs its base retriever

diff [95% CI] and the fraction of resamples with diff <= 0 (a one-sided bootstrap p-like number). Same resampled papers for both retrievers. Not corrected for multiple comparisons (11 comparisons x 4 metrics).

| retriever | recall@1 | recall@5 | recall@10 | mrr@20 |
|---|---|---|---|---|
| dense+rerank | +0.281 [+0.208, +0.353] (0.000) | +0.146 [+0.090, +0.204] (0.000) | +0.083 [+0.042, +0.126] (0.000) | +0.217 [+0.163, +0.272] (0.000) |
| bm25+rerank | -0.010 [-0.063, +0.040] (0.702) | -0.026 [-0.058, +0.005] (0.964) | +0.016 [-0.015, +0.047] (0.207) | -0.014 [-0.052, +0.019] (0.801) |
| hybrid-rrf+rerank | +0.167 [+0.094, +0.230] (0.000) | +0.052 [+0.000, +0.105] (0.028) | +0.036 [+0.000, +0.077] (0.034) | +0.121 [+0.068, +0.169] (0.000) |
| hybrid-convex+rerank | +0.078 [+0.026, +0.126] (0.002) | -0.016 [-0.052, +0.021] (0.823) | +0.010 [-0.021, +0.045] (0.340) | +0.038 [+0.005, +0.071] (0.014) |

## By span type

Point estimates only; small groups are noisy.

| group | n | retriever | recall@1 | recall@5 | recall@10 | mrr@20 |
|---|---|---|---|---|---|---|
| cross_paragraph | 43 | dense | 0.442 | 0.721 | 0.837 | 0.557 |
| cross_paragraph | 43 | bm25 | 0.767 | 0.953 | 0.953 | 0.847 |
| cross_paragraph | 43 | hybrid-rrf | 0.558 | 0.884 | 0.977 | 0.708 |
| cross_paragraph | 43 | hybrid-convex | 0.721 | 0.953 | 0.953 | 0.820 |
| cross_paragraph | 43 | dense+rerank | 0.651 | 0.837 | 0.907 | 0.735 |
| cross_paragraph | 43 | bm25+rerank | 0.651 | 0.837 | 0.930 | 0.740 |
| cross_paragraph | 43 | hybrid-rrf+rerank | 0.651 | 0.837 | 0.907 | 0.738 |
| cross_paragraph | 43 | hybrid-convex+rerank | 0.651 | 0.837 | 0.907 | 0.737 |
| paragraph | 114 | dense | 0.518 | 0.719 | 0.816 | 0.620 |
| paragraph | 114 | bm25 | 0.807 | 0.930 | 0.939 | 0.863 |
| paragraph | 114 | hybrid-rrf | 0.684 | 0.877 | 0.904 | 0.762 |
| paragraph | 114 | hybrid-convex | 0.763 | 0.912 | 0.939 | 0.834 |
| paragraph | 114 | dense+rerank | 0.798 | 0.895 | 0.912 | 0.842 |
| paragraph | 114 | bm25+rerank | 0.825 | 0.947 | 0.974 | 0.882 |
| paragraph | 114 | hybrid-rrf+rerank | 0.833 | 0.947 | 0.974 | 0.888 |
| paragraph | 114 | hybrid-convex+rerank | 0.833 | 0.947 | 0.974 | 0.888 |
| sentence | 35 | dense | 0.429 | 0.714 | 0.800 | 0.562 |
| sentence | 35 | bm25 | 0.886 | 1.000 | 1.000 | 0.943 |
| sentence | 35 | hybrid-rrf | 0.600 | 0.829 | 0.943 | 0.712 |
| sentence | 35 | hybrid-convex | 0.629 | 1.000 | 1.000 | 0.793 |
| sentence | 35 | dense+rerank | 0.800 | 0.800 | 0.857 | 0.806 |
| sentence | 35 | bm25+rerank | 0.914 | 0.943 | 1.000 | 0.934 |
| sentence | 35 | hybrid-rrf+rerank | 0.914 | 0.943 | 1.000 | 0.930 |
| sentence | 35 | hybrid-convex+rerank | 0.914 | 0.943 | 1.000 | 0.930 |

## By copy-like questions

`copy` = the judge marked the question as mostly repeating the evidence wording (favours lexical retrieval); items without a judge verdict count as non-copy. Point estimates only.

| group | n | retriever | recall@1 | recall@5 | recall@10 | mrr@20 |
|---|---|---|---|---|---|---|
| copy (judge.copy == true) | 47 | dense | 0.511 | 0.830 | 0.915 | 0.660 |
| copy (judge.copy == true) | 47 | bm25 | 0.915 | 1.000 | 1.000 | 0.957 |
| copy (judge.copy == true) | 47 | hybrid-rrf | 0.745 | 0.936 | 0.979 | 0.837 |
| copy (judge.copy == true) | 47 | hybrid-convex | 0.830 | 1.000 | 1.000 | 0.908 |
| copy (judge.copy == true) | 47 | dense+rerank | 0.915 | 0.936 | 0.979 | 0.931 |
| copy (judge.copy == true) | 47 | bm25+rerank | 0.936 | 0.957 | 1.000 | 0.952 |
| copy (judge.copy == true) | 47 | hybrid-rrf+rerank | 0.936 | 0.957 | 1.000 | 0.952 |
| copy (judge.copy == true) | 47 | hybrid-convex+rerank | 0.936 | 0.957 | 1.000 | 0.952 |
| non-copy (judge.copy == false) | 145 | dense | 0.476 | 0.683 | 0.786 | 0.574 |
| non-copy (judge.copy == false) | 145 | bm25 | 0.779 | 0.931 | 0.938 | 0.847 |
| non-copy (judge.copy == false) | 145 | hybrid-rrf | 0.607 | 0.848 | 0.910 | 0.710 |
| non-copy (judge.copy == false) | 145 | hybrid-convex | 0.697 | 0.917 | 0.938 | 0.796 |
| non-copy (judge.copy == false) | 145 | dense+rerank | 0.717 | 0.841 | 0.876 | 0.773 |
| non-copy (judge.copy == false) | 145 | bm25+rerank | 0.759 | 0.910 | 0.959 | 0.829 |
| non-copy (judge.copy == false) | 145 | hybrid-rrf+rerank | 0.766 | 0.910 | 0.952 | 0.833 |
| non-copy (judge.copy == false) | 145 | hybrid-convex+rerank | 0.766 | 0.910 | 0.952 | 0.832 |

## By lexical overlap

Split at the median `overlap` of the scored questions (0.234); `overlap` is the longest common word n-gram between question and evidence as a fraction of the question's words. A retriever that wins mainly in the high group is exploiting wording reuse. Point estimates only.

| group | n | retriever | recall@1 | recall@5 | recall@10 | mrr@20 |
|---|---|---|---|---|---|---|
| overlap < 0.234 (low) | 96 | dense | 0.490 | 0.771 | 0.823 | 0.608 |
| overlap < 0.234 (low) | 96 | bm25 | 0.781 | 0.958 | 0.958 | 0.855 |
| overlap < 0.234 (low) | 96 | hybrid-rrf | 0.635 | 0.854 | 0.917 | 0.741 |
| overlap < 0.234 (low) | 96 | hybrid-convex | 0.719 | 0.927 | 0.958 | 0.815 |
| overlap < 0.234 (low) | 96 | dense+rerank | 0.719 | 0.833 | 0.875 | 0.773 |
| overlap < 0.234 (low) | 96 | bm25+rerank | 0.760 | 0.896 | 0.958 | 0.827 |
| overlap < 0.234 (low) | 96 | hybrid-rrf+rerank | 0.771 | 0.906 | 0.948 | 0.834 |
| overlap < 0.234 (low) | 96 | hybrid-convex+rerank | 0.771 | 0.906 | 0.948 | 0.834 |
| overlap >= 0.234 (high) | 96 | dense | 0.479 | 0.667 | 0.812 | 0.582 |
| overlap >= 0.234 (high) | 96 | bm25 | 0.844 | 0.938 | 0.948 | 0.893 |
| overlap >= 0.234 (high) | 96 | hybrid-rrf | 0.646 | 0.885 | 0.938 | 0.740 |
| overlap >= 0.234 (high) | 96 | hybrid-convex | 0.740 | 0.948 | 0.948 | 0.832 |
| overlap >= 0.234 (high) | 96 | dense+rerank | 0.812 | 0.896 | 0.927 | 0.850 |
| overlap >= 0.234 (high) | 96 | bm25+rerank | 0.844 | 0.948 | 0.979 | 0.892 |
| overlap >= 0.234 (high) | 96 | hybrid-rrf+rerank | 0.844 | 0.938 | 0.979 | 0.890 |
| overlap >= 0.234 (high) | 96 | hybrid-convex+rerank | 0.844 | 0.938 | 0.979 | 0.890 |

## Rerank ceiling

Fraction of questions whose top-50 base candidates contain a hit: the best recall@k any reranker of that base retriever could reach.

| base retriever | ceiling (top 50) | base recall@10 | reranked recall@10 |
|---|---|---|---|
| dense | 0.911 | 0.818 [0.764, 0.870] | 0.901 [0.857, 0.940] |
| bm25 | 0.984 | 0.953 [0.922, 0.980] | 0.969 [0.945, 0.990] |
| hybrid-rrf | 0.984 | 0.927 [0.887, 0.959] | 0.964 [0.937, 0.989] |
| hybrid-convex | 0.984 | 0.953 [0.921, 0.980] | 0.964 [0.937, 0.989] |

Dense has the lowest-ceiling candidate pool (0.911 vs best 0.984), so reranking dense cannot reach what reranking a lexical or hybrid pool can.

## Reranked hybrids compared

`hybrid-rrf+rerank` vs `hybrid-convex+rerank` on the same 192 questions: identical top-1 chunk for 100.0% of questions, identical first-hit rank for 99.0%. They differ on the remainder, so they are not fully indistinguishable.

## By anchor rarity

For each question: the evidence-quote terms (`goldgen._terms(quote, heads=True)`) that the question reuses (`goldgen.is_anchored` matching rule, applied per term), the rarest by paper document frequency over the corpus chunks (papers with a chunk containing the term, same normalised matching). The df includes the gold paper itself, so df=1 means the term occurs only in the gold paper and that bucket dominates. Point estimates only.

| group | n | retriever | recall@1 | recall@5 | recall@10 |
|---|---|---|---|---|---|
| rarest anchor in 1 paper | 126 | dense | 0.484 | 0.714 | 0.833 |
| rarest anchor in 1 paper | 126 | bm25 | 0.825 | 0.960 | 0.960 |
| rarest anchor in 1 paper | 126 | hybrid-rrf | 0.635 | 0.881 | 0.929 |
| rarest anchor in 1 paper | 126 | hybrid-convex | 0.762 | 0.944 | 0.960 |
| rarest anchor in 1 paper | 126 | dense+rerank | 0.778 | 0.881 | 0.913 |
| rarest anchor in 1 paper | 126 | bm25+rerank | 0.817 | 0.929 | 0.976 |
| rarest anchor in 1 paper | 126 | hybrid-rrf+rerank | 0.817 | 0.921 | 0.960 |
| rarest anchor in 1 paper | 126 | hybrid-convex+rerank | 0.817 | 0.921 | 0.960 |
| rarest anchor in 2-5 papers | 37 | dense | 0.486 | 0.784 | 0.811 |
| rarest anchor in 2-5 papers | 37 | bm25 | 0.811 | 0.946 | 0.973 |
| rarest anchor in 2-5 papers | 37 | hybrid-rrf | 0.676 | 0.892 | 0.919 |
| rarest anchor in 2-5 papers | 37 | hybrid-convex | 0.730 | 0.946 | 0.946 |
| rarest anchor in 2-5 papers | 37 | dense+rerank | 0.757 | 0.838 | 0.892 |
| rarest anchor in 2-5 papers | 37 | bm25+rerank | 0.784 | 0.892 | 0.946 |
| rarest anchor in 2-5 papers | 37 | hybrid-rrf+rerank | 0.784 | 0.892 | 0.946 |
| rarest anchor in 2-5 papers | 37 | hybrid-convex+rerank | 0.784 | 0.892 | 0.946 |
| rarest anchor in >5 papers | 29 | dense | 0.483 | 0.655 | 0.759 |
| rarest anchor in >5 papers | 29 | bm25 | 0.759 | 0.897 | 0.897 |
| rarest anchor in >5 papers | 29 | hybrid-rrf | 0.621 | 0.793 | 0.931 |
| rarest anchor in >5 papers | 29 | hybrid-convex | 0.586 | 0.897 | 0.931 |
| rarest anchor in >5 papers | 29 | dense+rerank | 0.724 | 0.828 | 0.862 |
| rarest anchor in >5 papers | 29 | bm25+rerank | 0.759 | 0.931 | 0.966 |
| rarest anchor in >5 papers | 29 | hybrid-rrf+rerank | 0.793 | 0.966 | 1.000 |
| rarest anchor in >5 papers | 29 | hybrid-convex+rerank | 0.793 | 0.966 | 1.000 |

## Anchor-removed ablation

Each question is rewritten by deleting every whitespace-separated word for which a shared anchor term matches it under the `goldgen.is_anchored` rule (see above); 192 of 192 questions changed, 0 became empty (scored as a miss). Only the non-reranked base retrievers are rerun: reranked configs are skipped to avoid hours of new cross-encoder scoring. Point estimates; change = ablated - normal.

| retriever | recall@5 normal -> ablated (change) | recall@10 normal -> ablated (change) |
|---|---|---|
| dense | 0.719 -> 0.562 (-0.156) | 0.818 -> 0.615 (-0.203) |
| bm25 | 0.948 -> 0.839 (-0.109) | 0.953 -> 0.875 (-0.078) |
| hybrid-rrf | 0.870 -> 0.724 (-0.146) | 0.927 -> 0.776 (-0.151) |
| hybrid-convex | 0.938 -> 0.786 (-0.151) | 0.953 -> 0.854 (-0.099) |

## Caveats

- Lexical-anchor bias: every gold question was required to reuse a specific term from its evidence, which favours BM25/hybrid; absolute BM25 and hybrid numbers are optimistic. BM25 minus dense recall@5 is +0.229 on all questions, +0.197 on the 66 questions whose rarest anchor occurs in more than one paper, +0.276 after anchor removal. The BM25 lead persists for common anchors and after anchor removal, so it points to a broader lexical-match advantage of BM25 over the quantized bge-small on LLM-paraphrased factual questions; it cannot be fully separated from the gold-set construction.
- The embedding model is the quantized `Qdrant/bge-small-en-v1.5-onnx-Q` (fastembed default), not fp32.
- Reranker: `Xenova/ms-marco-MiniLM-L-6-v2`, candidates = base top 50. Truncation check: 0 of 17163 distinct (query, chunk) pairs exceed the model's 512-token limit (longest pair 306 tokens).
- No tuning on the gold set: BM25 k1/b, RRF k, hybrid depth, fusion weights and rerank depth are fixed defaults. Tuning them here would overfit 192 questions.
- BM25 tokenizer: lowercase alphanumeric runs plus hyphen/dot-joined compounds, no stemming.
- Questions are LLM-drafted and machine-verified (`auto_ok`: LLM judge + rule filters, no human review) unless the gold status line says otherwise.
- Paired comparisons and breakdowns are not corrected for multiple comparisons; breakdown cells are point estimates on small groups.
- The token-budget metric of Phase 2 is omitted: chunks are identical across retrievers. Ranking walks are capped at K_MAX = 20.
