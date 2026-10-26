# Phase 2: chunking strategies, dense retrieval

**Gold set: machine-verified (LLM judge + rule filters), not human-reviewed**

- Questions: 192 (rows read 192, stale 0, status-filtered 0, off-corpus 0, duplicates 0, unparseable 0)
- Corpus: 100 papers (93 with questions); gold: `eval\gold_v1.jsonl`
- Gold status (kept rows): auto_ok 192
- Chunking: max_tokens 256, overlap 32
- Embedding: BAAI/bge-small-en-v1.5 (fastembed-0.9.1), cosine, exact search
- Hit rule: same paper and chunk span covers >= 50% of the evidence span
- Cells: mean [95% percentile cluster bootstrap CI: papers resampled with replacement, all their questions kept; 1000 resamples, seed 0]
- Commit: 5e655db1334edda36dddf4592318ac41e76cea51

## Chunk recall

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 | mrr@20 |
|---|---|---|---|---|---|---|
| fixed | 0.495 [0.427, 0.563] | 0.661 [0.595, 0.728] | 0.719 [0.651, 0.789] | 0.781 [0.718, 0.843] | 0.812 [0.755, 0.867] | 0.593 [0.530, 0.654] |
| recursive | 0.469 [0.400, 0.537] | 0.667 [0.599, 0.733] | 0.714 [0.647, 0.773] | 0.781 [0.718, 0.838] | 0.854 [0.801, 0.900] | 0.582 [0.523, 0.640] |
| structure | 0.484 [0.416, 0.563] | 0.656 [0.589, 0.723] | 0.719 [0.657, 0.784] | 0.818 [0.764, 0.870] | 0.865 [0.817, 0.909] | 0.595 [0.538, 0.657] |
| structure+header | 0.411 [0.342, 0.483] | 0.568 [0.490, 0.645] | 0.625 [0.551, 0.699] | 0.734 [0.665, 0.800] | 0.792 [0.733, 0.850] | 0.513 [0.449, 0.577] |

## Paper recall (any chunk from the right paper)

| strategy | paper_recall@1 | paper_recall@5 | paper_recall@10 |
|---|---|---|---|
| fixed | 0.818 [0.758, 0.875] | 0.901 [0.854, 0.942] | 0.927 [0.888, 0.963] |
| recursive | 0.854 [0.801, 0.905] | 0.901 [0.856, 0.943] | 0.927 [0.886, 0.965] |
| structure | 0.854 [0.795, 0.907] | 0.932 [0.890, 0.965] | 0.948 [0.911, 0.978] |
| structure+header | 0.755 [0.687, 0.822] | 0.839 [0.775, 0.893] | 0.885 [0.831, 0.935] |

## Recall at a context budget (tokens of retrieved chunks)

| strategy | recall@512tok | recall@1024tok | recall@2048tok |
|---|---|---|---|
| fixed | 0.604 [0.534, 0.672] | 0.703 [0.635, 0.770] | 0.755 [0.689, 0.822] |
| recursive | 0.609 [0.546, 0.677] | 0.703 [0.634, 0.768] | 0.776 [0.712, 0.832] |
| structure | 0.620 [0.550, 0.689] | 0.729 [0.668, 0.793] | 0.823 [0.773, 0.873] |
| structure+header | 0.531 [0.455, 0.609] | 0.620 [0.544, 0.694] | 0.734 [0.663, 0.800] |

## Chunk statistics (tokens per chunk, as embedded)

| strategy | chunks | mean | median | p95 |
|---|---|---|---|---|
| fixed | 4415 | 252.6 | 256 | 256 |
| recursive | 4753 | 208.2 | 222 | 255 |
| structure | 5685 | 174.4 | 189 | 234 |
| structure+header | 5685 | 197.3 | 212 | 254 |

## Paired difference vs `fixed`

diff [95% CI] and the fraction of resamples with diff <= 0 (a one-sided bootstrap p-like number). Same resampled papers for both strategies. Paired comparisons are not corrected for multiple comparisons (3 strategies x 3 metrics).

| strategy | recall@5 | recall@10 | recall@1024tok |
|---|---|---|---|
| recursive | -0.005 [-0.064, +0.056] (0.605) | +0.000 [-0.064, +0.066] (0.548) | +0.000 [-0.056, +0.062] (0.535) |
| structure | +0.000 [-0.064, +0.067] (0.512) | +0.036 [-0.026, +0.096] (0.138) | +0.026 [-0.036, +0.092] (0.244) |
| structure+header | -0.094 [-0.162, -0.025] (0.997) | -0.047 [-0.109, +0.011] (0.949) | -0.083 [-0.151, -0.011] (0.990) |

## By span type

Point estimates only; small groups are noisy.

| group | n | strategy | recall@1 | recall@5 | recall@10 | mrr@20 | recall@1024tok |
|---|---|---|---|---|---|---|---|
| cross_paragraph | 43 | fixed | 0.558 | 0.860 | 0.930 | 0.680 | 0.814 |
| cross_paragraph | 43 | recursive | 0.419 | 0.744 | 0.837 | 0.554 | 0.744 |
| cross_paragraph | 43 | structure | 0.442 | 0.721 | 0.837 | 0.557 | 0.721 |
| cross_paragraph | 43 | structure+header | 0.419 | 0.721 | 0.744 | 0.534 | 0.698 |
| paragraph | 114 | fixed | 0.474 | 0.675 | 0.737 | 0.562 | 0.675 |
| paragraph | 114 | recursive | 0.500 | 0.702 | 0.737 | 0.594 | 0.684 |
| paragraph | 114 | structure | 0.518 | 0.719 | 0.816 | 0.620 | 0.728 |
| paragraph | 114 | structure+header | 0.430 | 0.579 | 0.719 | 0.515 | 0.579 |
| sentence | 35 | fixed | 0.486 | 0.686 | 0.743 | 0.585 | 0.657 |
| sentence | 35 | recursive | 0.429 | 0.714 | 0.857 | 0.574 | 0.714 |
| sentence | 35 | structure | 0.429 | 0.714 | 0.800 | 0.562 | 0.743 |
| sentence | 35 | structure+header | 0.343 | 0.657 | 0.771 | 0.481 | 0.657 |

## By title overlap

title_overlap > 0 means the question shares words with the paper title. If structure+header gains mainly there, the gain is title leakage, not better chunking.

| group | n | strategy | recall@1 | recall@5 | recall@10 | mrr@20 | recall@1024tok |
|---|---|---|---|---|---|---|---|
| title_overlap == 0 | 102 | fixed | 0.451 | 0.706 | 0.765 | 0.565 | 0.696 |
| title_overlap == 0 | 102 | recursive | 0.510 | 0.706 | 0.775 | 0.607 | 0.696 |
| title_overlap == 0 | 102 | structure | 0.480 | 0.706 | 0.804 | 0.581 | 0.716 |
| title_overlap == 0 | 102 | structure+header | 0.402 | 0.569 | 0.647 | 0.484 | 0.569 |
| title_overlap > 0 | 90 | fixed | 0.544 | 0.733 | 0.800 | 0.624 | 0.711 |
| title_overlap > 0 | 90 | recursive | 0.422 | 0.722 | 0.789 | 0.553 | 0.711 |
| title_overlap > 0 | 90 | structure | 0.489 | 0.733 | 0.833 | 0.612 | 0.744 |
| title_overlap > 0 | 90 | structure+header | 0.422 | 0.689 | 0.833 | 0.546 | 0.678 |

## Sensitivity: copy-like questions

`non-copy` drops items the judge marked `copy` (questions that mostly repeat the evidence wording, which favour lexical overlap); items without a judge verdict count as non-copy. Point estimates only.

| group | n | strategy | recall@5 | recall@10 | recall@1024tok |
|---|---|---|---|---|---|
| all | 192 | fixed | 0.719 | 0.781 | 0.703 |
| all | 192 | recursive | 0.714 | 0.781 | 0.703 |
| all | 192 | structure | 0.719 | 0.818 | 0.729 |
| all | 192 | structure+header | 0.625 | 0.734 | 0.620 |
| non-copy (judge.copy == false) | 145 | fixed | 0.676 | 0.752 | 0.669 |
| non-copy (judge.copy == false) | 145 | recursive | 0.697 | 0.779 | 0.690 |
| non-copy (judge.copy == false) | 145 | structure | 0.683 | 0.786 | 0.690 |
| non-copy (judge.copy == false) | 145 | structure+header | 0.593 | 0.710 | 0.586 |

## Caveats

- The embedding model is the quantized `Qdrant/bge-small-en-v1.5-onnx-Q` (fastembed default), not fp32.
- 3 of 100 papers have no detected sections, so the structure strategies treat each as one section.
- Gold questions are biased towards named, factual passages (see eval/README.md); absolute recall is optimistic.
- Questions are LLM-drafted. Rows count if human-reviewed (`ok`/`edit`) or machine-verified (`auto_ok`: LLM judge + rule filters, no human review); see the gold status line above.
- Chunks are only comparable by span; the header prefix of structure+header is not scored as evidence.
- `fixed` uses overlapping windows, so two of its top-k chunks can cover the same evidence (two chances at one hit); the other strategies do not overlap.
- The token-budget metric walks the ranking and stops at the first chunk that would overflow the budget (no skipping to smaller chunks). Chunk tokens include [CLS]/[SEP] and, for structure+header, the header prefix: header tokens are a real cost of that strategy.
- The ranking walk is capped at K_MAX = 20; budget metrics cannot see beyond rank 20.
