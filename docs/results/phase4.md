# Phase 4: answer generation with citations

> **PARTIAL: 31 questions answered under every config (of 192 in the gold set).** Not a full result.

**Gold set: machine-verified (LLM judge + rule filters), not human-reviewed**

- Questions scored: 31 of 192 gold questions (31 answered under every config; answers for 1 questions missing under some config, 0 answer rows stale or off-gold, 0 rows of other configs ignored)
- Corpus: 100 papers (31 with scored questions); chunks: structure, max_tokens 256, overlap 32
- Configs: `closed-book`, `dense-k5`, `bm25-k5`; passages numbered [1]..[k], no titles or ids shown
- Generator: `openai/gpt-oss-120b`, temperature 0
- Judge: `qwen/qwen3.8-27b`, temperature 0, sees the gold evidence quote as the reference
- Hit rule: same paper and chunk span covers >= 50% of the evidence span
- Cells: mean [95% percentile cluster bootstrap CI: papers resampled with replacement, all their questions kept; 1000 resamples, seed 0]
- Commit: e753166ab3a391e1194da474462ae638f58023f0

## Answer quality

`correct`: judge says yes; `correct or partial`: yes or partial; `faithful` (headline): every claim supported by the cited passages, over retrieval answers that are not refusals; an answer that cites no valid passage is forced to unfaithful, whatever the judge said. `faithful (cited)`: the judge's verdict among answers that cite at least one valid passage only (no forcing), so it is not penalised for uncited answers. `abstained`: the whole reply is INSUFFICIENT; prose refusals are counted separately below. Both count as incorrect. `context hit`: any of the k retrieved chunks hits the gold evidence; `citation hit`: any cited chunk does.

| config | correct | correct or partial | faithful | faithful (cited) | abstained | context hit | citation hit |
|---|---|---|---|---|---|---|---|
| closed-book | 0.129 [0.032, 0.258] | 0.226 [0.097, 0.387] | - | - | 0.323 [0.161, 0.484] | - | - |
| dense-k5 | 0.839 [0.710, 0.935] | 0.871 [0.742, 0.968] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.097 [0.000, 0.194] | 0.742 [0.581, 0.903] | 0.742 [0.581, 0.903] |
| bm25-k5 | 0.968 [0.903, 1.000] | 0.968 [0.903, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.032 [0.000, 0.097] | 0.903 [0.774, 1.000] | 0.903 [0.774, 1.000] |

## Abstentions and refusals

Re-derived from the answer text. Explicit abstention: the reply is exactly INSUFFICIENT. Prose refusal: a refusal phrasing ("the passages do not mention ...", "I cannot answer", ...) with no citation. Neither is judged; both score as incorrect.

| config | answers | explicit abstentions | prose refusals |
|---|---|---|---|
| closed-book | 31 | 10 | 0 |
| dense-k5 | 31 | 3 | 0 |
| bm25-k5 | 31 | 1 | 0 |

## Citation behaviour

Bracket groups are re-parsed from the answer text against the k retrieved passages; a group with any number that is 0 or above k is one invalid citation and none of its numbers count. `cites anything` is over non-refused answers only.

| config | answered | cites anything | invalid citations (total) | answers with an invalid citation |
|---|---|---|---|---|
| dense-k5 | 28 | 1.000 | 0 | 0 |
| bm25-k5 | 30 | 1.000 | 0 | 0 |

## Paired difference vs closed-book

diff [95% CI] and the fraction of resamples with diff <= 0. Same resampled papers for both configs. Not corrected for multiple comparisons (3 comparisons).

| comparison | correct | correct_or_partial |
|---|---|---|
| dense-k5 - closed-book | +0.710 [+0.548, +0.871] (0.000) | +0.645 [+0.484, +0.806] (0.000) |
| bm25-k5 - closed-book | +0.839 [+0.710, +0.968] (0.000) | +0.742 [+0.581, +0.903] (0.000) |

## Paired difference: BM25 vs dense

diff [95% CI] and the fraction of resamples with diff <= 0. Same resampled papers for both configs. Not corrected for multiple comparisons (3 comparisons).

| comparison | correct | correct_or_partial | context_hit | citation_hit |
|---|---|---|---|---|
| bm25-k5 - dense-k5 | +0.129 [+0.032, +0.258] (0.013) | +0.097 [+0.000, +0.194] (0.044) | +0.161 [+0.032, +0.290] (0.007) | +0.161 [+0.032, +0.290] (0.007) |

## Does retrieval failure explain wrong answers?

Correct rate split by whether any retrieved chunk hit the gold evidence. Point estimates; the `no context_hit` group is small when retrieval is good. A high `correct` there means the model answered from other passages or its own knowledge. `abstained` here includes prose refusals. `context_hit` uses the Phase 2/3 single-chunk 50% rule, so a gold span longer than one chunk (mostly `cross_paragraph`) can never be a context hit: the `no context_hit` group therefore includes some answerable questions.

Longest retrieved chunk: 1403 characters. Gold spans longer than that: 1 of 31 scored questions; of these, 0 are more than twice as long and cannot reach the 50% coverage a hit needs.

| config | group | n | correct | correct or partial | abstained |
|---|---|---|---|---|---|
| dense-k5 | context_hit | 23 | 1.000 | 1.000 | 0.000 |
| dense-k5 | no context_hit | 8 | 0.375 | 0.500 | 0.375 |
| bm25-k5 | context_hit | 28 | 1.000 | 1.000 | 0.000 |
| bm25-k5 | no context_hit | 3 | 0.667 | 0.667 | 0.333 |

## By span type

Point estimates only; small groups are noisy.

| span type | n | config | correct | context hit |
|---|---|---|---|---|
| paragraph | 31 | closed-book | 0.129 | n/a |
| paragraph | 31 | dense-k5 | 0.839 | 0.742 |
| paragraph | 31 | bm25-k5 | 0.968 | 0.903 |

## Token use (per call, for planning)

| call type | calls | mean prompt tokens | mean completion tokens |
|---|---|---|---|
| generation closed-book | 31 | 130.129 | 192.387 |
| generation dense-k5 | 31 | 1063.581 | 184.323 |
| generation bm25-k5 | 31 | 1081.839 | 187.258 |
| judge | 79 | 587.937 | 18.152 |

## Caveats

- Correctness and faithfulness are judged by one LLM (single judge, not human-checked) with a single generator model; both can share blind spots. The judge is a different family from the generator.
- The gold set is machine-verified, not human-reviewed (see eval/README.md), and biased towards named, factual passages; absolute numbers are optimistic.
- `correct` is evidence-relative: the judge sees the gold evidence quote, so an answer that is right but says something other than the quote provides can be marked partial or no. `faithful` ignores the reference and checks only the cited passages.
- The k retrieved passages may contain the right information in a chunk other than the gold span (another paper or a duplicate statement); `context hit` is stricter than 'the answer is available'.
- Abstentions and prose refusals count as incorrect and are excluded from the faithfulness rates; a config that refuses more can therefore look more faithful. Read the two together. The prose-refusal list is deliberately conservative, so some refusals phrased differently are still judged as answers.
- The headline `faithful` forces uncited answers to unfaithful; `faithful (cited)` does not. A large gap between them means the model often answers without citing.
- `context hit` needs one chunk to cover 50% of the gold span; spans longer than a chunk can never be a hit even when the answer is fully retrievable (see the span counts above).
- Paired comparisons and breakdowns are not corrected for multiple comparisons.
