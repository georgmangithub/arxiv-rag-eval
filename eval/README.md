# eval/

Files for the chunking study.

| File | What |
|---|---|
| `corpus_v1.txt` | Frozen list of arXiv ids (one per line, `#` comments allowed). Fetch with `python -m arxiv_rag.ingest --ids-file eval/corpus_v1.txt`. Create from a query run with `--write-ids eval/corpus_v1.txt`. |
| `gold_draft.jsonl` | LLM-drafted questions, all `"status": "draft"`. Made by `python -m arxiv_rag.goldgen --n 200 --seed 0 --out eval/gold_draft.jsonl`. Appends and resumes; `gold_draft.jsonl.rejected` lists spans the model skipped or whose question hit a banned phrase. |
| `gold_v1.jsonl` | The accepted subset: items with status `ok`, `edit` or `auto_ok`. This is what the evaluation uses. If it was built without any human review (only `auto_ok`), it is **machine-verified, not human-reviewed** (see below). |

## Review procedure

Open `gold_draft.jsonl` and set `status` on every item:

- `ok`: the question is answerable from `evidence_quote` alone, names its subject (a stranger searching ~100 papers would know what it is about), and is not a near copy of the quote.
- `edit`: fix the `question` text by hand, then set `ok`. Never change `evidence_quote`; it must stay an exact substring of the parsed document text (`Document.text`).
- `bad`: unanswerable, vague, ambiguous, or wrong. Dropped.

Sort by `overlap` descending to review shortcut-prone questions first (overlap is the longest common word n-gram as a fraction of the question's words; above ~0.5 the question mostly copies the quote). Then copy the `ok` items to `gold_v1.jsonl`.

## Schema (one JSON object per line)

```json
{"qid": "2610.31317v1:1204-1650", "paper_id": "2610.31317v1",
 "question": "...", "evidence_quote": "...",
 "char_start": 1204, "char_end": 1650, "doc_sha256": "<64 hex chars>",
 "span_type": "paragraph", "status": "draft", "overlap": 0.21, "title_overlap": 0.0}
```

- `qid`: `<paper_id>:<char_start>-<char_end>` of the quote in `Document.text`. Unique per file.
- `paper_id`: filename stem as in `data/parsed`.
- `char_start`, `char_end`: offsets of the quote in `Document.text`.
- `doc_sha256`: sha256 (UTF-8) of that `Document.text`. The eval ignores (and counts as stale) any row where `doc.text[char_start:char_end] != evidence_quote` or the sha differs from the current document, i.e. the parser or `document.py` changed since the row was drafted; it keeps the last row per `(qid, doc_sha256)` and errors only if no valid rows remain. Regenerate stale rows instead of editing offsets.
- `span_type`: `paragraph`, `cross_paragraph` (last sentence of a paragraph + first sentence of the next, same section, joined by the paragraph separator), or `sentence`. All spans start with an uppercase letter or digit and end with terminal punctuation (`.!?`, optionally followed by a closing quote or bracket), which filters out PDF column/page-break fragments.
- `status`: `draft`, `ok`, `edit`, `bad` (human review), `auto_ok`, `auto_bad` (machine judge).
- `judge` (optional, set by `arxiv_rag.judge`): `{"model", "answerable", "specific", "copy", "needs_both" (cross_paragraph only), "reason"}`.
- `overlap`: float in 0..1, see above.
- `title_overlap`: float in 0..1, fraction of the question's non-stopword lowercase word types that appear in the paper title. The question writer is not shown the title, but analysis should still split results by this (high values favour strategies that embed the title).

`goldgen` uses the ids in `eval/corpus_v1.txt` by default (`--ids-file` to override; all parsed papers if the default file is absent) and fails if an id has no parsed file.

Known bias: the sentence-edge filter requires a span to start with an uppercase letter or digit, so sentences that start with a lowercase term (`kNN`, `mBERT`) are never sampled. This is small but systematic.

Known bias: spans must contain a specific term (a name with an inner capital such as `ThreadRAG`, a letters-plus-digits token such as `GPT-4`, or a percentage; generic acronyms like `LLM` and `RAG`, citation years like `2026b`, ordinals, Roman numerals, short figure refs like `3a`/`S1` and short or stop-word all-caps tokens like `IN` don't count), and every question must reuse one of them (case, hyphens and spaces are ignored, so `BM25-based`, `GPT 4`, `BERTs` and `12.5 percent` all match). This keeps questions findable in a 100-paper corpus, but it skews the gold set towards named, factual passages, which are lexically easier to retrieve than conceptual ones. It applies equally to every chunking strategy, but absolute recall numbers are optimistic. Math-heavy spans (garbled by PDF extraction) and section outlines (`§2.1 ... §2.2`) are also excluded.

Known bias: sentence spans survive these filters least (about 37% of sampled sentences vs about 61% of paragraphs), so the surviving sentences skew more toward term-dense text than the surviving paragraphs do. Report results per `span_type` rather than only pooled.

## Maintaining the draft

- `python -m arxiv_rag.goldgen --out eval/gold_draft.jsonl --retry-rejected no_anchor,too_long [--dry-run]`: re-asks the model about exactly the spans in `gold_draft.jsonl.rejected` with those reasons (comma list; no re-sampling), rebuilt from their qid offsets against the current parsed documents. Successes are appended to `gold_draft.jsonl`; the others stay in the rejected file with their new reason. Rows whose `doc_sha256` no longer matches the current document are skipped (counted as `stale`, left untouched). A span that fails the current candidate filters (for example `no_specific_term`) is not sent to the model; its row gets the new reason. The rejected file is rewritten atomically (also if the run is interrupted), and a span already in the output is never written twice, so rerunning is safe. `--dry-run` prints how many rows would be retried per reason and how many fail the current filters, without LLM calls or writes.
- `python -m arxiv_rag.goldgen --out eval/gold_draft.jsonl --recheck`: no LLM calls. Recomputes, for every item, whether the current rules would still accept it (all the span and question filters: specific term in the quote, question at most 35 words, question anchored to the quote) and sets `"auto_flag"` to the first failing reason (`no_specific_term`, `too_long`, `no_anchor`, ...), or removes the field if the item is clean. A `cross_paragraph` item whose question is anchored in only one of its two sentences gets the hint `one_half_only`: check whether the question really needs both sentences (requiring a term from each half automatically rejected most good questions, so this is left to review). Rewrites the file atomically with row order and all other fields unchanged, never deletes a row, and prints the count per flag.
- `auto_flag` (optional field): present only on items that fail the current rules. It is a hint for review, not a verdict: `status` is untouched, so decide by hand whether to keep, edit or mark `bad`.

## Reviewing (terminal tool)

- `python -m arxiv_rag.review [--draft eval/gold_draft.jsonl] [--all] [--flagged]`: walks `draft` items one at a time (paper title from `data/metadata.jsonl`, `auto_flag` with an explanation, question, evidence with the paragraph boundary marked for `cross_paragraph`). Keys: `o` ok, `b` bad, `e` edit the question (evidence and offsets untouched; the old text goes to `question_original`, status `edit`), `s` skip (stays draft), `n` note (`review_note`), `u` undo, `q` quit. Every decision is saved at once (atomic rewrite; row order, other fields and unparseable lines are kept), so rerunning resumes. `--all` revisits reviewed items too, `--flagged` shows only items with an `auto_flag`. It refuses to start if `<draft>.lock` exists and removes it on exit, so do not run it while `goldgen --recheck/--retry-rejected` is running.
- `python -m arxiv_rag.review --export [--out eval/gold_v1.jsonl]`: writes the `ok`, `edit` and `auto_ok` items (the last row per `(qid, doc_sha256)` wins first, whatever its status, then the status filter is applied; LF endings), prints counts by status and `span_type`, and warns below 150 items.
- `python -m arxiv_rag.review --stats`: counts by status x `span_type`, and by `auto_flag`.

## Machine judge (replaces human review)

- `python -m arxiv_rag.judge [--draft eval/gold_draft.jsonl] [--model qwen/qwen3.8-27b] [--rpm 10] [--dry-run] [--limit N] [--no-json-mode]`: a second model, from a different family than the one that drafted the questions (`openai/gpt-oss-120b`), reads only the `question` and the `evidence_quote` (never the paper title or other fields) and returns strict JSON: `answerable` (the evidence alone fully answers the question), `specific` (the question names a method, dataset, model or number that would let a searcher pick this paper out of ~100 NLP/retrieval papers), `copy` (the question mostly copies the evidence wording) and, for `cross_paragraph` items, `needs_both` (the answer needs both sentences). `<think>` blocks in the reply are stripped.
- The verdict is stored in `judge`. `status` becomes `auto_ok` iff `answerable` and `specific` and the item's `auto_flag` is not `no_anchor`, `no_specific_term` or `too_long`; otherwise `auto_bad`. `copy` and `needs_both: false` are recorded but do not reject (`copy` feeds the sensitivity breakdown of `eval_chunking`: results on all items vs only `judge.copy == false`). `question`, `evidence_quote` and offsets are never changed.
- Only items with status `draft` and no `judge` field are asked, so rerunning resumes; malformed replies are counted and retried next run. Each verdict is saved at once (atomic rewrite; row order, other fields and unparseable lines kept). Like the review tool it refuses to run if `<draft>.lock` exists and removes the lock on exit (also on Ctrl-C or an exhausted quota, which stops cleanly with progress kept). `--dry-run` only counts pending items. At the end it prints judged / `auto_ok` / `auto_bad` (by reason) / malformed, and `auto_ok`/`auto_bad` by `span_type`.
- `python -m arxiv_rag.judge --restatus [--draft ...]`: no LLM calls. Recomputes `status` from the stored `judge` verdict and the current `auto_flag` for every item that has a `judge` field and status `auto_ok` or `auto_bad` (use after changing the rule or after `goldgen --recheck`). Items with human statuses (`ok`/`edit`/`bad`) and drafts without a verdict are never touched. Same lock and atomic write as the judge run; prints counts by status and `span_type`.
- Export: `python -m arxiv_rag.review --export` includes `auto_ok` together with `ok` and `edit`; `--stats` lists `auto_ok` and `auto_bad` rows.

**Provenance of `gold_v1`:** it is machine-verified (two model families plus the rule filters of `goldgen`), not human-reviewed. State this wherever results on it are reported. Judge and drafter can share blind spots, so treat absolute numbers with care; comparisons between chunkers on the same set are less affected.
