# eval/

Files for the chunking study.

| File | What |
|---|---|
| `corpus_v1.txt` | Frozen list of arXiv ids (one per line, `#` comments allowed). Fetch with `python -m arxiv_rag.ingest --ids-file eval/corpus_v1.txt`. Create from a query run with `--write-ids eval/corpus_v1.txt`. |
| `gold_draft.jsonl` | LLM-drafted questions, all `"status": "draft"`. Made by `python -m arxiv_rag.goldgen --n 200 --seed 0 --out eval/gold_draft.jsonl`. Appends and resumes; `gold_draft.jsonl.rejected` lists spans the model skipped or whose question hit a banned phrase. |
| `gold_v1.jsonl` | The reviewed subset: only items with `"status": "ok"`. This is what the evaluation uses. |

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
- `status`: `draft`, `ok`, `edit`, `bad`.
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
- `python -m arxiv_rag.review --export [--out eval/gold_v1.jsonl]`: writes the `ok` and `edit` items (one per `(qid, doc_sha256)`, last wins; LF endings), prints counts by status and `span_type`, and warns below 150 items.
- `python -m arxiv_rag.review --stats`: counts by status x `span_type`, and by `auto_flag`.
