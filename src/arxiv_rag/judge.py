"""Machine verification of the DRAFT gold set by a judge model from a different family.

The drafts are written by one model (see goldgen); nobody reviews them by hand, so a second,
independent model replaces the human pass. Results must say "machine-verified, not human-reviewed".

Design goals:
- Blind: the judge sees only the question and the evidence quote, never the paper title or any
  other field of the item.
- Strict: answerable and specific decide the status; the rule-based auto_flag from goldgen can
  still veto. `copy` (and, for cross_paragraph items, `needs_both`) is recorded only, for the
  sensitivity breakdown in eval_chunking.
- Re-decidable: --restatus recomputes the status from stored verdicts, without LLM calls.
- Safe: every verdict is written at once (atomic rewrite via review.DraftFile); row order, other
  fields and unparseable lines are kept; question, evidence and offsets are never touched.
- Resumable: only items with status "draft" and no `judge` field are asked; malformed replies are
  counted and retried next run.
- Exclusive: honours review.py's <draft>.lock, also removed on Ctrl-C / quota exhaustion.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from arxiv_rag.document import PARAGRAPH_SEP
from arxiv_rag.goldgen import LLMClient, MalformedReply, QuotaExhausted, load_dotenv
from arxiv_rag.review import DEFAULT_DRAFT, DraftFile, Lock

DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"
AUTO_BAD_FLAGS = ("no_anchor", "no_specific_term", "too_long")
MAX_REASON_CHARS = 200

SYSTEM_PROMPT = (
    "You check retrieval-evaluation questions about scientific papers. You get a QUESTION and the "
    "EVIDENCE passage it was written from. Judge:\n"
    '- "answerable": true if the evidence alone fully answers the question.\n'
    '- "specific": true if the question names something (a method, dataset, model, benchmark or '
    "number) that would let a searcher pick this passage's paper out of about 100 NLP/retrieval "
    "papers; false if it is generic and could be about many of them.\n"
    '- "copy": true if the question is mostly a verbatim copy of the evidence wording.\n'
    "@RULE@"
    'Reply with JSON only: {"answerable": true/false, "specific": true/false, "copy": true/false, '
    '@KEY@"reason": "<= 20 words"}'
)
CROSS_RULE = (
    '- "needs_both": true if answering requires information from BOTH sentences of the evidence '
    "(they are marked [1] and [2]).\n"
)
CROSS_KEY = '"needs_both": true/false, '


def build_system_prompt(span_type: str) -> str:
    cross = span_type == "cross_paragraph"
    return SYSTEM_PROMPT.replace("@RULE@", CROSS_RULE if cross else "").replace("@KEY@", CROSS_KEY if cross else "")


def build_user_prompt(question: str, quote: str, span_type: str) -> str:
    # Only the question and the evidence: no title, no other fields.
    if span_type == "cross_paragraph":
        evidence = "\n".join(f"[{i}] {p}" for i, p in enumerate(quote.split(PARAGRAPH_SEP), 1))
    else:
        evidence = quote
    return f'QUESTION: {question}\n\nEVIDENCE:\n"""\n{evidence}\n"""'


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(raw: str) -> str:
    text = _THINK_BLOCK.sub("", raw or "")
    if re.search(r"</think>", text, re.I):  # opening tag missing: keep what follows the last closing tag
        text = re.split(r"</think>", text, flags=re.I)[-1]
    elif re.search(r"<think>", text, re.I):  # never closed: the reply was cut off mid-reasoning
        text = re.split(r"<think>", text, flags=re.I)[0]
    return text.strip()


def parse_verdict(raw: str, span_type: str = "") -> dict:
    """{"answerable", "specific", "copy", ["needs_both"], "reason"}; raises MalformedReply."""
    text = strip_think(raw)
    if not text:
        raise MalformedReply("empty reply")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group() if m else text, strict=False)
    except json.JSONDecodeError as exc:
        raise MalformedReply(f"not JSON: {text[:80]!r}") from exc
    if not isinstance(obj, dict):
        raise MalformedReply(f"not an object: {text[:80]!r}")
    keys = ("answerable", "specific", "copy") + (("needs_both",) if span_type == "cross_paragraph" else ())
    out: dict = {}
    for key in keys:
        if not isinstance(obj.get(key), bool):
            raise MalformedReply(f"missing or non-boolean {key!r}: {text[:80]!r}")
        out[key] = obj[key]
    reason = obj.get("reason")
    out["reason"] = " ".join(reason.split())[:MAX_REASON_CHARS] if isinstance(reason, str) else ""
    return out


def bad_reasons(item: dict, verdict: dict) -> list[str]:
    """Why the item is auto_bad (empty: auto_ok). copy and needs_both are recorded, never reasons."""
    reasons = []
    if not verdict["answerable"]:
        reasons.append("not_answerable")
    if not verdict["specific"]:
        reasons.append("not_specific")
    if item.get("auto_flag") in AUTO_BAD_FLAGS:
        reasons.append(f"flag:{item['auto_flag']}")
    return reasons


def pending_indexes(df: DraftFile) -> list[int]:
    return [i for i, r in enumerate(df.rows) if r is not None and r.get("status") == "draft" and "judge" not in r]


def restatus(df: DraftFile) -> dict:
    """Recompute auto_ok/auto_bad from stored verdicts and the current auto_flag; one save, no LLM.

    Only rows with status auto_ok/auto_bad and a complete `judge` verdict are touched; human
    statuses, drafts without a verdict and unparseable lines are left as they are.
    """
    res = {"checked": 0, "changed": 0, "skipped_incomplete": 0, "by_status": Counter(), "by_span_type": Counter()}
    for i, row in enumerate(df.rows):
        if row is None or row.get("status") not in ("auto_ok", "auto_bad") or not isinstance(row.get("judge"), dict):
            continue
        j = row["judge"]
        if not all(isinstance(j.get(k), bool) for k in ("answerable", "specific", "copy")):
            res["skipped_incomplete"] += 1
            continue
        res["checked"] += 1
        status = "auto_bad" if bad_reasons(row, j) else "auto_ok"
        if status != row["status"]:
            df.set(i, dict(row, status=status), save=False)
            res["changed"] += 1
        res["by_status"][status] += 1
        res["by_span_type"][(row.get("span_type", ""), status)] += 1
    if res["changed"]:
        df.save()
    return res


def new_result() -> dict:
    return {"judged": 0, "auto_ok": 0, "auto_bad": 0, "malformed": 0,
            "auto_bad_by_reason": Counter(), "by_span_type": Counter()}


def judge_items(
    df: DraftFile,
    ask: Callable[[str, str], str],
    model: str,
    limit: int | None = None,
    res: dict | None = None,
) -> dict:
    """Judge pending items in file order, saving after each; counts accumulate in `res`.

    QuotaExhausted and KeyboardInterrupt propagate after the items so far have been saved.
    """
    res = new_result() if res is None else res
    todo = pending_indexes(df)
    for i in todo if limit is None else todo[:limit]:
        row = df.rows[i]
        span_type = row.get("span_type", "")
        user = build_user_prompt(str(row.get("question", "")), str(row.get("evidence_quote", "")), span_type)
        try:
            verdict = parse_verdict(ask(build_system_prompt(span_type), user), span_type)
        except MalformedReply as exc:
            print(f"malformed {row.get('qid')}: {exc}")
            res["malformed"] += 1
            continue
        reasons = bad_reasons(row, verdict)
        new = dict(row)
        new["judge"] = {"model": model, **verdict}
        new["status"] = "auto_bad" if reasons else "auto_ok"
        df.set(i, new)
        res["judged"] += 1
        res[new["status"]] += 1
        res["by_span_type"][(span_type, new["status"])] += 1
        res["auto_bad_by_reason"].update(reasons)
    return res


def print_summary(res: dict) -> None:
    print(f"judged={res['judged']} auto_ok={res['auto_ok']} auto_bad={res['auto_bad']} malformed={res['malformed']}")
    print("auto_bad by reason: " + str(dict(res["auto_bad_by_reason"])))
    for t in sorted({t for t, _ in res["by_span_type"]}):
        print(f"  {t or '?'}: auto_ok={res['by_span_type'][(t, 'auto_ok')]} auto_bad={res['by_span_type'][(t, 'auto_bad')]}")


def print_restatus(res: dict) -> None:
    print(f"restatus: checked={res['checked']} changed={res['changed']} "
          f"skipped_incomplete={res['skipped_incomplete']}")
    print("by status: " + str(dict(res["by_status"])))
    for t in sorted({t for t, _ in res["by_span_type"]}):
        print(f"  {t or '?'}: auto_ok={res['by_span_type'][(t, 'auto_ok')]} auto_bad={res['by_span_type'][(t, 'auto_bad')]}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Machine-verify draft gold questions with a second model")
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--rpm", type=float, default=10, help="max LLM requests per minute")
    parser.add_argument("--dry-run", action="store_true", help="count pending items, no LLM calls or writes")
    parser.add_argument("--limit", type=int, default=None, help="judge at most N items this run")
    parser.add_argument("--no-json-mode", action="store_true", help="omit response_format (if the model rejects it)")
    parser.add_argument("--restatus", action="store_true",
                        help="recompute auto_ok/auto_bad from stored verdicts and auto_flag; no LLM calls")
    args = parser.parse_args()

    if not args.draft.exists():
        raise SystemExit(f"{args.draft} not found")
    if args.restatus:
        with Lock(args.draft):
            res = restatus(DraftFile(args.draft))  # read after taking the lock
        print_restatus(res)
        return
    if args.dry_run:
        df = DraftFile(args.draft)
        todo = pending_indexes(df)
        print(f"{len(todo)} items to judge with {args.model}: "
              + str(dict(Counter(df.rows[i].get("span_type", "?") for i in todo))))
        return
    load_dotenv()
    client = LLMClient.from_env(args.rpm, model=args.model)

    def ask(system: str, user: str) -> str:
        return client.chat(system, user, json_mode=not args.no_json_mode, temperature=0.0)

    res = new_result()
    with Lock(args.draft):
        df = DraftFile(args.draft)  # read after taking the lock
        try:
            judge_items(df, ask, client.model, args.limit, res)
        except KeyboardInterrupt:
            print("\ninterrupted (all verdicts so far are saved)")
        except QuotaExhausted as exc:
            print(f"{exc} (verdicts so far are saved)")
    print_summary(res)


if __name__ == "__main__":
    main()
