"""Terminal tool to review the DRAFT gold set quickly, and export the reviewed set.

Design goals:
- Fast: one keypress per item (ok / bad / edit / skip / undo / note / quit), no mouse, no editor.
- Safe: every decision is written immediately and atomically (temp file + os.replace), row
  order and all other fields are kept, and lines that do not parse are written back
  byte-for-byte. A crash or Ctrl-C loses nothing.
- Resumable: only status "draft" items are walked, so rerunning continues where you stopped.
- Exclusive: <draft>.lock keeps it from running next to goldgen's --recheck/--retry-rejected,
  which rewrite the same file.
- Evidence untouched: edit changes only the question (the original goes to question_original);
  evidence_quote and offsets are never modified.
- Testable: falls back to line input when stdin is not a TTY.
"""
import argparse
import json
import os
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path

from arxiv_rag.document import PARAGRAPH_SEP, load_titles

DEFAULT_DRAFT = Path("eval/gold_draft.jsonl")
DEFAULT_OUT = Path("eval/gold_v1.jsonl")
MIN_GOLD = 150
EXPORT_STATUSES = ("ok", "edit")

RUBRIC = (
    "Rubric:",
    "  ok   = answerable from the evidence alone AND specific enough to find the paper among 100",
    "  edit = fixable by rewording (evidence stays as is)",
    "  bad  = vague, unanswerable, trivial copy of the quote, or a roadmap/outline sentence",
    "  cross_paragraph: prefer questions that need BOTH sentences",
    "  keys: [o]k [b]ad [e]dit [s]kip [n]ote [u]ndo [q]uit",
)
FLAG_HELP = {
    "one_half_only": "check that the question needs BOTH sentences (it is anchored in only one)",
    "no_anchor": "question shares no specific term with the evidence quote",
    "no_specific_term": "evidence has no term specific enough to identify the paper",
    "too_long": "question is longer than the word limit (likely a copied enumeration)",
}
KEYS = "obesnuq"


# --- file handling ----------------------------------------------------------

class DraftFile:
    """Rows of a JSONL file; unparseable lines are kept as raw bytes and never rewritten."""

    def __init__(self, path: Path):
        self.path = path
        data = path.read_bytes() if path.exists() else b""
        self.lines = data.splitlines(keepends=True)
        self.rows: list[dict | None] = [self._parse(ln) for ln in self.lines]

    @staticmethod
    def _parse(line: bytes) -> dict | None:
        try:
            row = json.loads(line.decode("utf-8"))
        except ValueError:  # includes UnicodeDecodeError
            return None
        return row if isinstance(row, dict) else None

    def set(self, i: int, row: dict) -> None:
        old = self.lines[i]
        eol = b"\r\n" if old.endswith(b"\r\n") else b"\n"
        self.rows[i] = row
        self.lines[i] = json.dumps(row, ensure_ascii=False).encode("utf-8") + eol
        self.save()

    def restore(self, i: int, row: dict, line: bytes) -> None:
        self.rows[i], self.lines[i] = row, line
        self.save()

    def save(self) -> None:
        atomic_write(self.path, b"".join(
            ln if ln.endswith(b"\n") else ln + b"\n" for ln in self.lines))


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Lock:
    def __init__(self, draft: Path):
        self.path = draft.with_name(draft.name + ".lock")

    def __enter__(self):
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            raise SystemExit(
                f"{self.path} exists: another review/goldgen session is using the file "
                "(delete the lock only if you are sure none is running)")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --- input ------------------------------------------------------------------

def read_key(valid: str = KEYS) -> str:
    """One key from a TTY (no Enter); a line of text (first char) otherwise."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            return "q"  # EOF
        return line.strip()[:1].lower()
    if os.name == "nt":
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow/function key prefix: swallow the second code
                msvcrt.getwch()
                continue
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch.lower() in valid:
                return ch.lower()
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # keeps Ctrl-C working
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return "q"
            if ch.lower() in valid:
                return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def ask_line(prompt: str) -> str:
    print(prompt, end="", flush=True)
    return sys.stdin.readline().rstrip("\r\n")


# --- display ----------------------------------------------------------------

def _style(code: str, text: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def _wrap(text: str, width: int, indent: str = "  ") -> str:
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)


def show_item(row: dict, pos: int, total: int, session: Counter, title: str | None) -> None:
    width = max(40, shutil.get_terminal_size((100, 24)).columns - 2)
    done = " ".join(f"{k}={session[k]}" for k in ("ok", "edit", "bad", "skip"))
    print("\n" + "=" * width)
    print(_style("1", f"[{pos}/{total}]") + f"  {done}   {row.get('span_type', '?')}   status={row.get('status')}")
    if title:
        print("Paper: " + title)
    flag = row.get("auto_flag")
    if flag:
        print(_style("1;33", f"FLAG {flag}") + ": " + FLAG_HELP.get(flag, "flagged by goldgen --recheck"))
    if row.get("review_note"):
        print("Note: " + row["review_note"])
    print("\n" + _style("1;36", "Q:") )
    print(_wrap(str(row.get("question", "")), width))
    if row.get("question_original"):
        print(_style("2", _wrap("(original) " + str(row["question_original"]), width)))
    print("\n" + _style("1;32", "Evidence:"))
    parts = str(row.get("evidence_quote", "")).split(PARAGRAPH_SEP)
    for k, part in enumerate(parts):
        if k:
            print(_style("1;35", "  " + "-" * 8 + " paragraph boundary " + "-" * 8))
        print(_wrap(" ".join(part.split()), width))


# --- review session ---------------------------------------------------------

def select_items(df: DraftFile, all_items: bool, flagged: bool) -> list[int]:
    out = []
    for i, row in enumerate(df.rows):
        if row is None:
            continue
        if not all_items and row.get("status") != "draft":
            continue
        if flagged and not row.get("auto_flag"):
            continue
        out.append(i)
    return out


def review(df: DraftFile, titles: dict[str, str], all_items: bool = False,
           flagged: bool = False) -> Counter:
    idxs = select_items(df, all_items, flagged)
    session: Counter = Counter()
    if not idxs:
        print("Nothing to review.")
        return session
    for line in RUBRIC:
        print(line)
    undo: list[tuple[int, int, dict, bytes, str]] = []  # (pos, row index, old row, old line, key)
    pos = 0
    while pos < len(idxs):
        i = idxs[pos]
        row = df.rows[i]
        show_item(row, pos + 1, len(idxs), session, titles.get(str(row.get("paper_id"))))
        print("[o]k [b]ad [e]dit [s]kip [n]ote [u]ndo [q]uit > ", end="", flush=True)
        key = read_key()
        print(key)
        if key == "q":
            break
        if key == "u":
            if not undo:
                print("Nothing to undo.")
                continue
            pos, j, old_row, old_line, k = undo.pop()
            if k != "s":
                df.restore(j, old_row, old_line)
            session[{"o": "ok", "b": "bad", "e": "edit", "s": "skip"}[k]] -= 1
            continue
        if key == "n":
            note = ask_line("note> ").strip()
            if note:
                new = dict(row, review_note=note)
                df.set(i, new)
            continue
        if key in "obe":
            old_row, old_line = row, df.lines[i]
            new = dict(row)
            if key == "o":
                new["status"] = "ok"
            elif key == "b":
                new["status"] = "bad"
            else:
                text = " ".join(ask_line("new question (empty = cancel)> ").split())
                if not text:
                    continue
                new.setdefault("question_original", row.get("question"))
                new["question"] = text
                new["status"] = "edit"
            df.set(i, new)
            session[{"o": "ok", "b": "bad", "e": "edit"}[key]] += 1
            undo.append((pos, i, old_row, old_line, key))
            pos += 1
        elif key == "s":
            session["skip"] += 1
            undo.append((pos, i, row, df.lines[i], "s"))
            pos += 1
        else:
            print("Unknown key.")
    return session


# --- export / stats ---------------------------------------------------------

def read_rows(path: Path) -> list[dict]:
    return [r for r in DraftFile(path).rows if r is not None]


def export(draft: Path, out: Path) -> dict[tuple, dict]:
    keep: dict[tuple, dict] = {}
    for r in read_rows(draft):
        if r.get("status") in EXPORT_STATUSES:
            keep[(r.get("qid"), r.get("doc_sha256"))] = r  # last wins
    rows = list(keep.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, b"".join(
        json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in rows))
    print(f"wrote {len(rows)} items to {out}")
    print("by status:    " + str(dict(Counter(r["status"] for r in rows))))
    print("by span_type: " + str(dict(Counter(r.get("span_type") for r in rows))))
    if len(rows) < MIN_GOLD:
        print(f"WARNING: only {len(rows)} items, fewer than {MIN_GOLD}; "
              "small gold sets give noisy comparisons")
    return keep


def stats(draft: Path) -> None:
    rows = read_rows(draft)
    types = sorted({r.get("span_type", "?") for r in rows})
    statuses = ["draft", "ok", "edit", "bad"] + sorted(
        {r.get("status") for r in rows} - {"draft", "ok", "edit", "bad"})
    cell = Counter((r.get("status"), r.get("span_type", "?")) for r in rows)
    print(f"{'status':<10}" + "".join(f"{t:>17}" for t in types) + f"{'total':>8}")
    for s in statuses:
        print(f"{s:<10}" + "".join(f"{cell[(s, t)]:>17}" for t in types)
              + f"{sum(cell[(s, t)] for t in types):>8}")
    print(f"{'total':<10}" + "".join(f"{sum(cell[(s, t)] for s in statuses):>17}" for t in types)
          + f"{len(rows):>8}")
    flags = Counter(r["auto_flag"] for r in rows if r.get("auto_flag"))
    print("auto_flag: " + (str(dict(flags)) if flags else "none"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Review draft gold questions, export the reviewed set")
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--all", action="store_true", help="revisit every item, not only drafts")
    parser.add_argument("--flagged", action="store_true", help="only items with an auto_flag")
    parser.add_argument("--export", action="store_true", help="write ok/edit items to --out")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", action="store_true", help="counts by status x span_type and auto_flag")
    args = parser.parse_args()

    if not args.draft.exists():
        raise SystemExit(f"{args.draft} not found")
    if args.stats:
        stats(args.draft)
        return
    if args.export:
        export(args.draft, args.out)
        return
    if os.name == "nt":
        os.system("")  # enables ANSI escapes in the Windows console
    with Lock(args.draft):
        df = DraftFile(args.draft)  # read after taking the lock
        titles = load_titles(args.data_dir)
        try:
            session = review(df, titles, args.all, args.flagged)
        except KeyboardInterrupt:
            print("\ninterrupted (all decisions so far are saved)")
            session = None
    if session is not None:
        print("session: " + str(dict(session)))
    stats(args.draft)


if __name__ == "__main__":
    main()
