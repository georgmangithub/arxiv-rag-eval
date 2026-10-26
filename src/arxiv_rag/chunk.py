"""Four chunking strategies over the canonical document text, for the recall@k study.

Every chunk is a character span of doc.text (paragraphs joined by "\\n\\n"), so chunks
from different strategies and gold evidence quotes share one coordinate system.
Chunk.text == doc.text[char_start:char_end], except "structure+header" where the text
is "<title> > <section>\\n\\n" followed by exactly that span.

    fixed             sliding word-aligned windows over the whole text, with overlap
    recursive         split on paragraphs, then sentences, then words; merge greedily
    structure         whole paragraphs packed per section, never across a section
    structure+header  identical spans, with the paper title and section prefixed to the text
                      (structure reserves the header's token budget too, so the two differ
                      only by the prefix and the ablation isolates the header)

The token counter is injected (it will be the embedding model's tokenizer), so it is
treated as a monotone black box: all boundary search is galloping + binary search over
atom boundaries, O(log n) counter calls per chunk. Chunks fit max_tokens, with one
exception: a single whitespace-free word longer than max_tokens cannot be split and is
emitted alone as an oversize chunk. Spans always start and end on non-whitespace.
"""
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from arxiv_rag.document import Document

CountTokens = Callable[[str], int]
Span = tuple[int, int]
Size = Callable[[int, int], int]  # tokens of the chunk text for span (start, end)

STRATEGIES: tuple[str, ...] = ("fixed", "recursive", "structure", "structure+header")

# Words that end in "." without ending the sentence (lowercased, no trailing period).
ABBREVIATIONS = {
    "e.g", "i.e", "al", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "tabs",
    "vs", "cf", "etc", "approx", "resp", "no",
    "dr", "mr", "mrs", "ms", "prof", "ref", "refs", "alg", "inc", "vol", "st",
}
DOTTED_ACRONYM = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")  # "U.S", "e.g", "i.e", "a.k.a"
MAX_WORD_SCAN = 64  # longer words are never abbreviations; bounds the backward scan
HEADER_FRACTION = 4  # the structure header may use at most max_tokens // HEADER_FRACTION tokens
SENTENCE_END = re.compile(r"[.!?][\"')\]”’]*\s+")
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")
WORD = re.compile(r"\S+")


@dataclass
class Chunk:
    chunk_id: str  # f"{paper_id}:{strategy}:{index}"
    paper_id: str
    strategy: str
    text: str  # text that gets embedded (may include a header prefix)
    char_start: int  # span in doc.text that this chunk covers
    char_end: int
    section: str | None  # section of the first paragraph covered
    page_start: int
    page_end: int


# --- splitting text into spans ---------------------------------------------------------

def _trimmed(text: str, pieces: list[Span]) -> list[Span]:
    """Strip whitespace from each span and drop the empty ones."""
    out = []
    for s, e in pieces:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out


def _split_at(text: str, s: int, e: int, pattern: re.Pattern, keep: int = 0) -> list[Span]:
    """Cut [s, e) at each match of pattern; `keep` chars of the match stay with the left piece."""
    pieces, pos = [], s
    for m in pattern.finditer(text, s, e):
        pieces.append((pos, m.start() + keep))
        pos = m.end()
    pieces.append((pos, e))
    return _trimmed(text, pieces)


def _paragraph_spans(text: str, s: int, e: int) -> list[Span]:
    return _split_at(text, s, e, PARAGRAPH_BREAK)


def _is_sentence_end(text: str, m: re.Match) -> bool:
    nxt = text[m.end():m.end() + 1]
    if not nxt or nxt.islower():
        return False  # "... et al. showed", "... e.g. foo"
    if text[m.start()] != ".":
        return True  # "?" and "!" never belong to an abbreviation
    # Word the period closes: scan back to whitespace (O(word), not O(prefix)).
    end = m.start()
    start = end
    while start > 0 and end - start <= MAX_WORD_SCAN and not text[start - 1].isspace():
        start -= 1
    word = text[start:end].lstrip("([\"'")
    if word.lower() in ABBREVIATIONS or DOTTED_ACRONYM.fullmatch(word):
        return False
    return not (len(word) == 1 and word.isupper())  # initials: "J. Smith"


def _sentence_spans(text: str, s: int, e: int) -> list[Span]:
    pieces, pos = [], s
    for m in SENTENCE_END.finditer(text, s, e):
        if _is_sentence_end(text, m):
            pieces.append((pos, m.start() + len(m.group().rstrip())))
            pos = m.end()
    pieces.append((pos, e))
    return _trimmed(text, pieces)


def _word_spans(text: str, s: int, e: int) -> list[Span]:
    return [m.span() for m in WORD.finditer(text, s, e)]


def sentence_spans(text: str) -> list[Span]:
    """(start, end) offsets of each sentence in text, stripped of surrounding whitespace.

    Single source of truth for sentence boundaries: the chunkers use the same splitter,
    so anything that needs boundaries consistent with chunks (e.g. gold generation)
    must call this rather than re-implement splitting. text[s:e] is the sentence.
    """
    return _sentence_spans(text, 0, len(text))


def split_sentences(text: str) -> list[str]:
    """Sentence strings of text; a thin wrapper over sentence_spans."""
    return [text[a:b] for a, b in sentence_spans(text)]


SPLITTERS = (_paragraph_spans, _sentence_spans, _word_spans)  # coarse to fine


def _atoms(text: str, s: int, e: int, level: int, size: Size, max_tokens: int) -> list[Span]:
    """Break [s, e) into the coarsest pieces that each fit max_tokens.

    A span that fits stays whole; otherwise it is split at the first level that
    actually divides it. A lone word that does not fit is returned as is (oversize).
    """
    if size(s, e) <= max_tokens:
        return [(s, e)]
    for lvl in range(level, len(SPLITTERS)):
        pieces = SPLITTERS[lvl](text, s, e)
        if len(pieces) > 1:
            return [a for p in pieces for a in _atoms(text, p[0], p[1], lvl + 1, size, max_tokens)]
    return [(s, e)]


# --- packing atoms into chunks ---------------------------------------------------------

def _furthest(atoms: list[Span], i: int, size: Size, max_tokens: int) -> int:
    """Largest j >= i such that atoms[i..j] fit max_tokens (j == i if even atom i does not)."""
    n = len(atoms)
    good, step = i, 1
    probe = i + 1
    while probe < n and size(atoms[i][0], atoms[probe][1]) <= max_tokens:
        good = probe
        step *= 2
        probe = good + step
    lo, hi = good, min(probe, n)  # atoms[hi] is known not to fit, or hi == n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if size(atoms[i][0], atoms[mid][1]) <= max_tokens:
            lo = mid
        else:
            hi = mid
    return lo


def _merge(atoms: list[Span], size: Size, max_tokens: int) -> list[Span]:
    """Greedily join adjacent atoms into maximal spans that fit (no overlap)."""
    out, i = [], 0
    while i < len(atoms):
        j = _furthest(atoms, i, size, max_tokens)
        out.append((atoms[i][0], atoms[j][1]))
        i = j + 1
    return out


def _fixed_spans(text: str, size: Size, max_tokens: int, overlap_tokens: int) -> list[Span]:
    words = _word_spans(text, 0, len(text))
    out, i = [], 0
    while words and i < len(words):
        j = _furthest(words, i, size, max_tokens)
        out.append((words[i][0], words[j][1]))
        if j == len(words) - 1:
            break

        def ok(k: int) -> bool:  # tail words[k..j] is a small enough overlap and the next word still fits
            if k > j:
                return True
            return (size(words[k][0], words[j][1]) <= overlap_tokens
                    and size(words[k][0], words[j + 1][1]) <= max_tokens)

        lo, hi = i + 1, j + 1  # smallest k in [i+1, j+1] with ok(k); ok is monotone in k
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        i = lo
    return out


# --- chunk assembly --------------------------------------------------------------------

def _header(doc: Document, section: str | None, count_tokens: CountTokens, max_tokens: int) -> str:
    """Header "title > section" plus a blank line, capped at max_tokens // 4 tokens.

    Drops parts (section first, then title) until it fits the cap; if none does, no header.
    """
    cap = max_tokens // HEADER_FRACTION
    for parts in ((doc.title, section), (doc.title,)):
        h = " > ".join(p for p in parts if p)
        if h and count_tokens(h + "\n\n") <= cap:
            return h + "\n\n"
    return ""


def _make_chunks(doc: Document, strategy: str, spans: list[Span], prefixes: list[str] | None = None) -> list[Chunk]:
    starts = [p.char_start for p in doc.paragraphs]
    chunks = []
    for idx, (s, e) in enumerate(spans):
        first = doc.paragraphs[max(bisect_right(starts, s) - 1, 0)]
        last = doc.paragraphs[max(bisect_left(starts, e) - 1, 0)]
        text = (prefixes[idx] if prefixes else "") + doc.text[s:e]
        chunks.append(Chunk(f"{doc.paper_id}:{strategy}:{idx}", doc.paper_id, strategy, text, s, e,
                            first.section, first.page, last.page))
    return chunks


def _sections(doc: Document) -> list[tuple[str | None, list[Span]]]:
    """Consecutive non-empty paragraphs grouped by section, as (section, paragraph spans)."""
    groups: list[tuple[str | None, list[Span]]] = []
    for p in doc.paragraphs:
        if not p.text.strip():
            continue
        if not groups or groups[-1][0] != p.section:
            groups.append((p.section, []))
        groups[-1][1].append((p.char_start, p.char_end))
    return groups


def chunk_document(doc: Document, strategy: str, count_tokens: CountTokens,
                   max_tokens: int = 256, overlap_tokens: int = 32) -> list[Chunk]:
    """Chunk one document. Raises ValueError for an unknown strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    text = doc.text

    def plain(s: int, e: int) -> int:
        return count_tokens(text[s:e])

    if strategy == "fixed":
        return _make_chunks(doc, strategy, _fixed_spans(text, plain, max_tokens, overlap_tokens))
    if strategy == "recursive":
        atoms = _atoms(text, 0, len(text), 0, plain, max_tokens) if text.strip() else []
        return _make_chunks(doc, strategy, _merge(atoms, plain, max_tokens))

    spans, prefixes = [], []
    for section, paragraphs in _sections(doc):
        # Both strategies reserve the header's budget so their spans are identical;
        # only structure+header actually emits it.
        header = _header(doc, section, count_tokens, max_tokens)
        prefix = header if strategy == "structure+header" else ""

        def size(s: int, e: int, header: str = header) -> int:
            return count_tokens(header + text[s:e])

        atoms = [a for s, e in paragraphs for a in _atoms(text, s, e, 1, size, max_tokens)]
        merged = _merge(atoms, size, max_tokens)
        spans += merged
        prefixes += [prefix] * len(merged)
    return _make_chunks(doc, strategy, spans, prefixes)
