"""Build a DRAFT gold set (question + exact evidence quote) for human review.

Design goals:
- Exact evidence: every quote is a verbatim slice of Document.text, so any chunker's
  output can be scored against it by character offsets.
- Deterministic: span sampling depends only on --seed and the parsed corpus.
- Mixed difficulty: whole paragraphs, single sentences, and sentence pairs that straddle
  a paragraph boundary (the case where chunk boundaries hurt most).
- No lexical shortcut: the LLM is told to paraphrase, and each question is scored by its
  word n-gram overlap with the span so reviewers can spot copy-paste questions.
- Quality spans only: every span starts at a sentence start and ends at a sentence end, so
  PDF column/page-break fragments are not used as evidence.
- Pinned: each item stores char offsets and a sha256 of Document.text, so the eval can
  refuse to run against a re-parsed (shifted) document.
- Resumable: output is append-only JSONL; span ids already present are not re-asked.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

from arxiv_rag.chunk import sentence_spans
from arxiv_rag.document import PARAGRAPH_SEP, Document, ids_to_stems, load_documents
from arxiv_rag.ingest import read_ids_file

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# (span_type, share of --n)
MIX = (("paragraph", 0.5), ("cross_paragraph", 0.25), ("sentence", 0.25))
MIN_WORDS = {"paragraph": 20, "sentence": 8}  # sentence limit also applies to each half of a cross_paragraph span
MAX_WORDS = 200
SKIP_SECTIONS = ("reference", "bibliograph", "acknowledg")
BANNED_PHRASES = ("this paper", "the authors", "the proposed", "this work", "this study")
_ROOT = Path(__file__).resolve().parents[2]  # repo root, independent of cwd
DEFAULT_IDS_FILE = _ROOT / "eval" / "corpus_v1.txt"
LOCAL_HOSTS = ("localhost", "127.0.0.1")  # the only hosts allowed over plain http (e.g. Ollama)
STOPWORDS = frozenset(
    "a an the and or of in on at to for with by from as is are was were be been it its this that these "
    "those what which who whom how why when where does do did can could would should will not no than "
    "then there their they we our us you your i he she his her but if into over under about between "
    "during per via vs".split()
)

_CAPTION_START = re.compile(r"^\s*(Table|Figure|Fig\.|Algorithm)\s*\d")
_TERMINAL_END = re.compile(r"[.!?][\"')\]\u201d\u2019]*$")


# --- text helpers -----------------------------------------------------------

def sentence_offsets(text: str) -> list[tuple[int, int]]:
    """The single place candidate_spans gets sentence boundaries from (swap the splitter here)."""
    return sentence_spans(text)


def ends_cleanly(text: str) -> bool:
    """Ends with terminal punctuation (optionally followed by a closing quote/bracket)."""
    return bool(_TERMINAL_END.search(text.rstrip()))


def starts_cleanly(text: str) -> bool:
    """Starts with an uppercase letter or a digit (not a mid-sentence fragment)."""
    t = text.lstrip()
    return bool(t) and (t[0].isupper() or t[0].isdigit())


def is_whole_sentences(text: str) -> bool:
    return starts_cleanly(text) and ends_cleanly(text)


def starts_with_caption(text: str) -> bool:
    return bool(_CAPTION_START.match(text))


# A term specific enough to point at one paper in a ~100-paper corpus: an inner capital
# (GLatM, FDBench, BERT), letters mixed with digits (GPT-4, BM25), or a percentage.
# Text is tokenized once on word characters + hyphens and each token is classified.
_TOKEN = re.compile(r"[\w\u2010\u2011-]+")
_PERCENT = re.compile(r"\d+(?:\.\d+)?%")
_CITATION = re.compile(r"\([^()]*\bet al\.?[^()]*\)")  # "(Wei et al., 2026b)"
_JUNK_TOKEN = re.compile(
    r"(?:(?:19|20)\d\d[a-z]"  # citation-year suffix: 2026b
    r"|\d+(?:st|nd|rd|th|s)"  # ordinals, decades: 3rd, 1990s
    r"|[ivxlc]{1,4}"  # Roman numerals
    r"|\d[a-z]|[a-z]\d)"  # short figure refs: 3a, S1
)
_ROADMAP_REF = re.compile(r"§\s*\d|\b(?:Sec(?:tion)?s?\.?|Appendix)\s+[A-Z\d]")
MATH_MAX_CHARS = 3  # parsed equations come out as Unicode math letters/operators, mostly garbled


# Acronyms that occur in most papers of an NLP/retrieval corpus, so they identify none of them.
GENERIC_TERMS = frozenset(
    "llm llms rag ai nlp qa ml dl gpu gpus api apis cpu sota ir lm lms vlm vlms mllm mllms "
    "sft dpo grpo ppo rl rlhf rlvr lora qlora pdf pdfs ocr asr tts nli vqa kv moe cot ner "
    "url urls json html sql csv".split()
)
MAX_QUESTION_WORDS = 35  # longer ones were enumerations copied from the span


def _piece_ok(piece: str) -> bool:
    low = piece.lower()
    letters = sum(c.isalpha() for c in piece)
    if not letters:
        return False
    if piece.isupper():  # all caps: BERT yes, IN / AND / LM no
        return letters >= 3 and low not in STOPWORDS and low not in GENERIC_TERMS
    return (
        any(c.isupper() for c in piece[1:])
        and low not in GENERIC_TERMS
        and low.removesuffix("s") not in GENERIC_TERMS
    )


def _token_ok(tok: str) -> bool:
    if not tok or _JUNK_TOKEN.fullmatch(tok.lower()):
        return False
    if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        return True
    return any(_piece_ok(p) for p in tok.replace("\u2010", "-").replace("\u2011", "-").split("-"))


def _terms(text: str, heads: bool = False) -> set[str]:
    """Lowercase specific terms; with `heads`, also the hyphen prefixes that are terms themselves
    (Llama-2 for Llama-2-7B)."""
    text = _CITATION.sub(" ", text)
    out = {m.group(0) for m in _PERCENT.finditer(text)}
    for m in _TOKEN.finditer(text):
        tok = m.group(0).strip("-\u2010\u2011").replace("\u2010", "-").replace("\u2011", "-")
        if _token_ok(tok):
            out.add(tok.lower())
        if heads and "-" in tok:
            parts = tok.split("-")
            out.update("-".join(parts[:i]).lower() for i in range(1, len(parts)) if _token_ok("-".join(parts[:i])))
    return out


def specific_terms(text: str) -> set[str]:
    return _terms(text)


_DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"


def _normalize(text: str) -> tuple[str, set[int]]:
    """Lowercase, drop spaces/dashes/punctuation, '%' -> 'percent'. Also returns the indices of the
    result where a word started in the original text."""
    out: list[str] = []
    starts: set[int] = set()
    prev_alnum = False
    for ch in text.lower():
        if ch == "%":
            out.extend("percent")
        elif ch.isalnum() or ch == ".":
            if ch.isalnum() and not prev_alnum:
                starts.add(len(out))
            out.append(ch)
        prev_alnum = ch.isalnum()
    return "".join(out), starts


def is_anchored(question: str, span_quote: str) -> bool:
    """True if some specific term of the span occurs in the question (at a word start), ignoring
    case, hyphens/dashes and spaces: BM25 ~ BM25-based, Llama-2 ~ Llama-2-7B, BERTs ~ BERT,
    GPT 4 ~ GPT-4, F1 score ~ F1-score, 12.5 percent ~ 12.5%."""
    q, starts = _normalize(question)
    ends = starts | {len(q)}  # spaces/punctuation are dropped, so a word ends where the next starts
    for term in _terms(span_quote, heads=True):
        norm = _normalize(term)[0].strip(".")
        for cand in {norm, norm.removesuffix("s")}:
            # Short all-letter acronyms must match a whole word: MAP != mapping, CLIP != clipping.
            whole = cand.isalpha() and len(cand) <= 4
            i = q.find(cand) if cand else -1
            while i >= 0:
                j = i + len(cand)
                if i in starts and (not whole or j in ends or (q[j : j + 1] == "s" and j + 1 in ends)):
                    return True
                i = q.find(cand, i + 1)
    return False


def _halves(span_type: str, quote: str) -> list[str]:
    return quote.split(PARAGRAPH_SEP) if span_type == "cross_paragraph" else [quote]


def span_content_reason(span_type: str, quote: str) -> str:
    """Why `quote` fails the content filters of candidate_spans ("" if it passes)."""
    if "\ufffd" in quote:
        return "replacement_char"
    if starts_with_caption(quote) or (
        span_type == "cross_paragraph" and starts_with_caption(quote.split(PARAGRAPH_SEP, 1)[-1])
    ):
        return "caption"
    if is_math_heavy(quote):
        return "math"
    if is_roadmap(quote):
        return "roadmap"
    if not specific_terms(quote):
        return "no_specific_term"
    return ""


def question_reject_reason(question: str | None, span_type: str, quote: str) -> str:
    """Why a generated question is dropped ("" if kept).

    Whether a cross_paragraph question really needs both sentences is left to the human reviewer
    (see auto_flag "one_half_only"): requiring a term from each half rejected most good ones."""
    if question is None:
        return "skip"
    if has_banned_phrase(question):
        return "banned_phrase"
    if n_words(question) > MAX_QUESTION_WORDS:
        return "too_long"
    if not is_anchored(question, quote):
        return "no_anchor"  # nothing in the question points at this paper
    return ""


_NOT_MATH = frozenset("\u2212\u2264\u2265\u2248\u223c\u2208\u2260\u00b1")  # minus and plain comparisons occur in prose


def is_math_heavy(text: str) -> bool:
    return sum(
        (0x1D400 <= ord(c) <= 0x1D7FF or 0x2200 <= ord(c) <= 0x22FF) and c not in _NOT_MATH for c in text
    ) > MATH_MAX_CHARS


def is_roadmap(text: str) -> bool:
    """'We first introduce ... (§2.1), then ... (§2.2)': an outline, not a fact."""
    return len(_ROADMAP_REF.findall(text)) >= 2


def n_words(text: str) -> int:
    return len(text.split())


def looks_like_boilerplate(text: str) -> bool:
    """URLs, emails, mostly-symbol text, or number-heavy (table-ish) text."""
    if re.search(r"https?://|www\.|arxiv\.org|@\w+\.\w+|doi\.org", text, re.I):
        return True
    chars = [c for c in text if not c.isspace()]
    if not chars or sum(c.isalpha() for c in chars) / len(chars) < 0.6:
        return True
    tokens = text.split()
    numeric = sum(bool(re.fullmatch(r"[\d.,%\u00b1+\-/()]+", t)) for t in tokens)
    return numeric / len(tokens) > 0.3


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def overlap_ratio(question: str, span: str) -> float:
    """Longest common contiguous word n-gram, as a fraction of the question's words."""
    q, s = _tokens(question), _tokens(span)
    if not q or not s:
        return 0.0
    best, prev = 0, [0] * (len(s) + 1)
    for qi in q:
        cur = [0] * (len(s) + 1)
        for j, sj in enumerate(s, 1):
            if qi == sj:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return round(best / len(q), 3)


def title_overlap(question: str, title: str) -> float:
    """Fraction of the question's non-stopword lowercase word types that also occur in the title."""
    words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOPWORDS}
    if not words:
        return 0.0
    title_words = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
    return round(len(words & title_words) / len(words), 3)


def has_banned_phrase(question: str) -> bool:
    low = question.lower()
    return any(p in low for p in BANNED_PHRASES)


# --- span sampling ----------------------------------------------------------

@dataclass(frozen=True)
class Span:
    paper_id: str
    span_type: str
    start: int  # char offsets into Document.text
    end: int
    quote: str
    doc_sha256: str = ""  # sha256 of Document.text (utf-8) the offsets refer to

    @property
    def qid(self) -> str:
        return f"{self.paper_id}:{self.start}-{self.end}"


def doc_sha256(doc: Document) -> str:
    return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()


def candidate_spans(doc: Document, rejects: Counter | None = None) -> dict[str, list[Span]]:
    """All valid spans in `doc`, by type.

    `rejects`, if given, counts spans that passed the structural checks but were dropped
    for content reasons ("caption", "replacement_char", "math", "roadmap", "no_specific_term").
    """
    out: dict[str, list[Span]] = {t: [] for t, _ in MIX}

    sha = doc_sha256(doc)

    def add(span_type: str, start: int, end: int) -> None:
        quote = doc.text[start:end]
        reason = span_content_reason(span_type, quote)
        if reason:
            if rejects is not None:
                rejects[reason] += 1
            return
        out[span_type].append(Span(doc.paper_id, span_type, start, end, quote, sha))

    paras = doc.paragraphs
    for i, p in enumerate(paras):
        if (p.section or "").lower().startswith(SKIP_SECTIONS):
            continue
        if (
            MIN_WORDS["paragraph"] <= n_words(p.text) <= MAX_WORDS
            and is_whole_sentences(p.text)
            and not looks_like_boilerplate(p.text)
        ):
            add("paragraph", p.char_start, p.char_end)
        sents = [(p.char_start + a, p.char_start + b) for a, b in sentence_offsets(p.text)]
        for a, b in sents:
            if (
                MIN_WORDS["sentence"] <= n_words(doc.text[a:b]) <= MAX_WORDS
                and is_whole_sentences(doc.text[a:b])
                and not looks_like_boilerplate(doc.text[a:b])
            ):
                add("sentence", a, b)
        nxt = paras[i + 1] if i + 1 < len(paras) else None
        if nxt and nxt.section == p.section and sents:
            nsents = sentence_offsets(nxt.text)
            if not nsents:
                continue
            a, b = sents[-1][0], nxt.char_start + nsents[0][1]
            halves = (doc.text[a : sents[-1][1]], doc.text[nxt.char_start : b])
            if (
                all(n_words(h) >= MIN_WORDS["sentence"] and is_whole_sentences(h) for h in halves)
                and n_words(doc.text[a:b]) <= MAX_WORDS
                and not looks_like_boilerplate(" ".join(halves))
            ):
                add("cross_paragraph", a, b)
    return out


def sample_spans(docs: list[Document], n: int, seed: int, rejects: Counter | None = None) -> list[Span]:
    """~n spans in the MIX proportions, spread evenly over papers (round-robin, per-paper cap).

    May return fewer than n if the corpus has too few valid spans. No two returned spans
    share (paper_id, start, end), even across span types (they would share a qid).
    """
    if not docs:
        return []
    rng = random.Random(seed)
    docs = sorted(docs, key=lambda d: d.paper_id)
    pools = {d.paper_id: candidate_spans(d, rejects) for d in docs}
    for by_type in pools.values():
        for spans in by_type.values():
            rng.shuffle(spans)
    cap = math.ceil(n / len(docs))
    taken = {d.paper_id: 0 for d in docs}
    picked: list[Span] = []
    seen: set[tuple[str, int, int]] = set()

    def take(pid: str, span_type: str) -> bool:
        pool = pools[pid][span_type]
        while pool:
            span = pool.pop()
            if (span.paper_id, span.start, span.end) not in seen:
                seen.add((span.paper_id, span.start, span.end))
                picked.append(span)
                return True
        return False

    quotas = [round(n * share) for _, share in MIX[:-1]]
    quotas.append(n - sum(quotas))  # last type takes the remainder so the total is exactly n
    for (span_type, _), quota in zip(MIX, quotas):
        got = 0
        order = [d.paper_id for d in docs]
        rng.shuffle(order)
        while got < quota:
            progressed = False
            for pid in order:
                if got < quota and taken[pid] < cap and take(pid, span_type):
                    taken[pid] += 1
                    got += 1
                    progressed = True
            if not progressed:
                break
    by_id = {d.paper_id: d for d in docs}
    for s in picked:
        assert by_id[s.paper_id].text[s.start : s.end] == s.quote
    return picked


# --- LLM --------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You write retrieval-evaluation questions about scientific papers. Given an evidence span, "
    "write ONE question such that:\n"
    "(a) the span alone is enough to answer it;\n"
    "(b) it is self-contained for someone searching a corpus of about 100 papers: use only "
    "names and terms that appear in the evidence span (the specific method, dataset, model or "
    "topic as the span names it); never say \"this paper\", \"the authors\", "
    "\"the proposed method\", \"this work\" or \"this study\";\n"
    "(c) it contains at least one specific term copied exactly from the span: a named method, "
    "model, dataset or benchmark, or a name containing an inner capital or a digit (e.g. GPT-4, "
    "BM25), or a percentage (e.g. 12.5%). A generic acronym such as LLM or RAG, or a citation year, "
    "does not count. This lets a searcher tell which paper it is about;\n"
    "(d) apart from those terms, it paraphrases: do not copy long phrases from the span.\n"
    "Skip the span if it states no answerable fact (boilerplate, a fragment, a bare list), if it "
    "is an outline of the paper's sections, or if it refers to its subject only as \"our system\", "
    "\"our method\", \"we\" or similar without naming it.\n"
    'Reply with JSON only: {"question": "..."} or {"question": "SKIP"}.'
)


CROSS_NOTE = (
    "This span is two consecutive sentences from adjacent paragraphs. Write a question whose "
    "answer needs information from BOTH sentences; if they are unrelated, skip."
)


def build_user_prompt(span: str, span_type: str = "") -> str:
    # The paper title is deliberately NOT shown: chunkers that embed the title would be favoured.
    note = CROSS_NOTE + "\n" if span_type == "cross_paragraph" else ""
    return f'{note}Evidence span:\n"""\n{span}\n"""'


class MalformedReply(ValueError):
    pass


class QuotaExhausted(RuntimeError):
    """A 429 whose wait is too long to sit out (e.g. a tokens-per-day limit)."""


MAX_RETRY_WAIT = 60.0  # seconds slept per retry at most
QUOTA_WAIT = 300.0  # a Retry-After longer than this means a daily quota, not a per-minute limit
DAILY_LIMIT_RE = re.compile(r"(?:tokens|requests) per day|\((?:TPD|RPD)\)", re.I)  # daily-limit phrasing in a 429 body
PROGRESS_EVERY = 10  # print a progress line every N successful LLM calls


def _is_skip(s: str) -> bool:
    return s.strip("\"' .").upper() == "SKIP"


def parse_question(raw: str) -> str | None:
    """Question text, or None for SKIP. Raises MalformedReply if nothing usable is found."""
    text = (raw or "").strip()
    if not text:
        raise MalformedReply("empty reply")
    if _is_skip(text):
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group() if m else text, strict=False)
    except json.JSONDecodeError as exc:
        raise MalformedReply(f"not JSON: {raw[:80]!r}") from exc
    if not isinstance(obj, dict):
        raise MalformedReply(f"not an object: {raw[:80]!r}")
    if obj.get("skip") is True:
        return None
    q = obj.get("question")
    if not isinstance(q, str) or not q.strip():
        raise MalformedReply(f"no question field: {raw[:80]!r}")
    return None if _is_skip(q) else " ".join(q.split())


def load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny KEY=VALUE parser. Real environment variables win over the file.

    Handles a UTF-8 BOM, a leading `export `, quoted values, and trailing ` # comments` on
    unquoted values.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if val[:1] in ("\"", "'"):
            close = val.find(val[0], 1)
            val = val[1:close] if close > 0 else val[1:]
        else:
            val = re.sub(r"\s+#.*$", "", val)
        os.environ.setdefault(key.strip(), val)


class LLMClient:
    """OpenAI-compatible chat completions with rate limiting and retry on 429/5xx."""

    def __init__(self, api_key: str, base_url: str, model: str, rpm: float = 25, max_retries: int = 6):
        parsed = urlparse(base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS):
            raise ValueError(f"LLM_BASE_URL must be https (plain http only for {'/'.join(LOCAL_HOSTS)})")
        api_key = api_key.strip()  # a pasted trailing newline would break the header
        if not (api_key.isascii() and api_key.isprintable()):
            # Never echo the key: a ValueError message may be logged.
            raise ValueError("LLM_API_KEY contains non-ASCII or non-printable characters; re-paste it")
        self.api_key, self.model, self.max_retries = api_key, model, max_retries
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0
        self.calls = 0

    @classmethod
    def from_env(cls, rpm: float, model: str | None = None) -> "LLMClient":
        key = os.environ.get("LLM_API_KEY", "").strip()
        if not key:
            raise SystemExit("LLM_API_KEY is not set (put it in .env or the environment)")
        return cls(
            key,
            os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
            model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL,
            rpm,
        )

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text

    def __call__(self, span: str, span_type: str = "") -> str:
        return self.chat(SYSTEM_PROMPT, build_user_prompt(span, span_type))

    def chat(self, system: str, user: str, json_mode: bool = True, temperature: float = 0.3) -> str:
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(self.max_retries + 1):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            failure = None
            try:
                resp = requests.post(
                    self.url, json=body, timeout=60, headers={"Authorization": f"Bearer {self.api_key}"}
                )
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                resp = None
            except (requests.RequestException, UnicodeError) as exc:
                failure = type(exc).__name__  # e.g. InvalidHeader: its message can contain the key
            if failure:
                # Raised outside the except block so the original exception is not even in __context__.
                raise RuntimeError(
                    f"LLM request failed: {failure} (not retried; check LLM_BASE_URL and LLM_API_KEY)"
                )
            if resp is not None and resp.status_code == 200:
                self.calls += 1
                if self.calls % PROGRESS_EVERY == 0:
                    print(f"  ... {self.calls} LLM calls", flush=True)
                try:
                    return resp.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise MalformedReply(f"unexpected 200 response shape: {type(exc).__name__}") from exc
            if resp is not None and resp.status_code in (401, 403):
                raise RuntimeError(f"LLM request failed: {resp.status_code} (check LLM_API_KEY)")
            retryable = resp is None or resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == self.max_retries:
                detail = "network error" if resp is None else f"{resp.status_code} {self._redact(resp.text)[:200]}"
                raise RuntimeError(f"LLM request failed: {detail}")
            retry_after = resp.headers.get("Retry-After", "") if resp is not None else ""
            has_wait = retry_after.replace(".", "", 1).isdigit()
            delay = float(retry_after) if has_wait else 2.0 ** (attempt + 1)
            if resp is not None and resp.status_code == 429 and (
                delay > QUOTA_WAIT
                or (DAILY_LIMIT_RE.search(resp.text) and not (has_wait and delay <= QUOTA_WAIT))
            ):
                # A daily quota: waiting a minute at a time only burns hours. Stop; the run resumes.
                raise QuotaExhausted(f"LLM quota exhausted (retry after ~{delay / 60:.0f} min); rerun later to resume")
            time.sleep(min(delay, MAX_RETRY_WAIT))
        raise AssertionError("unreachable")


# --- generation loop --------------------------------------------------------

def load_done_ids(*paths: Path) -> set[tuple[str, str]]:
    """(qid, doc_sha256) pairs already present; records without a sha get "".

    Undecodable lines (e.g. a write cut off by a crash) are skipped with a warning.
    """
    done: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec["qid"], rec.get("doc_sha256") or ""))
                except (ValueError, KeyError, TypeError, AttributeError):
                    print(f"WARNING: {path}:{n} is not a valid record (truncated write?); ignoring it")
    return done


def open_append(path: Path):
    """Open for appending; start on a fresh line if the file does not end with a newline."""
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    f = path.open("a", encoding="utf-8", newline="\n")
    if needs_newline:
        f.write("\n")
    return f


def make_item(span: Span, question: str, title: str = "") -> dict:
    return {
        "qid": span.qid,
        "paper_id": span.paper_id,
        "question": question,
        "evidence_quote": span.quote,
        "char_start": span.start,
        "char_end": span.end,
        "doc_sha256": span.doc_sha256,
        "span_type": span.span_type,
        "status": "draft",
        "overlap": overlap_ratio(question, span.quote),
        "title_overlap": title_overlap(question, title),
    }


def generate(
    spans: list[Span],
    titles: dict[str, str],
    ask: Callable[[str, str], str],
    out: Path,
    current_shas: dict[str, str] | None = None,
) -> dict[str, int]:
    """Append one JSONL item per answerable span.

    SKIP/banned-phrase spans are logged to `<out>.rejected` (qid + reason) so a rerun does
    not pay for them again. Done-ness is keyed on (qid, doc_sha256): records whose sha differs
    from the current document ("stale", counted in the summary, never deleted) do not count as
    done. `current_shas` maps paper_id -> sha (default: taken from `spans`). Malformed replies are not logged, so they are retried next run.
    `titles` is used only for the title_overlap field; the title is never shown to the model.
    """
    rejected = out.with_name(out.name + ".rejected")
    done = load_done_ids(out, rejected)
    out.parent.mkdir(parents=True, exist_ok=True)
    shas = {**{s.paper_id: s.doc_sha256 for s in spans}, **(current_shas or {})}
    stale = sum(1 for qid, sha in done if qid.rsplit(":", 1)[0] in shas and sha != shas[qid.rsplit(":", 1)[0]])
    counts = {"written": 0, "skipped_existing": 0, "rejected": 0, "malformed": 0, "stale": stale}
    with open_append(out) as f, open_append(rejected) as rf:
        for span in spans:
            if (span.qid, span.doc_sha256) in done:
                counts["skipped_existing"] += 1
                continue
            try:
                question = parse_question(ask(span.quote, span.span_type))
            except MalformedReply as exc:
                print(f"malformed {span.qid}: {exc}")
                counts["malformed"] += 1
                continue
            reason = question_reject_reason(question, span.span_type, span.quote)
            if reason:
                row = {"qid": span.qid, "doc_sha256": span.doc_sha256, "span_type": span.span_type, "reason": reason}
                rf.write(json.dumps(row) + "\n")
                rf.flush()
            else:
                item = make_item(span, question, titles.get(span.paper_id, ""))
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
            done.add((span.qid, span.doc_sha256))
            counts["rejected" if reason else "written"] += 1
    return counts


def atomic_write_lines(path: Path, lines: list[str]) -> None:
    """Replace `path` with `lines` (no trailing newlines) via temp file + os.replace."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f if line.strip()]


def _parse_qid(qid: str) -> tuple[str, int, int] | None:
    try:
        paper_id, offsets = qid.rsplit(":", 1)
        start, end = offsets.split("-")
        return paper_id, int(start), int(end)
    except ValueError:
        return None


def infer_span_type(doc: Document, start: int, end: int) -> str:
    """Older rejected rows do not store span_type; recover it from the offsets.

    A one-sentence paragraph of 20+ words is ambiguous (it may have been sampled as either type);
    it is reported as "paragraph". Shorter ones can only have been sampled as "sentence".
    """
    quote = doc.text[start:end]
    if PARAGRAPH_SEP in quote:
        return "cross_paragraph"
    if n_words(quote) >= MIN_WORDS["paragraph"] and any(
        p.char_start == start and p.char_end == end for p in doc.paragraphs
    ):
        return "paragraph"
    return "sentence"


def retry_rejected(
    docs: list[Document],
    ask: Callable[[str, str], str] | None,
    out: Path,
    reasons: set[str],
    dry_run: bool = False,
) -> dict:
    """Re-ask the LLM about spans in `<out>.rejected` whose reason is in `reasons`.

    The same spans are retried (no re-sampling), rebuilt from their qid offsets against the
    current documents. Rows whose document sha changed (or whose paper is gone) are left in the
    file untouched and counted as "stale". Spans failing the CURRENT candidate filters get that
    new reason. Successes are appended to `out`; then `<out>.rejected` is rewritten atomically
    (also if interrupted) without them. A row whose (qid, sha) is already in `out` (crash between
    the append and the rewrite) is dropped without a call, so nothing is duplicated or lost.
    """
    rejected = out.with_name(out.name + ".rejected")
    lines = _read_lines(rejected)
    by_id = {d.paper_id: d for d in docs}
    shas = {pid: doc_sha256(d) for pid, d in by_id.items()}
    titles = {d.paper_id: d.title for d in docs}
    done = load_done_ids(out)
    counts = {"targeted": 0, "stale": 0, "failed_filters": 0, "already_written": 0,
              "written": 0, "rejected": 0, "malformed": 0}
    by_reason: Counter = Counter()
    replace: dict[int, str | None] = {}  # line index -> new line (None: drop)

    def flush() -> None:
        if dry_run or not replace:
            return
        atomic_write_lines(rejected, [r for i, line in enumerate(lines) if (r := replace.get(i, line)) is not None])

    try:
        with (nullcontext() if dry_run else open_append(out)) as f:
            for i, line in enumerate(lines):
                try:
                    row = json.loads(line)
                    reason, qid, sha = row["reason"], row["qid"], row.get("doc_sha256") or ""
                except (ValueError, KeyError, TypeError):
                    continue
                if reason not in reasons:
                    continue
                counts["targeted"] += 1
                by_reason[reason] += 1
                parsed = _parse_qid(qid)
                doc = by_id.get(parsed[0]) if parsed else None
                if doc is None or sha != shas[doc.paper_id] or parsed[2] > len(doc.text):
                    counts["stale"] += 1
                    continue
                _, start, end = parsed
                quote = doc.text[start:end]
                span_type = row.get("span_type") or infer_span_type(doc, start, end)
                if (qid, sha) in done:
                    counts["already_written"] += 1
                    replace[i] = None
                    continue
                new_reason = span_content_reason(span_type, quote)
                if new_reason:
                    counts["failed_filters"] += 1
                    replace[i] = json.dumps({"qid": qid, "doc_sha256": sha, "reason": new_reason})
                    continue
                if dry_run:
                    continue
                try:
                    question = parse_question(ask(quote, span_type))
                except MalformedReply as exc:
                    print(f"malformed {qid}: {exc}")
                    counts["malformed"] += 1
                    continue
                new_reason = question_reject_reason(question, span_type, quote)
                if new_reason:
                    counts["rejected"] += 1
                    replace[i] = json.dumps({"qid": qid, "doc_sha256": sha, "reason": new_reason})
                else:
                    span = Span(doc.paper_id, span_type, start, end, quote, sha)
                    f.write(json.dumps(make_item(span, question, titles.get(doc.paper_id, "")), ensure_ascii=False) + "\n")
                    f.flush()
                    done.add((qid, sha))
                    counts["written"] += 1
                    replace[i] = None
    finally:
        flush()
    return {**counts, "by_reason": dict(by_reason)}


def auto_flag(item: dict) -> str:
    """Reason the current rules would reject this stored item ("" if clean).

    "one_half_only" is a hint, not a rejection: a cross_paragraph question anchored in only one of
    its two sentences may not need the other, which the reviewer should check.
    """
    span_type, quote, question = item.get("span_type", ""), item.get("evidence_quote", ""), item.get("question", "")
    reason = span_content_reason(span_type, quote) or question_reject_reason(question, span_type, quote)
    if reason:
        return reason
    halves = _halves(span_type, quote)
    # Only meaningful when both sentences have a term the question could have reused.
    if span_type == "cross_paragraph" and all(specific_terms(h) for h in halves) and not all(
        is_anchored(question, h) for h in halves
    ):
        return "one_half_only"
    return ""


def recheck(out: Path) -> Counter:
    """Set/clear `auto_flag` on every item of `out` under the current rules; rewrite atomically.

    Row order and all other fields are preserved; unchanged and unparseable lines are kept
    byte-for-byte. Returns the count per flag ("" counted as "clean").
    """
    lines = _read_lines(out)
    counts: Counter = Counter()
    result: list[str] = []
    for line in lines:
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError
        except ValueError:
            result.append(line)
            continue
        flag = auto_flag(item)
        counts[flag or "clean"] += 1
        if flag and item.get("auto_flag") != flag:
            item["auto_flag"] = flag
            line = json.dumps(item, ensure_ascii=False)
        elif not flag and "auto_flag" in item:
            del item["auto_flag"]
            line = json.dumps(item, ensure_ascii=False)
        result.append(line)
    atomic_write_lines(out, result)
    return counts


def type_counts(spans: list[Span]) -> dict[str, int]:
    return {t: sum(s.span_type == t for s in spans) for t, _ in MIX}


def main() -> None:
    # Windows consoles are often cp1252; never crash on a quote character.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Draft a gold question set from parsed papers")
    parser.add_argument("--n", type=int, default=200, help="number of spans to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("eval/gold_draft.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--ids-file", type=Path, default=None,
        help=f"arXiv ids to use (default {DEFAULT_IDS_FILE} if it exists, else every parsed paper)",
    )
    parser.add_argument("--rpm", type=float, default=25, help="max LLM requests per minute")
    parser.add_argument("--dry-run", action="store_true", help="print sampled spans, no LLM calls")
    parser.add_argument(
        "--retry-rejected", metavar="REASONS", default=None,
        help="comma list (e.g. no_anchor,too_long): re-ask the LLM about those spans in <out>.rejected",
    )
    parser.add_argument(
        "--recheck", action="store_true",
        help="recompute auto_flag on every item of --out under the current rules (no LLM calls)",
    )
    args = parser.parse_args()

    if args.recheck:
        counts = recheck(args.out)
        print(f"recheck {args.out}: {dict(counts)}")
        return

    ids_file = args.ids_file or (DEFAULT_IDS_FILE if DEFAULT_IDS_FILE.exists() else None)
    stems = None
    if ids_file:
        ids = read_ids_file(ids_file)
        if not ids:
            raise SystemExit(f"{ids_file} contains no ids")
        stems = ids_to_stems(ids)
    try:
        docs = load_documents(args.data_dir, stems)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    print(f"corpus: {len(docs)} ids from {ids_file}" if ids_file else f"corpus: ALL parsed ({len(docs)})")
    if args.retry_rejected is not None:
        reasons = {r.strip() for r in args.retry_rejected.split(",") if r.strip()}
        if not reasons:
            raise SystemExit("--retry-rejected needs at least one reason")
        ask = None
        if not args.dry_run:
            load_dotenv()
            ask = LLMClient.from_env(args.rpm)
        res = retry_rejected(docs, ask, args.out, reasons, args.dry_run)
        print(f"{'would retry' if args.dry_run else 'retry'}: {res}")
        return
    rejects: Counter = Counter()
    spans = sample_spans(docs, args.n, args.seed, rejects)
    summary = f"{len(spans)} spans from {len(docs)} papers: {type_counts(spans)}"
    if args.dry_run:
        for s in spans:
            print(f"[{s.span_type}] {s.qid} ({n_words(s.quote)} words)\n  {s.quote[:300]!r}")
        print(summary)
        print(f"rejected candidates by reason: {dict(rejects)}")
        return

    print(summary)
    load_dotenv()
    titles = {d.paper_id: d.title for d in docs}
    print(generate(spans, titles, LLMClient.from_env(args.rpm), args.out, {d.paper_id: doc_sha256(d) for d in docs}))


if __name__ == "__main__":
    try:
        main()
    except QuotaExhausted as exc:  # progress so far is saved; rerunning resumes
        raise SystemExit(str(exc)) from None
