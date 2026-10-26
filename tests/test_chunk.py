import pytest

from arxiv_rag.chunk import STRATEGIES, chunk_document, sentence_spans, split_sentences
from arxiv_rag.document import build_document


def words(text: str) -> int:
    return len(text.split())


def para(n_sentences: int, tag: str) -> str:
    return " ".join(f"Sentence {i} of {tag} has exactly eight plain words." for i in range(n_sentences))


def make_doc():
    recs = [
        {"text": "Abstract", "section": None, "page": 1},
        {"text": para(2, "abs"), "section": "Abstract", "page": 1},
        {"text": para(1, "intro-a"), "section": "1 Introduction", "page": 1},
        {"text": para(3, "intro-b"), "section": "1 Introduction", "page": 2},
        {"text": para(12, "big"), "section": "1 Introduction", "page": 2},  # > 40 tokens alone
        {"text": para(2, "methods"), "section": "2 Methods", "page": 3},
        {"text": para(4, "methods-b"), "section": "2 Methods", "page": 4},
        {"text": para(2, "results"), "section": "3 Results", "page": 5},
    ]
    return build_document("p1", "A Test Paper", recs)


MAX, OVERLAP = 40, 8
DOC = make_doc()


def run(strategy, doc=DOC, max_tokens=MAX, overlap=OVERLAP):
    return chunk_document(doc, strategy, words, max_tokens, overlap)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_text_matches_span(strategy):
    for c in run(strategy):
        body = DOC.text[c.char_start:c.char_end]
        if strategy == "structure+header":
            assert c.text.endswith(body) and c.text != body
        else:
            assert c.text == body


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_coverage(strategy):
    covered = set()
    for c in run(strategy):
        covered.update(range(c.char_start, c.char_end))
    missing = [i for i, ch in enumerate(DOC.text) if not ch.isspace() and i not in covered]
    assert missing == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_token_budget(strategy):
    chunks = run(strategy)
    assert chunks
    assert all(words(c.text) <= MAX for c in chunks)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_order_ids_and_boundaries(strategy):
    chunks = run(strategy)
    assert [c.chunk_id for c in chunks] == [f"p1:{strategy}:{i}" for i in range(len(chunks))]
    assert all(c.paper_id == "p1" and c.strategy == strategy for c in chunks)
    starts = [c.char_start for c in chunks]
    assert starts == sorted(set(starts))
    assert all(a.char_end <= b.char_end for a, b in zip(chunks, chunks[1:]))
    for c in chunks:  # spans start and end on non-whitespace
        assert not DOC.text[c.char_start].isspace() and not DOC.text[c.char_end - 1].isspace()


@pytest.mark.parametrize("strategy", ["structure", "structure+header"])
def test_structure_never_crosses_sections(strategy):
    for c in run(strategy):
        sections = {p.section for p in DOC.paragraphs if p.char_start < c.char_end and p.char_end > c.char_start}
        assert sections == {c.section}


def test_structure_packs_whole_paragraphs():
    chunks = run("structure")
    # Budget is 40 minus the 6-token header "A Test Paper > <section>" = 34.
    # 1 Introduction: paragraphs of 9 and 27 tokens do not fit together (36 > 34); the 108-token one is split.
    intro = [c for c in chunks if c.section == "1 Introduction"]
    assert (intro[0].char_start, intro[0].char_end) == (DOC.paragraphs[2].char_start, DOC.paragraphs[2].char_end)
    assert (intro[1].char_start, intro[1].char_end) == (DOC.paragraphs[3].char_start, DOC.paragraphs[3].char_end)
    big = DOC.paragraphs[4]
    assert all(big.char_start <= c.char_start and c.char_end <= big.char_end for c in intro[2:])
    # Methods: the 18-token paragraph packs with the first sentence of the 36-token one (18 + 9 + 9 > 34).
    methods = [c for c in chunks if c.section == "2 Methods"]
    assert methods[0].char_start == DOC.paragraphs[5].char_start
    assert DOC.paragraphs[5].char_end < methods[0].char_end < DOC.paragraphs[6].char_end
    assert methods[-1].char_end == DOC.paragraphs[6].char_end


def test_header_text_and_budget():
    plain = run("structure")
    headed = run("structure+header")
    for c in headed:
        sec = c.section
        assert c.text.startswith("A Test Paper > " + sec + "\n\n" if sec else "A Test Paper\n\n")
        assert words(c.text) <= MAX
    assert [(c.char_start, c.char_end) for c in headed] == [(c.char_start, c.char_end) for c in plain]
    assert headed[0].text.startswith("A Test Paper\n\n")  # section None -> no " > "


def test_structure_reserves_header_budget():
    # with a 6-token header, 8+24 = 32 body tokens fit in 40 - 6 but 40 would not
    doc = build_document("r", "T", [{"text": para(5, "a"), "section": "S", "page": 1}])  # 40 tokens
    plain = chunk_document(doc, "structure", words, 40, 0)
    assert len(plain) == 2  # "T > S" costs 3 tokens, so the 40-token paragraph no longer fits whole
    assert all(words(c.text) <= 37 for c in plain)


def test_header_without_title_and_oversize_header():
    d = build_document("p", "", [{"text": para(1, "x"), "section": "S", "page": 1}])
    assert chunk_document(d, "structure+header", words, 40, 0)[0].text.startswith("S\n\n")
    long_title = " ".join(["t"] * 50)
    d = build_document("p", long_title, [{"text": para(1, "x"), "section": "S", "page": 1}])
    c = chunk_document(d, "structure+header", words, 40, 0)
    assert [x.text for x in c] == [d.text]  # header dropped: it cannot fit


def test_fixed_overlaps_others_do_not():
    fixed = run("fixed")
    assert len(fixed) > 2
    for a, b in zip(fixed, fixed[1:]):
        assert b.char_start < a.char_end  # overlap
        assert b.char_start > a.char_start  # progress
        assert words(DOC.text[b.char_start:a.char_end]) <= OVERLAP
    for strategy in ("recursive", "structure"):
        chunks = run(strategy)
        assert all(a.char_end <= b.char_start for a, b in zip(chunks, chunks[1:]))


def test_fixed_zero_overlap():
    chunks = run("fixed", overlap=0)
    assert all(a.char_end <= b.char_start for a, b in zip(chunks, chunks[1:]))
    assert all(words(c.text) <= MAX for c in chunks)


def test_fixed_ignores_structure_and_cuts_words_cleanly():
    chunks = run("fixed")
    assert any(len({p.section for p in DOC.paragraphs
                    if p.char_start < c.char_end and p.char_end > c.char_start}) > 1 for c in chunks)
    for c in chunks:
        assert (c.char_start == 0 or DOC.text[c.char_start - 1].isspace())
        assert (c.char_end == len(DOC.text) or DOC.text[c.char_end].isspace())


def test_recursive_prefers_paragraph_boundaries():
    chunks = run("recursive")
    para_starts = {p.char_start for p in DOC.paragraphs}
    # the first chunk (abstract heading + abstract, then intro paragraphs) starts at a paragraph
    assert chunks[0].char_start == 0
    assert sum(c.char_start in para_starts for c in chunks) >= len(chunks) // 2


def test_metadata():
    chunks = run("structure")
    results = chunks[-1]
    assert (results.section, results.page_start, results.page_end) == ("3 Results", 5, 5)
    methods = [c for c in chunks if c.section == "2 Methods"]
    assert [(c.page_start, c.page_end) for c in methods] == [(3, 4), (4, 4)]
    intro = [c for c in chunks if c.section == "1 Introduction"]
    assert (intro[0].page_start, intro[0].page_end) == (1, 1)
    assert (intro[1].page_start, intro[1].page_end) == (2, 2)
    assert chunks[0].section is None and chunks[0].page_start == 1
    fixed = run("fixed")
    assert fixed[0].section is None
    assert fixed[-1].section == "3 Results" or fixed[-1].page_end == 5
    assert fixed[-1].page_end == 5 and all(c.page_start <= c.page_end for c in fixed)


def test_empty_document():
    empty = build_document("e", "T", [])
    for s in STRATEGIES:
        assert chunk_document(empty, s, words) == []
    blank = build_document("e", "T", [{"text": "  ", "section": None, "page": 1}])
    for s in STRATEGIES:
        assert chunk_document(blank, s, words) == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_single_huge_paragraph(strategy):
    doc = build_document("h", "T", [{"text": para(60, "huge"), "section": "S", "page": 1}])
    chunks = chunk_document(doc, strategy, words, 30, 5)
    assert len(chunks) >= 16
    assert all(words(c.text) <= 30 for c in chunks)
    covered = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    assert all(i in covered for i, ch in enumerate(doc.text) if not ch.isspace())


def test_sentence_split_in_huge_paragraph_prefers_sentence_boundaries():
    doc = build_document("h", "T", [{"text": para(20, "x"), "section": "S", "page": 1}])
    for c in chunk_document(doc, "structure", words, 30, 0):
        assert c.text.endswith("words.") and c.text.startswith("Sentence")


def test_unsplittable_word_is_its_own_oversize_chunk():
    giant = "x" * 50  # one "word"; counter below charges per character-run of 10
    doc = build_document("u", "T", [{"text": f"alpha beta {giant} gamma delta", "section": None, "page": 1}])

    def count(t: str) -> int:
        return sum(-(-len(w) // 10) for w in t.split())  # giant costs 5 tokens

    for s in ("fixed", "recursive", "structure"):
        chunks = chunk_document(doc, s, count, 3, 0)
        assert any(c.text == giant for c in chunks)
        assert all(count(c.text) <= 3 or c.text == giant for c in chunks)
        assert " ".join(c.text for c in chunks) == doc.text


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_document(DOC, "nope", words)


@pytest.mark.parametrize("text", [
    "We follow prior work, e.g. Smith et al. Then we extend it. Next step.",
])
def test_sentence_splitter_abbreviations(text):
    assert split_sentences(text) == [
        "We follow prior work, e.g. Smith et al. Then we extend it.",
        "Next step.",
    ]


@pytest.mark.parametrize("text", [
    "That is, i.e. Paris is the capital.",
    "See Fig. 3 for details and Eq. 2 for the loss.",
    "As shown in Sec. 4 and Tab. 2 below.",
    "Ours is faster vs. the baseline.",
    "Compare cf. Jones for more.",
    "The score was 3.5 versus 4.25 overall.",
    "Proposed by J. Smith and A. B. Jones in 2020.",
    "Brown et al. We confirm this.",
    "Results (e.g. Table 1) hold.",
])
def test_sentence_splitter_no_split(text):
    assert split_sentences(text) == [text]


def test_sentence_splitter_splits_normal_sentences():
    assert split_sentences("First one. Second one! Third one? \"Fourth.\" Fifth.") == [
        "First one.", "Second one!", "Third one?", "\"Fourth.\"", "Fifth.",
    ]
    assert split_sentences("") == []
    assert split_sentences("no terminal punctuation") == ["no terminal punctuation"]


@pytest.mark.parametrize("text", [
    "Ask Dr. Smith about it.",
    "Thanks to Mr. Jones and Mrs. Lee and Ms. Kim.",
    "Prof. Hinton agreed with Prof. Bengio.",
    "See Ref. Alpha and Refs. Beta for Alg. Two.",
    "Acme Inc. Holdings paid Vol. Three fees.",
    "Visit St. Andrews today.",
    "The U.S. Army and the U.K. Navy met.",
    "Many tools, e.g. Adam, i.e. Optimizers, work.",
    "Also known as a.k.a. Bob here.",
])
def test_sentence_splitter_more_abbreviations(text):
    assert split_sentences(text) == [text]


def test_sentence_spans_match_split_sentences():
    text = "  First one.  Second, e.g. Smith et al. Then more!\n\nThird.  "
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == split_sentences(text)
    assert split_sentences(text) == ["First one.", "Second, e.g. Smith et al. Then more!", "Third."]
    for s, e in spans:
        assert not text[s].isspace() and not text[e - 1].isspace()
    assert sentence_spans("") == []


def test_sentence_splitting_is_linear():
    import time
    sentence = "Sentence number %d has a few plain words, e.g. Smith et al. and J. Doe."
    big = " ".join(sentence % i for i in range(3000))  # ~200k chars, one paragraph
    assert len(big) > 190_000
    doc = build_document("perf", "T", [{"text": big, "section": "S", "page": 1}])
    for strategy in ("recursive", "structure"):
        t0 = time.perf_counter()
        chunks = chunk_document(doc, strategy, words, 64, 0)
        elapsed = time.perf_counter() - t0
        print(strategy, f"{elapsed:.2f}s")
        assert chunks and elapsed < 2


@pytest.mark.parametrize("max_tokens", [8, 12, 40])
def test_header_capped_at_quarter_of_budget(max_tokens):
    title = "one two three four"  # 4 tokens; with "\n\n" still 4 words
    cap = max_tokens // 4
    body = " ".join(f"w{i}" for i in range(60))
    doc = build_document("c", title, [{"text": body, "section": "Sec A", "page": 1}])
    plain = chunk_document(doc, "structure", words, max_tokens, 0)
    headed = chunk_document(doc, "structure+header", words, max_tokens, 0)
    assert [(c.char_start, c.char_end) for c in plain] == [(c.char_start, c.char_end) for c in headed]
    for c in headed:
        prefix = c.text[:len(c.text) - (c.char_end - c.char_start)]
        assert words(prefix) <= cap
        assert words(c.text) <= max_tokens
    # 4-word title + section (6 tokens) fits only when cap >= 6; title alone when cap >= 4
    expected = "one two three four > Sec A\n\n" if cap >= 6 else "one two three four\n\n" if cap >= 4 else ""
    assert all(c.text.startswith(expected) for c in headed)
    if not expected:
        assert [c.text for c in headed] == [doc.text[c.char_start:c.char_end] for c in headed]


def test_sentence_splitter_question_and_exclaim_ignore_abbreviations():
    assert split_sentences("Is it no? Yes it is. Really vs! Fine.") == [
        "Is it no?", "Yes it is.", "Really vs!", "Fine.",
    ]


def test_sentence_splitter_curly_closing_quotes():
    assert split_sentences("He said “stop.” Then he left. She said ‘go.’ Done.") == [
        "He said “stop.”", "Then he left.", "She said ‘go.’", "Done.",
    ]
