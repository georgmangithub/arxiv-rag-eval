from arxiv_rag.parse import is_heading_text, join_lines


def test_join_lines_repairs_soft_hyphen():
    assert join_lines(["retrieval bench-", "marking is hard"]) == "retrieval benchmarking is hard"


def test_join_lines_keeps_real_hyphen_after_uppercase_prefix():
    assert join_lines(["an LLM-", "based pipeline"]) == "an LLM-based pipeline"


def test_join_lines_does_not_merge_when_next_line_is_capitalised():
    assert join_lines(["see the well-", "Known result"]) == "see the well- Known result"


def test_join_lines_merges_hard_wraps_and_collapses_whitespace():
    assert join_lines(["one  two", "", "three"]) == "one two three"


def test_is_heading_text():
    assert is_heading_text("1 Introduction")
    assert is_heading_text("2.1 Clinical Summarization and Lay-Language Simplification")
    assert is_heading_text("Abstract")
    assert not is_heading_text("We propose a new method.")
    assert not is_heading_text("x" * 200)
