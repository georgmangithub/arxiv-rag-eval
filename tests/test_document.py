import json

import pytest

from arxiv_rag.document import PARAGRAPH_SEP, build_document, id_to_stem, ids_to_stems, load_documents, load_titles


def test_paragraph_offsets_slice_back_to_their_text():
    records = [
        {"text": "First para.", "section": "Abstract", "page": 1},
        {"text": "Second one here.", "section": "1 Introduction", "page": 1},
        {"text": "Third.", "section": "1 Introduction", "page": 2},
    ]
    doc = build_document("x", "T", records)
    assert doc.text == PARAGRAPH_SEP.join(r["text"] for r in records)
    for p, r in zip(doc.paragraphs, records):
        assert doc.text[p.char_start:p.char_end] == r["text"]
        assert (p.section, p.page) == (r["section"], r["page"])


def test_empty_document():
    doc = build_document("x", "T", [])
    assert doc.text == "" and doc.paragraphs == []


def test_id_to_stem_matches_pdf_filename_rule():
    assert ids_to_stems(["2401.12345v2", "cs/0112017v1"]) == ["2401.12345v2", "cs_0112017v1"]
    assert id_to_stem("hep-th/9901001") == "hep-th_9901001"


def test_load_documents_errors_on_missing_stem(tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "a.jsonl").write_text(json.dumps({"text": "Hi.", "section": "S", "page": 1}) + "\n", encoding="utf-8")
    assert [d.paper_id for d in load_documents(tmp_path, ["a"])] == ["a"]
    with pytest.raises(FileNotFoundError, match="b, c"):
        load_documents(tmp_path, ["a", "b", "c"])


def test_load_titles_skips_malformed_lines(tmp_path, capsys):
    good = json.dumps({"paper_id": "2401.00001v1", "title": "T1"})
    good2 = good.replace("00001", "00002")
    lines = [good, '{"paper_id": "2401.0', "", "[1]", '{"x": 1}', good2, '{"paper_id": "9']
    (tmp_path / "metadata.jsonl").write_text(chr(10).join(lines), encoding="utf-8")
    assert load_titles(tmp_path) == {"2401.00001v1": "T1", "2401.00002v1": "T1"}
    assert capsys.readouterr().out.count("WARNING") == 4
