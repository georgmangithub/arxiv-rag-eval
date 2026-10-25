import json

from arxiv_rag.ingest import load_known_ids, pdf_filename


def test_pdf_filename_handles_old_style_ids():
    assert pdf_filename("2401.12345v2") == "2401.12345v2.pdf"
    assert pdf_filename("cs/0112017v1") == "cs_0112017v1.pdf"


def test_load_known_ids_missing_file(tmp_path):
    assert load_known_ids(tmp_path / "nope.jsonl") == set()


def test_load_known_ids_reads_jsonl_and_ignores_blank_lines(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text(
        json.dumps({"paper_id": "a"}) + "\n\n" + json.dumps({"paper_id": "b"}) + "\n",
        encoding="utf-8",
    )
    assert load_known_ids(path) == {"a", "b"}
