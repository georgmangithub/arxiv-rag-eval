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


def test_read_ids_file_handles_comments_blanks_and_dupes(tmp_path):
    from arxiv_rag.ingest import read_ids_file

    path = tmp_path / "ids.txt"
    path.write_text("# corpus\n2401.00001v1\n\n  2401.00002v2  # inline\ncs/0112017v1\n2401.00001v1\n", encoding="utf-8")
    assert read_ids_file(path) == ["2401.00001v1", "2401.00002v2", "cs/0112017v1"]


def test_write_ids_file_sorted_roundtrip(tmp_path):
    from arxiv_rag.ingest import read_ids_file, write_ids_file

    path = tmp_path / "sub" / "ids.txt"
    write_ids_file(path, {"b", "a", "c"})
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"
    assert read_ids_file(path) == ["a", "b", "c"]


def test_write_ids_file_uses_lf_newlines(tmp_path):
    from arxiv_rag.ingest import write_ids_file

    path = tmp_path / "ids.txt"
    write_ids_file(path, ["b", "a"])
    assert path.read_bytes() == b"a\nb\n"


def test_ingest_rejects_empty_ids_without_searching(tmp_path, monkeypatch):
    import pytest

    from arxiv_rag import ingest as ing

    monkeypatch.setattr(ing.arxiv, "Search", lambda *a, **k: pytest.fail("must not search"))
    with pytest.raises(ValueError):
        ing.ingest(None, 0, tmp_path, ids=[])


def test_ingest_warns_on_ids_arxiv_did_not_return(tmp_path, monkeypatch, capsys):
    from arxiv_rag import ingest as ing

    class R:
        def get_short_id(self):
            return "2401.00001v2"

    class FakeClient:
        def __init__(self, **k):
            pass

        def results(self, search):
            return iter([R()])

    monkeypatch.setattr(ing.arxiv, "Client", FakeClient)
    monkeypatch.setattr(ing.arxiv, "Search", lambda **k: None)
    monkeypatch.setattr(ing, "_fetch_all", lambda results, *a: {"n": len(list(results))})
    ing.ingest(None, 0, tmp_path, ids=["2401.00001", "2401.99999v1"])
    out = capsys.readouterr().out
    assert "2401.99999v1" in out and "2401.00001 " not in out
