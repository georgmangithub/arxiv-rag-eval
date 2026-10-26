import numpy as np

from arxiv_rag import embed
from arxiv_rag.embed import BGE_QUERY_PREFIX, Embedder, FastEmbedBackend
from fakes import DIM, FakeBackend


def make(tmp_path, backend=None):
    return Embedder(cache_dir=tmp_path, backend=backend or FakeBackend())


def test_normalized_float32(tmp_path):
    e = make(tmp_path)
    m = e.embed_passages(["alpha beta", "gamma"])
    assert m.shape == (2, DIM) and m.dtype == np.float32
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-6)
    q = e.embed_query("alpha")
    assert q.shape == (DIM,) and np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)


def test_cache_hits_do_not_reembed(tmp_path):
    be = FakeBackend()
    e = make(tmp_path, be)
    a = e.embed_passages(["one", "two"])
    b = e.embed_passages(["two", "three", "one"])
    assert be.passage_calls == [["one", "two"], ["three"]]
    assert np.array_equal(b[0], a[1]) and np.array_equal(b[2], a[0])
    e.embed_passages(["one", "three"])
    assert len(be.passage_calls) == 2


def test_duplicates_in_one_call_embedded_once(tmp_path):
    be = FakeBackend()
    m = make(tmp_path, be).embed_passages(["x", "x", "y"])
    assert be.passage_calls == [["x", "y"]]
    assert np.array_equal(m[0], m[1])


def test_cache_survives_reload(tmp_path):
    first = make(tmp_path).embed_passages(["one", "two"])
    be = FakeBackend()
    again = make(tmp_path, be).embed_passages(["two", "one"])
    assert be.passage_calls == []
    assert np.array_equal(again, first[::-1])


def test_corrupt_cache_is_cold_not_fatal(tmp_path):
    make(tmp_path).embed_passages(["one"])
    (tmp_path / "BAAI__bge-small-en-v1.5.fake.npz").write_bytes(b"not a zip")
    be = FakeBackend()
    make(tmp_path, be).embed_passages(["one"])
    assert be.passage_calls == [["one"]]


def test_cache_keyed_by_model(tmp_path):
    be = FakeBackend()
    Embedder(model_name="m1", cache_dir=tmp_path, backend=be).embed_passages(["t"])
    Embedder(model_name="m2", cache_dir=tmp_path, backend=be).embed_passages(["t"])
    assert len(be.passage_calls) == 2


def test_empty_input(tmp_path):
    assert make(tmp_path).embed_passages([]).shape[0] == 0


def test_count_tokens_and_max_delegate(tmp_path):
    e = make(tmp_path)
    assert e.max_tokens == 512 and e.count_tokens("a b c") == 5


def test_query_prefix_applied_by_fastembed_backend():
    seen = []

    class M:
        def embed(self, texts):
            seen.append(list(texts))
            return iter([np.ones(4, dtype=np.float32)])

    be = FastEmbedBackend.__new__(FastEmbedBackend)  # skip model load
    be._model = M()
    be.embed_query("what is attention")
    assert seen == [[BGE_QUERY_PREFIX + "what is attention"]]


def test_saves_after_each_block(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "SAVE_BLOCK", 2)
    be = FakeBackend()
    make(tmp_path, be).embed_passages(["a", "b", "c", "d", "e"])
    assert [len(c) for c in be.passage_calls] == [2, 2, 1]

    class Crash(FakeBackend):
        def embed_passages(self, texts):
            if "z" in texts:
                raise RuntimeError("boom")
            return super().embed_passages(texts)

    e = make(tmp_path / "c", Crash())
    try:
        e.embed_passages(["p", "q", "z", "w"])
    except RuntimeError:
        pass
    be2 = FakeBackend()
    make(tmp_path / "c", be2).embed_passages(["p", "q"])
    assert be2.passage_calls == []  # first block survived the crash


def test_backend_row_count_checked(tmp_path):
    class Short(FakeBackend):
        def embed_passages(self, texts):
            return super().embed_passages(texts)[:-1]

    try:
        make(tmp_path, Short()).embed_passages(["a", "b"])
        assert False, "expected AssertionError"
    except AssertionError as ex:
        assert "returned 1 rows" in str(ex)


def test_empty_cache_file_is_cold_cache(tmp_path):
    make(tmp_path).embed_passages(["one"])
    (tmp_path / "BAAI__bge-small-en-v1.5.fake.npz").write_bytes(b"")
    be = FakeBackend()
    make(tmp_path, be).embed_passages(["one"])
    assert be.passage_calls == [["one"]]


def test_cache_keyed_by_backend_tag(tmp_path):
    make(tmp_path).embed_passages(["t"])
    other = FakeBackend()
    other.tag = "fake2"
    make(tmp_path, other).embed_passages(["t"])
    assert other.passage_calls == [["t"]]


def test_crash_mid_save_keeps_previous_complete_cache(tmp_path, monkeypatch):
    make(tmp_path).embed_passages(["a", "b"])

    def boom(src, dst):
        raise OSError("crash before replace")

    monkeypatch.setattr(embed.os, "replace", boom)
    try:
        make(tmp_path).embed_passages(["c"])
    except OSError:
        pass
    monkeypatch.undo()
    be = FakeBackend()
    make(tmp_path, be).embed_passages(["a", "b"])
    assert be.passage_calls == []  # earlier progress intact
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_incremental_saves_accumulate_and_reload(tmp_path):
    make(tmp_path).embed_passages(["a"])
    make(tmp_path).embed_passages(["a", "b"])
    be = FakeBackend()
    out = make(tmp_path, be).embed_passages(["b", "a"])
    assert be.passage_calls == [] and out.shape[0] == 2
