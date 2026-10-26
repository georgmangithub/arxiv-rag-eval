"""Embed passages and queries with a small local model, with an on-disk cache.

Design goals:
- CPU-friendly: bge-small via fastembed (onnxruntime), no torch, no GPU.
- Cheap to re-run: passage vectors are cached by sha256(model + text), so
  re-chunking only embeds chunks whose text actually changed.
- Testable: the model sits behind a small Backend seam; tests inject a fake and
  never download anything. The real model is loaded lazily on first use.
- Crash-safe: misses are embedded in blocks and the cache is saved after each.
- Cache namespace includes the fastembed version, so a library upgrade can't
  silently reuse vectors made by different code/weights.

Note: fastembed's default artifact for BAAI/bge-small-en-v1.5 is the quantized
Qdrant/bge-small-en-v1.5-onnx-Q, not the fp32 original.
"""
import hashlib
import os
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_ROOT = Path(__file__).resolve().parents[2]  # repo root, independent of cwd
DEFAULT_CACHE_DIR = _ROOT / "data" / "cache" / "embeddings"
MODEL_DIR = _ROOT / "data" / "cache" / "models"
BATCH_SIZE = 32
SAVE_BLOCK = 512  # cache misses embedded per save
# bge retrieval instruction; fastembed's query_embed does NOT add it for this model.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _fastembed_tag() -> str:
    return "fastembed-" + metadata.version("fastembed")


class Backend(Protocol):
    max_tokens: int
    tag: str  # cache namespace: identifies the library/weights producing vectors

    def count_tokens(self, text: str) -> int: ...
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedBackend:
    @property
    def tag(self) -> str:
        return _fastembed_tag()

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        # Default download dir is %TEMP%, which the OS may clean; keep it under data/.
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(MODEL_DIR))
        onnx = self._model.model
        onnx._ensure_tokenizer()  # fastembed has no public tokenizer accessor
        tok = onnx.tokenizer
        self.max_tokens: int = tok.truncation["max_length"]
        # fastembed's tokenizer truncates at max_tokens, which would hide overlong
        # text from the count. Use a private copy with truncation/padding off.
        self._tok = Tokenizer.from_str(tok.to_str())
        self._tok.no_truncation()
        self._tok.no_padding()

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=True).ids)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack(list(self._model.passage_embed(texts, batch_size=BATCH_SIZE)))

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed is a plain embed() for this model, so add the prefix ourselves
        return next(iter(self._model.embed([BGE_QUERY_PREFIX + text])))


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        backend: Backend | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._backend = backend
        # tag without loading the model, so construction stays lazy
        self._tag = backend.tag if backend is not None else _fastembed_tag()
        self._keys: dict[str, int] = {}  # cache key -> row in self._matrix
        self._matrix: np.ndarray | None = None
        self._load_cache()

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = FastEmbedBackend(self.model_name)
        return self._backend

    @property
    def max_tokens(self) -> int:
        return self.backend.max_tokens

    def count_tokens(self, text: str) -> int:
        """Token count INCLUDING special tokens ([CLS]/[SEP]), untruncated.

        Conservative on purpose: count_tokens(t) <= max_tokens guarantees the
        model sees the whole text, so chunkers can use it as a hard budget.
        """
        return self.backend.count_tokens(text)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """(n, d) float32, L2-normalized. Only cache misses hit the model."""
        keys = [self._key(t) for t in texts]
        # dict keyed by cache key also dedupes repeated texts within the call
        misses = {k: t for k, t in zip(keys, texts) if k not in self._keys}
        if misses:
            items = list(misses.items())
            for i in range(0, len(items), SAVE_BLOCK):
                block = items[i : i + SAVE_BLOCK]
                out = self.backend.embed_passages([t for _, t in block])
                assert len(out) == len(block), f"backend returned {len(out)} rows for {len(block)} texts"
                new = _normalize(out)
                base = 0 if self._matrix is None else len(self._matrix)
                self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
                for j, (k, _) in enumerate(block):
                    self._keys[k] = base + j
                self._save_cache()  # keep progress if a later block crashes
        if not texts:
            dim = 0 if self._matrix is None else self._matrix.shape[1]
            return np.zeros((0, dim), dtype=np.float32)
        return self._matrix[[self._keys[k] for k in keys]]

    def embed_query(self, text: str) -> np.ndarray:
        """(d,) float32, L2-normalized, with the bge query instruction prefix (added by the backend). Not cached."""
        return _normalize(self.backend.embed_query(text))

    # --- cache -------------------------------------------------------------

    def _key(self, text: str) -> str:
        return hashlib.sha256((self.model_name + "\0" + text).encode("utf-8")).hexdigest()

    def _cache_path(self) -> Path:
        stem = self.model_name.replace("/", "__") + "." + self._tag
        return self.cache_dir / f"{stem}.npz"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as z:
                matrix, keys = z["matrix"], [str(k) for k in z["keys"]]
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            return  # unreadable cache is just a cold cache
        if matrix.ndim != 2 or len(keys) != len(matrix):
            return
        self._matrix, self._keys = matrix, {k: i for i, k in enumerate(keys)}

    def _save_cache(self) -> None:
        """Write vectors and keys together in one .npz via temp file + atomic replace.

        A crash at any point leaves either the previous complete cache or the new one.
        """
        path = self._cache_path()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.part")
        with tmp.open("wb") as f:
            np.savez(f, matrix=self._matrix, keys=np.array(list(self._keys), dtype=str))
        os.replace(tmp, path)
