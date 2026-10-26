"""Cross-encoder reranking on top of any retriever, with an on-disk score cache.

Design goals:
- CPU-friendly: ms-marco MiniLM cross-encoder via fastembed (onnxruntime), no torch.
- Cheap to re-run: scores are cached by sha256(model + tag + query + text), so
  re-running an eval (or sharing candidates between retrievers) only scores new pairs.
- Testable: the model sits behind a small Backend seam; tests inject a fake and
  never download anything. The real model is loaded lazily on first use.
- Crash-safe: misses are scored in blocks and the cache is saved after each, as one
  atomic .npz (temp file + os.replace), same design as embed.py.
- Cache namespace includes the fastembed version (file name and key), so a library
  upgrade can't silently reuse scores made by different code/weights.

Truncation: fastembed's tokenizer for this model truncates (query, passage) pairs at
Backend.max_tokens (read from the tokenizer, 512 for MiniLM-L-6). Chunks here are
<= 256 tokens and queries are short, so pairs fit; `count_pair_tokens` exposes the
untruncated pair length so this can be asserted rather than assumed.
"""
import hashlib
import os
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Protocol, Sequence, TypeVar

import numpy as np

from arxiv_rag.retrieve import HasText

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_ROOT = Path(__file__).resolve().parents[2]  # repo root, independent of cwd
DEFAULT_CACHE_DIR = _ROOT / "data" / "cache" / "rerank"
MODEL_DIR = _ROOT / "data" / "cache" / "models"
BATCH_SIZE = 32
SAVE_BLOCK = 256  # cache misses scored per save


def _fastembed_tag() -> str:
    return "fastembed-" + metadata.version("fastembed")


class Backend(Protocol):
    max_tokens: int
    tag: str  # cache namespace: identifies the library/weights producing scores

    def count_pair_tokens(self, query: str, text: str) -> int: ...
    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


class FastEmbedBackend:
    @property
    def tag(self) -> str:
        return _fastembed_tag()

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        from tokenizers import Tokenizer

        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(MODEL_DIR))
        onnx = self._model.model
        onnx._ensure_tokenizer()  # fastembed has no public tokenizer accessor
        tok = onnx.tokenizer
        self.max_tokens: int = tok.truncation["max_length"]
        # private copy with truncation/padding off, so overlong pairs are visible
        self._tok = Tokenizer.from_str(tok.to_str())
        self._tok.no_truncation()
        self._tok.no_padding()

    def count_pair_tokens(self, query: str, text: str) -> int:
        return len(self._tok.encode(query, text, add_special_tokens=True).ids)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.rerank(query, texts, batch_size=BATCH_SIZE)), dtype=np.float32)


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        backend: Backend | None = None,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._backend = backend
        # tag without loading the model, so construction stays lazy
        self._tag = backend.tag if backend is not None else _fastembed_tag()
        self._keys: dict[str, int] = {}  # cache key -> index in self._scores
        self._scores = np.zeros(0, dtype=np.float32)
        self._load_cache()

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = FastEmbedBackend(self.model_name)
        return self._backend

    @property
    def max_tokens(self) -> int:
        return self.backend.max_tokens

    def count_pair_tokens(self, query: str, text: str) -> int:
        """Untruncated (query, text) pair length including special tokens."""
        return self.backend.count_pair_tokens(query, text)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """(n,) float32 relevance logits, higher = more relevant. Only misses hit the model."""
        texts = list(texts)
        keys = [self._key(query, t) for t in texts]
        # dict keyed by cache key also dedupes repeated texts within the call
        misses = {k: t for k, t in zip(keys, texts) if k not in self._keys}
        if misses:
            items = list(misses.items())
            for i in range(0, len(items), SAVE_BLOCK):
                block = items[i : i + SAVE_BLOCK]
                out = np.asarray(self.backend.score(query, [t for _, t in block]), dtype=np.float32)
                assert out.shape == (len(block),), f"backend returned {out.shape} for {len(block)} texts"
                base = len(self._scores)
                self._scores = np.concatenate([self._scores, out])
                for j, (k, _) in enumerate(block):
                    self._keys[k] = base + j
                self._save_cache()  # keep progress if a later block crashes
        if not texts:
            return np.zeros(0, dtype=np.float32)
        return self._scores[[self._keys[k] for k in keys]]

    # --- cache -------------------------------------------------------------

    def _key(self, query: str, text: str) -> str:
        raw = self.model_name + self._tag + "\0" + query + "\0" + text
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self) -> Path:
        stem = self.model_name.replace("/", "__") + "." + self._tag
        return self.cache_dir / f"{stem}.npz"

    def _load_cache(self) -> None:
        path = self._cache_path()
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as z:
                scores, keys = z["scores"], [str(k) for k in z["keys"]]
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            return  # unreadable cache is just a cold cache
        if scores.ndim != 1 or len(keys) != len(scores):
            return
        self._scores, self._keys = scores.astype(np.float32), {k: i for i, k in enumerate(keys)}

    def _save_cache(self) -> None:
        """Write scores and keys together in one .npz via temp file + atomic replace."""
        path = self._cache_path()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.part")
        with tmp.open("wb") as f:
            np.savez(f, scores=self._scores, keys=np.array(list(self._keys), dtype=str))
        os.replace(tmp, path)


T = TypeVar("T", bound=HasText)


class _Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[T, float]]: ...


class RerankedRetriever:
    def __init__(self, base: _Retriever, reranker: Reranker, depth: int = 50):
        self.base = base
        self.reranker = reranker
        self.depth = depth

    def search(self, query: str, k: int) -> list[tuple[T, float]]:
        """Top-k (item, cross-encoder score) pairs, best first.

        Candidates are the base retriever's top max(depth, k) (so k > depth still
        returns k); ties break by base rank.
        """
        if k <= 0:
            return []
        cands = self.base.search(query, max(self.depth, k))
        if not cands:
            return []
        scores = self.reranker.score(query, [it.text for it, _ in cands])
        order = np.lexsort((np.arange(len(cands)), -scores))[:k]
        return [(cands[i][0], float(scores[i])) for i in order]
