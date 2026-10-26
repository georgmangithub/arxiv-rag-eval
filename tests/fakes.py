import hashlib
import re

import numpy as np

DIM = 64


class FakeBackend:
    """Bag-of-words hashing embedder that counts how many passages it embedded."""

    max_tokens = 512
    tag = "fake"

    def __init__(self):
        self.passage_calls: list[list[str]] = []

    def count_tokens(self, text):
        return len(text.split()) + 2

    def _vec(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        return v * 3.0  # deliberately not unit length

    def embed_passages(self, texts):
        self.passage_calls.append(list(texts))
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeRerankBackend:
    """Scores = number of query words appearing in the text; records calls."""

    max_tokens = 512
    tag = "fake"

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def count_pair_tokens(self, query, text):
        return len(query.split()) + len(text.split()) + 3

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        q = set(re.findall(r"\w+", query.lower()))
        return np.array([len(q & set(re.findall(r"\w+", t.lower()))) for t in texts], dtype=np.float32)
