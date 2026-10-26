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
