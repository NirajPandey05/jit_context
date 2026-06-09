"""Embedding providers.

`LocalHashEmbedder` is deterministic, dependency-free, and offline so the whole
system is testable without network/keys. It is NOT semantically good — it's a
bag-of-words hash. Swap `APIEmbedder` in for real semantic quality.

The retriever depends only on the `embed(texts) -> list[vec]` interface.
"""
from __future__ import annotations
import hashlib
import math
import re

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class LocalHashEmbedder:
    """Hashed bag-of-words into a fixed-dim vector. Deterministic, offline."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class APIEmbedder:
    """Placeholder for a real embedding API (OpenAI, Voyage, etc.).

    Wire your client in `embed`. Kept separate so offline tests never need it.
    """

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Plug your embedding client here.")


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are pre-normalised
