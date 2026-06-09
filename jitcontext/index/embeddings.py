"""Embedding providers — offline default + configurable real backends.

Design:
- The retriever depends only on the `embed(texts) -> list[vec]` interface.
- `LocalHashEmbedder` is deterministic/offline so tests need no keys.
- Real backends are thin adapters. Clients are passed IN by the caller (no hard
  SDK imports), so importing this module never requires any provider package.
- Most providers (OpenAI, OpenRouter, Azure, Together, vLLM, …) share the
  OpenAI embeddings shape, so one adapter covers all of them via base_url.
  Cohere and Voyage get small dedicated adapters.
- Vectors are L2-normalised on the way out, so `cosine` is a plain dot product
  regardless of backend.

Select a backend by string via `make_embedder(spec)` where spec.provider is one
of: "local", "openai", "cohere", "voyage".
"""
from __future__ import annotations
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------- #
# Offline default
# --------------------------------------------------------------------------- #
class LocalHashEmbedder:
    """Hashed bag-of-words into a fixed-dim vector. Deterministic, offline.

    NOT semantically good — for tests/CI only. Use a real backend in practice.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalise(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


# --------------------------------------------------------------------------- #
# Real backends (clients injected; no hard imports)
# --------------------------------------------------------------------------- #
class OpenAICompatibleEmbedder:
    """Works with any OpenAI-shaped embeddings endpoint.

    Covers OpenAI, OpenRouter, Azure OpenAI, Together, local vLLM, etc. — just
    point the client's base_url at the provider. `client` must expose
    `client.embeddings.create(model=..., input=[...])` returning objects with
    `.data[i].embedding` (the official `openai` SDK shape).

    Example:
        from openai import OpenAI
        emb = OpenAICompatibleEmbedder(OpenAI(), model="text-embedding-3-small")
    """

    def __init__(self, client, model: str, batch_size: int = 128,
                 dimensions: int | None = None) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions  # 3-* models support truncation

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            kwargs = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            resp = self.client.embeddings.create(**kwargs)
            # resp.data preserves input order per the API contract.
            out.extend(_l2_normalise(list(d.embedding)) for d in resp.data)
        return out


class CohereEmbedder:
    """Cohere embed-v3/v4. Uses input_type to optimise query vs document.

    `client` is a cohere.Client(...). We embed index summaries as documents and
    queries as search queries (Cohere recommends distinct input types).
    """

    def __init__(self, client, model: str = "embed-v4.0",
                 input_type: str = "search_document", batch_size: int = 96) -> None:
        self.client = client
        self.model = model
        self.input_type = input_type
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embed(model=self.model, texts=batch,
                                     input_type=self.input_type)
            out.extend(_l2_normalise(list(e)) for e in resp.embeddings)
        return out


class VoyageEmbedder:
    """Voyage AI embeddings. `client` is a voyageai.Client(...)."""

    def __init__(self, client, model: str = "voyage-3",
                 input_type: str | None = "document", batch_size: int = 128) -> None:
        self.client = client
        self.model = model
        self.input_type = input_type
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embed(batch, model=self.model,
                                     input_type=self.input_type)
            out.extend(_l2_normalise(list(e)) for e in resp.embeddings)
        return out


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
@dataclass
class EmbedderSpec:
    provider: str = "local"            # local | openai | cohere | voyage
    model: str = ""                    # provider model id
    client: object | None = None       # provider SDK client (real backends)
    dimensions: int | None = None
    extra: dict = field(default_factory=dict)


def make_embedder(spec: EmbedderSpec) -> Embedder:
    p = spec.provider.lower()
    if p == "local":
        return LocalHashEmbedder(**({"dim": spec.dimensions} if spec.dimensions else {}))
    if spec.client is None:
        raise ValueError(f"provider '{p}' requires spec.client (an SDK client)")
    if p in ("openai", "openai_compatible", "openrouter", "azure", "together"):
        return OpenAICompatibleEmbedder(spec.client, spec.model or "text-embedding-3-small",
                                        dimensions=spec.dimensions, **spec.extra)
    if p == "cohere":
        return CohereEmbedder(spec.client, spec.model or "embed-v4.0", **spec.extra)
    if p == "voyage":
        return VoyageEmbedder(spec.client, spec.model or "voyage-3", **spec.extra)
    raise ValueError(f"unknown embedder provider: {spec.provider}")


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are pre-normalised
