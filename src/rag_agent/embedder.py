"""Embedding factory.

Three providers are supported:

* ``local`` (default): ``HuggingFaceEmbeddings`` with
  ``BAAI/bge-small-zh-v1.5`` — good Chinese + English coverage, runs fully
  offline once the model is cached.
* ``openai``: any OpenAI-compatible embedding endpoint
  (``OPENAI_EMBED_BASE_URL`` + ``OPENAI_EMBED_MODEL``).
* ``fake``: a deterministic bag-of-words embedding used for offline CLI smoke
  tests (e.g. ``EMBEDDING_PROVIDER=fake python main.py ingest --dir ./data``)
  and as the test fixture backing the in-memory Chroma store. No network, no
  model download — yet cosine-similarity semantics are real, so documents and
  queries sharing tokens actually score higher.
"""

from __future__ import annotations

import hashlib
import random

from langchain_core.embeddings import Embeddings

from .config import get_settings


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (no network, no model download).

    Each token maps to a seeded pseudo-random vector and a text is the average
    of its token vectors. Cosine similarity therefore reflects token overlap,
    which is enough for smoke-testing ingest / retrieve / ask without a real
    embedding model.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim
        self._cache: dict[str, list[float]] = {}

    def _token_vec(self, token: str) -> list[float]:
        if token in self._cache:
            return self._cache[token]
        seed = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        self._cache[token] = vec
        return vec

    def _embed(self, text: str) -> list[float]:
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return [0.0] * self.dim
        acc = [0.0] * self.dim
        for tok in tokens:
            v = self._token_vec(tok)
            for i in range(self.dim):
                acc[i] += v[i]
        for i in range(self.dim):
            acc[i] /= len(tokens)
        return acc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_embeddings() -> Embeddings:
    """Build the embedding function based on ``EMBEDDING_PROVIDER``."""
    settings = get_settings()
    provider = settings.embedding_provider

    if provider == "fake":
        # Offline deterministic embeddings — handy for CI / smoke tests.
        return FakeEmbeddings(dim=32)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        params: dict[str, object] = {"model": settings.openai_embed_model}
        if settings.openai_embed_base_url:
            params["base_url"] = settings.openai_embed_base_url
        if settings.openai_api_key:
            params["api_key"] = settings.openai_api_key
        return OpenAIEmbeddings(**params)  # type: ignore[arg-type]

    # Default: local HuggingFace embeddings.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.local_embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )
