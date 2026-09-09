"""Shared test fixtures.

Everything here is **offline and deterministic**:

* :class:`FakeEmbeddings` — a bag-of-words embedding: each token maps to a
  seeded pseudo-random vector and a text is the average of its token vectors.
  This gives real cosine-similarity semantics so that documents and queries
  sharing words actually score higher — without downloading any model.
* :class:`FakeLLM` — a duck-typed chat model. ``invoke`` accepts a prompt
  (string or prompt-value) and returns an object with ``.content``. Behaviour
  is controlled either by a ``responder`` callable or a fixed ``responses``
  list, so each test can script exact LLM answers.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import deque

import pytest

from langchain_core.embeddings import Embeddings

# Keep chromadb's posthog telemetry quiet and offline during tests.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (no network, no model download)."""

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


class FakeAIMessage:
    """Minimal stand-in for an AIMessage: only ``.content`` is consumed.

    Kept for backwards compatibility with older tests; new code should prefer a
    real :class:`langchain_core.messages.AIMessage` (see :class:`FakeLLM`).
    """

    def __init__(self, content: str):
        self.content = content
        self.type = "ai"


class FakeLLM:
    """Duck-typed chat model used in place of ``ChatOpenAI`` in tests.

    Parameters
    ----------
    responses:
        Fixed list of answer strings returned in order (cycled if exhausted).
    responder:
        Optional callable ``f(prompt_text: str) -> str``. When set it takes
        precedence over ``responses`` and lets a test branch on the prompt.
    """

    def __init__(self, responses=None, responder=None):
        self.responses = deque(responses or [])
        self.responder = responder
        self.calls: list[str] = []

    @staticmethod
    def _to_text(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        # ChatPromptValue / StringPromptValue expose .to_string()
        to_string = getattr(prompt, "to_string", None)
        if callable(to_string):
            return to_string()
        messages = getattr(prompt, "messages", None)
        if messages is not None:
            return "\n".join(
                getattr(m, "content", str(m)) for m in messages
            )
        if isinstance(prompt, list):
            return "\n".join(getattr(m, "content", str(m)) for m in prompt)
        return str(prompt)

    def invoke(self, prompt, config=None, **kwargs):  # noqa: D401
        text = self._to_text(prompt)
        self.calls.append(text)
        if self.responder is not None:
            content = self.responder(text)
        elif self.responses:
            content = self.responses.popleft()
            self.responses.append(content)  # cycle
        else:
            content = ""
        # Return a real AIMessage (a BaseMessage): langchain output-parser
        # pipelines (e.g. MultiQueryRetriever) detect BaseMessage and extract
        # ``.content``. Returning a plain object would break the parser chain.
        from langchain_core.messages import AIMessage

        return AIMessage(content=content)

    # Some langchain code paths call the model directly; keep them happy.
    def __call__(self, prompt, **kwargs):
        return self.invoke(prompt)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(dim=32)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(responder=lambda _prompt: "yes")


@pytest.fixture
def make_llm():
    """Factory for building :class:`FakeLLM` instances inside tests."""

    def _make(responder=None, responses=None):
        return FakeLLM(responder=responder, responses=responses)

    return _make


@pytest.fixture
def in_memory_vectorstore(fake_embeddings):
    """A fresh in-memory Chroma store (no disk persistence).

    chromadb's in-memory client is a process-level singleton, so we give each
    test a unique collection name to guarantee isolation between tests that
    would otherwise share the ``test_collection`` collection.
    """
    import uuid

    from langchain_community.vectorstores import Chroma

    unique_name = f"test_{uuid.uuid4().hex}"
    return Chroma(
        persist_directory=None,
        embedding_function=fake_embeddings,
        collection_name=unique_name,
    )


@pytest.fixture
def cs_vectorstore_lookup(fake_embeddings):
    """Per-category in-memory Chroma stores for the customer-service agent.

    Returns a callable ``(category: str) -> Chroma``. The same in-memory
    chromadb client backs every store, but each category lands in its own
    isolated collection (named ``cs_<category>_<uuid>``) so we can ingest
    different documents into ``product`` / ``policy`` / ``sop`` / ``general``
    and verify the router actually picks the right one.
    """
    import uuid

    from langchain_community.vectorstores import Chroma

    stores: dict[str, Chroma] = {}

    def _get(category: str):
        cat = category or "general"
        if cat not in stores:
            stores[cat] = Chroma(
                persist_directory=None,
                embedding_function=fake_embeddings,
                collection_name=f"cs_{cat}_{uuid.uuid4().hex}",
            )
        return stores[cat]

    return _get
