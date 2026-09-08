"""Retrieval strategies.

Two strategies are exposed:

* ``basic``: plain similarity search with ``top_k`` results — the naive-RAG
  baseline.
* ``multi_query``: :class:`MultiQueryRetriever` asks the LLM to rewrite the
  question into several variants, retrieves for each, and deduplicates. This
  boosts recall when the user's wording diverges from the corpus.

Both return a :class:`BaseRetriever`, so the agent can treat them uniformly.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from .config import get_settings
from .llm import create_llm
from .prompts import MULTI_QUERY_PROMPT

RetrievalStrategy = Literal["basic", "multi_query"]


def create_retriever(
    vectorstore,
    strategy: RetrievalStrategy = "basic",
    llm: BaseChatModel | None = None,
) -> BaseRetriever:
    """Build a retriever on top of ``vectorstore``.

    Parameters
    ----------
    vectorstore:
        Any object exposing ``.as_retriever(...)`` (typically a Chroma store).
    strategy:
        ``"basic"`` for similarity search, ``"multi_query"`` for the
        LLM-powered multi-query retriever.
    llm:
        Optional LLM for the multi-query strategy. Created from settings when
        omitted.
    """
    settings = get_settings()
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k},
    )

    if strategy == "basic":
        return base_retriever

    if strategy == "multi_query":
        # ``MultiQueryRetriever`` moved between langchain versions:
        # langchain <1.x exposed it as ``langchain.retrievers``; langchain 1.x
        # relocated classic retrievers to the ``langchain_classic`` package.
        try:
            from langchain.retrievers import MultiQueryRetriever
        except ImportError:  # langchain 1.x
            from langchain_classic.retrievers import MultiQueryRetriever

        if llm is None:
            llm = create_llm()
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            prompt=MULTI_QUERY_PROMPT,
        )
        retriever.include_original = True
        return retriever

    raise ValueError(
        f"Unknown retrieval strategy: {strategy!r}. "
        "Expected one of: 'basic', 'multi_query'."
    )
