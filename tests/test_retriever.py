"""Tests for the retriever (in-memory Chroma + fake embeddings)."""

from __future__ import annotations

import pytest
from langchain_core.retrievers import BaseRetriever

from rag_agent.retriever import create_retriever
from rag_agent.vectorstore import add_documents
from langchain_core.documents import Document


def _add_doc(vs, content, source):
    add_documents([Document(page_content=content, metadata={"source": source})], vectorstore=vs)


def test_create_retriever_returns_base_retriever(in_memory_vectorstore):
    _add_doc(in_memory_vectorstore, "langgraph conditional edges", "a.txt")
    retriever = create_retriever(in_memory_vectorstore, strategy="basic")
    assert isinstance(retriever, BaseRetriever)


def test_basic_retriever_returns_relevant_doc(in_memory_vectorstore, monkeypatch):
    monkeypatch.setenv("TOP_K", "4")
    _add_doc(in_memory_vectorstore, "langgraph conditional edges are useful", "a.txt")
    _add_doc(
        in_memory_vectorstore,
        "completely unrelated topic about cooking recipes",
        "b.txt",
    )
    retriever = create_retriever(in_memory_vectorstore, strategy="basic")
    docs = retriever.invoke("what are langgraph conditional edges")
    assert len(docs) >= 1
    # The relevant doc must be present in results.
    contents = [d.page_content for d in docs]
    assert any("langgraph conditional edges" in c for c in contents)


def test_basic_retriever_respects_top_k(in_memory_vectorstore, monkeypatch):
    monkeypatch.setenv("TOP_K", "1")
    for i in range(5):
        _add_doc(
            in_memory_vectorstore,
            f"document number {i} about langgraph features",
            f"f{i}.txt",
        )
    retriever = create_retriever(in_memory_vectorstore, strategy="basic")
    docs = retriever.invoke("langgraph features")
    assert len(docs) == 1


def test_unknown_strategy_raises(in_memory_vectorstore):
    with pytest.raises(ValueError):
        create_retriever(in_memory_vectorstore, strategy="unknown")


def test_multi_query_retriever_returns_docs(in_memory_vectorstore, fake_llm):
    # Build a FakeLLM whose responder emits 3 variant queries that all share
    # tokens with the stored document, guaranteeing a hit.
    _add_doc(
        in_memory_vectorstore,
        "langgraph conditional edges routing graph nodes",
        "a.txt",
    )

    def responder(prompt_text):
        # MultiQueryRetriever parses lines; return variants sharing tokens.
        return (
            "what is langgraph conditional edges\n"
            "how does langgraph route between nodes\n"
            "explain conditional edges in langgraph"
        )

    multi_llm = fake_llm.__class__(responder=responder)
    retriever = create_retriever(in_memory_vectorstore, strategy="multi_query", llm=multi_llm)
    assert isinstance(retriever, BaseRetriever)
    docs = retriever.invoke("langgraph conditional edges")
    assert len(docs) >= 1
    assert any("langgraph conditional edges" in d.page_content for d in docs)
