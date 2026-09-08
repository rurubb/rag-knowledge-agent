"""Tests for the chunker."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_agent.chunker import chunk_documents, get_splitter


def test_get_splitter_uses_settings(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    splitter = get_splitter()
    assert splitter._chunk_size == 500
    assert splitter._chunk_overlap == 50


def test_chunk_empty_input_returns_empty():
    assert chunk_documents([]) == []


def test_chunk_single_short_doc_returns_single_chunk():
    doc = Document(page_content="这是一段很短的文本，不足以切分。", metadata={"source": "a.txt"})
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == "这是一段很短的文本，不足以切分。"
    # metadata is preserved
    assert chunks[0].metadata["source"] == "a.txt"


def test_chunk_count_increases_with_long_text(monkeypatch):
    # Force a small chunk size so we can deterministically produce >1 chunk.
    monkeypatch.setenv("CHUNK_SIZE", "20")
    monkeypatch.setenv("CHUNK_OVERLAP", "5")
    long_text = "abcdefghij" * 30  # 300 chars, well over chunk_size=20
    doc = Document(page_content=long_text, metadata={"source": "b.txt"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    # Every chunk must be at most chunk_size characters (separator-aware).
    for c in chunks:
        assert len(c.page_content) <= 20 + 10  # allow small slack for separators
    # Overlap means consecutive chunks share some characters.
    if len(chunks) >= 2:
        overlap = chunks[0].page_content[-5:] if len(chunks[0].page_content) >= 5 else chunks[0].page_content
        # The overlap region should appear at the start of the next chunk.
        assert overlap[:3] in chunks[1].page_content


def test_chunk_preserves_metadata_across_chunks():
    base = "x" * 600
    doc = Document(page_content=base, metadata={"source": "c.md", "page": 2})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source"] == "c.md"
        assert c.metadata["page"] == 2
