"""Text chunking built on ``RecursiveCharacterTextSplitter``.

``RecursiveCharacterTextSplitter`` is the workhorse splitter from the LangChain
ecosystem: it tries a list of separators (``["\n\n", "\n", " ", ""]``) in order
and only falls back to the next one when a chunk still exceeds ``chunk_size``.
This keeps paragraphs and sentences intact whenever possible.
"""

from __future__ import annotations

from langchain_core.documents import Document

from .config import get_settings


def get_splitter():
    """Build a splitter from current settings (size/overlap)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split ``docs`` into chunked ``Document`` objects.

    Each chunk inherits the source document's metadata so downstream retrieval
    can cite provenance.
    """
    if not docs:
        return []
    splitter = get_splitter()
    return splitter.split_documents(docs)
