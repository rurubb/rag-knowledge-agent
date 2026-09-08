"""Chroma vector store helpers.

Wraps the persistent Chroma client so callers don't have to thread the
embedding function and persist directory through their code. Chroma writes to
``CHROMA_PERSIST_DIR`` on disk, which means a one-time ``ingest`` is enough for
all subsequent ``ask`` calls to reuse the index.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import get_settings
from .embedder import create_embeddings


def _build_chroma(
    embedding_function: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_knowledge",
):
    from langchain_community.vectorstores import Chroma

    settings = get_settings()
    if embedding_function is None:
        embedding_function = create_embeddings()
    if persist_directory is None:
        persist_directory = settings.chroma_persist_dir
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )


def get_vectorstore(
    embedding_function: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_knowledge",
):
    """Return a (persistent) Chroma vector store handle."""
    return _build_chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def add_documents(
    docs: list[Document],
    vectorstore=None,
    embedding_function: Optional[Embeddings] = None,
):
    """Embed and persist ``docs`` into the vector store.

    Returns the vector store so callers can chain ``add_documents(...).as_retriever()``.
    """
    if vectorstore is None:
        vectorstore = get_vectorstore(embedding_function=embedding_function)
    ids = vectorstore.add_documents(docs)
    # Chroma persists automatically when persist_directory is set; call
    # ``persist`` defensively for older chromadb versions that need it.
    persist_fn = getattr(vectorstore, "persist", None)
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            # Newer langchain-community Chroma wrappers may not support
            # persist(); in that case persistence is already handled.
            pass
    return ids


def clear(vectorstore=None) -> None:
    """Delete the underlying collection so the store can be re-ingested."""
    if vectorstore is None:
        vectorstore = get_vectorstore()
    vectorstore.delete_collection()
