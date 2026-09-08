"""Document loading utilities.

Supports ``.txt``/``.md`` (TextLoader) and ``.pdf`` (PyPDFLoader). A
directory walk recursively ingests every supported file. Each returned
``Document`` carries ``source`` (and ``page`` for PDFs) metadata.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_file(file_path: str) -> list[Document]:
    """Load a single file and return its ``Document`` chunks.

    Raises ``ValueError`` for unsupported extensions and ``FileNotFoundError``
    when the path does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        # TextLoader is the most robust option for plain text and markdown.
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
        return docs

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
        docs = loader.load()
        # PyPDFLoader already populates source + page metadata.
        return docs

    raise ValueError(
        f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def load_directory(dir_path: str) -> list[Document]:
    """Recursively load every supported document under ``dir_path``."""
    base = Path(dir_path)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if base.is_file():
        return load_file(str(base))

    docs: list[Document] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(load_file(str(path)))
    return docs
