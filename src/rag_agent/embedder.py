"""Embedding factory.

Two providers are supported:

* ``local`` (default): ``HuggingFaceEmbeddings`` with
  ``BAAI/bge-small-zh-v1.5`` — good Chinese + English coverage, runs fully
  offline once the model is cached.
* ``openai``: any OpenAI-compatible embedding endpoint
  (``OPENAI_EMBED_BASE_URL`` + ``OPENAI_EMBED_MODEL``).
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import get_settings


def create_embeddings() -> Embeddings:
    """Build the embedding function based on ``EMBEDDING_PROVIDER``."""
    settings = get_settings()
    provider = settings.embedding_provider

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
