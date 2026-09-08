"""Application configuration.

Settings are loaded from environment variables (optionally backed by a
``.env`` file via python-dotenv) with sensible defaults so the project can be
imported and tested even when no configuration is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value != "" else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Runtime configuration for the RAG agent.

    All fields default to values read from environment variables, but the
    dataclass can also be constructed directly (handy for tests and monkey
    patching).
    """

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "")
    )
    openai_model: str = field(
        default_factory=lambda: _env("OPENAI_MODEL", "gpt-3.5-turbo")
    )
    embedding_provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "local").lower()
    )
    openai_embed_base_url: str = field(
        default_factory=lambda: _env("OPENAI_EMBED_BASE_URL", "")
    )
    openai_embed_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )
    local_embed_model: str = field(
        default_factory=lambda: _env("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: _env("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 2))


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    return Settings()


def load_dotenv_if_present() -> None:
    """Load a ``.env`` file from the current directory if it exists.

    Kept optional and defensive so tests can run without python-dotenv or a
    ``.env`` file.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return
    load_dotenv(override=False)
