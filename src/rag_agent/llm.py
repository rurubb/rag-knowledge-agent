"""LLM factory.

Builds a chat model that speaks the OpenAI-compatible Chat Completions API
(DeepSeek, Moonshot, OpenAI, vLLM, Ollama's OpenAI endpoint, etc.). Keeping
construction in one place makes the model trivially mockable in tests via
``monkeypatch``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .config import get_settings


def create_llm(**overrides: Any) -> BaseChatModel:
    """Create an OpenAI-compatible chat model from settings.

    Extra keyword arguments are forwarded to ``ChatOpenAI`` so callers can
    override temperature, model name, etc.
    """
    settings = get_settings()
    # Imported lazily so that simply importing the package never requires the
    # openai/network stack to be available.
    from langchain_openai import ChatOpenAI

    params: dict[str, Any] = {
        "model": settings.openai_model,
        "temperature": overrides.pop("temperature", 0),
    }
    if settings.openai_api_key:
        params["api_key"] = settings.openai_api_key
    if settings.openai_base_url:
        params["base_url"] = settings.openai_base_url
    params.update(overrides)
    return ChatOpenAI(**params)
