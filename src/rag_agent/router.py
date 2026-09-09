"""Multi-knowledge-base router for the customer-service agent.

The router solves a concrete customer-service problem: the user's question
("3 天没发货怎么办") usually maps to one specific corpus (product FAQ), and
searching across a giant mixed collection dilutes recall and adds noise from
unrelated corpora (policy / sop). By classifying the question first, we can
hit a focused Chroma collection and get tighter, more relevant hits.

Two helpers are exposed:

* :func:`route_query` — the LangGraph node closed over an LLM. It formats
  :data:`~rag_agent.prompts.ROUTING_PROMPT`, asks the LLM to emit a JSON
  object ``{"category": ..., "reason": ...}``, and falls back to
  ``"general"`` on any error.
* :func:`classify_filename` — a heuristic used by the CLI ``ingest``
  sub-command to bucket files under ``data/`` into the matching collection
  based on their filename (so ``product_faq.txt`` → ``product``, etc.).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from .prompts import ROUTING_PROMPT

#: Valid categories. Anything else from the LLM falls back to ``"general"``.
VALID_CATEGORIES = ("product", "policy", "sop", "general")

#: Map a category to the Chroma collection name. Kept in one place so the
#: CLI ingest and the ask-time retriever agree on the convention.
CATEGORY_TO_COLLECTION: dict[str, str] = {
    "product": "rag_knowledge_product",
    "policy": "rag_knowledge_policy",
    "sop": "rag_knowledge_sop",
    "general": "rag_knowledge_general",
}


def collection_for_category(category: str) -> str:
    """Return the Chroma collection name for ``category``.

    Unknown categories fall back to the ``general`` collection so a typo in
    the LLM response can never crash the pipeline.
    """
    return CATEGORY_TO_COLLECTION.get(category, CATEGORY_TO_COLLECTION["general"])


def parse_routing_response(text: str) -> Tuple[str, str]:
    """Parse the LLM routing response into ``(category, reason)``.

    Tolerates JSON wrapped in ```json ... ``` code fences, surrounded by prose,
    or with extra whitespace. Falls back to ``("general", "<raw>")`` when the
    response is empty or unparseable — this is the spec'd "失败兜底".
    """
    if not text or not text.strip():
        return "general", "empty response"

    cleaned = text.strip()

    # Strip ```json ... ``` code fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)

    # Find the first {...} block to tolerate surrounding prose like
    # "Sure, here is the routing: {...}".
    brace = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if brace:
        cleaned = brace.group(0)

    try:
        data = json.loads(cleaned)
    except Exception:
        return "general", f"unparseable: {text[:80]}"

    if not isinstance(data, dict):
        return "general", f"non-dict: {text[:80]}"

    category = str(data.get("category", "general")).lower().strip()
    if category not in VALID_CATEGORIES:
        # LLM produced an out-of-vocabulary label — fall back rather than crash.
        category = "general"
    reason = str(data.get("reason", "")).strip()
    return category, reason


def route_query(llm: BaseChatModel) -> Callable[[dict], dict]:
    """Build the ``route_query`` LangGraph node closed over ``llm``.

    The node reads ``question`` (or ``rewritten_query`` if a previous round
    rewrote it) and writes ``category`` into state.
    """

    def _node(state: dict) -> dict:
        question = state.get("rewritten_query") or state.get("question", "")
        prompt = ROUTING_PROMPT.format(question=question)
        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
        except Exception:
            # LLM blew up — fall back to "general" so the pipeline still runs.
            return {"category": "general"}
        category, _reason = parse_routing_response(text)
        return {"category": category}

    return _node


def classify_filename(name: str) -> str:
    """Heuristically bucket a filename under ``data/`` into a category.

    Used by the CLI ``ingest`` sub-command. The mapping is intentionally
    simple and rule-based (no LLM call) so it is fast and deterministic.

    * ``faq`` or ``product`` in the name → ``product``
    * ``policy`` / ``refund`` / ``退`` in the name → ``policy``
    * ``sop`` / ``process`` / ``操作`` in the name → ``sop``
    * otherwise → ``general``
    """
    n = (name or "").lower()
    if "faq" in n or "product" in n:
        return "product"
    if "policy" in n or "refund" in n or "退" in name:
        return "policy"
    if "sop" in n or "process" in n or "操作" in name:
        return "sop"
    return "general"
