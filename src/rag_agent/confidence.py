"""Confidence assessment for customer-service answers.

After the hallucination check passes, the customer-service agent runs one
more gate: a **confidence score** in ``[0, 1]`` that combines three cheap,
network-free signals:

1. **Relevant document count** — `grade_documents` already filtered the
   noise; the remaining count is the strongest relevance signal. 0 docs →
   0, 3+ docs → full sub-score.
2. **Citation presence** — a verifiable ``**参考来源**`` block is required
   for the answer to be auditable. Missing it halves the sub-score.
3. **Uncertainty phrasing** — "无法确定 / 建议转人工 / 未找到" etc. heavily
   penalise confidence, since the LLM is signalling it is guessing.

A hallucinated citation (source not in retrieved documents) is treated as a
hard failure: ``-0.3`` on top of the weighted average.

Below :data:`~rag_agent.config.Settings.confidence_threshold` (default 0.6),
the node replaces ``generation`` with the :data:`FALLBACK_TEMPLATE` so the
end user sees a clear "未找到明确答案，建议转人工客服" message instead of a
hallucinated answer.
"""

from __future__ import annotations

from typing import Callable, List

from langchain_core.documents import Document

from .citation import extract_citations, has_hallucinated_citations

#: Phrases that signal the LLM is uncertain or hedging. Hit-counted and
#: converted into a per-answer penalty (each hit subtracts 0.25, floored at
#: 0). Lower-case comparison so the check is case-insensitive.
UNCERTAINTY_PHRASES: tuple[str, ...] = (
    "无法确定",
    "建议联系人工",
    "建议转人工",
    "未找到",
    "没有找到",
    "找不到",
    "无法回答",
    "无法找到",
    "建议联系客服",
    "请咨询人工",
    "i don't know",
    "i do not know",
    "no clear answer",
    "not enough information",
    "insufficient information",
    "不确定",
    "不清楚",
)

#: The template shown when confidence is below threshold. Mentions "转人工"
#: so downstream consumers (CLI / orchestrator) can grep on it.
FALLBACK_TEMPLATE: str = (
    "未找到明确答案，建议转人工客服。\n\n"
    "**说明**：当前知识库中未检索到与您问题高度匹配的内容，"
    "为避免给您提供不准确的信息，已为您转接人工客服。"
    "您也可以重新描述问题或提供订单号 / 商品链接以提升检索精度。"
)


def _score_documents(documents: List[Document]) -> float:
    """0..1 — 0 docs = 0, 3+ docs = 1, linear in between."""
    n = len(documents or [])
    if n <= 0:
        return 0.0
    return min(1.0, n / 3.0)


def _score_citations(citations: List[dict], enable_citation: bool) -> float:
    """0..1 — full mark when ≥1 verifiable citation; 0.5 when no citations."""
    if not enable_citation:
        # Citation feature disabled — neutral score so we don't unfairly
        # tank confidence on answers that legitimately don't need citations.
        return 0.5
    if not citations:
        return 0.0
    verified = [c for c in citations if c.get("verified") is True]
    if verified:
        return 1.0
    # Citations present but none verified against documents — partial credit
    # because the model at least attempted to cite.
    return 0.4


def _score_uncertainty(generation: str) -> float:
    """1.0 = no uncertainty phrase, 0.0 = 4+ uncertainty hits."""
    if not generation:
        return 0.0
    gen_lower = generation.lower()
    hits = sum(1 for p in UNCERTAINTY_PHRASES if p.lower() in gen_lower)
    return max(0.0, 1.0 - 0.25 * hits)


def assess_confidence(
    confidence_threshold: float = 0.6,
    enable_citation: bool = True,
) -> Callable[[dict], dict]:
    """Build the ``assess_confidence`` LangGraph node.

    Reads ``documents``, ``generation``, ``citations`` from state. Writes:

    * ``confidence`` — float in ``[0, 1]``.
    * ``citations`` — parsed citations (re-parsed here if not already set,
      so the node works whether or not the generate node pre-parsed them).
    * ``low_confidence`` — bool flag.
    * ``generation`` — replaced with :data:`FALLBACK_TEMPLATE` when below
      threshold so downstream consumers see the "建议转人工" message.
    """

    def _node(state: dict) -> dict:
        documents: List[Document] = state.get("documents", []) or []
        generation: str = state.get("generation", "") or ""

        # Reuse citations parsed by an earlier node if present; otherwise
        # parse them now from the generation text.
        citations = state.get("citations") or []
        if not citations and enable_citation:
            citations = extract_citations(generation, documents)

        doc_score = _score_documents(documents)
        cite_score = _score_citations(citations, enable_citation)
        uncertainty_score = _score_uncertainty(generation)

        confidence = (
            0.4 * doc_score
            + 0.3 * cite_score
            + 0.3 * uncertainty_score
        )

        # Hard penalty for fabricated references: a citation whose source is
        # not in the retrieved documents is a hallucination signal.
        if has_hallucinated_citations(citations):
            confidence -= 0.3

        confidence = max(0.0, min(1.0, confidence))

        result = {
            "confidence": confidence,
            "citations": citations,
            "low_confidence": confidence < confidence_threshold,
            # Always echo the (possibly replaced) generation back so direct
            # callers (tests) get a stable contract: the node either keeps
            # the original generation (above threshold) or swaps it for the
            # fallback template (below threshold). LangGraph merges this
            # into state, so the effect on the graph is identical either way.
            "generation": FALLBACK_TEMPLATE
            if confidence < confidence_threshold
            else generation,
        }
        return result

    return _node
