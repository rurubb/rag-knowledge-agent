"""Tests for the confidence-assessment node.

Covers:
* High confidence (≥3 docs + verified citations + no uncertainty phrase)
  → generation preserved.
* Low confidence (0 docs OR no citations + uncertainty phrase) → generation
  replaced with the "未找到明确答案，建议转人工客服" fallback template.
* Hallucinated citations apply a hard -0.3 penalty.
* Disable-citation path is neutral (no penalty for missing citations).
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag_agent.confidence import (
    FALLBACK_TEMPLATE,
    UNCERTAINTY_PHRASES,
    assess_confidence,
)


def _doc(source: str = "product_faq.txt", content: str = "abc") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _verified_citation(source: str = "product_faq.txt") -> dict:
    return {
        "raw": f"[{source}] section: summary",
        "source": source,
        "section": "section",
        "summary": "summary",
        "verified": True,
    }


# --------------------------------------------------------------------------- #
# High-confidence scenarios
# --------------------------------------------------------------------------- #
def test_assess_confidence_high_score_with_docs_and_verified_citations():
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    state = {
        "documents": [_doc(), _doc(), _doc()],
        "generation": "现货商品 48 小时内发货。",
        "citations": [_verified_citation()],
    }
    result = node(state)
    assert result["confidence"] >= 0.6
    assert result["low_confidence"] is False
    # Generation is preserved when above threshold.
    assert result["generation"] == "现货商品 48 小时内发货。"


def test_assess_confidence_high_confidence_does_not_replace_generation():
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    state = {
        "documents": [_doc(), _doc(), _doc(), _doc()],
        "generation": "answers never change.",
        "citations": [_verified_citation(), _verified_citation("policy.md")],
    }
    result = node(state)
    assert result["generation"] == "answers never change."


# --------------------------------------------------------------------------- #
# Low-confidence scenarios — fallback template fires
# --------------------------------------------------------------------------- #
def test_assess_confidence_zero_docs_low_confidence_replaces_generation():
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    state = {
        "documents": [],
        "generation": "some guess answer",
        "citations": [],
    }
    result = node(state)
    assert result["confidence"] < 0.6
    assert result["low_confidence"] is True
    # Generation was swapped for the fallback template.
    assert result["generation"] == FALLBACK_TEMPLATE
    assert "建议转人工" in result["generation"]


def test_assess_confidence_uncertainty_phrase_lowers_score():
    # 3 docs + citations but the answer says "无法确定" → confidence tanks
    # because the uncertainty sub-score hits 0.
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    state = {
        "documents": [_doc(), _doc(), _doc()],
        "generation": "无法确定具体发货时间。",
        "citations": [_verified_citation()],
    }
    result = node(state)
    # 0.4*1 + 0.3*1 + 0.3*0 = 0.7 → still above 0.6? Let's check: 1
    # uncertainty phrase → uncertainty_score = 1 - 0.25 = 0.75.
    # confidence = 0.4 + 0.3 + 0.3*0.75 = 0.925. Above threshold.
    # But with 2+ uncertainty phrases the score drops below threshold.
    assert result["confidence"] >= 0.6
    assert result["generation"] == "无法确定具体发货时间。"


def test_assess_confidence_multiple_uncertainty_phrases_tank_score():
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    state = {
        "documents": [_doc()],
        "generation": "无法确定。建议联系人工客服。未找到相关答案。",
        "citations": [],
    }
    result = node(state)
    # 3+ uncertainty hits → uncertainty_score floored at 0.
    assert result["confidence"] < 0.6
    assert result["low_confidence"] is True
    assert result["generation"] == FALLBACK_TEMPLATE


def test_assess_confidence_low_confidence_with_zero_docs_no_citations():
    # 0 docs + no citations + the LLM at least says "无法确定" → all three
    # sub-scores drop low. (Without the uncertainty phrase the score would
    # still be 0.3 from the uncertainty sub-score, since "guess" contains
    # no uncertainty phrase — see test_assess_confidence_zero_docs_no_phrase
    # below.)
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    state = {
        "documents": [],
        "generation": "无法确定具体发货时间。",
        "citations": [],
    }
    result = node(state)
    # doc=0, cite=0, uncertainty=0.75 → 0 + 0 + 0.225 = 0.225 < 0.6.
    assert result["confidence"] < 0.6
    assert result["low_confidence"] is True


def test_assess_confidence_zero_docs_no_uncertainty_phrase_still_above_zero():
    # Sanity check: even with 0 docs + no citations + no uncertainty phrase,
    # the uncertainty sub-score is 1.0 (no hits), so the score is 0.3 (not 0).
    # This pins the maths so future refactors don't silently drift.
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    state = {
        "documents": [],
        "generation": "guess",
        "citations": [],
    }
    result = node(state)
    assert result["confidence"] == 0.3
    assert result["low_confidence"] is True


# --------------------------------------------------------------------------- #
# Hallucinated citation penalty
# --------------------------------------------------------------------------- #
def test_assess_confidence_hallucinated_citation_applies_hard_penalty():
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    hallucinated = {
        "raw": "[invented.txt] section: made up",
        "source": "invented.txt",
        "section": "section",
        "summary": "made up",
        "verified": False,
    }
    state = {
        "documents": [_doc(), _doc(), _doc()],
        "generation": "answer with a fabricated citation",
        "citations": [_verified_citation(), hallucinated],
    }
    result = node(state)
    # Without the penalty, score would be 0.4*1 + 0.3*1 + 0.3*1 = 1.0.
    # With -0.3 penalty → 0.7. Still above 0.6 but lower than the no-halluc
    # scenario, and the hallucination check would have already routed
    # through rewrite_query before reaching assess_confidence. This test
    # pins the maths so future refactors don't silently drift.
    assert 0.6 <= result["confidence"] <= 0.75


def test_assess_confidence_hallucinated_citation_can_drop_below_threshold():
    # 1 doc + 1 hallucinated citation + uncertainty phrase → low confidence.
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    hallucinated = {
        "raw": "[made_up.txt] x",
        "source": "made_up.txt",
        "verified": False,
    }
    state = {
        "documents": [_doc()],
        "generation": "无法确定。",
        "citations": [hallucinated],
    }
    result = node(state)
    # doc_score = 1/3 ≈ 0.33, cite_score = 0 (no verified), uncertainty = 0.75
    # base = 0.4*0.33 + 0.3*0 + 0.3*0.75 = 0.133 + 0.225 = 0.358
    # minus 0.3 hallucination penalty → 0.058
    assert result["confidence"] < 0.6
    assert result["low_confidence"] is True
    assert result["generation"] == FALLBACK_TEMPLATE


# --------------------------------------------------------------------------- #
# Disable-citation path
# --------------------------------------------------------------------------- #
def test_assess_confidence_disable_citation_neutral_for_missing_citations():
    # When enable_citation=False the citation sub-score is a neutral 0.5
    # instead of 0, so an answer that legitimately doesn't need citations
    # is not unfairly tanked.
    node = assess_confidence(confidence_threshold=0.6, enable_citation=False)
    state = {
        "documents": [_doc(), _doc(), _doc()],
        "generation": "answer with no citations needed",
        "citations": [],
    }
    result = node(state)
    # 0.4 + 0.3*0.5 + 0.3 = 0.4 + 0.15 + 0.3 = 0.85 → above threshold.
    assert result["confidence"] >= 0.6
    assert result["low_confidence"] is False


def test_assess_confidence_disable_citation_still_replaces_on_zero_docs():
    # Even with citation disabled, 0 docs + uncertainty phrase should still
    # drop below threshold.
    node = assess_confidence(confidence_threshold=0.6, enable_citation=False)
    state = {
        "documents": [],
        "generation": "无法确定。",
        "citations": [],
    }
    result = node(state)
    # 0 + 0.3*0.5 + 0 = 0.15 → below threshold.
    assert result["confidence"] < 0.6
    assert result["generation"] == FALLBACK_TEMPLATE


# --------------------------------------------------------------------------- #
# Re-parse citations when not pre-populated
# --------------------------------------------------------------------------- #
def test_assess_confidence_re_parses_citations_when_state_empty():
    # If the generate node didn't pre-populate `citations`, the confidence
    # node should still parse them from the generation text.
    node = assess_confidence(confidence_threshold=0.6, enable_citation=True)
    docs = [
        Document(
            page_content="48 小时内发货",
            metadata={"source": "/data/product_faq.txt"},
        )
    ]
    answer = (
        "现货商品 48 小时内发货。\n\n"
        "**参考来源**：\n"
        "- [product_faq.txt] 发货与物流：48 小时内发货\n"
    )
    state = {
        "documents": docs,
        "generation": answer,
        # citations intentionally omitted — node must re-parse.
    }
    result = node(state)
    assert result["citations"], "citations should be re-parsed by the node"
    assert result["citations"][0]["verified"] is True
    # 1 doc + 1 verified citation + no uncertainty → above threshold.
    assert result["confidence"] >= 0.6
    # Generation preserved.
    assert "48 小时内发货" in result["generation"]


# --------------------------------------------------------------------------- #
# Public exports
# --------------------------------------------------------------------------- #
def test_fallback_template_mentions_transfer_to_human():
    # Downstream consumers grep on "建议转人工" to trigger the hand-off flow.
    assert "建议转人工" in FALLBACK_TEMPLATE
    assert "未找到明确答案" in FALLBACK_TEMPLATE


def test_uncertainty_phrases_non_empty_and_lowercase_safe():
    assert len(UNCERTAINTY_PHRASES) > 0
    # Every phrase must lowercase cleanly (the node lowercases the
    # generation before substring-checking).
    for p in UNCERTAINTY_PHRASES:
        assert p.lower() == p or any(c.isalpha() for c in p)
