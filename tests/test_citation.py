"""Tests for citation extraction and verification.

Covers:
* :func:`extract_citations` — parsing the ``**参考来源**`` block in various
  formats (bracketed source, with/without section, no colon, empty).
* Citation verification against retrieved documents — fuzzy basename match
  so ``[product_faq.txt]`` matches ``source=/path/to/product_faq.txt``.
* :func:`has_citations` — boolean helper used by the confidence node.
* :func:`has_hallucinated_citations` — flags answers whose cited sources
  are not in the retrieved documents (drives the hallucination retry).
* :func:`build_citation_context` — renders documents with ``[文档名] 章节``
  headers that round-trip cleanly through :func:`extract_citations`.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag_agent.citation import (
    build_citation_context,
    extract_citations,
    has_citations,
    has_hallucinated_citations,
)


# --------------------------------------------------------------------------- #
# extract_citations — parsing
# --------------------------------------------------------------------------- #
ANSWER_WITH_CITATIONS = """正常情况下现货商品会在付款后 48 小时内发货，超过承诺时间未发货可点"催发货"按钮催办商家。

**参考来源**：
- [product_faq.txt] 发货与物流：付款后 48 小时内发货
- [policy.md] 第一节：7 天无理由退货政策
- [sop.md] 第四章：退款操作流程
"""


def test_extract_citations_parses_bracketed_sources():
    citations = extract_citations(ANSWER_WITH_CITATIONS)
    assert len(citations) == 3
    assert citations[0]["source"] == "product_faq.txt"
    assert citations[0]["section"] == "发货与物流"
    assert "48 小时内发货" in citations[0]["summary"]
    assert citations[1]["source"] == "policy.md"
    assert citations[2]["source"] == "sop.md"


def test_extract_citations_no_block_returns_empty():
    text = "这是一段没有引用来源的答案。"
    assert extract_citations(text) == []


def test_extract_citations_empty_answer_returns_empty():
    assert extract_citations("") == []
    assert extract_citations(None) == []  # type: ignore[arg-type]


def test_extract_citations_with_unstructured_bullets():
    # Bullet lines without a [source] bracket should still parse, but the
    # source field stays None (the confidence node penalises this).
    text = (
        "答案正文。\n\n"
        "**参考来源**：\n"
        "- 一些手写的说明文字\n"
    )
    citations = extract_citations(text)
    assert len(citations) == 1
    assert citations[0]["source"] is None
    assert citations[0]["summary"] == "一些手写的说明文字"


def test_extract_citations_tolerates_half_width_colon():
    text = (
        "答案。\n\n"
        "**参考来源**:\n"  # half-width colon
        "- [product_faq.txt] section: summary text\n"
    )
    citations = extract_citations(text)
    assert len(citations) == 1
    assert citations[0]["source"] == "product_faq.txt"


def test_extract_citations_tolerates_no_bold_marker():
    # The "**" markers around 参考来源 should be optional.
    text = (
        "答案。\n\n"
        "参考来源：\n"
        "- [product_faq.txt] section: summary text\n"
    )
    citations = extract_citations(text)
    assert len(citations) == 1
    assert citations[0]["source"] == "product_faq.txt"


# --------------------------------------------------------------------------- #
# extract_citations — verification against documents
# --------------------------------------------------------------------------- #
def test_extract_citations_verifies_source_against_documents_by_basename():
    docs = [
        Document(
            page_content="...",
            metadata={"source": "/some/path/product_faq.txt"},
        ),
        Document(
            page_content="...",
            metadata={"source": "/some/path/policy.md"},
        ),
    ]
    citations = extract_citations(ANSWER_WITH_CITATIONS, documents=docs)
    assert len(citations) == 3
    # product_faq.txt and policy.md match → verified True
    assert citations[0]["verified"] is True
    assert citations[1]["verified"] is True
    # sop.md is NOT in retrieved documents → verified False (hallucinated)
    assert citations[2]["verified"] is False


def test_extract_citations_substring_match_for_path_with_dirs():
    docs = [
        Document(
            page_content="...",
            metadata={"source": "data/sub/product_faq.txt"},
        ),
    ]
    text = (
        "答案。\n\n"
        "**参考来源**：\n"
        "- [product_faq.txt] section: summary\n"
    )
    citations = extract_citations(text, documents=docs)
    assert citations[0]["verified"] is True


def test_extract_citations_no_documents_returns_unverified_citations():
    # documents=None → no verification attempted, verified stays None.
    citations = extract_citations(ANSWER_WITH_CITATIONS, documents=None)
    assert all(c.get("verified") is None for c in citations)


def test_extract_citations_empty_documents_marks_all_unverified():
    # documents=[] → verification runs but nothing matches.
    citations = extract_citations(ANSWER_WITH_CITATIONS, documents=[])
    assert all(c.get("verified") is False for c in citations)


# --------------------------------------------------------------------------- #
# has_citations
# --------------------------------------------------------------------------- #
def test_has_citations_true_when_block_present():
    assert has_citations(ANSWER_WITH_CITATIONS) is True


def test_has_citations_false_when_block_absent():
    assert has_citations("一段没有引用的答案。") is False


def test_has_citations_false_for_empty_string():
    assert has_citations("") is False


# --------------------------------------------------------------------------- #
# has_hallucinated_citations
# --------------------------------------------------------------------------- #
def test_has_hallucinated_citations_true_when_source_not_in_documents():
    docs = [
        Document(page_content="...", metadata={"source": "product_faq.txt"}),
    ]
    # Citation list mixes one verified + one unverified source.
    citations = [
        {"source": "product_faq.txt", "verified": True, "raw": "..."},
        {"source": "made_up_doc.txt", "verified": False, "raw": "..."},
    ]
    assert has_hallucinated_citations(citations) is True


def test_has_hallucinated_citations_false_when_all_verified():
    citations = [
        {"source": "product_faq.txt", "verified": True, "raw": "..."},
        {"source": "policy.md", "verified": True, "raw": "..."},
    ]
    assert has_hallucinated_citations(citations) is False


def test_has_hallucinated_citations_false_when_no_citations():
    assert has_hallucinated_citations([]) is False


def test_has_hallucinated_citations_false_when_source_is_none():
    # Citations without an explicit source field don't count as hallucinations
    # (they're penalised separately via the no-citation sub-score).
    citations = [
        {"source": None, "verified": False, "raw": "no bracket text"},
    ]
    assert has_hallucinated_citations(citations) is False


def test_has_hallucinated_citations_end_to_end_with_extract_citations():
    docs = [Document(page_content="...", metadata={"source": "real.txt"})]
    answer = (
        "答案。\n\n"
        "**参考来源**：\n"
        "- [real.txt] section: real summary\n"
        "- [invented.txt] section: made-up summary\n"
    )
    citations = extract_citations(answer, documents=docs)
    assert has_hallucinated_citations(citations) is True


# --------------------------------------------------------------------------- #
# build_citation_context
# --------------------------------------------------------------------------- #
def test_build_citation_context_empty_documents_returns_placeholder():
    out = build_citation_context([])
    assert "no relevant context" in out


def test_build_citation_context_includes_basename_and_section():
    docs = [
        Document(
            page_content="Q：怎么退款？\nA：联系客服。",
            metadata={"source": "/data/product_faq.txt", "section": "退款"},
        )
    ]
    out = build_citation_context(docs)
    assert "[product_faq.txt] 退款" in out
    assert "Q：怎么退款？" in out


def test_build_citation_context_handles_missing_section():
    docs = [
        Document(
            page_content="some content",
            metadata={"source": "/data/note.txt"},
        )
    ]
    out = build_citation_context(docs)
    assert "[note.txt]" in out
    assert "some content" in out


def test_build_citation_context_round_trips_through_extract_citations():
    # A doc rendered with build_citation_context should produce a header
    # that the LLM can echo back as a citation verifiable against the
    # same document set.
    docs = [
        Document(
            page_content="48 小时内发货",
            metadata={"source": "/data/product_faq.txt", "section": "发货与物流"},
        )
    ]
    context = build_citation_context(docs)
    # Simulate the LLM echoing back the citation header.
    answer = (
        "现货商品 48 小时内发货。\n\n"
        "**参考来源**：\n"
        "- [product_faq.txt] 发货与物流：48 小时内发货\n"
    )
    citations = extract_citations(answer, documents=docs)
    assert len(citations) == 1
    assert citations[0]["verified"] is True
    # And the [文档名] in the answer matches the [文档名] in the context.
    assert "[product_faq.txt] 发货与物流" in context
