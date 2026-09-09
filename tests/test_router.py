"""Tests for the multi-knowledge-base router.

Covers:
* :func:`parse_routing_response` — JSON parsing with code fences, prose,
  invalid category fallback, empty / unparseable inputs.
* :func:`route_query` — the LangGraph node, including the LLM-exception
  fallback.
* :func:`classify_filename` — the heuristic used by the CLI ingest.
* :func:`collection_for_category` — the category → collection map.
"""

from __future__ import annotations

import json

import pytest

from rag_agent.router import (
    CATEGORY_TO_COLLECTION,
    VALID_CATEGORIES,
    classify_filename,
    collection_for_category,
    parse_routing_response,
    route_query,
)


# --------------------------------------------------------------------------- #
# parse_routing_response
# --------------------------------------------------------------------------- #
def test_parse_routing_response_valid_json():
    text = json.dumps({"category": "product", "reason": "订单咨询"})
    cat, reason = parse_routing_response(text)
    assert cat == "product"
    assert reason == "订单咨询"


def test_parse_routing_response_code_fence():
    text = '```json\n{"category": "policy", "reason": "退换货政策"}\n```'
    cat, _ = parse_routing_response(text)
    assert cat == "policy"


def test_parse_routing_response_with_surrounding_prose():
    text = 'Sure, here is the routing: {"category": "sop", "reason": "退款流程"} Hope that helps.'
    cat, _ = parse_routing_response(text)
    assert cat == "sop"


def test_parse_routing_response_invalid_category_falls_back_to_general():
    text = '{"category": "billing", "reason": "unknown"}'
    cat, _ = parse_routing_response(text)
    # Unknown category must collapse to "general" — never propagate upstream.
    assert cat == "general"


def test_parse_routing_response_empty_string():
    cat, reason = parse_routing_response("")
    assert cat == "general"
    assert reason == "empty response"


def test_parse_routing_response_whitespace_only():
    cat, _ = parse_routing_response("   \n\t  ")
    assert cat == "general"


def test_parse_routing_response_unparseable_text():
    cat, reason = parse_routing_response("I cannot classify this question.")
    assert cat == "general"
    assert "unparseable" in reason


def test_parse_routing_response_missing_category_field():
    text = '{"reason": "forgot the category"}'
    cat, _ = parse_routing_response(text)
    # Missing field defaults to "general" via .get("category", "general").
    assert cat == "general"


def test_parse_routing_response_non_dict_payload():
    # Valid JSON but not an object — must fall back, not crash.
    cat, _ = parse_routing_response('["product", "policy"]')
    assert cat == "general"


# --------------------------------------------------------------------------- #
# route_query node
# --------------------------------------------------------------------------- #
def test_route_query_node_returns_category_from_llm(make_llm):
    llm = make_llm(responder=lambda _p: '{"category": "product", "reason": "x"}')
    node = route_query(llm)
    state = {"question": "下单 3 天了还没发货怎么办"}
    result = node(state)
    assert result["category"] == "product"


def test_route_query_node_falls_back_on_unparseable_response(make_llm):
    llm = make_llm(responder=lambda _p: "我无法分类这个问题")
    node = route_query(llm)
    result = node({"question": "anything"})
    assert result["category"] == "general"


def test_route_query_node_falls_back_on_llm_exception():
    class _BoomLLM:
        def invoke(self, prompt, config=None, **kwargs):
            raise RuntimeError("LLM is down")

        def __call__(self, prompt, **kwargs):
            return self.invoke(prompt)

    node = route_query(_BoomLLM())  # type: ignore[arg-type]
    result = node({"question": "any question"})
    assert result["category"] == "general"


def test_route_query_node_uses_rewritten_query_when_present(make_llm):
    captured: list[str] = []

    def responder(prompt_text):
        captured.append(prompt_text)
        return '{"category": "policy", "reason": "rewrite"}'

    llm = make_llm(responder=responder)
    node = route_query(llm)
    # rewritten_query should take precedence over question.
    result = node({"question": "原始问题", "rewritten_query": "重写后的退货政策问题"})
    assert result["category"] == "policy"
    assert "重写后的退货政策问题" in captured[0]


# --------------------------------------------------------------------------- #
# classify_filename (used by the CLI ingest sub-command)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,expected",
    [
        ("product_faq.txt", "product"),
        ("ProductFAQ.md", "product"),
        ("faq_v2.txt", "product"),
        ("policy.md", "policy"),
        ("refund_policy.txt", "policy"),
        ("退货政策.md", "policy"),
        ("sop.md", "sop"),
        ("customer_service_sop.txt", "sop"),
        ("操作流程.md", "sop"),
        ("random_notes.txt", "general"),
        ("readme.md", "general"),
        ("", "general"),
    ],
)
def test_classify_filename(name, expected):
    assert classify_filename(name) == expected


# --------------------------------------------------------------------------- #
# collection_for_category
# --------------------------------------------------------------------------- #
def test_collection_for_category_returns_mapped_name():
    assert collection_for_category("product") == "rag_knowledge_product"
    assert collection_for_category("policy") == "rag_knowledge_policy"
    assert collection_for_category("sop") == "rag_knowledge_sop"
    assert collection_for_category("general") == "rag_knowledge_general"


def test_collection_for_category_unknown_falls_back_to_general():
    assert collection_for_category("billing") == "rag_knowledge_general"
    assert collection_for_category("") == "rag_knowledge_general"
    assert collection_for_category("unknown_category") == "rag_knowledge_general"


def test_valid_categories_and_collection_map_consistent():
    # Every valid category must have a collection mapping.
    for cat in VALID_CATEGORIES:
        assert cat in CATEGORY_TO_COLLECTION
        assert collection_for_category(cat) == CATEGORY_TO_COLLECTION[cat]
