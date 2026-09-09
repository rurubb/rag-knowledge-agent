"""Tests for the self-reflective LangGraph agent.

All tests use the in-memory Chroma store + fake embeddings from conftest and a
scriptable FakeLLM, so they never touch the network or a real model.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langgraph.graph import END

from rag_agent.agent import (
    AgentState,
    build_customer_service_nodes,
    build_nodes,
    create_agent,
    create_customer_service_agent,
    create_customer_service_agent_from_vectorstore,
    parse_yes_no,
)
from rag_agent.confidence import FALLBACK_TEMPLATE
from rag_agent.retriever import create_retriever
from rag_agent.vectorstore import add_documents


# --------------------------------------------------------------------------- #
# Pure helper
# --------------------------------------------------------------------------- #
def test_parse_yes_no():
    assert parse_yes_no("yes") is True
    assert parse_yes_no("Yes.") is True
    assert parse_yes_no("  YES\n") is True
    assert parse_yes_no("no") is False
    assert parse_yes_no("nope") is False
    assert parse_yes_no("") is False
    assert parse_yes_no(None) is False


# --------------------------------------------------------------------------- #
# grade_documents
# --------------------------------------------------------------------------- #
def test_grade_documents_keeps_relevant(make_llm):
    # Mark a doc as irrelevant when its content contains the word "cooking".
    llm = make_llm(responder=lambda p: "no" if "cooking" in p else "yes")
    nodes = build_nodes(retriever=None, llm=llm, max_retries=2)
    docs = [
        Document(page_content="langgraph conditional edges", metadata={"source": "a"}),
        Document(page_content="a cooking recipe for pasta", metadata={"source": "b"}),
    ]
    state = {"question": "what are conditional edges", "documents": docs}
    result = nodes["grade_documents"](state)
    assert len(result["documents"]) == 1
    assert "langgraph" in result["documents"][0].page_content


def test_grade_documents_filters_all_when_all_irrelevant(make_llm):
    llm = make_llm(responder=lambda _p: "no")
    nodes = build_nodes(retriever=None, llm=llm, max_retries=2)
    docs = [
        Document(page_content="cooking pasta", metadata={"source": "a"}),
        Document(page_content="gardening tips", metadata={"source": "b"}),
    ]
    state = {"question": "what is langgraph", "documents": docs}
    result = nodes["grade_documents"](state)
    assert result["documents"] == []


# --------------------------------------------------------------------------- #
# decide_after_grading
# --------------------------------------------------------------------------- #
def _grade_node(make_llm):
    return build_nodes(retriever=None, llm=make_llm(), max_retries=2)


def test_decide_after_grading_routes_to_rewrite_when_no_docs(make_llm):
    nodes = _grade_node(make_llm)
    state = {"documents": [], "retry_count": 0}
    assert nodes["decide_after_grading"](state) == "rewrite_query"


def test_decide_after_grading_routes_to_generate_when_docs_present(make_llm):
    nodes = _grade_node(make_llm)
    state = {"documents": [Document(page_content="x")], "retry_count": 0}
    assert nodes["decide_after_grading"](state) == "generate"


def test_decide_after_grading_falls_back_to_generate_when_retries_exhausted(make_llm):
    nodes = _grade_node(make_llm)
    # No docs but retry_count == max_retries -> must NOT loop forever.
    state = {"documents": [], "retry_count": 2}
    assert nodes["decide_after_grading"](state) == "generate"


# --------------------------------------------------------------------------- #
# rewrite_query
# --------------------------------------------------------------------------- #
def test_rewrite_query_sets_query_and_increments_retry(make_llm):
    llm = make_llm(responder=lambda _p: "improved langgraph query")
    nodes = build_nodes(retriever=None, llm=llm, max_retries=2)
    state = {"question": "langgraph", "retry_count": 0, "rewritten_query": ""}
    result = nodes["rewrite_query"](state)
    assert result["rewritten_query"] == "improved langgraph query"
    assert result["retry_count"] == 1


def test_rewrite_query_strips_surrounding_quotes(make_llm):
    llm = make_llm(responder=lambda _p: '"a rewritten question"')
    nodes = build_nodes(retriever=None, llm=llm, max_retries=2)
    result = nodes["rewrite_query"]({"question": "q", "retry_count": 3})
    assert result["rewritten_query"] == "a rewritten question"
    assert result["retry_count"] == 4


# --------------------------------------------------------------------------- #
# decide_after_hallucination
# --------------------------------------------------------------------------- #
def test_decide_after_hallucination_ends_when_grounded(make_llm):
    nodes = _grade_node(make_llm)
    state = {"grounded": True, "retry_count": 0}
    assert nodes["decide_after_hallucination"](state) == END


def test_decide_after_hallucination_rewrites_when_not_grounded(make_llm):
    nodes = _grade_node(make_llm)
    state = {"grounded": False, "retry_count": 0}
    assert nodes["decide_after_hallucination"](state) == "rewrite_query"


def test_decide_after_hallucination_ends_when_retries_exhausted(make_llm):
    nodes = _grade_node(make_llm)
    # Hallucination detected but no retries left -> stop to avoid infinite loop.
    state = {"grounded": False, "retry_count": 2}
    assert nodes["decide_after_hallucination"](state) == END


# --------------------------------------------------------------------------- #
# End-to-end graph runs
# --------------------------------------------------------------------------- #
def _store_doc(vs, content="langgraph conditional edges routing", source="a.txt"):
    add_documents(
        [Document(page_content=content, metadata={"source": source})],
        vectorstore=vs,
    )


def test_full_agent_happy_path(in_memory_vectorstore, make_llm, monkeypatch):
    monkeypatch.setenv("TOP_K", "4")
    _store_doc(in_memory_vectorstore)
    retriever = create_retriever(in_memory_vectorstore, strategy="basic")

    def responder(prompt):
        if "Is this document relevant" in prompt:
            return "yes"
        if "Output ONLY the rewritten question" in prompt or "improved version" in prompt:
            return "rewritten query"
        if "Is the answer grounded" in prompt:
            return "yes"
        return "Conditional edges route between nodes."

    llm = make_llm(responder=responder)
    agent = create_agent(retriever, llm=llm, max_retries=2)
    initial = {
        "question": "what are langgraph conditional edges",
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "rewritten_query": "",
    }
    result = agent.invoke(initial, {"recursion_limit": 50})

    assert result["generation"] == "Conditional edges route between nodes."
    assert result.get("grounded") is True
    # No rewrite was needed in the happy path.
    assert result["retry_count"] == 0
    # The relevant doc was kept.
    assert len(result["documents"]) >= 1


def test_full_agent_triggers_rewrite_branch(in_memory_vectorstore, make_llm, monkeypatch):
    """First grading round rejects everything -> rewrite -> re-retrieve -> accept."""
    monkeypatch.setenv("TOP_K", "4")
    _store_doc(in_memory_vectorstore)
    retriever = create_retriever(in_memory_vectorstore, strategy="basic")

    counter = {"grade": 0}

    def responder(prompt):
        if "Is this document relevant" in prompt:
            counter["grade"] += 1
            # Round 1: reject the single retrieved doc to force a rewrite.
            if counter["grade"] == 1:
                return "no"
            return "yes"
        if "Output ONLY the rewritten question" in prompt or "improved version" in prompt:
            return "rewritten langgraph conditional edges query"
        if "Is the answer grounded" in prompt:
            return "yes"
        return "final answer about conditional edges"

    llm = make_llm(responder=responder)
    agent = create_agent(retriever, llm=llm, max_retries=2)
    initial = {
        "question": "what are langgraph conditional edges",
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "rewritten_query": "",
    }
    result = agent.invoke(initial, {"recursion_limit": 50})

    assert result["generation"] == "final answer about conditional edges"
    # A rewrite must have happened.
    assert result["retry_count"] >= 1
    assert result["rewritten_query"]
    assert "rewritten" in result["rewritten_query"]


def test_full_agent_hallucination_loop_triggers_rewrite(in_memory_vectorstore, make_llm, monkeypatch):
    """Generate produces an answer, hallucination check fails -> rewrite."""
    monkeypatch.setenv("TOP_K", "4")
    _store_doc(in_memory_vectorstore)
    retriever = create_retriever(in_memory_vectorstore, strategy="basic")

    counter = {"hall": 0}

    def responder(prompt):
        if "Is this document relevant" in prompt:
            return "yes"
        if "Output ONLY the rewritten question" in prompt or "improved version" in prompt:
            return "better langgraph conditional edges query"
        if "Is the answer grounded" in prompt:
            counter["hall"] += 1
            # First hallucination check fails, second passes.
            return "no" if counter["hall"] == 1 else "yes"
        return "generated grounded answer"

    llm = make_llm(responder=responder)
    agent = create_agent(retriever, llm=llm, max_retries=2)
    initial = {
        "question": "what are langgraph conditional edges",
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "rewritten_query": "",
    }
    result = agent.invoke(initial, {"recursion_limit": 50})

    assert result["generation"] == "generated grounded answer"
    # The hallucination failure must have triggered at least one rewrite.
    assert counter["hall"] >= 2
    assert result["retry_count"] >= 1
    assert result.get("grounded") is True


# =========================================================================== #
# Customer-service graph — routing + citation + confidence
# =========================================================================== #
def _store_doc_in(vs, content, source, section=None):
    meta = {"source": source, "doc_name": source}
    if section:
        meta["section"] = section
    add_documents(
        [Document(page_content=content, metadata=meta)],
        vectorstore=vs,
    )


def _cs_responder_for_product_question(make_llm):
    """FakeLLM responder that drives a happy customer-service run.

    * Routing prompt → JSON {"category": "product"}
    * Grading prompt → "yes"
    * Rewrite prompt → returns a rewritten query (only used on the rewrite
      branch; the happy path never invokes it).
    * Generation prompt → an answer with a 参考来源 block citing product_faq.txt
    * Hallucination prompt → "yes" (answer is grounded)
    """

    def responder(prompt: str) -> str:
        if "路由分类器" in prompt or '"category"' in prompt:
            return '{"category": "product", "reason": "订单咨询"}'
        if "Is this document relevant" in prompt:
            return "yes"
        if "Output ONLY the rewritten question" in prompt or "improved version" in prompt:
            return "订单 3 天还没发货怎么办 重写"
        if "Is the answer grounded" in prompt:
            return "yes"
        if "你是企业智能客服知识库 Agent" in prompt:
            return (
                '正常情况下现货商品会在付款后 48 小时内发货，'
                '超过承诺时间未发货可点"催发货"按钮催办商家。\n\n'
                "**参考来源**：\n"
                "- [product_faq.txt] 发货与物流：付款后 48 小时内发货\n"
            )
        # Fallback so unexpected prompts still return something rather than "".
        return "yes"

    return make_llm(responder=responder)


def test_customer_service_agent_routes_product_question_to_product_collection(
    cs_vectorstore_lookup, make_llm, monkeypatch
):
    """End-to-end: a product question routes to the product collection and
    the final state carries ``category == "product"`` plus a confidence score
    that meets the threshold.
    """
    monkeypatch.setenv("TOP_K", "4")
    # Ingest a real answer into the PRODUCT collection only. The other
    # collections stay empty so we can prove the router really picked
    # "product" (a wrong route would retrieve 0 docs and fall back).
    product_vs = cs_vectorstore_lookup("product")
    _store_doc_in(
        product_vs,
        content="Q：下单 3 天了还没发货怎么办？\nA：付款后 48 小时内发货。",
        source="product_faq.txt",
        section="发货与物流",
    )

    llm = _cs_responder_for_product_question(make_llm)
    agent = create_customer_service_agent(
        cs_vectorstore_lookup, llm=llm, max_retries=2, confidence_threshold=0.6
    )
    initial = {
        "question": "下单 3 天了还没发货怎么办",
        "documents": [],
        "retry_count": 0,
    }
    result = agent.invoke(initial, {"recursion_limit": 80})

    # The router must have classified into "product".
    assert result["category"] == "product"
    # The final answer must mention the 48-hour SLA.
    assert "48 小时" in result["generation"]
    # Confidence must be above the threshold (3+ relevant docs not strictly
    # required; the citations + no uncertainty phrase is enough).
    assert result["confidence"] >= 0.6
    assert result["low_confidence"] is False
    # Citations were parsed and verified against the retrieved document.
    assert result["citations"], "expected at least one citation"
    assert all(c.get("verified") is True for c in result["citations"])
    # No retry was needed in the happy path.
    assert result["retry_count"] == 0
    assert result.get("grounded") is True


def test_customer_service_agent_low_confidence_triggers_fallback_template(
    cs_vectorstore_lookup, make_llm, monkeypatch
):
    """End-to-end: when retrieval returns nothing useful, the confidence
    node drops below threshold and replaces ``generation`` with the
    "未找到明确答案，建议转人工客服" template.
    """
    monkeypatch.setenv("TOP_K", "4")
    # Ingest into the PRODUCT collection only, but route to POLICY so the
    # retrieve node returns 0 docs → grade_documents filters everything →
    # generate runs with empty context → confidence tanks.
    product_vs = cs_vectorstore_lookup("product")
    _store_doc_in(
        product_vs,
        content="订单发货相关内容（不会被检索到）",
        source="product_faq.txt",
    )

    def responder(prompt: str) -> str:
        if "路由分类器" in prompt or '"category"' in prompt:
            # Force the route to POLICY, which has no documents at all.
            return '{"category": "policy", "reason": "用户问政策"}'
        if "Is this document relevant" in prompt:
            return "yes"  # not reached: 0 docs come back
        if "Output ONLY the rewritten question" in prompt or "improved version" in prompt:
            return "退货政策 重写"
        if "Is the answer grounded" in prompt:
            return "yes"
        if "你是企业智能客服知识库 Agent" in prompt:
            # Empty-context branch: the prompt instructs the LLM to emit the
            # fallback phrasing. We script it to include "未找到" so the
            # uncertainty sub-score fires.
            return "未找到明确答案，建议联系人工客服。"
        return "yes"

    llm = make_llm(responder=responder)
    agent = create_customer_service_agent(
        cs_vectorstore_lookup, llm=llm, max_retries=0, confidence_threshold=0.6
    )
    initial = {
        "question": "7 天无理由怎么操作",
        "documents": [],
        "retry_count": 0,
    }
    result = agent.invoke(initial, {"recursion_limit": 80})

    # Routed to POLICY (no docs ingested there) → 0 relevant docs.
    assert result["category"] == "policy"
    # Confidence must be below threshold → fallback template replaces the
    # raw "未找到明确答案，建议联系人工客服。" generation.
    assert result["confidence"] < 0.6
    assert result["low_confidence"] is True
    assert result["generation"] == FALLBACK_TEMPLATE
    # The fallback template must mention the human-hand-off trigger.
    assert "建议转人工" in result["generation"]


def test_customer_service_agent_hallucinated_citation_triggers_rewrite(
    cs_vectorstore_lookup, make_llm, monkeypatch
):
    """End-to-end: when the LLM invents a citation not in the retrieved
    documents, the hallucination check must fail and route back through
    rewrite_query, then a clean second attempt produces a verified answer.
    """
    monkeypatch.setenv("TOP_K", "4")
    product_vs = cs_vectorstore_lookup("product")
    _store_doc_in(
        product_vs,
        content="Q：退货流程？\nA：在订单页点申请退货。",
        source="product_faq.txt",
        section="退换货",
    )

    counter = {"gen": 0}

    def responder(prompt: str) -> str:
        if "路由分类器" in prompt or '"category"' in prompt:
            return '{"category": "product", "reason": "退货"}'
        if "Is this document relevant" in prompt:
            return "yes"
        if "Output ONLY the rewritten question" in prompt or "improved version" in prompt:
            return "退货流程 重写"
        if "Is the answer grounded" in prompt:
            return "yes"
        if "你是企业智能客服知识库 Agent" in prompt:
            counter["gen"] += 1
            if counter["gen"] == 1:
                # First attempt: invent a source that doesn't exist → hallucination.
                return (
                    "在订单页点申请退货。\n\n"
                    "**参考来源**：\n"
                    "- [fabricated_doc.md] 退款：编造的来源\n"
                )
            # Second attempt: cite the real source.
            return (
                "在订单页点申请退货。\n\n"
                "**参考来源**：\n"
                "- [product_faq.txt] 退换货：申请退货流程\n"
            )
        return "yes"

    llm = make_llm(responder=responder)
    agent = create_customer_service_agent(
        cs_vectorstore_lookup, llm=llm, max_retries=2, confidence_threshold=0.6
    )
    initial = {
        "question": "退货流程怎么操作",
        "documents": [],
        "retry_count": 0,
    }
    result = agent.invoke(initial, {"recursion_limit": 80})

    # The first generation had a hallucinated citation; the rewrite loop
    # must have fired at least once.
    assert counter["gen"] >= 2
    assert result["retry_count"] >= 1
    # Final answer cites the real source only.
    assert "product_faq.txt" in result["generation"]
    assert "fabricated_doc.md" not in result["generation"]
    # All final citations verified.
    assert all(c.get("verified") is True for c in result["citations"])


def test_customer_service_agent_from_vectorstore_single_store(
    in_memory_vectorstore, make_llm, monkeypatch
):
    """The single-vectorstore convenience wrapper builds a working graph.

    All categories resolve to the same in-memory Chroma, which is fine for
    tests that don't need per-collection isolation.
    """
    monkeypatch.setenv("TOP_K", "4")
    _store_doc_in(
        in_memory_vectorstore,
        content="Q：怎么开发票？\nA：下单时选电子普票。",
        source="product_faq.txt",
        section="支付与发票",
    )
    llm = _cs_responder_for_product_question(make_llm)
    agent = create_customer_service_agent_from_vectorstore(
        in_memory_vectorstore, llm=llm, max_retries=2
    )
    initial = {"question": "怎么开发票", "documents": [], "retry_count": 0}
    result = agent.invoke(initial, {"recursion_limit": 80})
    assert result["category"] == "product"
    assert result["confidence"] >= 0.6
    assert result["low_confidence"] is False


def test_build_customer_service_nodes_exposes_new_keys(make_llm):
    """``build_customer_service_nodes`` must include the new node keys."""
    llm = make_llm()
    nodes = build_customer_service_nodes(
        vectorstore_lookup=None,  # not invoked in this test
        llm=llm,
        max_retries=2,
        confidence_threshold=0.6,
        enable_citation=True,
        strategy="basic",
    )
    for key in (
        "route_query",
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "generate",
        "hallucination_check",
        "assess_confidence",
        "decide_after_grading",
        "decide_after_hallucination",
    ):
        assert key in nodes, f"missing node key: {key}"


def test_decide_after_hallucination_routes_to_assess_confidence_in_cs_graph(make_llm):
    """In the customer-service graph, a grounded answer flows to
    ``assess_confidence`` (NOT ``END``) so the confidence gate always runs.
    """
    llm = make_llm()
    nodes = build_customer_service_nodes(
        vectorstore_lookup=None, llm=llm, max_retries=2
    )
    state = {"grounded": True, "retry_count": 0}
    assert nodes["decide_after_hallucination"](state) == "assess_confidence"


def test_decide_after_hallucination_routes_to_rewrite_when_not_grounded_cs(make_llm):
    llm = make_llm()
    nodes = build_customer_service_nodes(
        vectorstore_lookup=None, llm=llm, max_retries=2
    )
    state = {"grounded": False, "retry_count": 0}
    assert nodes["decide_after_hallucination"](state) == "rewrite_query"
