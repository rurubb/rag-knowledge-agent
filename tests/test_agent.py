"""Tests for the self-reflective LangGraph agent.

All tests use the in-memory Chroma store + fake embeddings from conftest and a
scriptable FakeLLM, so they never touch the network or a real model.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langgraph.graph import END

from rag_agent.agent import (
    AgentState,
    build_nodes,
    create_agent,
    parse_yes_no,
)
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
