"""Self-reflective RAG agent built on LangGraph.

Pipeline (Corrective RAG / CRAG flavour):

    START
      └─ retrieve ─────────► grade_documents ──(decide_after_grading)──┐
                                 │                                     │
                                 │ relevant docs >= 1                  │ no relevant docs
                                 ▼                                     │ & retries left
                               generate ──► hallucination_check        │
                                              │                         ▼
                                    (decide_after_hallucination)   rewrite_query ──► retrieve
                                              │
                                              ▼
                                             END

Key behaviours that distinguish this from "naive RAG":

* **Document grading** — each retrieved chunk is judged relevant/irrelevant;
  irrelevant chunks are dropped before generation.
* **Query rewriting + re-retrieval** — when nothing relevant comes back the
  query is rewritten (synonyms / better keywords) and retrieval is retried,
  up to ``MAX_RETRIES`` times.
* **Hallucination check** — after generation we verify the answer is actually
  grounded in the retrieved documents; if not, we loop back through query
  rewriting.
"""

from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from .config import get_settings
from .llm import create_llm
from .prompts import (
    GENERATION_PROMPT,
    GRADING_PROMPT,
    HALLUCINATION_PROMPT,
    REWRITE_PROMPT,
)
from .retriever import RetrievalStrategy, create_retriever


class AgentState(TypedDict, total=False):
    """State flowing through the graph.

    ``question`` / ``documents`` / ``generation`` / ``retry_count`` /
    ``rewritten_query`` are the primary fields. ``grounded`` is an auxiliary
    flag written by the hallucination check and read by its conditional edge.
    """

    question: str
    documents: List[Document]
    generation: str
    retry_count: int
    rewritten_query: str
    grounded: bool


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_yes_no(text: str) -> bool:
    """Return ``True`` for a "yes" answer, ``False`` otherwise.

    Robust to surrounding whitespace / punctuation / extra tokens
    ("Yes.", "yes, the document is relevant", "YES\n", etc.).
    """
    if text is None:
        return False
    cleaned = text.strip().lower()
    if not cleaned:
        return False
    # Take the first token to tolerate "yes, the document is relevant".
    first_token = cleaned.split()[0]
    # Strip trailing punctuation like "yes." or "yes,".
    first_token = first_token.strip(".,!?;:\"'()[]")
    return first_token in {"yes", "y", "true", "1"}


def format_context(documents: List[Document]) -> str:
    """Render documents into a single context string with provenance."""
    if not documents:
        return "(no relevant context was retrieved)"
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        loc = f" (page {page})" if page is not None else ""
        blocks.append(f"[{i}] source={source}{loc}\n{doc.page_content}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Node factory
# --------------------------------------------------------------------------- #
def build_nodes(retriever: BaseRetriever, llm: BaseChatModel, max_retries: int):
    """Return the node callables closed over ``retriever`` and ``llm``."""

    def retrieve(state: AgentState) -> dict:
        # Prefer the rewritten query when a previous round failed.
        query = state.get("rewritten_query") or state["question"]
        documents = retriever.invoke(query)
        return {"documents": documents}

    def grade_documents(state: AgentState) -> dict:
        question = state["question"]
        documents = state.get("documents", [])
        relevant: List[Document] = []
        for doc in documents:
            prompt = GRADING_PROMPT.format(
                question=question, content=doc.page_content
            )
            try:
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
            except Exception:
                # If the grader blows up, keep the document conservatively.
                answer = "yes"
            if parse_yes_no(answer):
                relevant.append(doc)
        return {"documents": relevant}

    def rewrite_query(state: AgentState) -> dict:
        question = state.get("rewritten_query") or state["question"]
        prompt = REWRITE_PROMPT.format(question=question)
        response = llm.invoke(prompt)
        rewritten = (
            response.content if hasattr(response, "content") else str(response)
        )
        rewritten = rewritten.strip().strip('"').strip("'")
        return {
            "rewritten_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def generate(state: AgentState) -> dict:
        context = format_context(state.get("documents", []))
        prompt = GENERATION_PROMPT.format(
            context=context, question=state["question"]
        )
        response = llm.invoke(prompt)
        generation = (
            response.content if hasattr(response, "content") else str(response)
        )
        return {"generation": generation}

    def hallucination_check(state: AgentState) -> dict:
        context = format_context(state.get("documents", []))
        generation = state.get("generation", "")
        prompt = HALLUCINATION_PROMPT.format(
            context=context, generation=generation
        )
        try:
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            answer = "yes"
        grounded = parse_yes_no(answer)
        return {"grounded": grounded}

    # ------------------------------------------------------------------ #
    # Conditional edges
    # ------------------------------------------------------------------ #
    def decide_after_grading(state: AgentState) -> str:
        """Route after grading: rewrite & retry, or proceed to generate."""
        docs = state.get("documents", [])
        retries = state.get("retry_count", 0)
        if len(docs) == 0 and retries < max_retries:
            return "rewrite_query"
        return "generate"

    def decide_after_hallucination(state: AgentState) -> str:
        """Route after the hallucination check."""
        grounded = state.get("grounded", True)
        retries = state.get("retry_count", 0)
        if not grounded and retries < max_retries:
            return "rewrite_query"
        return END

    return {
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "rewrite_query": rewrite_query,
        "generate": generate,
        "hallucination_check": hallucination_check,
        "decide_after_grading": decide_after_grading,
        "decide_after_hallucination": decide_after_hallucination,
    }


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def create_agent(
    retriever: BaseRetriever,
    llm: BaseChatModel | None = None,
    max_retries: int | None = None,
):
    """Compile and return the LangGraph agent.

    Parameters
    ----------
    retriever:
        Any :class:`BaseRetriever` (from :func:`rag_agent.retriever.create_retriever`).
    llm:
        Chat model used for grading / rewriting / generation / hallucination
        check. Defaults to :func:`rag_agent.llm.create_llm`.
    max_retries:
        Maximum number of re-retrieval attempts. Defaults to ``MAX_RETRIES``.
    """
    if llm is None:
        llm = create_llm()
    if max_retries is None:
        max_retries = get_settings().max_retries

    nodes = build_nodes(retriever, llm, max_retries)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("grade_documents", nodes["grade_documents"])
    graph.add_node("rewrite_query", nodes["rewrite_query"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("hallucination_check", nodes["hallucination_check"])

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        nodes["decide_after_grading"],
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "hallucination_check")
    graph.add_conditional_edges(
        "hallucination_check",
        nodes["decide_after_hallucination"],
        {"rewrite_query": "rewrite_query", END: END},
    )

    return graph.compile()


def create_agent_from_vectorstore(
    vectorstore,
    strategy: RetrievalStrategy = "basic",
    llm: BaseChatModel | None = None,
    max_retries: int | None = None,
):
    """Convenience wrapper: build retriever + agent in one call."""
    retriever = create_retriever(vectorstore, strategy=strategy, llm=llm)
    return create_agent(retriever, llm=llm, max_retries=max_retries)
