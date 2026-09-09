"""Self-reflective RAG agent built on LangGraph.

Two graphs are exposed:

1. :func:`create_agent` — the original Corrective RAG (CRAG) loop, kept for
   backward compatibility with the existing tests::

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

2. :func:`create_customer_service_agent` — the customer-service graph that
   wraps the CRAG loop with multi-knowledge-base routing, citation-enforcing
   generation, and a confidence-gated fallback to a "转人工" template::

       START → route_query → retrieve(category-aware) → grade_documents
                            └─(no relevant docs)─→ rewrite_query → retrieve
       grade_documents ──(relevant)─→ generate(citations) → hallucination_check
                            └─(not grounded)─→ rewrite_query → retrieve
       hallucination_check ──(grounded)─→ assess_confidence → END
       assess_confidence ──(low confidence)─→ END (with fallback template)

The customer-service graph adds three nodes on top of the CRAG backbone:

* ``route_query`` — LLM classifies the question into ``product`` / ``policy``
  / ``sop`` / ``general`` so the retrieve node picks the right Chroma
  collection. (Falls back to ``general`` on any error.)
* ``generate`` — uses :data:`~rag_agent.prompts.CITATION_GENERATION_PROMPT`
  which mandates a ``**参考来源**`` block.
* ``assess_confidence`` — combines document-count + citation-presence +
  uncertainty-phrase signals into a 0-1 score. Below
  :attr:`~rag_agent.config.Settings.confidence_threshold` the generation is
  replaced with the "未找到明确答案，建议转人工客服" template.

Key behaviours that distinguish this from "naive RAG":

* **Multi-knowledge-base routing** — different corpora live in different
  Chroma collections; the router picks the right one per question.
* **Document grading** — each retrieved chunk is judged relevant/irrelevant;
  irrelevant chunks are dropped before generation.
* **Query rewriting + re-retrieval** — when nothing relevant comes back the
  query is rewritten (synonyms / better keywords) and retrieval is retried,
  up to ``MAX_RETRIES`` times.
* **Citation enforcement + hallucination cross-check** — every answer must
  carry a ``**参考来源**`` block, and citations are verified against the
  retrieved documents; fabricated references trigger the rewrite loop.
* **Confidence gate** — below threshold, the answer is replaced with a
  "建议转人工" template instead of being shown to the user.
"""

from __future__ import annotations

from typing import Any, Callable, List, TypedDict, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from .citation import (
    build_citation_context,
    extract_citations,
    has_hallucinated_citations,
)
from .config import get_settings
from .confidence import assess_confidence
from .llm import create_llm
from .prompts import (
    CITATION_GENERATION_PROMPT,
    GENERATION_PROMPT,
    GRADING_PROMPT,
    HALLUCINATION_PROMPT,
    REWRITE_PROMPT,
)
from .retriever import RetrievalStrategy, create_retriever
from .router import route_query


class AgentState(TypedDict, total=False):
    """State flowing through the graph.

    ``question`` / ``documents`` / ``generation`` / ``retry_count`` /
    ``rewritten_query`` are the primary fields. ``grounded`` is an auxiliary
    flag written by the hallucination check and read by its conditional edge.

    Customer-service extension fields (all optional — the basic CRAG graph
    in :func:`create_agent` does not set them, so existing tests that build
    a basic agent continue to pass):

    * ``category`` — output of the ``route_query`` node.
    * ``citations`` — parsed ``**参考来源**`` records from the answer.
    * ``confidence`` — 0-1 score from the ``assess_confidence`` node.
    * ``low_confidence`` — bool flag set when confidence < threshold.
    """

    question: str
    documents: List[Document]
    generation: str
    retry_count: int
    rewritten_query: str
    grounded: bool
    # Customer-service extension
    category: str
    citations: List[dict]
    confidence: float
    low_confidence: bool


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
    """Convenience wrapper: build retriever + agent in one call.

    Kept for backward compatibility — builds the basic CRAG agent (no
    routing / citation / confidence). For the customer-service graph use
    :func:`create_customer_service_agent`.
    """
    retriever = create_retriever(vectorstore, strategy=strategy, llm=llm)
    return create_agent(retriever, llm=llm, max_retries=max_retries)


# =========================================================================== #
# Customer-service graph (route_query + citation-aware generate + confidence)
# =========================================================================== #
#: Type alias for the vectorstore lookup. Either a callable that returns a
#: vectorstore for a given category, or a single vectorstore used for every
#: category (handy for tests).
VectorstoreLookup = Union[
    Callable[[str], Any],
    Any,
]


def _resolve_lookup(vectorstore_lookup: VectorstoreLookup) -> Callable[[str], Any]:
    """Coerce ``vectorstore_lookup`` into a callable(category) -> vectorstore."""
    if callable(vectorstore_lookup):
        return vectorstore_lookup
    return lambda _category: vectorstore_lookup


def build_customer_service_nodes(
    vectorstore_lookup: VectorstoreLookup,
    llm: BaseChatModel,
    max_retries: int,
    confidence_threshold: float = 0.6,
    enable_citation: bool = True,
    strategy: RetrievalStrategy = "basic",
):
    """Return node callables for the customer-service graph.

    The returned dict mirrors :func:`build_nodes` (so the basic CRAG tests can
    still call ``nodes["grade_documents"]`` etc. on the basic builder) plus
    the three new nodes ``route_query`` / ``assess_confidence`` and the new
    ``decide_after_hallucination`` that routes to ``assess_confidence``
    instead of ``END``.

    Parameters
    ----------
    vectorstore_lookup:
        Either a callable ``(category: str) -> vectorstore`` (the production
        path — each category resolves to its own Chroma collection) or a
        single vectorstore (the test path — one in-memory Chroma is shared
        across categories).
    llm:
        Chat model used for routing / grading / rewriting / generation /
        hallucination check.
    max_retries:
        Maximum number of re-retrieval attempts.
    confidence_threshold:
        Below this score the generation is replaced with the fallback
        template.
    enable_citation:
        Whether the generate node uses the citation-enforcing prompt and the
        hallucination check verifies citations against retrieved documents.
    strategy:
        Retrieval strategy (``basic`` / ``multi_query``) used inside the
        ``retrieve`` node.
    """
    get_vs = _resolve_lookup(vectorstore_lookup)

    def retrieve(state: AgentState) -> dict:
        # Prefer the rewritten query when a previous round failed; otherwise
        # use the original question. The category tells us which collection
        # to hit.
        query = state.get("rewritten_query") or state["question"]
        category = state.get("category", "general")
        vs = get_vs(category)
        retriever = create_retriever(vs, strategy=strategy, llm=llm)
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
        if enable_citation:
            context = build_citation_context(state.get("documents", []))
            prompt = CITATION_GENERATION_PROMPT.format(
                context=context, question=state["question"]
            )
        else:
            context = format_context(state.get("documents", []))
            prompt = GENERATION_PROMPT.format(
                context=context, question=state["question"]
            )
        response = llm.invoke(prompt)
        generation = (
            response.content if hasattr(response, "content") else str(response)
        )
        result: dict = {"generation": generation}
        # Pre-parse citations so the hallucination check + assess_confidence
        # node can read them without re-parsing.
        if enable_citation:
            result["citations"] = extract_citations(
                generation, state.get("documents", [])
            )
        return result

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
        # Hard hallucination signal: a cited source is NOT in the retrieved
        # documents. The LLM was told not to do this — if it does, force a
        # rewrite round.
        if enable_citation:
            citations = state.get("citations") or extract_citations(
                generation, state.get("documents", [])
            )
            if has_hallucinated_citations(citations):
                grounded = False
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
        """Route after the hallucination check.

        On success the customer-service graph continues to
        ``assess_confidence`` rather than terminating — the confidence gate
        runs as the final step.
        """
        grounded = state.get("grounded", True)
        retries = state.get("retry_count", 0)
        if not grounded and retries < max_retries:
            return "rewrite_query"
        return "assess_confidence"

    return {
        "route_query": route_query(llm),
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "rewrite_query": rewrite_query,
        "generate": generate,
        "hallucination_check": hallucination_check,
        "assess_confidence": assess_confidence(
            confidence_threshold=confidence_threshold,
            enable_citation=enable_citation,
        ),
        "decide_after_grading": decide_after_grading,
        "decide_after_hallucination": decide_after_hallucination,
    }


def create_customer_service_agent(
    vectorstore_lookup: VectorstoreLookup,
    llm: BaseChatModel | None = None,
    max_retries: int | None = None,
    confidence_threshold: float | None = None,
    enable_citation: bool | None = None,
    strategy: RetrievalStrategy = "basic",
):
    """Compile and return the customer-service LangGraph agent.

    Graph::

        START → route_query → retrieve → grade_documents
                                    │
                                    ├─(no relevant docs)─→ rewrite_query → retrieve
                                    └─(relevant)─→ generate → hallucination_check
                                                                      │
                                                                      ├─(not grounded)─→ rewrite_query
                                                                      └─(grounded)─→ assess_confidence → END

    Parameters
    ----------
    vectorstore_lookup:
        Callable ``(category) -> vectorstore`` (production) or a single
        vectorstore (tests). The retrieve node uses the ``category`` written
        by ``route_query`` to pick the right Chroma collection.
    llm, max_retries:
        Same semantics as :func:`create_agent`.
    confidence_threshold, enable_citation:
        Customer-service settings. Default to :class:`Settings` values.
    strategy:
        Retrieval strategy passed into the per-call retriever.
    """
    if llm is None:
        llm = create_llm()
    if max_retries is None:
        max_retries = get_settings().max_retries
    if confidence_threshold is None:
        confidence_threshold = get_settings().confidence_threshold
    if enable_citation is None:
        enable_citation = get_settings().enable_citation

    nodes = build_customer_service_nodes(
        vectorstore_lookup,
        llm,
        max_retries,
        confidence_threshold=confidence_threshold,
        enable_citation=enable_citation,
        strategy=strategy,
    )

    graph = StateGraph(AgentState)
    graph.add_node("route_query", nodes["route_query"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("grade_documents", nodes["grade_documents"])
    graph.add_node("rewrite_query", nodes["rewrite_query"])
    graph.add_node("generate", nodes["generate"])
    graph.add_node("hallucination_check", nodes["hallucination_check"])
    graph.add_node("assess_confidence", nodes["assess_confidence"])

    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "retrieve")
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
        {"rewrite_query": "rewrite_query", "assess_confidence": "assess_confidence"},
    )
    graph.add_edge("assess_confidence", END)

    return graph.compile()


def create_customer_service_agent_from_vectorstore(
    vectorstore,
    strategy: RetrievalStrategy = "basic",
    llm: BaseChatModel | None = None,
    max_retries: int | None = None,
    confidence_threshold: float | None = None,
    enable_citation: bool | None = None,
):
    """Convenience wrapper: single-vectorstore customer-service agent.

    Used by tests where one in-memory Chroma is shared across categories.
    Production callers should pass a ``vectorstore_lookup`` callable to
    :func:`create_customer_service_agent` so each category resolves to its
    own collection.
    """
    return create_customer_service_agent(
        vectorstore,
        llm=llm,
        max_retries=max_retries,
        confidence_threshold=confidence_threshold,
        enable_citation=enable_citation,
        strategy=strategy,
    )
