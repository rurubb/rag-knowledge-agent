"""Prompt templates for every LLM call inside the agent.

Using ``ChatPromptTemplate`` keeps the prompts versionable and makes it easy to
unit test them in isolation (``prompt.format(...)`` returns plain text).
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

#: Grade whether a retrieved document is relevant to the user question.
#: The model must reply with a single token ``yes`` or ``no``.
GRADING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing the relevance of a retrieved document "
            "to a user question.\n"
            "If the document contains keywords or semantic meaning related to "
            "the user question, grade it as relevant.\n"
            "Give a binary score: respond with exactly 'yes' or 'no'.",
        ),
        (
            "human",
            "User question:\n{question}\n\n"
            "Retrieved document:\n----------------\n{content}\n----------------\n\n"
            "Is this document relevant to the question? (yes/no)",
        ),
    ]
)

#: Rewrite the user question to improve retrieval recall. Often a poorly
#: phrased question causes the first round of retrieval to miss; the rewriter
#: reformulates it with synonyms / better keywords.
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriter. Given an input question, produce an "
            "improved version that is optimised for vector retrieval: add "
            "relevant synonyms, expand abbreviations, and keep it concise.\n"
            "Output ONLY the rewritten question, no preamble.",
        ),
        ("human", "{question}"),
    ]
)

#: Final answer generation grounded strictly in the retrieved context.
GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for a knowledge base Q&A task.\n"
            "Use ONLY the pieces of context below to answer the question.\n"
            "If the context is empty or does not contain enough information, "
            "say you don't know — never make up facts.\n"
            "Cite the source filename(s) when possible.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)

#: Hallucination check: is the generated answer actually supported by the
#: retrieved documents? Prevents the model from inventing facts.
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader checking whether an answer is grounded in the "
            "retrieved documents.\n"
            "Respond with exactly 'yes' if the answer is supported by the "
            "documents, otherwise 'no'.",
        ),
        (
            "human",
            "Documents:\n----------------\n{context}\n----------------\n\n"
            "Answer: {generation}\n\n"
            "Is the answer grounded in the documents? (yes/no)",
        ),
    ]
)

#: Prompt used by MultiQueryRetriever to generate alternative queries.
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI language model assistant. Your task is to generate "
            "3 different versions of the given user question to retrieve "
            "relevant documents from a vector database.\n"
            "Provide these alternative questions separated by newlines.",
        ),
        ("human", "{question}"),
    ]
)
