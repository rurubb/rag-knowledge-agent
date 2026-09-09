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
#: Used by the basic CRAG agent (backward-compat). The customer-service agent
#: uses :data:`CITATION_GENERATION_PROMPT` which mandates a citations block.
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


#: Customer-service generation prompt: enforces a "**参考来源**" citations
#: block so every answer is auditable. The model is told to ONLY cite sources
#: that actually appear in the context — fabricating references is forbidden.
CITATION_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业智能客服知识库 Agent。请仅根据下方提供的知识库内容回答用户问题，"
            "不得编造任何不在上下文中的事实。\n"
            "回答格式必须如下，缺一不可：\n"
            "<答案正文：用 1-3 段话直接回答用户问题，必要时给出操作步骤>\n\n"
            "**参考来源**：\n"
            "- [文档名] 章节：相关段落摘要（30 字以内）\n"
            "- [文档名] 章节：相关段落摘要\n"
            "\n"
            "约束：\n"
            "1. 引用来源必须真实出现在下方 Context 中，文档名取 Context 中标注的 [文档名]；"
            "如果 Context 中没有标出文档名，则用 source 文件名。\n"
            "2. 如果 Context 为空或不足以回答，输出：未找到明确答案，建议联系人工客服。\n"
            "3. 不要引用未在 Context 中出现的文档名，否则视为幻觉。\n"
            "4. 答案正文控制在 200 字以内，要点清晰、可执行。",
        ),
        (
            "human",
            "Context:\n{context}\n\n用户问题: {question}",
        ),
    ]
)


#: Multi-knowledge-base routing: classify the user question into one of
#: product / policy / sop / general so the retrieve node can pick the right
#: Chroma collection. Output must be a single JSON object on one line so it
#: is trivial to parse — the router falls back to "general" on any error.
#:
#: NOTE: the literal JSON example below uses ``{{`` / ``}}`` escapes because
#: :class:`ChatPromptTemplate` runs the system message through
#: ``str.format``; unescaped ``{...}`` would be parsed as a format
#: placeholder and crash with ``KeyError``.
ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是企业智能客服知识库的路由分类器。\n"
            "知识库分为 4 类：\n"
            "- product：产品 FAQ，如订单查询、发货物流、退换货操作、支付发票、会员账户等用户使用产品时遇到的问题；\n"
            "- policy：退换货政策，如 7 天无理由、质量问题、三包、保修、运费、运费险、特殊场景（大促 / 跨境 / 大件）等政策性条款；\n"
            "- sop：客服操作 SOP，如工单受理、投诉分级、退款操作、升级转接、转人工时机、知识库维护等客服内部流程；\n"
            "- general：无法明确归类的问题，如闲聊、跨类复合问题、与客服业务无关的问题。\n"
            "请输出一个 JSON 对象，格式如下，禁止输出多余文字：\n"
            '{{"category": "<product|policy|sop|general>", "reason": "<不超过 30 字的中文理由>"}}',
        ),
        ("human", "{question}"),
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
