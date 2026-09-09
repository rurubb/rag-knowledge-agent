"""Citation extraction and verification for customer-service answers.

A customer-service answer is auditable **only if** every cited source actually
exists in the retrieved documents. This module:

* :func:`extract_citations` — parses the ``**参考来源**`` block from the
  generated answer into structured records.
* :func:`has_hallucinated_citations` — flags citations whose ``source`` is
  not present in the retrieved documents (i.e. the LLM fabricated a
  reference). The hallucination-check node treats this as a hard failure and
  routes back through ``rewrite_query``.
* :func:`build_citation_context` — renders retrieved documents with explicit
  ``[文档名] 章节`` headers so the generation prompt can produce citations
  that round-trip through :func:`extract_citations`.
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.documents import Document

#: Match the "**参考来源**：" header and capture the bullet list that follows.
#: The list ends at the first blank line or end of string.
_CITATION_HEADER_RE = re.compile(
    r"\*{0,2}参考来源\*{0,2}\s*[：:]\s*\n((?:[ \t]*[-*•][ \t]*.+\n?)+)",
    re.IGNORECASE,
)

#: Match a single bullet line: "- [文档名] 章节：摘要" or "- [文档名]：摘要"
#: or "- 文本（无方括号）".
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$")


def _basename(path: str) -> str:
    """Cross-platform basename: handles both / and \\ separators."""
    if not path:
        return "unknown"
    return re.split(r"[\\/]", path)[-1]


def extract_citations(
    answer: str,
    documents: Optional[List[Document]] = None,
) -> List[dict]:
    """Parse the ``**参考来源**`` block from ``answer``.

    Returns a list of dicts with keys:

    * ``raw`` — the full bullet text (minus the leading ``-``).
    * ``source`` — the document name parsed from ``[...]``, or ``None``.
    * ``section`` — the section/title text before the colon, or ``None``.
    * ``summary`` — the summary text after the colon (or the whole bullet
      if no colon).
    * ``verified`` — ``True`` / ``False`` if ``documents`` was supplied and
      the source matched one of the retrieved documents; ``None`` if
      ``documents`` was ``None`` (no verification requested).

    The match is intentionally fuzzy on the source name: we compare
    basenames and accept substring matches so a citation of
    ``product_faq.txt`` against a retrieved document with
    ``source=/path/to/product_faq.txt`` is accepted.
    """
    if not answer:
        return []

    match = _CITATION_HEADER_RE.search(answer)
    if not match:
        return []

    block = match.group(1)
    citations: List[dict] = []
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        bullet = _BULLET_RE.match(line)
        if not bullet:
            continue
        body = bullet.group(1).strip()
        citation: dict = {
            "raw": body,
            "source": None,
            "section": None,
            "summary": body,
            "verified": None,
        }

        # Parse "[文档名] 章节：摘要" or "[文档名]：摘要".
        bracket = re.match(r"\[([^\]]+)\]\s*(.*)", body)
        if bracket:
            citation["source"] = bracket.group(1).strip()
            rest = bracket.group(2).strip()
            # Split section / summary on the first full-width or half-width
            # colon.
            sec_match = re.match(r"([^：:]+?)\s*[：:]\s*(.+)", rest)
            if sec_match:
                citation["section"] = sec_match.group(1).strip()
                citation["summary"] = sec_match.group(2).strip()
            else:
                citation["summary"] = rest or body
        else:
            # No bracket: try splitting on the first colon.
            sec_match = re.match(r"([^：:]+?)\s*[：:]\s*(.+)", body)
            if sec_match:
                citation["source"] = sec_match.group(1).strip()
                citation["summary"] = sec_match.group(2).strip()
        citations.append(citation)

    if documents is not None:
        valid_sources = set()
        for d in documents:
            src = d.metadata.get("source") if hasattr(d, "metadata") else None
            if src:
                valid_sources.add(src)
                valid_sources.add(_basename(src))
            # Also accept an explicit metadata "doc_name" if the loader set one.
            doc_name = d.metadata.get("doc_name") if hasattr(d, "metadata") else None
            if doc_name:
                valid_sources.add(doc_name)
                valid_sources.add(_basename(doc_name))
        for c in citations:
            src = c.get("source")
            if src is None:
                c["verified"] = False
                continue
            src_lower = src.lower()
            src_base = _basename(src).lower()
            matched = any(
                src_lower == v.lower()
                or src_lower in v.lower()
                or v.lower() in src_lower
                or src_base == v.lower()
                or src_base in v.lower()
                or v.lower() in src_base
                for v in valid_sources
            )
            c["verified"] = matched

    return citations


def has_citations(answer: str) -> bool:
    """True if ``answer`` contains a non-empty ``**参考来源**`` block."""
    return bool(extract_citations(answer))


def has_hallucinated_citations(citations: List[dict]) -> bool:
    """True if any citation's source is NOT in the retrieved documents.

    A citation is considered hallucinated when:

    * it has a ``source`` field (i.e. the LLM produced a ``[docname]``), AND
    * ``verified`` is exactly ``False`` (i.e. we did run verification and it
      failed).

    Citations without a ``source`` field are not counted as hallucinations —
    they are just unstructured prose and the confidence node penalises them
    separately via the "no citation" sub-score.
    """
    if not citations:
        return False
    return any(
        c.get("source") is not None and c.get("verified") is False
        for c in citations
    )


def build_citation_context(documents: List[Document]) -> str:
    """Render documents with explicit ``[文档名] 章节`` headers.

    Each retrieved chunk becomes a block whose first line is the citation
    header the generation prompt is told to echo back, e.g.::

        [product_faq.txt] 发货与物流
        Q：下单 3 天了还没发货怎么办？
        A：正常情况下现货商品会在付款后 48 小时内发货...

    The generation prompt then produces ``**参考来源**`` bullets whose
    ``[文档名]`` round-trips cleanly through :func:`extract_citations`.
    """
    if not documents:
        return "(no relevant context was retrieved)"
    blocks = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
        source_name = _basename(str(source))
        section = ""
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        for key in ("section", "heading", "title"):
            v = meta.get(key)
            if v:
                section = str(v)
                break
        header = f"[{source_name}]"
        if section:
            header += f" {section}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)
