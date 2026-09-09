"""Command-line interface: ``ingest`` and ``ask`` sub-commands.

Examples
--------
::

    # Ingest every supported file under data/, each routed to the matching
    # Chroma collection by filename (product_faq.txt -> product, policy.md
    # -> policy, sop.md -> sop, anything else -> general).
    python main.py ingest --dir ./data

    # Offline smoke-test (no model download, no network):
    $env:EMBEDDING_PROVIDER = "fake"; python main.py ingest --dir ./data

    # Ask the customer-service agent. --show-trace prints the route ->
    # retrieve -> grade -> generate -> confidence flow; --show-citations
    # prints only the answer + its references block.
    python main.py ask "下单 3 天了还没发货怎么办" --show-trace
    python main.py ask "7 天无理由怎么操作" --show-citations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import get_settings, load_dotenv_if_present


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-agent",
        description=(
            "企业智能客服知识库 Agent (LangChain + LangGraph): "
            "multi-knowledge-base routing + citation + confidence gate."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest -----------------------------------------------------------------
    p_ingest = sub.add_parser(
        "ingest",
        help="Load, chunk, embed and persist documents into per-category "
        "Chroma collections under data/.",
    )
    p_ingest.add_argument(
        "--dir", default="./data", help="Directory of documents to ingest."
    )
    p_ingest.add_argument(
        "--clear",
        action="store_true",
        help="Clear each affected collection before ingesting.",
    )

    # ask --------------------------------------------------------------------
    p_ask = sub.add_parser(
        "ask", help="Ask the customer-service RAG agent a question."
    )
    p_ask.add_argument("question", help="The question to ask.")
    p_ask.add_argument(
        "--strategy",
        choices=["basic", "multi_query"],
        default="basic",
        help="Retrieval strategy.",
    )
    p_ask.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Max re-retrieval attempts (defaults to MAX_RETRIES).",
    )
    p_ask.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override CONFIDENCE_THRESHOLD for this run.",
    )
    p_ask.add_argument(
        "--no-citation",
        action="store_true",
        help="Disable the citation-enforcing prompt and the citation "
        "hallucination check.",
    )
    p_ask.add_argument(
        "--show-trace",
        action="store_true",
        default=None,
        help="Print route -> retrieve -> grade -> generate -> confidence flow.",
    )
    p_ask.add_argument(
        "--show-citations",
        action="store_true",
        help="Only print the answer and its 参考来源 block (no trace).",
    )
    return parser


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def cmd_ingest(dir_path: str, clear: bool = False) -> int:
    """Ingest every supported file under ``dir_path``.

    Each file is bucketed into a knowledge-base category by filename (see
    :func:`rag_agent.router.classify_filename`) and persisted into the
    matching Chroma collection (e.g. ``rag_knowledge_product``). Files that
    don't match a known pattern land in ``rag_knowledge_general``.
    """
    from .loader import SUPPORTED_SUFFIXES, load_file
    from .chunker import chunk_documents
    from .router import classify_filename, collection_for_category
    from .vectorstore import (
        add_documents,
        clear as clear_store,
        get_vectorstore,
    )

    base = Path(dir_path)
    if not base.exists():
        print(f"[ingest] directory not found: {dir_path}")
        return 1

    files = sorted(
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        print("[ingest] no supported documents found (.txt/.md/.pdf).")
        return 1

    print(f"[ingest] scanning {len(files)} file(s) under {dir_path} ...")
    counts: dict[str, int] = {}
    for path in files:
        category = classify_filename(path.name)
        collection = collection_for_category(category)
        try:
            docs = load_file(str(path))
        except Exception as exc:  # pragma: no cover - depends on file content
            print(f"[ingest] skip {path} ({exc})")
            continue
        if not docs:
            continue
        # Tag each doc with the category + collection so downstream citation
        # verification can resolve the source name.
        for d in docs:
            d.metadata.setdefault("source", str(path))
            d.metadata.setdefault("doc_name", path.name)
            d.metadata.setdefault("category", category)
            d.metadata.setdefault("collection", collection)

        chunks = chunk_documents(docs)
        print(
            f"[ingest] {path.name} -> category={category} "
            f"collection={collection} chunks={len(chunks)}"
        )

        vs = get_vectorstore(collection_name=collection)
        if clear:
            try:
                clear_store(vs)
            except Exception as exc:  # pragma: no cover - chroma state
                print(f"[ingest] clear skipped for {collection} ({exc})")
            vs = get_vectorstore(collection_name=collection)
        add_documents(chunks, vectorstore=vs)
        counts[category] = counts.get(category, 0) + len(chunks)

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
    print(f"[ingest] done. Per-category chunk counts: {summary}")
    return 0


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #
def _vectorstore_lookup(category: str):
    """Production lookup: resolve a category to its Chroma collection.

    Lazy-imported here so the CLI startup cost stays low and so tests can
    monkeypatch :func:`rag_agent.vectorstore.get_vectorstore` if needed.
    """
    from .router import collection_for_category
    from .vectorstore import get_vectorstore

    return get_vectorstore(collection_name=collection_for_category(category))


def cmd_ask(
    question: str,
    strategy: str = "basic",
    max_retries: int | None = None,
    confidence_threshold: float | None = None,
    no_citation: bool = False,
    show_trace: bool | None = None,
    show_citations: bool = False,
) -> int:
    from .agent import create_customer_service_agent
    from .citation import extract_citations

    settings = get_settings()
    if show_trace is None:
        show_trace = settings.show_trace
    enable_citation = not no_citation and settings.enable_citation

    agent = create_customer_service_agent(
        _vectorstore_lookup,
        strategy=strategy,
        max_retries=max_retries,
        confidence_threshold=confidence_threshold,
        enable_citation=enable_citation,
    )

    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "rewritten_query": "",
        "category": "",
        "citations": [],
        "confidence": 0.0,
    }

    # Fast path: --show-citations skips the per-step trace and prints only
    # the final answer + 参考来源 block.
    if show_citations and not show_trace:
        final = agent.invoke(initial_state, {"recursion_limit": 50})
        print("\n=== Answer ===")
        print(final.get("generation", ""))
        citations = final.get("citations") or extract_citations(
            final.get("generation", "")
        )
        print("\n=== Citations ===")
        if not citations:
            print("(no citations)")
        for i, c in enumerate(citations, start=1):
            mark = "OK" if c.get("verified") else "?"
            print(f"  [{i}] [{mark}] {c.get('raw', '')}")
        conf = final.get("confidence")
        if conf is not None:
            print(f"\nconfidence: {conf:.2f}")
        return 0

    if not show_trace:
        # --no-trace path: just run and print the answer.
        final = agent.invoke(initial_state, {"recursion_limit": 50})
        print("\n=== Answer ===")
        print(final.get("generation", ""))
        conf = final.get("confidence")
        if conf is not None:
            print(f"\nconfidence: {conf:.2f}")
        return 0

    # --show-trace: stream the state deltas per step.
    print(f"\n[ask] question: {question}")
    print(f"[ask] strategy: {strategy}")
    print("[ask] running customer-service agent (streaming state per step) ...\n")
    for output in agent.stream(initial_state, {"recursion_limit": 50}):
        for node_name, state_delta in output.items():
            print(f"--- step: {node_name} ---")
            if "category" in state_delta and state_delta["category"]:
                print(f"  category: {state_delta['category']}")
            if "rewritten_query" in state_delta and state_delta["rewritten_query"]:
                print(f"  rewritten_query: {state_delta['rewritten_query']}")
            if "retry_count" in state_delta:
                print(f"  retry_count: {state_delta['retry_count']}")
            if "documents" in state_delta:
                docs = state_delta["documents"] or []
                print(f"  documents: {len(docs)} kept")
                for i, d in enumerate(docs, start=1):
                    src = d.metadata.get("source", "?")
                    snippet = d.page_content[:80].replace("\n", " ")
                    print(f"    [{i}] ({src}) {snippet}...")
            if "generation" in state_delta and state_delta["generation"]:
                print(f"  generation: {state_delta['generation']}")
            if "grounded" in state_delta:
                print(f"  grounded: {state_delta['grounded']}")
            if "citations" in state_delta and state_delta["citations"]:
                cs = state_delta["citations"]
                print(f"  citations: {len(cs)} parsed")
                for i, c in enumerate(cs, start=1):
                    mark = "OK" if c.get("verified") else "?"
                    print(f"    [{i}] [{mark}] {c.get('raw', '')}")
            if "confidence" in state_delta:
                print(f"  confidence: {state_delta['confidence']:.2f}")
            if "low_confidence" in state_delta:
                print(f"  low_confidence: {state_delta['low_confidence']}")

    # Re-invoke once to grab the final state for the summary; the streamed
    # deltas above don't expose the merged final state.
    final = agent.invoke(initial_state, {"recursion_limit": 50})
    print("\n=== Final Answer ===")
    print(final.get("generation", ""))
    conf = final.get("confidence")
    if conf is not None:
        print(f"\nconfidence: {conf:.2f}")
    cat = final.get("category")
    if cat:
        print(f"category: {cat}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return cmd_ingest(args.dir, clear=args.clear)
    if args.command == "ask":
        return cmd_ask(
            args.question,
            strategy=args.strategy,
            max_retries=args.max_retries,
            confidence_threshold=args.confidence_threshold,
            no_citation=args.no_citation,
            show_trace=args.show_trace,
            show_citations=args.show_citations,
        )
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
