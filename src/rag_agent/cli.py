"""Command-line interface: ``ingest`` and ``ask`` sub-commands.

Examples
--------
::

    python main.py ingest --dir ./data
    python main.py ask "什么是 LangGraph 的条件边？" --strategy multi_query
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import load_dotenv_if_present


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-agent",
        description=(
            "Self-reflective RAG knowledge agent (LangChain + LangGraph)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest -----------------------------------------------------------------
    p_ingest = sub.add_parser(
        "ingest", help="Load, chunk, embed and persist documents into Chroma."
    )
    p_ingest.add_argument(
        "--dir", default="./data", help="Directory of documents to ingest."
    )
    p_ingest.add_argument(
        "--clear",
        action="store_true",
        help="Clear the existing collection before ingesting.",
    )

    # ask --------------------------------------------------------------------
    p_ask = sub.add_parser(
        "ask", help="Ask the self-reflective RAG agent a question."
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
        "--no-stream",
        action="store_true",
        help="Disable step-by-step state printing (only print the answer).",
    )
    return parser


def cmd_ingest(dir_path: str, clear: bool = False) -> int:
    from .loader import load_directory
    from .vectorstore import add_documents, clear as clear_store, get_vectorstore

    print(f"[ingest] loading documents from {dir_path} ...")
    docs = load_directory(dir_path)
    if not docs:
        print("[ingest] no supported documents found (.txt/.md/.pdf).")
        return 1
    print(f"[ingest] loaded {len(docs)} document(s).")

    from .chunker import chunk_documents

    chunks = chunk_documents(docs)
    print(f"[ingest] chunked into {len(chunks)} pieces.")

    vs = get_vectorstore()
    if clear:
        print("[ingest] clearing existing collection ...")
        try:
            clear_store(vs)
        except Exception as exc:  # pragma: no cover - depends on chroma state
            print(f"[ingest] clear skipped ({exc})")
        vs = get_vectorstore()

    add_documents(chunks, vectorstore=vs)
    print("[ingest] done. Vector store persisted.")
    return 0


def cmd_ask(
    question: str,
    strategy: str = "basic",
    max_retries: int | None = None,
    no_stream: bool = False,
) -> int:
    from .agent import create_agent_from_vectorstore
    from .vectorstore import get_vectorstore

    vs = get_vectorstore()
    agent = create_agent_from_vectorstore(
        vs, strategy=strategy, max_retries=max_retries
    )

    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "rewritten_query": "",
    }

    if no_stream:
        final = agent.invoke(
            initial_state, {"recursion_limit": 50}
        )
        print("\n=== Answer ===")
        print(final.get("generation", ""))
        return 0

    print(f"\n[ask] question: {question}")
    print(f"[ask] strategy: {strategy}")
    print("[ask] running agent (streaming state per step) ...\n")
    final_state = None
    for output in agent.stream(initial_state, {"recursion_limit": 50}):
        for node_name, state_delta in output.items():
            print(f"--- step: {node_name} ---")
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
            final_state = state_delta

    print("\n=== Final Answer ===")
    # The last state delta may not contain generation; query the full state.
    final = agent.invoke(initial_state, {"recursion_limit": 50}) if final_state is None else None
    if final is not None:
        print(final.get("generation", ""))
    else:
        # Re-derive from the streamed deltas: generation was printed above.
        print("(see generation printed above)")
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
            no_stream=args.no_stream,
        )
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
