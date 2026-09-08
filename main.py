#!/usr/bin/env python
"""Entry point for the RAG knowledge agent CLI.

Usage
-----
::

    python main.py ingest --dir ./data
    python main.py ask "什么是 LangGraph 的条件边？" --strategy multi_query

This shim makes sure the ``src`` directory is on ``sys.path`` so the package
can be run directly from a source checkout (without an editable install).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
