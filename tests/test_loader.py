"""Tests for the document loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.loader import load_directory, load_file, SUPPORTED_SUFFIXES


def test_load_txt_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world\nsecond line", encoding="utf-8")
    docs = load_file(str(f))
    assert len(docs) == 1
    assert "hello world" in docs[0].page_content
    assert docs[0].metadata["source"] == str(f)


def test_load_md_file(tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome markdown content.", encoding="utf-8")
    docs = load_file(str(f))
    assert len(docs) == 1
    assert "Title" in docs[0].page_content
    assert docs[0].metadata["source"] == str(f)


def test_load_unsupported_extension_raises(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(str(f))


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_file("/nonexistent/path/does_not_exist.txt")


def test_load_directory_recursive(tmp_path):
    (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("beta content", encoding="utf-8")
    # An unsupported file should be silently skipped, not crash.
    (sub / "ignore.csv").write_text("x,y", encoding="utf-8")

    docs = load_directory(str(tmp_path))
    sources = {d.metadata["source"] for d in docs}
    assert any(s.endswith("a.txt") for s in sources)
    assert any(s.endswith("b.md") for s in sources)
    assert all(not s.endswith("ignore.csv") for s in sources)
    assert len(docs) == 2


def test_load_directory_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_directory("/nonexistent/directory/here")


def test_supported_suffixes_set():
    assert SUPPORTED_SUFFIXES == {".txt", ".md", ".pdf"}
