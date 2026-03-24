"""Tests for document loaders."""

from __future__ import annotations

import tempfile
from pathlib import Path

from catchfly.loaders import load_glob, resolve_documents


class TestLoadGlob:
    def test_loads_text_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        docs = load_glob(str(tmp_path / "*.txt"))
        assert len(docs) == 2
        contents = {d.content for d in docs}
        assert contents == {"hello", "world"}

    def test_no_match_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            docs = load_glob(str(Path(d) / "*.xyz"))
            assert docs == []

    def test_document_fields(self, tmp_path: Path) -> None:
        (tmp_path / "doc.txt").write_text("content here")
        docs = load_glob(str(tmp_path / "*.txt"))
        assert docs[0].id == "doc.txt"
        assert docs[0].source is not None


class TestResolveDocuments:
    def test_passes_through_documents(self) -> None:
        from catchfly._types import Document

        docs = [Document(content="hi")]
        assert resolve_documents(docs) is docs

    def test_resolves_glob_strings(self, tmp_path: Path) -> None:
        (tmp_path / "x.txt").write_text("data")
        result = resolve_documents([str(tmp_path / "*.txt")])
        assert len(result) == 1
        assert result[0].content == "data"

    def test_empty_list(self) -> None:
        assert resolve_documents([]) == []
