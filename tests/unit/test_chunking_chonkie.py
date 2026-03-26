"""Tests for chonkie-backed chunking strategies."""

from __future__ import annotations

import pytest

from catchfly._types import Document
from catchfly.extraction.chunking_strategy import ChunkingStrategy

chonkie = pytest.importorskip("chonkie")


def _has_recipe_deps() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        import jsonschema  # noqa: F401

        return True
    except ImportError:
        return False


from catchfly.extraction.chunking_chonkie import (  # noqa: E402
    RecursiveChunking,
    SentenceChunking,
    TokenChunking,
)


def _assert_offset_invariant(content: str, chunks: list[Document]) -> None:
    """Verify that char_start/char_end map back to the chunk content."""
    for chunk in chunks:
        cs = chunk.metadata["char_start"]
        ce = chunk.metadata["char_end"]
        assert content[cs:ce] == chunk.content, (
            f"Offset invariant violated for chunk {chunk.id}: "
            f"content[{cs}:{ce}] = {content[cs:ce]!r} != {chunk.content!r}"
        )


class TestTokenChunking:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(TokenChunking(), ChunkingStrategy)

    def test_short_doc_unchanged(self) -> None:
        doc = Document(content="short text", id="doc1")
        strategy = TokenChunking(chunk_size=512)
        chunks = strategy.chunk(doc)
        assert len(chunks) == 1

    def test_splits_long_doc(self) -> None:
        content = "This is a test sentence. " * 200
        doc = Document(content=content, id="doc1")
        strategy = TokenChunking(chunk_size=64, overlap=8)
        chunks = strategy.chunk(doc)
        assert len(chunks) > 1

    def test_preserves_offsets(self) -> None:
        content = "This is a test sentence. " * 200
        doc = Document(content=content, id="doc1")
        strategy = TokenChunking(chunk_size=64, overlap=8)
        chunks = strategy.chunk(doc)
        _assert_offset_invariant(content, chunks)

    def test_preserves_metadata(self) -> None:
        content = "Hello world. " * 200
        doc = Document(content=content, id="doc1", metadata={"lang": "en"})
        strategy = TokenChunking(chunk_size=64, overlap=8)
        chunks = strategy.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["lang"] == "en"
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["parent_id"] == "doc1"

    def test_chunk_batch(self) -> None:
        docs = [Document(content="Some text here. " * 100, id=f"doc{i}") for i in range(3)]
        strategy = TokenChunking(chunk_size=64, overlap=8)
        all_chunks = strategy.chunk_batch(docs)
        assert len(all_chunks) >= 3

    def test_estimate_matches_actual(self) -> None:
        docs = [Document(content="Test sentence. " * 100, id="doc1")]
        strategy = TokenChunking(chunk_size=64, overlap=8)
        estimated = strategy.estimate_chunks(docs)
        actual = len(strategy.chunk_batch(docs))
        assert estimated == actual


class TestSentenceChunking:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(SentenceChunking(), ChunkingStrategy)

    def test_preserves_offsets(self) -> None:
        content = (
            "This is sentence one. This is sentence two. "
            "And here is a third sentence. The fourth follows. "
            "A fifth to be safe. Sixth sentence here. "
            "Seventh for good measure. Eighth keeps going. "
        ) * 5
        doc = Document(content=content, id="doc1")
        strategy = SentenceChunking(chunk_size=64, overlap=8)
        chunks = strategy.chunk(doc)
        _assert_offset_invariant(content, chunks)

    def test_metadata_preserved(self) -> None:
        content = "First sentence. Second sentence. Third sentence. " * 50
        doc = Document(content=content, id="doc1", metadata={"source_type": "report"})
        strategy = SentenceChunking(chunk_size=64, overlap=8)
        chunks = strategy.chunk(doc)
        for c in chunks:
            assert c.metadata["source_type"] == "report"
            assert "chunk_index" in c.metadata
            assert c.metadata["parent_id"] == "doc1"


class TestRecursiveChunking:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(RecursiveChunking(), ChunkingStrategy)

    @pytest.mark.skipif(
        not _has_recipe_deps(),
        reason="huggingface_hub and jsonschema required for recipe-based chunking",
    )
    def test_markdown_recipe(self) -> None:
        md_text = (
            "# Section 1\n\n"
            "Paragraph one with some content here.\n\n"
            "# Section 2\n\n"
            "Paragraph two with different content.\n\n"
            "## Subsection 2.1\n\n"
            "More detailed content in subsection.\n\n"
        ) * 5
        doc = Document(content=md_text, id="doc1")
        strategy = RecursiveChunking(recipe="markdown", chunk_size=64)
        chunks = strategy.chunk(doc)
        assert len(chunks) >= 1
        _assert_offset_invariant(md_text, chunks)

    def test_no_recipe(self) -> None:
        content = "Simple text. " * 200
        doc = Document(content=content, id="doc1")
        strategy = RecursiveChunking(recipe=None, chunk_size=64)
        chunks = strategy.chunk(doc)
        assert len(chunks) >= 1
        _assert_offset_invariant(content, chunks)

    def test_chunk_ids(self) -> None:
        content = "Some content here. " * 100
        doc = Document(content=content, id="report1")
        strategy = RecursiveChunking(chunk_size=64)
        chunks = strategy.chunk(doc)
        if len(chunks) > 1:
            assert chunks[0].id == "report1__chunk_0"
            assert chunks[1].id == "report1__chunk_1"
