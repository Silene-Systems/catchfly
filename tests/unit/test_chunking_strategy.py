"""Tests for ChunkingStrategy protocol and FixedSizeChunking."""

from __future__ import annotations

from catchfly._types import Document
from catchfly.extraction.chunking_fixed import FixedSizeChunking
from catchfly.extraction.chunking_strategy import ChunkingStrategy


class TestChunkingStrategyProtocol:
    def test_fixed_size_satisfies_protocol(self) -> None:
        assert isinstance(FixedSizeChunking(), ChunkingStrategy)

    def test_protocol_rejects_non_conforming(self) -> None:
        assert not isinstance("not a strategy", ChunkingStrategy)


class TestFixedSizeChunking:
    def test_short_doc_unchanged(self) -> None:
        strategy = FixedSizeChunking(chunk_size=100)
        doc = Document(content="short text", id="doc1")
        chunks = strategy.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0] is doc

    def test_splits_long_doc(self) -> None:
        strategy = FixedSizeChunking(chunk_size=400, overlap=100)
        doc = Document(content="a" * 1000, id="doc1")
        chunks = strategy.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 400

    def test_preserves_offsets(self) -> None:
        content = "hello world this is a test document with some content here"
        strategy = FixedSizeChunking(chunk_size=20, overlap=5)
        doc = Document(content=content, id="doc1")
        chunks = strategy.chunk(doc)
        for chunk in chunks:
            cs = chunk.metadata["char_start"]
            ce = chunk.metadata["char_end"]
            assert content[cs:ce] == chunk.content

    def test_preserves_metadata(self) -> None:
        strategy = FixedSizeChunking(chunk_size=30, overlap=5)
        doc = Document(content="a" * 100, id="doc1", metadata={"lang": "en"})
        chunks = strategy.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["lang"] == "en"
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["parent_id"] == "doc1"

    def test_chunk_batch(self) -> None:
        strategy = FixedSizeChunking(chunk_size=50, overlap=10)
        docs = [Document(content="a" * 200, id=f"doc{i}") for i in range(3)]
        all_chunks = strategy.chunk_batch(docs)
        assert len(all_chunks) > 3

    def test_estimate_matches_actual(self) -> None:
        strategy = FixedSizeChunking(chunk_size=400, overlap=100)
        docs = [Document(content="a" * 1000, id="doc1")]
        estimated = strategy.estimate_chunks(docs)
        actual = len(strategy.chunk_batch(docs))
        assert estimated == actual

    def test_estimate_short_docs(self) -> None:
        strategy = FixedSizeChunking(chunk_size=100)
        docs = [Document(content="short", id=str(i)) for i in range(5)]
        assert strategy.estimate_chunks(docs) == 5

    def test_estimate_empty(self) -> None:
        strategy = FixedSizeChunking()
        assert strategy.estimate_chunks([]) == 0

    def test_default_config(self) -> None:
        strategy = FixedSizeChunking()
        assert strategy.chunk_size == 4000
        assert strategy.overlap == 200
