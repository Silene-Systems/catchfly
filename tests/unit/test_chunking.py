"""Tests for document chunking."""

from __future__ import annotations

from catchfly._types import Document
from catchfly.extraction.chunking import chunk_document, estimate_chunks


class TestChunkDocument:
    def test_short_doc_unchanged(self) -> None:
        doc = Document(content="short text", id="doc1")
        chunks = chunk_document(doc, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] is doc

    def test_splits_long_doc(self) -> None:
        doc = Document(content="a" * 1000, id="doc1")
        chunks = chunk_document(doc, chunk_size=400, overlap=100)
        assert len(chunks) > 1
        # Each chunk should be at most chunk_size
        for chunk in chunks:
            assert len(chunk.content) <= 400

    def test_overlap_preserved(self) -> None:
        doc = Document(content="abcdefghij" * 10, id="doc1")  # 100 chars
        chunks = chunk_document(doc, chunk_size=30, overlap=10)
        # Check that consecutive chunks overlap
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i].content[-10:]
            start_of_next = chunks[i + 1].content[:10]
            assert end_of_current == start_of_next

    def test_preserves_offsets(self) -> None:
        content = "hello world this is a test document with some content"
        doc = Document(content=content, id="doc1")
        chunks = chunk_document(doc, chunk_size=20, overlap=5)

        for chunk in chunks:
            char_start = chunk.metadata["char_start"]
            char_end = chunk.metadata["char_end"]
            assert content[char_start:char_end] == chunk.content

    def test_preserves_metadata(self) -> None:
        doc = Document(content="a" * 100, id="doc1", metadata={"lang": "en"})
        chunks = chunk_document(doc, chunk_size=30, overlap=5)
        for chunk in chunks:
            assert chunk.metadata["lang"] == "en"
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["parent_id"] == "doc1"

    def test_chunk_ids(self) -> None:
        doc = Document(content="a" * 100, id="doc1")
        chunks = chunk_document(doc, chunk_size=30, overlap=5)
        ids = [c.id for c in chunks]
        assert ids[0] == "doc1__chunk_0"
        assert ids[1] == "doc1__chunk_1"

    def test_covers_entire_document(self) -> None:
        content = "abcdefghijklmnopqrstuvwxyz" * 4  # 104 chars
        doc = Document(content=content, id="doc1")
        chunks = chunk_document(doc, chunk_size=30, overlap=5)
        # Reconstruct — every char in original should be in at least one chunk
        covered = set()
        for chunk in chunks:
            start = chunk.metadata["char_start"]
            end = chunk.metadata["char_end"]
            covered.update(range(start, end))
        assert covered == set(range(len(content)))


class TestEstimateChunks:
    def test_short_docs(self) -> None:
        docs = [Document(content="short", id=str(i)) for i in range(5)]
        assert estimate_chunks(docs, chunk_size=100) == 5

    def test_long_docs(self) -> None:
        docs = [Document(content="a" * 1000, id="doc1")]
        count = estimate_chunks(docs, chunk_size=400, overlap=100)
        # Verify against actual chunking
        actual = len(chunk_document(docs[0], chunk_size=400, overlap=100))
        assert count == actual

    def test_empty(self) -> None:
        assert estimate_chunks([], chunk_size=100) == 0
