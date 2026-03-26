"""Chunking strategies backed by the chonkie library."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from catchfly._types import Document

logger = logging.getLogger(__name__)


def _import_chonkie(class_name: str) -> Any:
    """Lazy-import a chonkie chunker class with a helpful error message."""
    try:
        import chonkie  # noqa: F811
    except ImportError as e:
        raise ImportError(
            f"chonkie is required for {class_name}-based chunking. "
            "Install it with: pip install 'catchfly[chunking]'"
        ) from e
    return getattr(chonkie, class_name)


def _import_semantic_chunker() -> Any:
    """Lazy-import chonkie's SemanticChunker (requires chonkie[semantic])."""
    try:
        from chonkie import SemanticChunker
    except ImportError as e:
        raise ImportError(
            "chonkie[semantic] is required for SemanticChunking. "
            "Install it with: pip install 'catchfly[semantic-chunking]'"
        ) from e
    return SemanticChunker


def _chunks_to_documents(
    chunks: list[Any],
    parent: Document,
) -> list[Document]:
    """Convert chonkie Chunk objects to catchfly Documents with provenance metadata."""
    results: list[Document] = []
    for i, chunk in enumerate(chunks):
        meta = {
            **parent.metadata,
            "chunk_index": i,
            "char_start": chunk.start_index,
            "char_end": chunk.end_index,
            "parent_id": parent.id,
        }
        results.append(
            Document(
                content=chunk.text,
                id=f"{parent.id or 'doc'}__chunk_{i}",
                source=parent.source,
                metadata=meta,
            )
        )
    return results


class TokenChunking(BaseModel):
    """Token-based fixed-size chunking via chonkie.

    Splits documents into chunks of a fixed number of tokens,
    with configurable overlap.
    """

    chunk_size: int = 512
    overlap: int = 64

    def chunk(self, document: Document) -> list[Document]:
        """Split a document into token-based chunks."""
        TokenChunker = _import_chonkie("TokenChunker")
        chunker = TokenChunker(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        chunks = chunker(document.content)
        if not chunks:
            return [document]
        return _chunks_to_documents(chunks, document)

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents, returning all chunks flattened."""
        return [c for doc in documents for c in self.chunk(doc)]

    def estimate_chunks(self, documents: list[Document]) -> int:
        """Estimate total chunk count by performing actual chunking."""
        return sum(len(self.chunk(doc)) for doc in documents)


class SentenceChunking(BaseModel):
    """Sentence-boundary chunking via chonkie.

    Splits documents at natural sentence boundaries, grouping
    sentences into chunks up to the specified token size.
    """

    chunk_size: int = 512
    overlap: int = 64

    def chunk(self, document: Document) -> list[Document]:
        """Split a document at sentence boundaries."""
        SentenceChunker = _import_chonkie("SentenceChunker")
        chunker = SentenceChunker(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        chunks = chunker(document.content)
        if not chunks:
            return [document]
        return _chunks_to_documents(chunks, document)

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents, returning all chunks flattened."""
        return [c for doc in documents for c in self.chunk(doc)]

    def estimate_chunks(self, documents: list[Document]) -> int:
        """Estimate total chunk count by performing actual chunking."""
        return sum(len(self.chunk(doc)) for doc in documents)


class RecursiveChunking(BaseModel):
    """Hierarchical recursive chunking via chonkie.

    Splits documents using a hierarchy of delimiters (e.g. markdown
    headers, paragraphs, sentences). Ideal for structured documents
    like clinical case reports or research papers.
    """

    chunk_size: int = 512
    recipe: str | None = None

    def chunk(self, document: Document) -> list[Document]:
        """Split a document using hierarchical delimiter rules."""
        RecursiveChunker = _import_chonkie("RecursiveChunker")
        if self.recipe:
            chunker = RecursiveChunker.from_recipe(self.recipe, chunk_size=self.chunk_size)
        else:
            chunker = RecursiveChunker(chunk_size=self.chunk_size)
        chunks = chunker(document.content)
        if not chunks:
            return [document]
        return _chunks_to_documents(chunks, document)

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents, returning all chunks flattened."""
        return [c for doc in documents for c in self.chunk(doc)]

    def estimate_chunks(self, documents: list[Document]) -> int:
        """Estimate total chunk count by performing actual chunking."""
        return sum(len(self.chunk(doc)) for doc in documents)


class SemanticChunking(BaseModel):
    """Embedding-based semantic chunking via chonkie.

    Splits documents at semantic boundaries by measuring
    embedding similarity between consecutive sentences. Requires
    ``chonkie[semantic]`` (install via ``pip install 'catchfly[semantic-chunking]'``).
    """

    threshold: float = 0.5
    model_name: str = "minishlab/potion-base-8M"

    def chunk(self, document: Document) -> list[Document]:
        """Split a document at semantic boundaries."""
        SemanticChunker = _import_semantic_chunker()
        chunker = SemanticChunker(
            threshold=self.threshold,
            embedding_model=self.model_name,
        )
        chunks = chunker(document.content)
        if not chunks:
            return [document]
        return _chunks_to_documents(chunks, document)

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents, returning all chunks flattened."""
        return [c for doc in documents for c in self.chunk(doc)]

    def estimate_chunks(self, documents: list[Document]) -> int:
        """Estimate total chunk count by performing actual chunking."""
        return sum(len(self.chunk(doc)) for doc in documents)
