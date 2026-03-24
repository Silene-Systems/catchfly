"""Fixed-size character chunking strategy — wraps existing chunk_document()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from catchfly.extraction.chunking import chunk_document, estimate_chunks

if TYPE_CHECKING:
    from catchfly._types import Document


class FixedSizeChunking(BaseModel):
    """Fixed-size character chunking with overlap.

    Wraps the existing ``chunk_document()`` function as a
    ``ChunkingStrategy``-compatible class. This is the default
    fallback when no explicit chunking strategy is provided.
    """

    chunk_size: int = 4000
    overlap: int = 200

    def chunk(self, document: Document) -> list[Document]:
        """Split a document into fixed-size character chunks with overlap."""
        return chunk_document(document, self.chunk_size, self.overlap)

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents, returning all chunks flattened."""
        return [c for doc in documents for c in self.chunk(doc)]

    def estimate_chunks(self, documents: list[Document]) -> int:
        """Estimate total chunk count using fast arithmetic."""
        return estimate_chunks(documents, self.chunk_size, self.overlap)
