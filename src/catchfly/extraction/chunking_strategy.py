"""Chunking strategy protocol for pluggable document splitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from catchfly._types import Document


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Protocol for document chunking strategies.

    Each returned Document must have metadata keys:
    ``chunk_index``, ``char_start``, ``char_end``, ``parent_id``.
    """

    def chunk(self, document: Document) -> list[Document]:
        """Split a single document into chunks."""
        ...

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents, returning all chunks flattened."""
        ...

    def estimate_chunks(self, documents: list[Document]) -> int:
        """Estimate total chunk count without performing full chunking."""
        ...
