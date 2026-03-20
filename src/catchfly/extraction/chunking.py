"""Document chunking for long-document extraction."""

from __future__ import annotations

import logging

from catchfly._types import Document

logger = logging.getLogger(__name__)


def chunk_document(
    doc: Document,
    chunk_size: int = 4000,
    overlap: int = 200,
) -> list[Document]:
    """Split a document into overlapping chunks.

    Each chunk preserves char_start/char_end offsets relative to the
    original document via metadata, enabling provenance tracking.

    Returns the original document unchanged if it fits within chunk_size.
    """
    content = doc.content

    if len(content) <= chunk_size:
        return [doc]

    chunks: list[Document] = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunk_text = content[start:end]

        chunk_meta = {
            **doc.metadata,
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
            "parent_id": doc.id,
        }

        chunks.append(
            Document(
                content=chunk_text,
                id=f"{doc.id or 'doc'}__chunk_{chunk_index}",
                source=doc.source,
                metadata=chunk_meta,
            )
        )

        # Advance by chunk_size - overlap, but at least 1 char to avoid infinite loop
        step = max(chunk_size - overlap, 1)
        start += step
        chunk_index += 1

    logger.debug(
        "Chunked document '%s' (%d chars) into %d chunks (size=%d, overlap=%d)",
        doc.id,
        len(content),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


def estimate_chunks(
    documents: list[Document],
    chunk_size: int = 4000,
    overlap: int = 200,
) -> int:
    """Estimate total number of chunks across all documents."""
    total = 0
    step = max(chunk_size - overlap, 1)
    for doc in documents:
        length = len(doc.content)
        if length <= chunk_size:
            total += 1
        else:
            # Chunks start at 0, step, 2*step, ... while start < length
            total += (length - 1) // step + 1
    return total
