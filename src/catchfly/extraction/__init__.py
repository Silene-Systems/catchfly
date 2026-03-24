"""Data extraction strategies."""

from __future__ import annotations

from typing import Any

from catchfly.extraction.base import ExtractionStrategy
from catchfly.extraction.chunking import chunk_document, estimate_chunks
from catchfly.extraction.chunking_fixed import FixedSizeChunking
from catchfly.extraction.chunking_strategy import ChunkingStrategy
from catchfly.extraction.llm_direct import LLMDirectExtraction

__all__ = [
    "ChunkingStrategy",
    "ExtractionStrategy",
    "FixedSizeChunking",
    "LLMDirectExtraction",
    "RecursiveChunking",
    "SemanticChunking",
    "SentenceChunking",
    "TokenChunking",
    "chunk_document",
    "estimate_chunks",
]


def __getattr__(name: str) -> Any:
    _chonkie_strategies = {
        "TokenChunking",
        "SentenceChunking",
        "RecursiveChunking",
        "SemanticChunking",
    }
    if name in _chonkie_strategies:
        from catchfly.extraction import chunking_chonkie

        return getattr(chunking_chonkie, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
