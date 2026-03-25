"""Tests for embedding client."""

from __future__ import annotations

from catchfly.providers.embeddings import OpenAIEmbeddingClient


class TestOpenAIEmbeddingClient:
    def test_clear_cache(self) -> None:
        client = OpenAIEmbeddingClient()
        client._cache["test"] = [0.1, 0.2]
        client.clear_cache()
        assert len(client._cache) == 0
