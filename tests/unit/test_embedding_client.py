"""Tests for embedding client."""

from __future__ import annotations

from catchfly.providers.embeddings import OpenAIEmbeddingClient
from tests.conftest import MockEmbeddingClient


class TestMockEmbeddingClient:
    async def test_aembed(self) -> None:
        client = MockEmbeddingClient(dimensions=4)
        vectors = await client.aembed(["hello", "world"])

        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        assert len(vectors[1]) == 4

    async def test_deterministic(self) -> None:
        client = MockEmbeddingClient(dimensions=4)
        v1 = await client.aembed(["hello"])
        v2 = await client.aembed(["hello"])
        assert v1 == v2

    async def test_different_texts_different_vectors(self) -> None:
        client = MockEmbeddingClient(dimensions=8)
        vectors = await client.aembed(["hello", "world"])
        assert vectors[0] != vectors[1]

    async def test_records_calls(self) -> None:
        client = MockEmbeddingClient()
        await client.aembed(["a", "b"])
        await client.aembed(["c"])
        assert len(client.calls) == 2
        assert client.calls[0] == ["a", "b"]

    async def test_empty_input(self) -> None:
        client = MockEmbeddingClient()
        vectors = await client.aembed([])
        assert vectors == []


class TestOpenAIEmbeddingClient:
    def test_init_defaults(self) -> None:
        client = OpenAIEmbeddingClient()
        assert client.model == "text-embedding-3-small"
        assert client.batch_size == 2048

    def test_init_custom(self) -> None:
        client = OpenAIEmbeddingClient(
            model="nomic-embed",
            base_url="http://localhost:11434/v1",
            batch_size=512,
        )
        assert client.model == "nomic-embed"
        assert client.base_url == "http://localhost:11434/v1"

    def test_clear_cache(self) -> None:
        client = OpenAIEmbeddingClient()
        client._cache["test"] = [0.1, 0.2]
        client.clear_cache()
        assert len(client._cache) == 0
