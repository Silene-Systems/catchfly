"""Tests for KLLMeansClustering."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np

from catchfly.normalization.kllmeans import KLLMeansClustering
from catchfly.providers.llm import LLMResponse


class MockKLLMeansLLM:
    """Mock LLM for cluster summarization."""

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        user_msg = messages[-1]["content"] if messages else ""
        # Extract first member from the cluster list
        lines = [ln.strip("- ").strip() for ln in user_msg.split("\n") if ln.startswith("-")]
        name = lines[0] if lines else "cluster"
        return LLMResponse(content=name, input_tokens=50, output_tokens=10)

    async def astructured_complete(
        self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kwargs: Any
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class MockKLLMeansEmbedder:
    """Mock embedder that maps texts to deterministic vectors."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(self.dim)]


def _make_clustered_embedder(groups: dict[str, list[str]], dim: int = 8) -> MockKLLMeansEmbedder:
    """Embedder where values in same group produce nearby vectors."""

    class ClusteredEmbedder(MockKLLMeansEmbedder):
        def __init__(self) -> None:
            super().__init__(dim)
            self._vectors: dict[str, list[float]] = {}
            rng = np.random.default_rng(42)
            for i, (canonical, members) in enumerate(groups.items()):
                center = np.zeros(dim)
                center[i % dim] = 1.0
                for m in members:
                    vec = center + rng.normal(0, 0.02, dim)
                    self._vectors[m] = vec.tolist()
                self._vectors[canonical] = center.tolist()

        async def aembed(self, texts: list[str]) -> list[list[float]]:
            fallback = self._vec
            return [self._vectors.get(t, fallback(t)) for t in texts]

    return ClusteredEmbedder()


class TestKLLMeansClustering:
    async def test_empty_values(self) -> None:
        normalizer = KLLMeansClustering()
        result = await normalizer.anormalize([])
        assert result.mapping == {}

    async def test_single_value(self) -> None:
        normalizer = KLLMeansClustering()
        result = await normalizer.anormalize(["hello"])
        assert result.mapping == {"hello": "hello"}

    async def test_two_values(self) -> None:
        normalizer = KLLMeansClustering()
        result = await normalizer.anormalize(["a", "b"])
        assert len(result.mapping) == 2

    async def test_basic_clustering(self) -> None:
        """Values with similar embeddings cluster together."""
        groups = {
            "New York": ["NYC", "NY", "New York"],
            "Los Angeles": ["LA", "L.A.", "Los Angeles"],
        }
        embedder = _make_clustered_embedder(groups)
        mock_llm = MockKLLMeansLLM()

        normalizer = KLLMeansClustering(
            num_clusters=2,
            num_iterations=5,
            summarize_every=2,
        )

        embed_patch = "catchfly.normalization.kllmeans.OpenAIEmbeddingClient"
        llm_patch = "catchfly.normalization.kllmeans.OpenAICompatibleClient"
        with (
            patch(embed_patch, return_value=embedder),
            patch(llm_patch, return_value=mock_llm),
        ):
            all_values = [v for members in groups.values() for v in members]
            result = await normalizer.anormalize(all_values, context_field="city")

        assert len(result.mapping) == 6
        # NY variants should map to same canonical
        ny_canonical = result.mapping["NYC"]
        assert result.mapping["NY"] == ny_canonical
        assert result.mapping["New York"] == ny_canonical

    async def test_schema_seeded_init(self) -> None:
        """seed_from_schema uses field_metadata descriptions as initial centroids."""
        groups = {"Cat": ["cat", "cats", "kitten"]}
        embedder = _make_clustered_embedder(groups)
        mock_llm = MockKLLMeansLLM()

        normalizer = KLLMeansClustering(
            num_clusters=1,
            seed_from_schema=True,
            num_iterations=3,
            summarize_every=10,  # no LLM summary during iterations
        )

        field_metadata = {
            "description": "Type of animal",
            "examples": ["cat", "dog"],
            "synonyms": ["feline", "canine"],
        }

        with (
            patch("catchfly.normalization.kllmeans.OpenAIEmbeddingClient", return_value=embedder),
            patch("catchfly.normalization.kllmeans.OpenAICompatibleClient", return_value=mock_llm),
        ):
            result = await normalizer.anormalize(
                ["cat", "cats", "kitten"],
                context_field="animal",
                field_metadata=field_metadata,
            )

        assert len(result.mapping) == 3
        # All should map to same canonical
        assert len(set(result.mapping.values())) == 1

    async def test_convergence(self) -> None:
        """Algorithm should converge (assignments stabilize)."""
        groups = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}
        embedder = _make_clustered_embedder(groups, dim=4)
        mock_llm = MockKLLMeansLLM()

        normalizer = KLLMeansClustering(
            num_clusters=2,
            num_iterations=20,
            summarize_every=100,  # disable LLM summaries
        )

        with (
            patch("catchfly.normalization.kllmeans.OpenAIEmbeddingClient", return_value=embedder),
            patch("catchfly.normalization.kllmeans.OpenAICompatibleClient", return_value=mock_llm),
        ):
            result = await normalizer.anormalize(
                ["a1", "a2", "a3", "b1", "b2", "b3"],
                context_field="test",
            )

        assert len(result.mapping) == 6
        assert result.metadata["k"] == 2

    def test_kmeans_pp_init(self) -> None:
        embeddings = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        centroids = KLLMeansClustering._kmeans_pp_init(embeddings, 2)
        assert centroids.shape == (2, 2)

    def test_assign(self) -> None:
        embeddings = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        centroids = np.array([[0, 0], [1, 1]], dtype=np.float64)
        assignments = KLLMeansClustering._assign(embeddings, centroids)
        assert assignments[0] == 0  # [0,0] closest to [0,0]
        assert assignments[3] == 1  # [1,1] closest to [1,1]

    def test_update_centroids(self) -> None:
        embeddings = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float64)
        assignments = np.array([0, 0, 1, 1])
        centroids = np.array([[0, 0], [0, 0]], dtype=np.float64)
        updated = KLLMeansClustering._update_centroids(embeddings, assignments, 2, centroids)
        np.testing.assert_array_almost_equal(updated[0], [1, 0])
        np.testing.assert_array_almost_equal(updated[1], [1, 2])

    def test_build_result(self) -> None:
        from collections import Counter

        values = ["a", "b", "c", "d"]
        counts = Counter(values)
        assignments = np.array([0, 0, 1, 1])
        names = ["Group A", "Group B"]

        result = KLLMeansClustering._build_result(values, counts, assignments, names, 2, "test")

        assert result.mapping["a"] == "Group A"
        assert result.mapping["c"] == "Group B"
        assert result.metadata["k"] == 2

    def test_trivial_result(self) -> None:
        result = KLLMeansClustering._trivial_result(["x", "y"], "f")
        assert result.mapping == {"x": "x", "y": "y"}

    def test_sync_wrapper(self) -> None:
        normalizer = KLLMeansClustering()
        result = normalizer.normalize(["a"], context_field="test")
        assert result.mapping == {"a": "a"}
