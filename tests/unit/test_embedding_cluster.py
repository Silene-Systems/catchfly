"""Tests for EmbeddingClustering normalization."""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from catchfly.exceptions import NormalizationError
from catchfly.normalization.embedding_cluster import EmbeddingClustering


def _make_clustered_embeddings(groups: list[list[str]], dim: int = 8) -> dict[str, list[float]]:
    """Create embeddings where texts in the same group are close together."""
    embeddings: dict[str, list[float]] = {}
    for group_idx, group in enumerate(groups):
        center = np.zeros(dim)
        center[group_idx % dim] = 1.0  # each group in a different direction
        for _i, text in enumerate(group):
            # Small perturbation within group
            vec = center + np.random.default_rng(hash(text) % 2**31).normal(0, 0.05, dim)
            embeddings[text] = vec.tolist()
    return embeddings


class TestEmbeddingClustering:
    async def test_empty_values(self) -> None:
        normalizer = EmbeddingClustering()
        result = await normalizer.anormalize([], context_field="test")
        assert result.mapping == {}

    async def test_single_value(self) -> None:
        normalizer = EmbeddingClustering()
        result = await normalizer.anormalize(["hello"], context_field="test")
        assert result.mapping == {"hello": "hello"}

    async def test_clusters_similar_values(self) -> None:
        """Values with similar embeddings should cluster together."""
        groups = [
            ["New York", "NYC", "NY"],
            ["Los Angeles", "LA", "L.A."],
        ]
        mock_embeddings = _make_clustered_embeddings(groups)

        async def mock_embed(texts: list[str]) -> list[list[float]]:
            return [mock_embeddings[t] for t in texts]

        normalizer = EmbeddingClustering(
            clustering_algorithm="agglomerative",
            similarity_threshold=0.5,
            reduce_dimensions=False,
        )

        with patch.object(normalizer, "_get_embeddings", side_effect=mock_embed):
            all_values = [v for group in groups for v in group]
            result = await normalizer.anormalize(all_values, context_field="city")

        # All NY variants should map to the same canonical
        ny_canonical = result.mapping["New York"]
        assert result.mapping["NYC"] == ny_canonical
        assert result.mapping["NY"] == ny_canonical

        # All LA variants should map to the same canonical
        la_canonical = result.mapping["Los Angeles"]
        assert result.mapping["LA"] == la_canonical

        # NY and LA should have different canonicals
        assert ny_canonical != la_canonical

    async def test_explain(self) -> None:
        groups = [["cat", "cats", "kitten"]]
        mock_embeddings = _make_clustered_embeddings(groups)

        async def mock_embed(texts: list[str]) -> list[list[float]]:
            return [mock_embeddings[t] for t in texts]

        normalizer = EmbeddingClustering(
            clustering_algorithm="agglomerative",
            similarity_threshold=0.5,
            reduce_dimensions=False,
        )

        with patch.object(normalizer, "_get_embeddings", side_effect=mock_embed):
            result = await normalizer.anormalize(["cat", "cats", "kitten"], context_field="animal")

        explanation = result.explain("cats")
        assert "cats" in explanation

    async def test_unknown_algorithm_raises(self) -> None:
        normalizer = EmbeddingClustering(
            clustering_algorithm="unknown",
            reduce_dimensions=False,
        )

        async def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 8 for _ in texts]

        with (
            patch.object(normalizer, "_get_embeddings", side_effect=mock_embed),
            pytest.raises(NormalizationError, match="Unknown clustering"),
        ):
            await normalizer.anormalize(["a", "b"], context_field="test")

    async def test_frequency_selects_canonical(self) -> None:
        """Most frequent value should become canonical."""
        groups = [["NY", "New York", "NYC"]]
        mock_embeddings = _make_clustered_embeddings(groups)

        async def mock_embed(texts: list[str]) -> list[list[float]]:
            return [mock_embeddings[t] for t in texts]

        normalizer = EmbeddingClustering(
            clustering_algorithm="agglomerative",
            similarity_threshold=0.5,
            reduce_dimensions=False,
        )

        # "New York" appears 5 times, others once
        values = ["NY", "New York", "New York", "New York", "New York", "New York", "NYC"]
        with patch.object(normalizer, "_get_embeddings", side_effect=mock_embed):
            result = await normalizer.anormalize(values, context_field="city")

        # Most frequent "New York" should be canonical
        assert result.mapping["NY"] == "New York"
        assert result.mapping["NYC"] == "New York"

    def test_build_result_noise_points(self) -> None:
        """HDBSCAN noise points (label=-1) should map to themselves."""
        unique = ["alpha", "beta", "gamma"]
        counts: Counter[str] = Counter({"alpha": 1, "beta": 1, "gamma": 1})
        labels = np.array([-1, -1, 0])  # alpha, beta are noise; gamma is cluster 0

        result = EmbeddingClustering._build_result(unique, counts, labels, "test")

        assert result.mapping["alpha"] == "alpha"
        assert result.mapping["beta"] == "beta"
        assert "unclustered" in result.metadata["explanations"]["alpha"]

    def test_build_result_multiple_clusters(self) -> None:
        unique = ["a", "b", "c", "d"]
        counts: Counter[str] = Counter({"a": 3, "b": 1, "c": 2, "d": 1})
        labels = np.array([0, 0, 1, 1])

        result = EmbeddingClustering._build_result(unique, counts, labels, "test")

        # Cluster 0: a(3), b(1) → canonical=a
        assert result.mapping["a"] == "a"
        assert result.mapping["b"] == "a"
        # Cluster 1: c(2), d(1) → canonical=c
        assert result.mapping["c"] == "c"
        assert result.mapping["d"] == "c"
        assert result.metadata["n_clusters"] == 2
