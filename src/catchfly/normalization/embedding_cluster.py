"""EmbeddingClustering — normalize values via embedding + clustering."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from pydantic import BaseModel

from catchfly._compat import run_sync
from catchfly._types import NormalizationResult
from catchfly.exceptions import NormalizationError

logger = logging.getLogger(__name__)


def _import_numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError as e:
        raise ImportError(
            "numpy is required for EmbeddingClustering. "
            "Install it with: pip install catchfly[clustering]"
        ) from e


def _import_sklearn_hdbscan() -> Any:
    try:
        from sklearn.cluster import HDBSCAN  # type: ignore[import-untyped]

        return HDBSCAN
    except ImportError as e:
        raise ImportError(
            "scikit-learn>=1.3 is required for HDBSCAN clustering. "
            "Install it with: pip install catchfly[clustering]"
        ) from e


def _import_sklearn_agglomerative() -> Any:
    try:
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for agglomerative clustering. "
            "Install it with: pip install catchfly[clustering]"
        ) from e


def _import_umap() -> Any:
    try:
        import umap  # type: ignore[import-untyped]

        return umap
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for dimensionality reduction. "
            "Install it with: pip install catchfly[clustering]"
        ) from e


class EmbeddingClustering(BaseModel):
    """Normalize values by embedding them and clustering similar ones.

    Process:
    1. Deduplicate input values
    2. Embed all unique values
    3. Optional: UMAP dimensionality reduction (for high-dim embeddings)
    4. Cluster via HDBSCAN or agglomerative clustering
    5. Select canonical name per cluster (most frequent value)
    """

    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.8
    clustering_algorithm: str = "hdbscan"
    min_cluster_size: int = 2
    reduce_dimensions: bool = True
    umap_n_components: int = 32
    base_url: str | None = None
    api_key: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values asynchronously."""
        if not values:
            return NormalizationResult(mapping={})

        np = _import_numpy()

        # Deduplicate while preserving frequency info
        value_counts = Counter(values)
        unique_values = list(value_counts.keys())

        if len(unique_values) == 1:
            canonical = unique_values[0]
            return NormalizationResult(
                mapping={canonical: canonical},
                clusters={canonical: [canonical]},
                metadata={"strategy": "embedding_clustering", "field": context_field},
            )

        logger.info(
            "EmbeddingClustering: normalizing %d values (%d unique) for field '%s'",
            len(values),
            len(unique_values),
            context_field,
        )

        # Embed
        embeddings = await self._get_embeddings(unique_values)
        embedding_matrix = np.array(embeddings)

        # Optional dimensionality reduction
        if self.reduce_dimensions and embedding_matrix.shape[1] > self.umap_n_components:
            embedding_matrix = self._reduce_dims(embedding_matrix)

        # Cluster
        labels = self._cluster(embedding_matrix, len(unique_values))

        # Build mapping
        return self._build_result(unique_values, value_counts, labels, context_field)

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values synchronously."""
        return run_sync(self.anormalize(values, context_field=context_field, **kwargs))

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings via the configured provider."""
        from catchfly.providers.embeddings import OpenAIEmbeddingClient

        client = OpenAIEmbeddingClient(
            model=self.embedding_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return await client.aembed(texts)

    def _reduce_dims(self, matrix: Any) -> Any:
        """Reduce dimensionality with UMAP."""
        umap_mod = _import_umap()

        n_samples = matrix.shape[0]
        # UMAP needs enough samples to build a meaningful neighborhood graph
        if n_samples < 10:
            logger.debug("Too few samples (%d) for UMAP reduction, skipping", n_samples)
            return matrix

        n_components = min(self.umap_n_components, n_samples - 1)
        n_neighbors = min(15, n_samples - 1)

        logger.debug(
            "UMAP: reducing %d dims to %d (n_samples=%d)",
            matrix.shape[1],
            n_components,
            n_samples,
        )
        reducer = umap_mod.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=42,
        )
        return reducer.fit_transform(matrix)

    def _cluster(self, matrix: Any, n_values: int) -> Any:
        """Run clustering algorithm on embedding matrix."""
        if self.clustering_algorithm == "hdbscan":
            HDBSCAN = _import_sklearn_hdbscan()
            min_size = min(self.min_cluster_size, max(2, n_values // 3))
            clusterer = HDBSCAN(
                min_cluster_size=min_size,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(matrix)
        elif self.clustering_algorithm == "agglomerative":
            AgglomerativeClustering = _import_sklearn_agglomerative()
            clusterer = AgglomerativeClustering(
                distance_threshold=1.0 - self.similarity_threshold,
                n_clusters=None,
                metric="cosine",
                linkage="average",
            )
            labels = clusterer.fit_predict(matrix)
        else:
            raise NormalizationError(
                f"Unknown clustering algorithm: '{self.clustering_algorithm}'. "
                "Supported: 'hdbscan', 'agglomerative'."
            )

        return labels

    @staticmethod
    def _build_result(
        unique_values: list[str],
        value_counts: Counter[str],
        labels: Any,
        context_field: str,
    ) -> NormalizationResult:
        """Build NormalizationResult from cluster labels."""
        # Group values by cluster label
        cluster_groups: dict[int, list[str]] = {}
        for value, label in zip(unique_values, labels, strict=True):
            label_int = int(label)
            if label_int not in cluster_groups:
                cluster_groups[label_int] = []
            cluster_groups[label_int].append(value)

        # Select canonical name per cluster (most frequent value)
        mapping: dict[str, str] = {}
        clusters: dict[str, list[str]] = {}
        explanations: dict[str, str] = {}

        for label, members in cluster_groups.items():
            if label == -1:
                # Noise points (HDBSCAN) — each maps to itself
                for member in members:
                    mapping[member] = member
                    explanations[member] = "unclustered (noise)"
                continue

            # Pick canonical = most frequent member
            canonical = max(members, key=lambda v: value_counts[v])
            clusters[canonical] = members

            for member in members:
                mapping[member] = canonical
                if member != canonical:
                    explanations[member] = (
                        f"clustered with '{canonical}' (cluster #{label}, {len(members)} members)"
                    )
                else:
                    explanations[member] = (
                        f"canonical representative of cluster #{label} ({len(members)} members)"
                    )

        n_clusters = len([lbl for lbl in set(labels) if lbl != -1])
        logger.info(
            "EmbeddingClustering: %d unique values → %d clusters for '%s'",
            len(unique_values),
            n_clusters,
            context_field,
        )

        return NormalizationResult(
            mapping=mapping,
            clusters=clusters,
            metadata={
                "strategy": "embedding_clustering",
                "field": context_field,
                "n_clusters": n_clusters,
                "explanations": explanations,
            },
        )
