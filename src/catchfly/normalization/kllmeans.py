"""KLLMeansClustering — k-means with LLM-generated textual centroids.

Implements the k-LLMmeans algorithm: standard k-means in embedding space
with periodic LLM-generated textual summaries as centroids.

Best suited for large-scale surface-form deduplication (e.g., brand name
variants). For semantic normalization, LLMCanonicalization is recommended.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

from pydantic import BaseModel

from catchfly._compat import run_sync
from catchfly._types import NormalizationResult
from catchfly.providers.embeddings import OpenAIEmbeddingClient
from catchfly.providers.llm import OpenAICompatibleClient

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """\
You are a cluster summarization assistant. Given a list of text values \
that belong to the same cluster, generate a short canonical name that \
best represents the entire group.

Rules:
- Output ONLY the canonical name (one line, no explanation)
- Use the most formal/standard form
- Keep it concise (1-5 words)\
"""


def _import_numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError as e:
        raise ImportError(
            "numpy is required for KLLMeansClustering. "
            "Install it with: pip install catchfly[clustering]"
        ) from e


class KLLMeansClustering(BaseModel):
    """Normalize values using k-LLMmeans: k-means + LLM textual centroids.

    Algorithm:
    1. Embed all unique values
    2. Initialize centroids via k-means++
    3. Iterate:
       a. Assign values to nearest centroid (standard k-means)
       b. Update centroids as mean of assigned embeddings
       c. Every `summarize_every` iterations: LLM generates textual
          summary for each cluster → embed summary → replace centroid
    4. Final LLM pass generates canonical names from cluster members

    Best for surface-form deduplication where embeddings separate
    variants well. For semantic normalization, use LLMCanonicalization.
    """

    num_clusters: int | None = None
    embedding_model: str = "text-embedding-3-small"
    summarization_model: str = "gpt-5.4-mini"
    num_iterations: int = 10
    summarize_every: int = 3
    max_members_in_prompt: int = 50
    base_url: str | None = None
    api_key: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values asynchronously using k-LLMmeans."""
        if not values:
            return NormalizationResult(mapping={})

        np = _import_numpy()
        value_counts = Counter(values)
        unique_values = list(value_counts.keys())

        if len(unique_values) <= 2:
            return self._trivial_result(unique_values, context_field)

        # Embed all values
        embed_client = OpenAIEmbeddingClient(
            model=self.embedding_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        raw_embeddings = await embed_client.aembed(unique_values)
        embeddings = np.array(raw_embeddings, dtype=np.float64)

        # Determine k
        k = self._determine_k(embeddings, len(unique_values))

        # Initialize centroids via k-means++
        centroids = self._kmeans_pp_init(embeddings, k)

        logger.info(
            "KLLMeansClustering: starting %d iterations, k=%d, %d values for '%s'",
            self.num_iterations,
            k,
            len(unique_values),
            context_field,
        )

        # k-LLMmeans loop
        llm_client = OpenAICompatibleClient(
            model=self.summarization_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        assignments = np.zeros(len(unique_values), dtype=int)
        for iteration in range(1, self.num_iterations + 1):
            # Assignment step
            new_assignments = self._assign(embeddings, centroids)

            # Check convergence
            changed = int(np.sum(new_assignments != assignments))
            assignments = new_assignments

            if changed == 0 and iteration > 1:
                logger.info("KLLMeansClustering: converged at iteration %d", iteration)
                break

            # Update step (mean of assigned embeddings)
            centroids = self._update_centroids(embeddings, assignments, k, centroids)

            # LLM summary step
            if iteration % self.summarize_every == 0:
                centroids = await self._summarize_centroids(
                    unique_values, assignments, k, embed_client, llm_client
                )
                logger.debug("KLLMeansClustering: LLM summary at iteration %d", iteration)

            logger.debug(
                "KLLMeansClustering: iteration %d, %d assignments changed",
                iteration,
                changed,
            )

        # Generate canonical names via LLM
        canonical_names = await self._generate_canonical_names(
            unique_values, assignments, k, llm_client
        )

        return self._build_result(
            unique_values, value_counts, assignments, canonical_names, k, context_field
        )

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values synchronously."""
        return run_sync(
            self.anormalize(values, context_field=context_field, **kwargs)
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _determine_k(self, embeddings: Any, n: int) -> int:
        """Determine number of clusters."""
        if self.num_clusters is not None:
            return min(self.num_clusters, n)

        # Auto-detect: try range [2, sqrt(n)] and pick best silhouette
        max_k = max(2, min(int(math.sqrt(n)), 20))
        if max_k <= 2:
            return 2

        try:
            from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

            best_k = 2
            best_score = -1.0

            for k in range(2, max_k + 1):
                centroids = self._kmeans_pp_init(embeddings, k)
                assignments = self._assign(embeddings, centroids)

                # Need at least 2 clusters with members
                if len(set(assignments.tolist())) < 2:
                    continue

                score: float = silhouette_score(embeddings, assignments)
                if score > best_score:
                    best_score = score
                    best_k = k

            logger.info(
                "KLLMeansClustering: auto-detected k=%d (silhouette=%.3f)",
                best_k,
                best_score,
            )
            return best_k

        except ImportError:
            # No sklearn — use heuristic
            return max(2, min(int(math.sqrt(n / 2)), 10))

    @staticmethod
    def _kmeans_pp_init(embeddings: Any, k: int) -> Any:
        """k-means++ initialization."""
        np = _import_numpy()
        n = embeddings.shape[0]
        rng = np.random.default_rng(42)

        # First centroid: random
        centroids = [embeddings[rng.integers(n)]]

        for _ in range(1, k):
            # Distance to nearest centroid
            dists = np.min([np.sum((embeddings - c) ** 2, axis=1) for c in centroids], axis=0)
            # Probability proportional to distance squared
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centroids.append(embeddings[idx])

        return np.array(centroids)

    # ------------------------------------------------------------------
    # k-means steps
    # ------------------------------------------------------------------

    @staticmethod
    def _assign(embeddings: Any, centroids: Any) -> Any:
        """Assign each point to nearest centroid."""
        np = _import_numpy()
        # Compute distances: (n, k)
        dists = np.array([np.sum((embeddings - c) ** 2, axis=1) for c in centroids]).T
        return np.argmin(dists, axis=1)

    @staticmethod
    def _update_centroids(embeddings: Any, assignments: Any, k: int, prev_centroids: Any) -> Any:
        """Update centroids as mean of assigned points."""
        np = _import_numpy()
        new_centroids = np.copy(prev_centroids)
        for i in range(k):
            mask = assignments == i
            if np.any(mask):
                new_centroids[i] = embeddings[mask].mean(axis=0)
        return new_centroids

    async def _summarize_centroids(
        self,
        values: list[str],
        assignments: Any,
        k: int,
        embed_client: OpenAIEmbeddingClient,
        llm_client: OpenAICompatibleClient,
    ) -> Any:
        """LLM generates textual summary for each cluster → embed → new centroids."""
        np = _import_numpy()
        summaries: list[str] = []

        for i in range(k):
            mask = assignments == i
            members = [values[j] for j in range(len(values)) if mask[j]]

            if not members:
                summaries.append(f"cluster_{i}")
                continue

            # Ask LLM for summary
            member_list = "\n".join(f"- {m}" for m in members[:self.max_members_in_prompt])
            try:
                response = await llm_client.acomplete(
                    [
                        {"role": "system", "content": _SUMMARY_SYSTEM},
                        {"role": "user", "content": f"Cluster members:\n{member_list}"},
                    ],
                    temperature=0.0,
                )
                summaries.append(response.content.strip())
            except Exception:
                logger.warning(
                    "KLLMeansClustering: summary failed for cluster %d, "
                    "falling back to first member",
                    i,
                    exc_info=True,
                )
                summaries.append(members[0])

        # Embed summaries to get new centroids
        summary_embeddings = await embed_client.aembed(summaries)
        return np.array(summary_embeddings, dtype=np.float64)

    async def _generate_canonical_names(
        self,
        values: list[str],
        assignments: Any,
        k: int,
        llm_client: OpenAICompatibleClient,
    ) -> list[str]:
        """Generate final canonical name for each cluster."""
        names: list[str] = []

        for i in range(k):
            mask = assignments == i
            members = [values[j] for j in range(len(values)) if mask[j]]

            if not members:
                names.append(f"cluster_{i}")
                continue

            if len(members) == 1:
                names.append(members[0])
                continue

            member_list = "\n".join(f"- {m}" for m in members[:self.max_members_in_prompt])
            try:
                response = await llm_client.acomplete(
                    [
                        {"role": "system", "content": _SUMMARY_SYSTEM},
                        {"role": "user", "content": f"Cluster members:\n{member_list}"},
                    ],
                    temperature=0.0,
                )
                names.append(response.content.strip())
            except Exception:
                logger.warning(
                    "KLLMeansClustering: canonical name generation failed for cluster %d, "
                    "falling back to most frequent member",
                    i,
                    exc_info=True,
                )
                counter = Counter(members)
                names.append(counter.most_common(1)[0][0])

        return names

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @staticmethod
    def _trivial_result(values: list[str], context_field: str) -> NormalizationResult:
        """Handle trivial cases (0-2 values)."""
        mapping = {v: v for v in values}
        clusters = {v: [v] for v in values}
        return NormalizationResult(
            mapping=mapping,
            clusters=clusters,
            metadata={"strategy": "kllmeans", "field": context_field},
        )

    @staticmethod
    def _build_result(
        values: list[str],
        value_counts: Counter[str],
        assignments: Any,
        canonical_names: list[str],
        k: int,
        context_field: str,
    ) -> NormalizationResult:
        """Build NormalizationResult from cluster assignments."""
        mapping: dict[str, str] = {}
        clusters: dict[str, list[str]] = {}
        explanations: dict[str, str] = {}

        for i in range(k):
            mask = assignments == i
            members = [values[j] for j in range(len(values)) if mask[j]]
            canonical = canonical_names[i] if i < len(canonical_names) else f"cluster_{i}"

            clusters[canonical] = members
            for member in members:
                mapping[member] = canonical
                explanations[member] = (
                    f"kLLMmeans cluster #{i} → '{canonical}' ({len(members)} members)"
                )

        logger.info(
            "KLLMeansClustering: %d values → %d clusters for '%s'",
            len(values),
            k,
            context_field,
        )

        return NormalizationResult(
            mapping=mapping,
            clusters=clusters,
            metadata={
                "strategy": "kllmeans",
                "field": context_field,
                "k": k,
                "explanations": explanations,
            },
        )
