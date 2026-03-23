"""OntologyIndex — embedding-based nearest-neighbor search over ontology terms."""

from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003
from typing import Any

from catchfly.ontology.types import OntologyEntry  # noqa: TC001
from catchfly.providers.embeddings import OpenAIEmbeddingClient  # noqa: TC001

logger = logging.getLogger(__name__)


def _import_numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError as e:
        raise ImportError(
            "numpy is required for OntologyIndex. "
            "Install it with: pip install catchfly[medical]"
        ) from e


_CACHE_VERSION = 1


class OntologyIndex:
    """Embedding index over ontology entries for nearest-neighbor search.

    Builds an embedding matrix from entry names and synonyms, then
    supports batch cosine-similarity search. Optionally caches the
    embedding matrix to disk.
    """

    def __init__(
        self,
        entries: list[OntologyEntry],
        embedding_client: OpenAIEmbeddingClient,
        cache_path: Path | None = None,
    ) -> None:
        self.entries = entries
        self._embedding_client = embedding_client
        self._cache_path = cache_path

        self._texts: list[str] = []
        self._text_to_entry: list[OntologyEntry] = []
        self._embedding_matrix: Any = None  # np.ndarray (n_texts, dim)
        self._norms: Any = None  # np.ndarray (n_texts, 1)
        self._built = False

    async def build(self) -> None:
        """Build the embedding index. Loads from cache if available."""
        np = _import_numpy()

        # Expand entries to searchable texts
        for entry in self.entries:
            for text in entry.all_texts:
                self._texts.append(text)
                self._text_to_entry.append(entry)

        if not self._texts:
            self._embedding_matrix = np.zeros((0, 1), dtype=np.float32)
            self._norms = np.ones((0, 1), dtype=np.float32)
            self._built = True
            return

        # Try loading from cache
        if self._cache_path and self._load_cache(np):
            self._built = True
            logger.info(
                "OntologyIndex: loaded %d embeddings from cache", len(self._texts)
            )
            return

        # Embed all texts
        logger.info("OntologyIndex: embedding %d texts", len(self._texts))
        raw = await self._embedding_client.aembed(self._texts)
        self._embedding_matrix = np.array(raw, dtype=np.float32)
        self._norms = np.linalg.norm(
            self._embedding_matrix, axis=1, keepdims=True
        )
        self._norms = np.where(self._norms == 0, 1.0, self._norms)
        self._built = True

        if self._cache_path:
            self._save_cache()

    def search(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 5,
    ) -> list[list[tuple[OntologyEntry, float]]]:
        """Batch cosine-similarity search.

        Returns *top_k* ``(entry, score)`` pairs per query, deduplicated
        by entry ID (highest score wins).
        """
        if not self._built:
            raise RuntimeError("OntologyIndex.build() must be called first")

        np = _import_numpy()
        if len(self._texts) == 0:
            return [[] for _ in query_embeddings]

        q = np.array(query_embeddings, dtype=np.float32)
        q_norms = np.linalg.norm(q, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1.0, q_norms)

        # (n_queries, n_texts)
        similarities = (q / q_norms) @ (self._embedding_matrix / self._norms).T

        results: list[list[tuple[OntologyEntry, float]]] = []
        for i in range(len(query_embeddings)):
            scores = similarities[i]
            # Get more indices than top_k to allow for deduplication
            n_candidates = min(len(self._texts), top_k * 3)
            top_indices = np.argpartition(scores, -n_candidates)[-n_candidates:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            seen_ids: set[str] = set()
            candidates: list[tuple[OntologyEntry, float]] = []
            for idx in top_indices:
                entry = self._text_to_entry[idx]
                if entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    candidates.append((entry, float(scores[idx])))
                if len(candidates) >= top_k:
                    break
            results.append(candidates)

        return results

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _save_cache(self) -> None:
        assert self._cache_path is not None
        data = {
            "version": _CACHE_VERSION,
            "model": self._embedding_client.model,
            "n_texts": len(self._texts),
            "embeddings": self._embedding_matrix.tolist(),
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(data), encoding="utf-8"
        )
        logger.info("OntologyIndex: saved cache to %s", self._cache_path)

    def _load_cache(self, np: Any) -> bool:
        assert self._cache_path is not None
        if not self._cache_path.exists():
            return False

        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("OntologyIndex: corrupt cache, rebuilding")
            return False

        if data.get("version") != _CACHE_VERSION:
            return False
        if data.get("model") != self._embedding_client.model:
            logger.info("OntologyIndex: model changed, rebuilding cache")
            return False
        if data.get("n_texts") != len(self._texts):
            logger.info("OntologyIndex: ontology size changed, rebuilding cache")
            return False

        self._embedding_matrix = np.array(data["embeddings"], dtype=np.float32)
        self._norms = np.linalg.norm(
            self._embedding_matrix, axis=1, keepdims=True
        )
        self._norms = np.where(self._norms == 0, 1.0, self._norms)
        return True
