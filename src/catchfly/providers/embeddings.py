"""Embedding client abstraction over OpenAI-compatible endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

from catchfly._defaults import DEFAULT_EMBEDDING_MODEL
from catchfly.exceptions import ProviderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    async def aembed(self, texts: list[str]) -> list[list[float]]: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible implementation
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE = 2048
"""Max texts per OpenAI embeddings API call (API limit is 2048)."""


class OpenAIEmbeddingClient:
    """Embedding client for any OpenAI-compatible embeddings endpoint.

    Works with OpenAI, Ollama, vLLM, and any other provider
    exposing an OpenAI-compatible /v1/embeddings endpoint.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        timeout: float = 120.0,
        max_cache_size: int = 10_000,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_cache_size = max_cache_size
        self._cache: dict[str, list[float]] = {}

    def _make_async_client(self) -> Any:
        """Create a fresh AsyncOpenAI client per call."""
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEmbeddingClient. "
                "Install it with: pip install catchfly[openai]"
            ) from e

        kwargs: dict[str, Any] = {
            "api_key": self.api_key or "not-needed",
            "timeout": self.timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return openai.AsyncOpenAI(**kwargs)

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning vectors.

        Uses an in-memory cache and batches large requests.
        """
        if not texts:
            return []

        async_client = self._make_async_client()

        # Split into cached and uncached
        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.debug(
                "Embedding %d texts (%d cached, %d new) model=%s",
                len(texts),
                len(texts) - len(uncached_texts),
                len(uncached_texts),
                self.model,
            )

            # Batch requests
            all_embeddings: list[list[float]] = []
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start : batch_start + self.batch_size]
                try:
                    response = await async_client.embeddings.create(
                        input=batch,
                        model=self.model,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    await async_client.close()
                    raise ProviderError(
                        f"Embedding call failed for batch starting at {batch_start}: {e}"
                    ) from e

            # Populate cache and results (evict oldest entries if over limit)
            for idx, embedding in zip(uncached_indices, all_embeddings, strict=True):
                text = texts[idx]
                if len(self._cache) >= self.max_cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[text] = embedding
                results[idx] = embedding

            await async_client.close()

        # Return in original order
        return [results[i] for i in range(len(texts))]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts synchronously."""
        from catchfly._compat import run_sync

        return run_sync(self.aembed(texts))

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Local sentence-transformers implementation
# ---------------------------------------------------------------------------


class SentenceTransformerEmbeddingClient:
    """Local embedding client via sentence-transformers.

    Default model: ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext``
    (768-dim, CLS pooling, optimized for biomedical entity linking).

    The model is downloaded on first use (~420 MB) and cached by
    HuggingFace Hub. Subsequent loads are fast.

    Requires: ``pip install catchfly[embeddings]``
    """

    def __init__(
        self,
        model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        *,
        batch_size: int = 256,
        device: str | None = None,
        max_cache_size: int = 10_000,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._device = device
        self.max_cache_size = max_cache_size
        self._cache: dict[str, list[float]] = {}
        self._model_instance: Any = None

    def _load_model(self) -> Any:
        """Load the SentenceTransformer model (lazy, on first embed call)."""
        if self._model_instance is not None:
            return self._model_instance

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for "
                "SentenceTransformerEmbeddingClient. "
                "Install it with: pip install catchfly[embeddings]"
            ) from e

        device = self._device or self._detect_device()
        logger.info(
            "Loading SentenceTransformer model '%s' on device '%s'",
            self.model,
            device,
        )
        self._model_instance = SentenceTransformer(self.model, device=device)
        return self._model_instance

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect best available device: CUDA > MPS > CPU."""
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts synchronously via sentence-transformers."""
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result: list[list[float]] = embeddings.tolist()
        return result

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts asynchronously.

        Runs the synchronous ``SentenceTransformer.encode()`` in a
        thread pool via :func:`asyncio.to_thread` to avoid blocking
        the event loop.
        """
        import asyncio

        if not texts:
            return []

        # Split into cached and uncached
        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.debug(
                "Embedding %d texts (%d cached, %d new) model=%s",
                len(texts),
                len(texts) - len(uncached_texts),
                len(uncached_texts),
                self.model,
            )

            all_embeddings = await asyncio.to_thread(self._encode, uncached_texts)

            for idx, embedding in zip(uncached_indices, all_embeddings, strict=True):
                text = texts[idx]
                if len(self._cache) >= self.max_cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[text] = embedding
                results[idx] = embedding

        return [results[i] for i in range(len(texts))]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts synchronously."""
        if not texts:
            return []

        results: dict[int, list[float]] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.debug(
                "Embedding %d texts (%d cached, %d new) model=%s",
                len(texts),
                len(texts) - len(uncached_texts),
                len(uncached_texts),
                self.model,
            )

            all_embeddings = self._encode(uncached_texts)

            for idx, embedding in zip(uncached_indices, all_embeddings, strict=True):
                text = texts[idx]
                if len(self._cache) >= self.max_cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[text] = embedding
                results[idx] = embedding

        return [results[i] for i in range(len(texts))]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
