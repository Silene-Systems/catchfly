"""Embedding client abstraction over OpenAI-compatible endpoints."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

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


class OpenAIEmbeddingClient:
    """Embedding client for any OpenAI-compatible embeddings endpoint.

    Works with OpenAI, Ollama, vLLM, and any other provider
    exposing an OpenAI-compatible /v1/embeddings endpoint.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.batch_size = batch_size
        self.timeout = timeout
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

            # Populate cache and results
            for idx, embedding in zip(uncached_indices, all_embeddings, strict=True):
                text = texts[idx]
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
