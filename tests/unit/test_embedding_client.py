"""Tests for embedding clients."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from catchfly.providers.embeddings import (
    EmbeddingClient,
    OpenAIEmbeddingClient,
    SentenceTransformerEmbeddingClient,
)


class TestOpenAIEmbeddingClient:
    def test_clear_cache(self) -> None:
        client = OpenAIEmbeddingClient()
        client._cache["test"] = [0.1, 0.2]
        client.clear_cache()
        assert len(client._cache) == 0


# ---------------------------------------------------------------------------
# SentenceTransformerEmbeddingClient
# ---------------------------------------------------------------------------


def _make_mock_st_class(dim: int = 8) -> MagicMock:
    """Create a mock SentenceTransformer class."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kw: np.random.default_rng(42).random(
        (len(texts), dim)
    )
    mock_class = MagicMock(return_value=mock_model)
    return mock_class


class TestSentenceTransformerEmbeddingClient:
    def test_protocol_compliance(self) -> None:
        """SentenceTransformerEmbeddingClient satisfies EmbeddingClient protocol."""
        client = SentenceTransformerEmbeddingClient()
        assert isinstance(client, EmbeddingClient)

    def test_constructor_stores_config(self) -> None:
        client = SentenceTransformerEmbeddingClient(
            model="test-model",
            batch_size=64,
            device="cpu",
            max_cache_size=100,
        )
        assert client.model == "test-model"
        assert client.batch_size == 64
        assert client._device == "cpu"
        assert client.max_cache_size == 100

    def test_lazy_model_loading(self) -> None:
        """Model is NOT loaded in __init__."""
        client = SentenceTransformerEmbeddingClient()
        assert client._model_instance is None

    def test_load_model_on_first_embed(self) -> None:
        mock_class = _make_mock_st_class()
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_class)}):
            client = SentenceTransformerEmbeddingClient(device="cpu")
            client._load_model()
            assert client._model_instance is not None
            mock_class.assert_called_once()

    def test_import_error_message(self) -> None:
        """Helpful error when sentence-transformers is not installed."""
        client = SentenceTransformerEmbeddingClient()
        client._model_instance = None
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                client._load_model()

    def test_device_detection_cuda(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert SentenceTransformerEmbeddingClient._detect_device() == "cuda"

    def test_device_detection_mps(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert SentenceTransformerEmbeddingClient._detect_device() == "mps"

    def test_device_detection_cpu_fallback(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert SentenceTransformerEmbeddingClient._detect_device() == "cpu"

    def test_device_detection_no_torch(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            assert SentenceTransformerEmbeddingClient._detect_device() == "cpu"

    def test_embed_empty_input(self) -> None:
        client = SentenceTransformerEmbeddingClient()
        assert client.embed([]) == []

    async def test_aembed_empty_input(self) -> None:
        client = SentenceTransformerEmbeddingClient()
        assert await client.aembed([]) == []

    def test_embed_returns_vectors(self) -> None:
        import numpy as np

        client = SentenceTransformerEmbeddingClient(device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        client._model_instance = mock_model

        result = client.embed(["hello", "world"])
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])

    async def test_aembed_returns_vectors(self) -> None:
        import numpy as np

        client = SentenceTransformerEmbeddingClient(device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        client._model_instance = mock_model

        result = await client.aembed(["hello", "world"])
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])

    def test_caching(self) -> None:
        import numpy as np

        client = SentenceTransformerEmbeddingClient(device="cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        client._model_instance = mock_model

        # First call
        result1 = client.embed(["hello"])
        assert mock_model.encode.call_count == 1

        # Second call — should use cache
        result2 = client.embed(["hello"])
        assert mock_model.encode.call_count == 1  # not called again
        assert result1 == result2

    def test_caching_partial(self) -> None:
        """Mixed cached + uncached texts."""
        import numpy as np

        client = SentenceTransformerEmbeddingClient(device="cpu")
        mock_model = MagicMock()
        client._model_instance = mock_model

        # Pre-populate cache
        client._cache["cached"] = [1.0, 2.0]

        mock_model.encode.return_value = np.array([[3.0, 4.0]])
        result = client.embed(["cached", "new"])

        assert result[0] == [1.0, 2.0]  # from cache
        assert result[1] == pytest.approx([3.0, 4.0])  # from encode
        # Only "new" was encoded
        mock_model.encode.assert_called_once_with(
            ["new"], batch_size=256, show_progress_bar=False, convert_to_numpy=True
        )

    def test_cache_eviction(self) -> None:
        import numpy as np

        client = SentenceTransformerEmbeddingClient(
            device="cpu", max_cache_size=2
        )
        mock_model = MagicMock()
        client._model_instance = mock_model

        # Fill cache to max
        client._cache["a"] = [1.0]
        client._cache["b"] = [2.0]

        # Add a new entry — should evict "a" (oldest)
        mock_model.encode.return_value = np.array([[3.0]])
        client.embed(["c"])

        assert "a" not in client._cache
        assert "b" in client._cache
        assert "c" in client._cache

    def test_clear_cache(self) -> None:
        client = SentenceTransformerEmbeddingClient()
        client._cache["test"] = [0.1, 0.2]
        client.clear_cache()
        assert len(client._cache) == 0

    def test_model_attribute(self) -> None:
        """Model name is accessible as .model for cache compatibility."""
        client = SentenceTransformerEmbeddingClient(model="my-model")
        assert client.model == "my-model"

    def test_default_model(self) -> None:
        client = SentenceTransformerEmbeddingClient()
        assert client.model == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
