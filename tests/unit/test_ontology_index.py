"""Tests for OntologyIndex."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from catchfly.ontology.index import OntologyIndex
from catchfly.ontology.types import OntologyEntry

if TYPE_CHECKING:
    from pathlib import Path


class MockOntologyEmbedder:
    """Deterministic embedder: hash-based vectors with controlled similarity."""

    def __init__(self, dim: int = 8, overrides: dict[str, list[float]] | None = None) -> None:
        self.dim = dim
        self.model = "mock-embedder"
        self._overrides = overrides or {}

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        if text in self._overrides:
            return self._overrides[text]
        h = hash(text) & 0xFFFFFFFF
        return [((h >> (i * 4)) & 0xF) / 15.0 for i in range(self.dim)]


def _make_entries() -> list[OntologyEntry]:
    return [
        OntologyEntry(id="HP:001", name="Seizure", synonyms=("Seizures", "Epileptic seizure")),
        OntologyEntry(id="HP:002", name="Ataxia", synonyms=("Incoordination",)),
        OntologyEntry(id="HP:003", name="Fever"),
    ]


class TestOntologyIndex:
    async def test_build_expands_synonyms(self) -> None:
        entries = _make_entries()
        embedder = MockOntologyEmbedder()
        index = OntologyIndex(entries, embedder)  # type: ignore[arg-type]
        await index.build()

        # 3 entries: "Seizure" + 2 synonyms, "Ataxia" + 1 synonym, "Fever" = 6 texts
        assert len(index._texts) == 6
        assert index._embedding_matrix.shape[0] == 6

    async def test_search_returns_top_k(self) -> None:
        entries = _make_entries()
        # Make "my query" close to Seizure
        overrides = {
            "Seizure": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Seizures": [0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Epileptic seizure": [0.98, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Ataxia": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Incoordination": [0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Fever": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        embedder = MockOntologyEmbedder(overrides=overrides)
        index = OntologyIndex(entries, embedder)  # type: ignore[arg-type]
        await index.build()

        query = [[1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]  # close to Seizure
        results = index.search(query, top_k=2)

        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0][0].id == "HP:001"  # Seizure is top match

    async def test_search_deduplicates_entries(self) -> None:
        """Same entry matched via name and synonym should appear once."""
        overrides = {
            "Seizure": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Seizures": [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Epileptic seizure": [0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Ataxia": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Incoordination": [0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "Fever": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        embedder = MockOntologyEmbedder(overrides=overrides)
        entries = _make_entries()
        index = OntologyIndex(entries, embedder)  # type: ignore[arg-type]
        await index.build()

        query = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        results = index.search(query, top_k=3)

        entry_ids = [r[0].id for r in results[0]]
        # HP:001 should appear only once despite matching name + 2 synonyms
        assert entry_ids.count("HP:001") == 1
        assert len(results[0]) == 3  # 3 distinct entries

    async def test_empty_entries(self) -> None:
        embedder = MockOntologyEmbedder()
        index = OntologyIndex([], embedder)  # type: ignore[arg-type]
        await index.build()
        results = index.search([[0.5] * 8], top_k=3)
        assert results == [[]]

    async def test_cache_roundtrip(self, tmp_path: Path) -> None:
        entries = _make_entries()
        embedder = MockOntologyEmbedder()
        cache_path = tmp_path / "cache.json"

        # Build and save
        index1 = OntologyIndex(entries, embedder, cache_path=cache_path)  # type: ignore[arg-type]
        await index1.build()
        assert cache_path.exists()

        # Load from cache
        index2 = OntologyIndex(entries, embedder, cache_path=cache_path)  # type: ignore[arg-type]
        await index2.build()

        # Same results
        query = [[0.5] * 8]
        r1 = index1.search(query, top_k=2)
        r2 = index2.search(query, top_k=2)
        assert len(r1[0]) == len(r2[0])
        assert r1[0][0][0].id == r2[0][0][0].id

    async def test_cache_invalidation_on_model_change(self, tmp_path: Path) -> None:
        entries = _make_entries()
        cache_path = tmp_path / "cache.json"

        embedder1 = MockOntologyEmbedder()
        embedder1.model = "model-a"
        index1 = OntologyIndex(entries, embedder1, cache_path=cache_path)  # type: ignore[arg-type]
        await index1.build()

        # Different model — should rebuild, not use cache
        embedder2 = MockOntologyEmbedder()
        embedder2.model = "model-b"
        index2 = OntologyIndex(entries, embedder2, cache_path=cache_path)  # type: ignore[arg-type]
        await index2.build()

        # Cache should now reflect model-b
        cache_data = json.loads(cache_path.read_text())
        assert cache_data["model"] == "model-b"

    async def test_build_must_be_called_before_search(self) -> None:
        entries = _make_entries()
        embedder = MockOntologyEmbedder()
        index = OntologyIndex(entries, embedder)  # type: ignore[arg-type]

        import pytest

        with pytest.raises(RuntimeError, match="build"):
            index.search([[0.5] * 8])
