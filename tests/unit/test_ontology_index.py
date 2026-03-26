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
        index = OntologyIndex(entries, embedder)
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
        index = OntologyIndex(entries, embedder)
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
        index = OntologyIndex(entries, embedder)
        await index.build()

        query = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        results = index.search(query, top_k=3)

        entry_ids = [r[0].id for r in results[0]]
        # HP:001 should appear only once despite matching name + 2 synonyms
        assert entry_ids.count("HP:001") == 1
        assert len(results[0]) == 3  # 3 distinct entries

    async def test_empty_entries(self) -> None:
        embedder = MockOntologyEmbedder()
        index = OntologyIndex([], embedder)
        await index.build()
        results = index.search([[0.5] * 8], top_k=3)
        assert results == [[]]

    async def test_cache_roundtrip(self, tmp_path: Path) -> None:
        entries = _make_entries()
        embedder = MockOntologyEmbedder()
        cache_path = tmp_path / "cache.json"

        # Build and save
        index1 = OntologyIndex(entries, embedder, cache_path=cache_path)
        await index1.build()
        assert cache_path.exists()

        # Load from cache
        index2 = OntologyIndex(entries, embedder, cache_path=cache_path)
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
        index1 = OntologyIndex(entries, embedder1, cache_path=cache_path)
        await index1.build()

        # Different model — should rebuild, not use cache
        embedder2 = MockOntologyEmbedder()
        embedder2.model = "model-b"
        index2 = OntologyIndex(entries, embedder2, cache_path=cache_path)
        await index2.build()

        # Cache should now reflect model-b
        cache_data = json.loads(cache_path.read_text())
        assert cache_data["model"] == "model-b"

    async def test_cache_works_with_client_without_model_attr(self, tmp_path: Path) -> None:
        """Client without .model attribute uses 'unknown' in cache metadata."""

        class BareEmbedder:
            async def aembed(self, texts: list[str]) -> list[list[float]]:
                return [[((hash(t) + i) & 0xF) / 15.0 for i in range(8)] for t in texts]

        entries = _make_entries()
        cache_path = tmp_path / "cache.json"
        embedder = BareEmbedder()
        index = OntologyIndex(entries, embedder, cache_path=cache_path)
        await index.build()

        assert cache_path.exists()
        cache_data = json.loads(cache_path.read_text())
        assert cache_data["model"] == "unknown"


class TestHomonymDisambiguation:
    """Tests for homonym detection and disambiguation in OntologyIndex."""

    async def test_homonym_detection(self) -> None:
        """Entries with same name but different IDs get disambiguated."""
        entries = [
            OntologyEntry(id="HP:100", name="Mercury", synonyms=("Hg", "Quicksilver")),
            OntologyEntry(id="HP:200", name="Mercury", synonyms=("Planet Mercury",)),
            OntologyEntry(id="HP:300", name="Fever"),
        ]
        embedder = MockOntologyEmbedder()
        index = OntologyIndex(entries, embedder)
        await index.build()

        # The two "Mercury" texts should be disambiguated
        mercury_texts = [t for t in index._texts if t.startswith("Mercury")]
        assert len(mercury_texts) == 2
        # Each should have context appended
        for t in mercury_texts:
            assert "(" in t  # disambiguation context added
            assert "[HP:" in t  # ID included

        # "Fever" should be unchanged
        assert "Fever" in index._texts

    async def test_no_disambiguation_unique_names(self) -> None:
        """Entries with distinct names are not modified."""
        entries = _make_entries()  # Seizure, Ataxia, Fever — all unique
        embedder = MockOntologyEmbedder()
        index = OntologyIndex(entries, embedder)
        await index.build()

        # No text should have disambiguation context
        for t in index._texts:
            assert "also known as:" not in t

    async def test_same_entry_synonym_not_homonym(self) -> None:
        """Entry where name matches another text from same entry is not a homonym."""
        entries = [
            OntologyEntry(id="HP:001", name="seizure", synonyms=("Seizure", "Epileptic seizure")),
        ]
        embedder = MockOntologyEmbedder()
        index = OntologyIndex(entries, embedder)
        await index.build()

        # "seizure" and "Seizure" match case-insensitively but same entry ID
        # No disambiguation should happen
        for t in index._texts:
            assert "[HP:" not in t

    async def test_disambiguated_texts_include_synonyms(self) -> None:
        """Disambiguated texts include synonym context."""
        entries = [
            OntologyEntry(id="HP:100", name="Cold", synonyms=("Common cold", "Rhinitis")),
            OntologyEntry(id="HP:200", name="Cold", synonyms=("Low temperature",)),
        ]
        embedder = MockOntologyEmbedder()
        index = OntologyIndex(entries, embedder)
        await index.build()

        cold_texts = [t for t in index._texts if t.startswith("Cold")]
        assert len(cold_texts) == 2

        # HP:100's Cold should mention "Common cold, Rhinitis"
        hp100_text = [t for t in cold_texts if "HP:100" in t]
        assert len(hp100_text) == 1
        assert "Common cold" in hp100_text[0]

        # HP:200's Cold should mention "Low temperature"
        hp200_text = [t for t in cold_texts if "HP:200" in t]
        assert len(hp200_text) == 1
        assert "Low temperature" in hp200_text[0]


class TestOntologyIndexMisc:
    async def test_build_must_be_called_before_search(self) -> None:
        entries = _make_entries()
        embedder = MockOntologyEmbedder()
        index = OntologyIndex(entries, embedder)
        import pytest

        with pytest.raises(RuntimeError, match="build"):
            index.search([[0.5] * 8])
