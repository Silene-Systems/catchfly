"""Tests for LearnedDictionaryCache and NormalizationResult.to_dictionary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from catchfly._types import NormalizationResult
from catchfly.normalization.learned_cache import LearnedDictionaryCache

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# NormalizationResult.to_dictionary
# ---------------------------------------------------------------------------


class TestToDictionary:
    def test_filters_identity_mappings(self) -> None:
        result = NormalizationResult(
            mapping={"a": "A", "b": "b", "c": "C"},
            metadata={"per_value": {"a": {"confidence": 1.0}, "c": {"confidence": 0.9}}},
        )
        d = result.to_dictionary()
        assert d == {"a": "A", "c": "C"}
        assert "b" not in d  # identity mapping excluded

    def test_min_confidence_filtering(self) -> None:
        result = NormalizationResult(
            mapping={"a": "A", "b": "B"},
            metadata={
                "per_value": {
                    "a": {"confidence": 0.95},
                    "b": {"confidence": 0.5},
                }
            },
        )
        d = result.to_dictionary(min_confidence=0.8)
        assert d == {"a": "A"}
        assert "b" not in d  # below threshold

    def test_no_per_value_defaults_to_1(self) -> None:
        """Without per_value metadata, confidence defaults to 1.0."""
        result = NormalizationResult(
            mapping={"a": "A", "b": "b"},
            metadata={},
        )
        d = result.to_dictionary(min_confidence=0.5)
        assert d == {"a": "A"}

    def test_empty_mapping(self) -> None:
        result = NormalizationResult(mapping={})
        assert result.to_dictionary() == {}


# ---------------------------------------------------------------------------
# LearnedDictionaryCache
# ---------------------------------------------------------------------------


class TestLearnedDictionaryCache:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache = LearnedDictionaryCache(path=str(cache_path), min_confidence=0.5)

        result = NormalizationResult(
            mapping={"seizures": "Seizure", "fever": "Fever", "unknown": "unknown"},
            metadata={
                "strategy": "ontology_mapping",
                "per_value": {
                    "seizures": {"confidence": 0.95},
                    "fever": {"confidence": 0.85},
                    "unknown": {"confidence": 0.0},
                },
            },
        )
        cache.save({"phenotype": result})

        # Load back
        dn = cache.load_dictionary("phenotype")
        assert dn is not None
        assert dn.mapping["seizures"] == "Seizure"
        assert dn.mapping["fever"] == "Fever"
        assert "unknown" not in dn.mapping  # identity mapping excluded

    def test_min_confidence_filtering(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache = LearnedDictionaryCache(path=str(cache_path), min_confidence=0.9)

        result = NormalizationResult(
            mapping={"a": "A", "b": "B"},
            metadata={
                "strategy": "test",
                "per_value": {
                    "a": {"confidence": 0.95},
                    "b": {"confidence": 0.7},
                },
            },
        )
        cache.save({"field": result})

        dn = cache.load_dictionary("field")
        assert dn is not None
        assert "a" in dn.mapping  # 0.95 >= 0.9
        assert "b" not in dn.mapping  # 0.7 < 0.9

    def test_merge_existing_cache(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache = LearnedDictionaryCache(path=str(cache_path), min_confidence=0.5)

        # First save
        r1 = NormalizationResult(
            mapping={"a": "A"},
            metadata={"strategy": "s1", "per_value": {"a": {"confidence": 0.8}}},
        )
        cache.save({"field": r1})

        # Second save — adds new, keeps higher confidence
        r2 = NormalizationResult(
            mapping={"a": "A_better", "b": "B"},
            metadata={
                "strategy": "s2",
                "per_value": {
                    "a": {"confidence": 0.95},  # higher → overwrites
                    "b": {"confidence": 0.9},
                },
            },
        )
        cache.save({"field": r2})

        dn = cache.load_dictionary("field")
        assert dn is not None
        assert dn.mapping["a"] == "A_better"  # higher confidence wins
        assert dn.mapping["b"] == "B"

    def test_merge_keeps_higher_confidence(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache = LearnedDictionaryCache(path=str(cache_path), min_confidence=0.5)

        r1 = NormalizationResult(
            mapping={"a": "A_good"},
            metadata={"strategy": "s1", "per_value": {"a": {"confidence": 0.95}}},
        )
        cache.save({"field": r1})

        # Lower confidence should NOT overwrite
        r2 = NormalizationResult(
            mapping={"a": "A_worse"},
            metadata={"strategy": "s2", "per_value": {"a": {"confidence": 0.6}}},
        )
        cache.save({"field": r2})

        dn = cache.load_dictionary("field")
        assert dn is not None
        assert dn.mapping["a"] == "A_good"

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "nonexistent.json"
        cache = LearnedDictionaryCache(path=str(cache_path))
        assert cache.load_dictionary("field") is None

    def test_load_all(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache = LearnedDictionaryCache(path=str(cache_path), min_confidence=0.5)

        r1 = NormalizationResult(
            mapping={"a": "A"},
            metadata={"strategy": "s", "per_value": {"a": {"confidence": 0.9}}},
        )
        r2 = NormalizationResult(
            mapping={"x": "X"},
            metadata={"strategy": "s", "per_value": {"x": {"confidence": 0.8}}},
        )
        cache.save({"field1": r1, "field2": r2})

        all_dicts = cache.load_all()
        assert "field1" in all_dicts
        assert "field2" in all_dicts
        assert all_dicts["field1"].mapping["a"] == "A"
        assert all_dicts["field2"].mapping["x"] == "X"

    def test_corrupt_cache_returns_none(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("not valid json", encoding="utf-8")

        cache = LearnedDictionaryCache(path=str(cache_path))
        assert cache.load_dictionary("field") is None

    def test_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"version": 999, "field_mappings": {}}), encoding="utf-8")

        cache = LearnedDictionaryCache(path=str(cache_path))
        assert cache.load_dictionary("field") is None

    def test_cache_json_structure(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.json"
        cache = LearnedDictionaryCache(path=str(cache_path))

        result = NormalizationResult(
            mapping={"a": "A"},
            metadata={"strategy": "test", "per_value": {"a": {"confidence": 0.9}}},
        )
        cache.save({"field": result})

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert "created" in data
        assert "updated" in data
        assert "field_mappings" in data
        assert data["field_mappings"]["field"]["a"]["canonical"] == "A"
        assert data["field_mappings"]["field"]["a"]["confidence"] == 0.9
        assert data["field_mappings"]["field"]["a"]["source"] == "test"
