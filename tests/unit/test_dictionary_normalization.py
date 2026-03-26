"""Tests for DictionaryNormalization."""

from __future__ import annotations

import pytest

from catchfly.exceptions import NormalizationError
from catchfly.normalization.dictionary import DictionaryNormalization


class TestDictionaryNormalization:
    async def test_exact_match(self) -> None:
        norm = DictionaryNormalization(mapping={"NYC": "New York", "LA": "Los Angeles"})
        result = await norm.anormalize(["NYC", "LA"], context_field="city")
        assert result.mapping["NYC"] == "New York"
        assert result.mapping["LA"] == "Los Angeles"

    async def test_case_insensitive(self) -> None:
        norm = DictionaryNormalization(
            mapping={"NYC": "New York"},
            case_insensitive=True,
        )
        result = await norm.anormalize(["nyc", "Nyc", "NYC"], context_field="city")
        assert result.mapping["nyc"] == "New York"
        assert result.mapping["Nyc"] == "New York"
        assert result.mapping["NYC"] == "New York"

    async def test_passthrough_unmapped(self) -> None:
        norm = DictionaryNormalization(mapping={"NYC": "New York"})
        result = await norm.anormalize(["NYC", "Chicago"], context_field="city")
        assert result.mapping["NYC"] == "New York"
        assert result.mapping["Chicago"] == "Chicago"

    async def test_passthrough_false_raises(self) -> None:
        norm = DictionaryNormalization(
            mapping={"NYC": "New York"},
            passthrough_unmapped=False,
        )
        with pytest.raises(NormalizationError, match="Chicago"):
            await norm.anormalize(["NYC", "Chicago"], context_field="city")

    async def test_empty_input(self) -> None:
        norm = DictionaryNormalization(mapping={"a": "b"})
        result = await norm.anormalize([])
        assert result.mapping == {}

    def test_sync_wrapper(self) -> None:
        norm = DictionaryNormalization(mapping={"NYC": "New York", "LA": "Los Angeles"})
        result = norm.normalize(["NYC", "LA"], context_field="city")
        assert result.mapping["NYC"] == "New York"

    async def test_clusters_built(self) -> None:
        norm = DictionaryNormalization(
            mapping={"NYC": "New York", "NY": "New York", "LA": "Los Angeles"}
        )
        result = await norm.anormalize(["NYC", "NY", "LA"], context_field="city")
        assert "New York" in result.clusters
        assert set(result.clusters["New York"]) == {"NYC", "NY"}
        assert result.clusters["Los Angeles"] == ["LA"]

    async def test_explain(self) -> None:
        norm = DictionaryNormalization(mapping={"NYC": "New York", "NY": "New York"})
        result = await norm.anormalize(["NYC", "NY"], context_field="city")
        explanation = result.explain("NYC")
        assert "New York" in explanation

    async def test_per_value_confidence_metadata(self) -> None:
        """Matched values get confidence 1.0, passthrough gets 0.0."""
        norm = DictionaryNormalization(mapping={"NYC": "New York"})
        result = await norm.anormalize(["NYC", "Chicago"], context_field="city")

        per_value = result.metadata["per_value"]
        assert per_value["NYC"]["confidence"] == 1.0
        assert per_value["Chicago"]["confidence"] == 0.0
