"""Tests for CompositeNormalization."""

from __future__ import annotations

from typing import Any

import pytest

from catchfly._types import NormalizationResult
from catchfly.exceptions import NormalizationError
from catchfly.normalization.composite import CompositeNormalization
from catchfly.normalization.dictionary import DictionaryNormalization


class _UppercaseNorm:
    """Test strategy that uppercases all values."""

    async def anormalize(
        self, values: list[str], context_field: str = "", **kwargs: Any
    ) -> NormalizationResult:
        mapping = {v: v.upper() for v in set(values)}
        return NormalizationResult(mapping=mapping)

    def normalize(
        self, values: list[str], context_field: str = "", **kwargs: Any
    ) -> NormalizationResult:
        from catchfly._compat import run_sync

        return run_sync(self.anormalize(values, context_field=context_field))


class TestCompositeNormalization:
    async def test_routes_to_field_strategy(self) -> None:
        composite = CompositeNormalization(
            field_strategies={
                "city": DictionaryNormalization(
                    mapping={"NYC": "New York"}
                ),
                "name": _UppercaseNorm(),
            }
        )
        city_result = await composite.anormalize(["NYC"], context_field="city")
        assert city_result.mapping["NYC"] == "New York"

        name_result = await composite.anormalize(["alice"], context_field="name")
        assert name_result.mapping["alice"] == "ALICE"

    async def test_falls_back_to_default(self) -> None:
        composite = CompositeNormalization(
            field_strategies={"city": DictionaryNormalization(mapping={})},
            default=_UppercaseNorm(),
        )
        result = await composite.anormalize(["hello"], context_field="unknown")
        assert result.mapping["hello"] == "HELLO"

    async def test_no_strategy_raises(self) -> None:
        composite = CompositeNormalization(field_strategies={})
        with pytest.raises(NormalizationError, match="No normalization strategy"):
            await composite.anormalize(["val"], context_field="missing")

    def test_sync_wrapper(self) -> None:
        composite = CompositeNormalization(
            field_strategies={"x": _UppercaseNorm()}
        )
        result = composite.normalize(["abc"], context_field="x")
        assert result.mapping["abc"] == "ABC"

    async def test_pipeline_dict_autowrap(self) -> None:
        """Pipeline wraps dict normalization into CompositeNormalization."""
        from catchfly._types import Document
        from catchfly.pipeline import Pipeline

        from tests.unit.test_pipeline import (
            MockDiscovery,
            MockExtraction,
        )

        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization={
                "title": DictionaryNormalization(
                    mapping={"Extracted from doc0": "Normalized"}
                ),
            },
        )
        assert isinstance(pipeline.normalization, CompositeNormalization)

        docs = [Document(content="test", id="doc0")]
        result = await pipeline.arun(
            docs, domain_hint="test", normalize_fields=["title"]
        )
        assert "title" in result.normalizations
        assert result.normalizations["title"].mapping["Extracted from doc0"] == "Normalized"
