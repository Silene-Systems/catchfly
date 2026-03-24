"""Tests for CascadeNormalization."""

from __future__ import annotations

from typing import Any

from catchfly._types import NormalizationResult
from catchfly.normalization.base import NormalizationStrategy
from catchfly.normalization.cascade import CascadeNormalization
from catchfly.normalization.dictionary import DictionaryNormalization


class MockPassthroughStrategy:
    """Mock strategy that maps specific values and passes through the rest."""

    def __init__(self, known: dict[str, str]) -> None:
        self._known = known

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        mapping = {}
        for v in set(values):
            mapping[v] = self._known.get(v, v)
        return NormalizationResult(mapping=mapping, clusters=None, metadata={})

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        from catchfly._compat import run_sync

        return run_sync(self.anormalize(values, context_field=context_field, **kwargs))


class TestCascadeNormalization:
    def test_satisfies_protocol(self) -> None:
        cascade = CascadeNormalization(steps=[])
        assert isinstance(cascade, NormalizationStrategy)

    async def test_empty_values(self) -> None:
        cascade = CascadeNormalization(
            steps=[DictionaryNormalization(mapping={"a": "b"})]
        )
        result = await cascade.anormalize([])
        assert result.mapping == {}

    async def test_single_step_dictionary(self) -> None:
        cascade = CascadeNormalization(
            steps=[
                DictionaryNormalization(
                    mapping={"NYC": "New York", "LA": "Los Angeles"},
                    case_insensitive=True,
                    passthrough_unmapped=True,
                )
            ]
        )
        result = await cascade.anormalize(
            ["NYC", "LA", "Chicago"], context_field="city"
        )
        assert result.mapping["NYC"] == "New York"
        assert result.mapping["LA"] == "Los Angeles"
        assert result.mapping["Chicago"] == "Chicago"  # passthrough

    async def test_cascade_two_steps(self) -> None:
        """First step catches some, second step catches the rest."""
        step1 = DictionaryNormalization(
            mapping={"NYC": "New York"},
            passthrough_unmapped=True,
        )
        step2 = MockPassthroughStrategy(
            known={"Chi-town": "Chicago", "LA": "Los Angeles"}
        )
        cascade = CascadeNormalization(steps=[step1, step2])

        result = await cascade.anormalize(
            ["NYC", "Chi-town", "LA", "Boston"], context_field="city"
        )
        assert result.mapping["NYC"] == "New York"  # step 1
        assert result.mapping["Chi-town"] == "Chicago"  # step 2
        assert result.mapping["LA"] == "Los Angeles"  # step 2
        assert result.mapping["Boston"] == "Boston"  # passthrough

    async def test_unmapped_passthrough(self) -> None:
        """Values no step can handle pass through unchanged."""
        cascade = CascadeNormalization(
            steps=[
                DictionaryNormalization(
                    mapping={"a": "A"}, passthrough_unmapped=True
                )
            ]
        )
        result = await cascade.anormalize(["a", "b", "c"])
        assert result.mapping["a"] == "A"
        assert result.mapping["b"] == "b"
        assert result.mapping["c"] == "c"

    async def test_step_metadata(self) -> None:
        """Metadata tracks per-step statistics."""
        step1 = DictionaryNormalization(
            mapping={"x": "X"}, passthrough_unmapped=True
        )
        step2 = MockPassthroughStrategy(known={"y": "Y"})
        cascade = CascadeNormalization(steps=[step1, step2])

        result = await cascade.anormalize(["x", "y", "z"])
        steps = result.metadata["steps"]
        assert len(steps) == 2

        # Step 1 maps "x", leaves "y" and "z"
        assert steps[0]["strategy"] == "DictionaryNormalization"
        assert steps[0]["mapped_count"] == 1
        assert steps[0]["remaining_count"] == 2

        # Step 2 maps "y", leaves "z"
        assert steps[1]["strategy"] == "MockPassthroughStrategy"
        assert steps[1]["mapped_count"] == 1
        assert steps[1]["remaining_count"] == 1

    async def test_remaining_decreases(self) -> None:
        """Each step receives fewer values than the previous one."""
        calls: list[int] = []

        class TrackingStrategy:
            async def anormalize(
                self, values: list[str], **kwargs: Any
            ) -> NormalizationResult:
                calls.append(len(values))
                mapping = {v: v for v in values}
                # Map the first value to something else
                if values:
                    mapping[values[0]] = f"mapped_{values[0]}"
                return NormalizationResult(mapping=mapping)

        cascade = CascadeNormalization(
            steps=[TrackingStrategy(), TrackingStrategy(), TrackingStrategy()]
        )
        await cascade.anormalize(["a", "b", "c"])
        assert calls == [3, 2, 1]

    async def test_clusters_built_from_merged_mapping(self) -> None:
        """Clusters aggregate all raw values per canonical."""
        step1 = DictionaryNormalization(
            mapping={"NYC": "New York", "NY": "New York"},
            passthrough_unmapped=True,
        )
        step2 = MockPassthroughStrategy(known={"Big Apple": "New York"})
        cascade = CascadeNormalization(steps=[step1, step2])

        result = await cascade.anormalize(["NYC", "NY", "Big Apple"])
        assert "New York" in result.clusters
        members = sorted(result.clusters["New York"])
        assert members == ["Big Apple", "NY", "NYC"]

    async def test_usage_callback_propagated(self) -> None:
        """Usage callback is set on sub-strategies."""
        callbacks_received: list[str] = []

        class CallbackTracker:
            _usage_callback: Any = None

            async def anormalize(
                self, values: list[str], **kwargs: Any
            ) -> NormalizationResult:
                if self._usage_callback:
                    callbacks_received.append("got_callback")
                return NormalizationResult(
                    mapping={v: v for v in values}
                )

        cascade = CascadeNormalization(steps=[CallbackTracker()])
        cascade._usage_callback = lambda *a: None  # type: ignore[attr-defined]
        await cascade.anormalize(["test"])
        assert len(callbacks_received) == 1

    async def test_default_factory_llm_only(self) -> None:
        """default() with no dictionary/ontology creates LLM-only cascade."""
        cascade = CascadeNormalization.default(model="gpt-test")
        assert len(cascade.steps) == 1
        assert type(cascade.steps[0]).__name__ == "LLMCanonicalization"

    async def test_default_factory_with_dictionary(self) -> None:
        """default() with dictionary creates 2-step cascade."""
        cascade = CascadeNormalization.default(
            dictionary={"a": "A"}, model="gpt-test"
        )
        assert len(cascade.steps) == 2
        assert type(cascade.steps[0]).__name__ == "DictionaryNormalization"
        assert type(cascade.steps[1]).__name__ == "LLMCanonicalization"

    async def test_default_factory_full(self) -> None:
        """default() with all options creates 3-step cascade."""
        cascade = CascadeNormalization.default(
            dictionary={"a": "A"},
            model="gpt-test",
            ontology="tests/fixtures/test_ontology.json",
        )
        assert len(cascade.steps) == 3
        assert type(cascade.steps[0]).__name__ == "DictionaryNormalization"
        assert type(cascade.steps[1]).__name__ == "LLMCanonicalization"
        assert type(cascade.steps[2]).__name__ == "OntologyMapping"

    def test_sync_wrapper(self) -> None:
        """normalize() synchronous wrapper works."""
        cascade = CascadeNormalization(
            steps=[
                DictionaryNormalization(
                    mapping={"a": "A"}, passthrough_unmapped=True
                )
            ]
        )
        result = cascade.normalize(["a", "b"])
        assert result.mapping["a"] == "A"
        assert result.mapping["b"] == "b"

    async def test_no_steps(self) -> None:
        """Cascade with no steps passes everything through."""
        cascade = CascadeNormalization(steps=[])
        result = await cascade.anormalize(["a", "b"])
        assert result.mapping["a"] == "a"
        assert result.mapping["b"] == "b"

    async def test_metadata_strategy_name(self) -> None:
        """Metadata identifies this as cascade strategy."""
        cascade = CascadeNormalization(steps=[])
        result = await cascade.anormalize(["a"])
        assert result.metadata["strategy"] == "cascade"

    async def test_deduplicates_input(self) -> None:
        """Duplicate input values are deduplicated internally."""
        cascade = CascadeNormalization(
            steps=[
                DictionaryNormalization(
                    mapping={"a": "A"}, passthrough_unmapped=True
                )
            ]
        )
        result = await cascade.anormalize(["a", "a", "a", "b"])
        assert len(result.mapping) == 2
        assert result.mapping["a"] == "A"

    async def test_early_exit_when_all_mapped(self) -> None:
        """Later steps are skipped when all values are already mapped."""
        calls: list[str] = []

        class NeverCalledStrategy:
            async def anormalize(
                self, values: list[str], **kwargs: Any
            ) -> NormalizationResult:
                calls.append("called")
                return NormalizationResult(mapping={v: v for v in values})

        cascade = CascadeNormalization(
            steps=[
                DictionaryNormalization(
                    mapping={"a": "A", "b": "B"}, passthrough_unmapped=True
                ),
                NeverCalledStrategy(),
            ]
        )
        await cascade.anormalize(["a", "b"])
        assert len(calls) == 0  # second step never called
