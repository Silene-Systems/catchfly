"""Tests for LLMCanonicalization."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from catchfly.exceptions import NormalizationError
from catchfly.normalization.llm_canonical import LLMCanonicalization
from catchfly.providers.llm import LLMResponse


class MockCanonicalizationLLM:
    """Mock LLM that returns grouping JSON."""

    def __init__(self, groups: list[dict[str, Any]] | None = None) -> None:
        self._groups = groups or [
            {
                "canonical": "New York",
                "members": ["New York", "NYC", "NY"],
                "rationale": "Same city",
            },
            {
                "canonical": "Los Angeles",
                "members": ["Los Angeles", "LA", "L.A."],
                "rationale": "Same city",
            },
        ]

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content=json.dumps({"groups": self._groups}),
            input_tokens=200,
            output_tokens=150,
        )

    async def astructured_complete(
        self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kwargs: Any
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class TestLLMCanonicalization:
    async def test_basic_normalization(self) -> None:
        mock_llm = MockCanonicalizationLLM()
        normalizer = LLMCanonicalization(model="mock")

        import catchfly.normalization.llm_canonical as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            values = ["New York", "NYC", "NY", "Los Angeles", "LA", "L.A."]
            result = await normalizer.anormalize(values, context_field="city")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        assert result.mapping["NYC"] == "New York"
        assert result.mapping["LA"] == "Los Angeles"
        assert len(result.clusters) == 2

    async def test_empty_values(self) -> None:
        normalizer = LLMCanonicalization()
        result = await normalizer.anormalize([])
        assert result.mapping == {}

    async def test_single_value(self) -> None:
        normalizer = LLMCanonicalization()
        result = await normalizer.anormalize(["hello"])
        assert result.mapping == {"hello": "hello"}

    async def test_explain(self) -> None:
        mock_llm = MockCanonicalizationLLM()
        normalizer = LLMCanonicalization(model="mock")

        import catchfly.normalization.llm_canonical as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            result = await normalizer.anormalize(["NYC", "New York"], context_field="city")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        explanation = result.explain("NYC")
        assert "New York" in explanation

    async def test_map_reduce_splits_batches(self) -> None:
        """Values exceeding max_values_per_prompt trigger map-reduce."""
        mock_llm = MockCanonicalizationLLM(
            groups=[
                {"canonical": "A", "members": ["a1", "a2"], "rationale": "group A"},
            ]
        )
        normalizer = LLMCanonicalization(model="mock", max_values_per_prompt=5, batch_size=3)

        import catchfly.normalization.llm_canonical as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            values = [f"val_{i}" for i in range(10)]
            result = await normalizer.anormalize(values, context_field="test")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        # All values should be mapped (ungrouped ones become singletons)
        assert len(result.mapping) >= len(set(values))

    def test_parse_groups_valid(self) -> None:
        content = json.dumps(
            {
                "groups": [
                    {"canonical": "X", "members": ["x1", "x2"], "rationale": "same"},
                ]
            }
        )
        normalizer = LLMCanonicalization()
        groups = normalizer._parse_groups(content, ["x1", "x2"])
        assert len(groups) == 1
        assert groups[0]["canonical"] == "X"

    def test_parse_groups_missing_values_become_singletons(self) -> None:
        content = json.dumps(
            {
                "groups": [
                    {"canonical": "X", "members": ["x1"], "rationale": "known"},
                ]
            }
        )
        normalizer = LLMCanonicalization()
        groups = normalizer._parse_groups(content, ["x1", "orphan"])
        # "orphan" should be added as singleton
        orphan_group = [g for g in groups if "orphan" in g["members"]]
        assert len(orphan_group) == 1
        assert orphan_group[0]["canonical"] == "orphan"

    def test_parse_groups_invalid_json(self) -> None:
        normalizer = LLMCanonicalization()
        with pytest.raises(NormalizationError, match="not valid JSON"):
            normalizer._parse_groups("not json", ["a"])

    def test_parse_groups_with_fences(self) -> None:
        inner = json.dumps({"groups": [{"canonical": "Y", "members": ["y1"], "rationale": "ok"}]})
        content = f"```json\n{inner}\n```"
        normalizer = LLMCanonicalization()
        groups = normalizer._parse_groups(content, ["y1"])
        assert groups[0]["canonical"] == "Y"

    def test_merge_groups_case_insensitive(self) -> None:
        groups = [
            {"canonical": "New York", "members": ["NYC"], "rationale": "batch 1"},
            {"canonical": "new york", "members": ["NY"], "rationale": "batch 2"},
        ]
        merged = LLMCanonicalization._merge_groups(groups)
        assert len(merged) == 1
        assert "NYC" in merged[0]["members"]
        assert "NY" in merged[0]["members"]

    def test_merge_groups_distinct(self) -> None:
        groups = [
            {"canonical": "New York", "members": ["NYC"], "rationale": "a"},
            {"canonical": "Los Angeles", "members": ["LA"], "rationale": "b"},
        ]
        merged = LLMCanonicalization._merge_groups(groups)
        assert len(merged) == 2

    def test_build_result(self) -> None:
        groups = [
            {"canonical": "Cat", "members": ["cat", "cats", "kitten"], "rationale": "felines"},
        ]
        counts: Counter[str] = Counter({"cat": 5, "cats": 3, "kitten": 1})
        result = LLMCanonicalization._build_result(groups, counts, "animal")

        assert result.mapping["cats"] == "Cat"
        assert result.mapping["kitten"] == "Cat"
        assert result.clusters["Cat"] == ["cat", "cats", "kitten"]
        assert result.metadata["n_groups"] == 1

    def test_sync_wrapper(self) -> None:
        mock_llm = MockCanonicalizationLLM()
        normalizer = LLMCanonicalization(model="mock")

        import catchfly.normalization.llm_canonical as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            result = normalizer.normalize(
                ["NYC", "New York", "LA", "Los Angeles"], context_field="city"
            )
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        assert result.mapping["NYC"] == "New York"
