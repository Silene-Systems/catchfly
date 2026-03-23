"""Tests for LLMCanonicalization."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from catchfly.exceptions import NormalizationError
from catchfly.normalization.llm_canonical import (
    LLMCanonicalization,
    _build_system_prompt,
    _build_hierarchical_system_prompt,
)
from catchfly.providers.llm import LLMResponse


class MockCanonicalizationLLM:
    """Mock LLM that returns grouping JSON.

    Supports two modes:
    - Single response: pass ``groups`` (backward compatible).
    - Call-indexed responses: pass ``responses`` list — each call returns the
      next entry (cycling if exhausted).
    """

    def __init__(
        self,
        groups: list[dict[str, Any]] | None = None,
        responses: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        if responses is not None:
            self._responses = responses
        else:
            self._responses = [
                groups
                or [
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
            ]
        self.call_count = 0
        self.captured_messages: list[list[dict[str, str]]] = []

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.captured_messages.append(messages)
        groups = self._responses[self.call_count % len(self._responses)]
        self.call_count += 1
        return LLMResponse(
            content=json.dumps({"groups": groups}),
            input_tokens=200,
            output_tokens=150,
        )

    async def astructured_complete(
        self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kwargs: Any
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


def _patch_client(mod: Any, mock_llm: MockCanonicalizationLLM):
    """Context-manager-style helper for monkey-patching OpenAICompatibleClient."""
    original = mod.OpenAICompatibleClient
    mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    return original


class TestLLMCanonicalization:
    async def test_basic_normalization(self) -> None:
        mock_llm = MockCanonicalizationLLM()
        normalizer = LLMCanonicalization(model="mock")

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
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

        original = _patch_client(mod, mock_llm)
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
        normalizer = LLMCanonicalization(
            model="mock", max_values_per_prompt=5, batch_size=3, hierarchical_merge=False
        )

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
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

        original = _patch_client(mod, mock_llm)
        try:
            result = normalizer.normalize(
                ["NYC", "New York", "LA", "Los Angeles"], context_field="city"
            )
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        assert result.mapping["NYC"] == "New York"


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt and schema-aware prompting."""

    def test_without_metadata_returns_base_prompt(self) -> None:
        prompt = _build_system_prompt("city")
        assert "field \"city\"" in prompt
        assert "Schema context" not in prompt

    def test_with_none_metadata(self) -> None:
        prompt = _build_system_prompt("city", None)
        assert "Schema context" not in prompt

    def test_with_empty_metadata(self) -> None:
        prompt = _build_system_prompt("city", {})
        assert "Schema context" not in prompt

    def test_with_full_metadata(self) -> None:
        metadata = {
            "description": "Type of product",
            "examples": ["Rings/Bands", "Necklaces", "Earrings"],
            "synonyms": ["product category", "item type"],
            "constraints": "Must be a standard product category name",
        }
        prompt = _build_system_prompt("product_type", metadata)
        assert "Schema context for this field:" in prompt
        assert "Field description: Type of product" in prompt
        assert "Example canonical values: Rings/Bands, Necklaces, Earrings" in prompt
        assert "Field aliases: product category, item type" in prompt
        assert "Constraints: Must be a standard product category name" in prompt

    def test_with_partial_metadata(self) -> None:
        metadata = {"description": "Brand name"}
        prompt = _build_system_prompt("brand", metadata)
        assert "Field description: Brand name" in prompt
        assert "Example canonical values" not in prompt
        assert "Field aliases" not in prompt

    def test_examples_capped_at_10(self) -> None:
        metadata = {"examples": [f"ex_{i}" for i in range(20)]}
        prompt = _build_system_prompt("field", metadata)
        assert "ex_9" in prompt
        assert "ex_10" not in prompt

    def test_hierarchical_prompt_with_metadata(self) -> None:
        metadata = {"description": "Type of product"}
        prompt = _build_hierarchical_system_prompt("product_type", metadata)
        assert "data consolidation assistant" in prompt
        assert "Field description: Type of product" in prompt


class TestSchemaAwareIntegration:
    """Integration tests: field_metadata flows into LLM prompts."""

    async def test_schema_context_reaches_llm(self) -> None:
        """When field_metadata is passed, the system prompt includes schema context."""
        mock_llm = MockCanonicalizationLLM()
        normalizer = LLMCanonicalization(model="mock")

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
        try:
            metadata = {
                "description": "Type of jewelry product",
                "examples": ["Rings", "Necklaces"],
            }
            await normalizer.anormalize(
                ["NYC", "New York"], context_field="product_type", field_metadata=metadata
            )
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        assert mock_llm.call_count == 1
        system_msg = mock_llm.captured_messages[0][0]["content"]
        assert "Schema context for this field:" in system_msg
        assert "Type of jewelry product" in system_msg
        assert "Rings, Necklaces" in system_msg

    async def test_no_metadata_no_schema_context(self) -> None:
        """When field_metadata is not passed, system prompt has no schema context."""
        mock_llm = MockCanonicalizationLLM()
        normalizer = LLMCanonicalization(model="mock")

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
        try:
            await normalizer.anormalize(["NYC", "New York"], context_field="city")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        system_msg = mock_llm.captured_messages[0][0]["content"]
        assert "Schema context" not in system_msg


class TestApplyHierarchicalMerge:
    """Tests for _apply_hierarchical_merge static method."""

    def test_merges_two_groups(self) -> None:
        original = [
            {"canonical": "Wedding Rings", "members": ["wedding band", "bridal ring"], "rationale": "rings"},
            {"canonical": "Rings & Bands", "members": ["ring", "band"], "rationale": "bands"},
            {"canonical": "Earrings", "members": ["earring", "stud"], "rationale": "ear jewelry"},
        ]
        instructions = [
            {
                "canonical": "Rings",
                "members": ["Wedding Rings", "Rings & Bands"],
                "rationale": "same concept",
            },
            {
                "canonical": "Earrings",
                "members": ["Earrings"],
                "rationale": "distinct",
            },
        ]
        result = LLMCanonicalization._apply_hierarchical_merge(original, instructions)

        # Should be 2 groups: "Rings" (merged) and "Earrings"
        assert len(result) == 2
        canonicals = {g["canonical"] for g in result}
        assert canonicals == {"Rings", "Earrings"}

        rings_group = [g for g in result if g["canonical"] == "Rings"][0]
        assert set(rings_group["members"]) == {"wedding band", "bridal ring", "ring", "band"}

    def test_no_merges_keeps_all_groups(self) -> None:
        original = [
            {"canonical": "A", "members": ["a1"], "rationale": "r1"},
            {"canonical": "B", "members": ["b1"], "rationale": "r2"},
        ]
        instructions = [
            {"canonical": "A", "members": ["A"], "rationale": ""},
            {"canonical": "B", "members": ["B"], "rationale": ""},
        ]
        result = LLMCanonicalization._apply_hierarchical_merge(original, instructions)
        assert len(result) == 2

    def test_preserves_unconsumed_groups(self) -> None:
        """Groups not mentioned in instructions are preserved."""
        original = [
            {"canonical": "A", "members": ["a1"], "rationale": "r1"},
            {"canonical": "B", "members": ["b1"], "rationale": "r2"},
            {"canonical": "C", "members": ["c1"], "rationale": "r3"},
        ]
        # Only merge A and B; C is not in any instruction
        instructions = [
            {"canonical": "AB", "members": ["A", "B"], "rationale": "merged"},
        ]
        result = LLMCanonicalization._apply_hierarchical_merge(original, instructions)
        assert len(result) == 2
        canonicals = {g["canonical"] for g in result}
        assert canonicals == {"AB", "C"}

    def test_ignores_unknown_canonical_names(self) -> None:
        original = [
            {"canonical": "A", "members": ["a1"], "rationale": "r1"},
        ]
        instructions = [
            {"canonical": "A", "members": ["A", "NonExistent"], "rationale": ""},
        ]
        result = LLMCanonicalization._apply_hierarchical_merge(original, instructions)
        assert len(result) == 1
        assert result[0]["members"] == ["a1"]


class TestHierarchicalMerge:
    """Integration tests for hierarchical merge in map-reduce pipeline."""

    async def test_hierarchical_merge_reduces_groups(self) -> None:
        """Map-reduce with hierarchical merge consolidates semantically similar groups."""
        # Batch responses: each batch produces its own groups
        batch_response = [
            {"canonical": "Wedding Rings", "members": ["wedding band", "bridal ring"], "rationale": "rings"},
        ]
        # Hierarchical merge response: merges the duplicate canonicals
        merge_response = [
            {
                "canonical": "Rings",
                "members": ["Wedding Rings", "Rings & Bands"],
                "rationale": "same concept",
            },
            {
                "canonical": "Earrings",
                "members": ["Earrings"],
                "rationale": "distinct",
            },
        ]
        # First 4 calls are map batches, 5th call is hierarchical merge
        mock_llm = MockCanonicalizationLLM(
            responses=[batch_response, batch_response, batch_response, batch_response, merge_response]
        )
        normalizer = LLMCanonicalization(
            model="mock", max_values_per_prompt=3, batch_size=2, hierarchical_merge=True
        )

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
        try:
            values = [f"val_{i}" for i in range(8)]
            result = await normalizer.anormalize(values, context_field="product_type")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        # Hierarchical merge should have been called (more than just batch calls)
        assert mock_llm.call_count > 4

    async def test_hierarchical_merge_disabled(self) -> None:
        """When hierarchical_merge=False, no extra LLM call is made."""
        mock_llm = MockCanonicalizationLLM(
            groups=[
                {"canonical": "A", "members": ["a1", "a2"], "rationale": "group A"},
            ]
        )
        normalizer = LLMCanonicalization(
            model="mock", max_values_per_prompt=3, batch_size=2, hierarchical_merge=False
        )

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
        try:
            values = [f"val_{i}" for i in range(8)]
            await normalizer.anormalize(values, context_field="test")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        # 4 batch calls (8 values / batch_size=2), no hierarchical merge call
        assert mock_llm.call_count == 4

    async def test_hierarchical_merge_skipped_for_single_batch(self) -> None:
        """When values fit in one batch, hierarchical merge is not invoked."""
        mock_llm = MockCanonicalizationLLM(
            groups=[
                {"canonical": "A", "members": ["a1", "a2"], "rationale": "group A"},
                {"canonical": "B", "members": ["b1", "b2"], "rationale": "group B"},
            ]
        )
        normalizer = LLMCanonicalization(
            model="mock", max_values_per_prompt=200, hierarchical_merge=True
        )

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
        try:
            values = ["a1", "a2", "b1", "b2"]
            await normalizer.anormalize(values, context_field="test")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        # Only 1 call — single batch, no map-reduce, no hierarchical merge
        assert mock_llm.call_count == 1

    async def test_hierarchical_merge_preserves_all_members(self) -> None:
        """After hierarchical merge, every original value is in exactly one group."""
        # Batch response: 2 values per group
        batch_groups = [
            {"canonical": "Group A", "members": ["a1", "a2"], "rationale": "A"},
        ]
        # Merge: consolidate all into one
        merge_groups = [
            {"canonical": "Consolidated", "members": ["Group A"], "rationale": "all same"},
        ]
        mock_llm = MockCanonicalizationLLM(
            responses=[batch_groups, batch_groups, batch_groups, merge_groups]
        )
        normalizer = LLMCanonicalization(
            model="mock", max_values_per_prompt=3, batch_size=2, hierarchical_merge=True
        )

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
        try:
            values = [f"val_{i}" for i in range(6)]
            result = await normalizer.anormalize(values, context_field="test")
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        # Every input value must appear in the mapping
        for v in set(values):
            assert v in result.mapping

    async def test_hierarchical_merge_uses_schema_context(self) -> None:
        """Hierarchical merge prompt includes schema context when field_metadata is provided."""
        batch_groups = [
            {"canonical": "A", "members": ["a1"], "rationale": ""},
            {"canonical": "B", "members": ["b1"], "rationale": ""},
        ]
        merge_groups = [
            {"canonical": "A", "members": ["A"], "rationale": ""},
            {"canonical": "B", "members": ["B"], "rationale": ""},
        ]
        mock_llm = MockCanonicalizationLLM(
            responses=[batch_groups, batch_groups, merge_groups]
        )
        normalizer = LLMCanonicalization(
            model="mock", max_values_per_prompt=3, batch_size=2, hierarchical_merge=True
        )

        import catchfly.normalization.llm_canonical as mod

        original = _patch_client(mod, mock_llm)
        try:
            metadata = {"description": "Product category"}
            values = [f"val_{i}" for i in range(4)]
            await normalizer.anormalize(
                values, context_field="product_type", field_metadata=metadata
            )
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        # The last call is the hierarchical merge — check its system prompt
        merge_system_msg = mock_llm.captured_messages[-1][0]["content"]
        assert "data consolidation assistant" in merge_system_msg
        assert "Product category" in merge_system_msg
