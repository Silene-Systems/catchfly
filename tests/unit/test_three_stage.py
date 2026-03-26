"""Tests for ThreeStageDiscovery."""

from __future__ import annotations

import json
from typing import Any

import pytest

from catchfly._types import Document
from catchfly.discovery.three_stage import ThreeStageDiscovery
from catchfly.exceptions import DiscoveryError
from catchfly.providers.llm import LLMResponse


class MockThreeStageLLM:
    """Mock LLM that returns stage-appropriate responses."""

    def __init__(self) -> None:
        self._call_count = 0

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self._call_count += 1
        user_msg = messages[-1]["content"] if messages else ""
        sys_msg = messages[0]["content"] if messages else ""

        # Stage 1: initial schema (via SinglePassDiscovery)
        if "propose" in user_msg.lower() and "json schema" in user_msg.lower():
            return LLMResponse(
                content=json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "rating": {"type": "number"},
                            "category": {"type": "string"},
                        },
                        "required": ["title"],
                    }
                ),
                input_tokens=500,
                output_tokens=200,
            )

        # Refinement/expansion — detect by system prompt
        if "refinement" in sys_msg.lower() or "expansion" in sys_msg.lower():
            return LLMResponse(
                content=json.dumps(
                    {
                        "add_fields": {
                            "price": {"type": "number", "description": "Price in USD"},
                        },
                        "remove_fields": [],
                        "modify_fields": {},
                        "rationale": "Price appears in most documents",
                    }
                ),
                input_tokens=300,
                output_tokens=150,
            )

        # Extraction calls (default)
        return LLMResponse(
            content=json.dumps(
                {
                    "title": "Sample",
                    "rating": 4.5,
                    "category": "electronics",
                }
            ),
            input_tokens=200,
            output_tokens=100,
        )

    async def astructured_complete(
        self,
        messages: list[dict[str, str]],
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


def _make_docs(n: int = 20) -> list[Document]:
    return [
        Document(
            content=f"Product {i}: Great item, rating {i % 5 + 3}/5, price ${i * 10 + 99}",
            id=f"doc{i}",
        )
        for i in range(n)
    ]


def _patch(mock_llm: MockThreeStageLLM) -> tuple[Any, Any]:
    """Patch both SinglePass and ThreeStage to use mock."""
    import catchfly.discovery.single_pass as sp_mod
    import catchfly.discovery.three_stage as ts_mod

    orig_sp = sp_mod.OpenAICompatibleClient
    orig_ts = ts_mod.OpenAICompatibleClient
    sp_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    ts_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
    return orig_sp, orig_ts


def _unpatch(orig_sp: Any, orig_ts: Any) -> None:
    import catchfly.discovery.single_pass as sp_mod
    import catchfly.discovery.three_stage as ts_mod

    sp_mod.OpenAICompatibleClient = orig_sp  # type: ignore[assignment]
    ts_mod.OpenAICompatibleClient = orig_ts  # type: ignore[assignment]


class TestThreeStageDiscovery:
    async def test_full_three_stages(self) -> None:
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(model="mock")

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = await discovery.adiscover(_make_docs(60), domain_hint="products")
        finally:
            _unpatch(orig_sp, orig_ts)

        assert schema.json_schema.get("properties") is not None
        # Stage 2+3 should have added "price" field
        assert "price" in schema.json_schema["properties"]
        assert "ThreeStageDiscovery:stage3" in schema.lineage
        # Report metadata should be in field_metadata
        report = schema.field_metadata.get("_discovery_report", {})
        assert report.get("stages_completed") == 3

    async def test_human_review_stops_at_stage2(self) -> None:
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(model="mock", human_review=True)

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = await discovery.adiscover(_make_docs(20))
        finally:
            _unpatch(orig_sp, orig_ts)

        assert "ThreeStageDiscovery:stage2" in schema.lineage
        report = schema.field_metadata.get("_discovery_report", {})
        assert report.get("stages_completed") == 2

    async def test_empty_docs_raises(self) -> None:
        discovery = ThreeStageDiscovery()
        with pytest.raises(DiscoveryError, match="No documents"):
            await discovery.adiscover([])

    async def test_few_docs_still_works(self) -> None:
        """Works even with fewer docs than stage sample sizes."""
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(
            model="mock",
            stage1_samples=3,
            stage2_samples=10,
            stage3_samples=50,
        )

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = await discovery.adiscover(_make_docs(5))
        finally:
            _unpatch(orig_sp, orig_ts)

        assert schema.json_schema.get("properties") is not None

    def test_sync_wrapper(self) -> None:
        mock = MockThreeStageLLM()
        discovery = ThreeStageDiscovery(model="mock")

        orig_sp, orig_ts = _patch(mock)
        try:
            schema = discovery.discover(_make_docs(10))
        finally:
            _unpatch(orig_sp, orig_ts)

        assert schema.json_schema.get("properties") is not None

    def test_compute_coverage(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string"},
                "rare": {"type": "string"},
            }
        }
        extracted = [
            {"name": "A", "rare": None},
            {"name": "B", "rare": None},
            {"name": "C", "rare": "val"},
        ]
        coverage = ThreeStageDiscovery._compute_coverage(schema, extracted)
        assert coverage["name"] == pytest.approx(1.0)
        assert coverage["rare"] == pytest.approx(1 / 3)

    def test_compute_coverage_empty(self) -> None:
        assert ThreeStageDiscovery._compute_coverage({}, []) == {}

    def test_apply_changes_add(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        changes = {
            "add_fields": {"age": {"type": "integer"}},
            "remove_fields": [],
            "modify_fields": {},
        }
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert "age" in result["properties"]
        assert "name" in result["properties"]

    def test_apply_changes_remove(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "junk": {"type": "string"},
            },
            "required": ["name", "junk"],
        }
        changes = {"remove_fields": ["junk"]}
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert "junk" not in result["properties"]
        assert "junk" not in result["required"]

    def test_apply_changes_modify(self) -> None:
        schema = {
            "type": "object",
            "properties": {"rating": {"type": "string"}},
        }
        changes = {"modify_fields": {"rating": {"type": "number"}}}
        result = ThreeStageDiscovery._apply_changes(schema, changes)
        assert result["properties"]["rating"]["type"] == "number"

    def test_parse_changes_valid(self) -> None:
        content = json.dumps({"add_fields": {"x": {"type": "string"}}})
        result = ThreeStageDiscovery._parse_changes(content)
        assert "x" in result["add_fields"]

    def test_parse_changes_invalid_json(self) -> None:
        result = ThreeStageDiscovery._parse_changes("not json")
        assert result == {}

    def test_parse_changes_with_fences(self) -> None:
        inner = json.dumps({"add_fields": {"y": {"type": "integer"}}})
        content = f"```json\n{inner}\n```"
        result = ThreeStageDiscovery._parse_changes(content)
        assert "y" in result["add_fields"]

    def test_sample_fewer_than_n(self) -> None:
        docs = _make_docs(3)
        result = ThreeStageDiscovery._sample(docs, 10)
        assert len(result) == 3

    def test_sample_exact_n(self) -> None:
        docs = _make_docs(10)
        result = ThreeStageDiscovery._sample(docs, 5)
        assert len(result) == 5

    async def test_discovery_raises_on_empty_schema(self) -> None:
        """ThreeStageDiscovery should raise DiscoveryError if schema has no fields.

        Previously it silently returned an empty Schema with model=None,
        causing downstream extraction to crash with AttributeError.
        """

        class EmptySchemaLLM:
            async def acomplete(self, messages: list[dict[str, str]], **kw: Any) -> LLMResponse:
                return LLMResponse(
                    content=json.dumps({"type": "object", "properties": {}}),
                    input_tokens=100,
                    output_tokens=50,
                )

            async def astructured_complete(self, messages: Any, **kw: Any) -> LLMResponse:
                return await self.acomplete(messages, **kw)

        mock = EmptySchemaLLM()
        discovery = ThreeStageDiscovery(model="mock")
        orig_sp, orig_ts = _patch(mock)  # type: ignore[arg-type]
        try:
            with pytest.raises(DiscoveryError):
                await discovery.adiscover(_make_docs(10), domain_hint="test")
        finally:
            _unpatch(orig_sp, orig_ts)

    def test_build_schema_raises_on_empty_properties(self) -> None:
        """_build_schema raises DiscoveryError when properties is empty."""
        empty_schema: dict[str, Any] = {"type": "object", "properties": {}}
        with pytest.raises(DiscoveryError, match="no properties"):
            ThreeStageDiscovery._build_schema(empty_schema, {}, {"stages_completed": 1}, stage=1)

    def test_build_schema_raises_on_conversion_failure(self) -> None:
        """_build_schema raises DiscoveryError when Pydantic conversion fails."""
        bad_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": "not_a_dict"},
        }
        with pytest.raises(DiscoveryError, match="failed to build Pydantic model"):
            ThreeStageDiscovery._build_schema(bad_schema, {}, {"stages_completed": 2}, stage=2)
