"""Tests for SinglePassDiscovery."""

from __future__ import annotations

import json
from typing import Any

import pytest

from catchfly._types import Document
from catchfly.discovery.single_pass import SinglePassDiscovery
from catchfly.exceptions import DiscoveryError
from catchfly.providers.llm import LLMResponse


class MockDiscoveryLLM:
    """Mock LLM client that returns a valid JSON Schema."""

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.schema = schema or {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "score": {"type": "number"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
        }
        self.calls: list[Any] = []

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            content=json.dumps(self.schema),
            input_tokens=500,
            output_tokens=200,
            model="mock",
        )

    async def astructured_complete(
        self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kwargs: Any
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class TestSinglePassDiscovery:
    def _make_docs(self, n: int = 3) -> list[Document]:
        return [
            Document(content=f"Document {i} content about topic {i}", id=f"doc{i}")
            for i in range(n)
        ]

    async def test_discovers_schema(self) -> None:
        mock_llm = MockDiscoveryLLM()
        discovery = SinglePassDiscovery(model="mock")

        # Monkey-patch the client creation
        original_adiscover = discovery.adiscover

        async def patched_adiscover(
            documents: list[Document], *, domain_hint: str | None = None, **kwargs: Any
        ) -> Any:
            # Replace client with mock
            import catchfly.discovery.single_pass as sp

            original_client_class = sp.OpenAICompatibleClient
            sp.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
            try:
                return await original_adiscover(documents, domain_hint=domain_hint, **kwargs)
            finally:
                sp.OpenAICompatibleClient = original_client_class  # type: ignore[assignment]

        schema = await patched_adiscover(self._make_docs())

        assert schema.json_schema["properties"]["title"]["type"] == "string"
        assert schema.model is not None
        assert "SinglePassDiscovery" in schema.lineage

    async def test_empty_docs_raises(self) -> None:
        discovery = SinglePassDiscovery()
        with pytest.raises(DiscoveryError, match="No documents"):
            await discovery.adiscover([])

    def test_parse_schema_valid(self) -> None:
        schema_json = json.dumps({"type": "object", "properties": {"name": {"type": "string"}}})
        result = SinglePassDiscovery._parse_schema(schema_json)
        assert "name" in result["properties"]

    def test_parse_schema_with_fences(self) -> None:
        schema_json = '```json\n{"type": "object", "properties": {"x": {"type": "integer"}}}\n```'
        result = SinglePassDiscovery._parse_schema(schema_json)
        assert "x" in result["properties"]

    def test_parse_schema_invalid_json(self) -> None:
        with pytest.raises(DiscoveryError, match="not valid JSON"):
            SinglePassDiscovery._parse_schema("not json at all")

    def test_parse_schema_no_properties(self) -> None:
        with pytest.raises(DiscoveryError, match="properties"):
            SinglePassDiscovery._parse_schema('{"type": "object"}')

    def test_parse_schema_nested_wrapper(self) -> None:
        """Handle case where LLM wraps schema in a top-level key."""
        wrapped = json.dumps(
            {
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            }
        )
        result = SinglePassDiscovery._parse_schema(wrapped)
        assert "name" in result["properties"]

    def test_sample_documents(self) -> None:
        discovery = SinglePassDiscovery(num_samples=3)
        docs = self._make_docs(10)
        sample = discovery._sample_documents(docs)
        assert len(sample) == 3

    def test_sample_fewer_than_limit(self) -> None:
        discovery = SinglePassDiscovery(num_samples=10)
        docs = self._make_docs(3)
        sample = discovery._sample_documents(docs)
        assert len(sample) == 3

    def test_build_pydantic_model_success(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        model = SinglePassDiscovery._build_pydantic_model(schema)
        assert model is not None
        instance = model(name="test")
        assert instance.name == "test"  # type: ignore[attr-defined]

    def test_build_pydantic_model_failure(self) -> None:
        # Empty properties should cause failure
        model = SinglePassDiscovery._build_pydantic_model({"type": "object"})
        assert model is None
