"""Tests for SchemaOptimizer."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from catchfly._types import Document, Schema
from catchfly.discovery.optimizer import SchemaOptimizer
from catchfly.exceptions import DiscoveryError
from catchfly.providers.llm import LLMResponse


class MockOptimizerLLM:
    """Mock LLM that returns extraction results and enrichment suggestions."""

    def __init__(self) -> None:
        self._call_index = 0

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self._call_index += 1
        user_msg = messages[-1]["content"] if messages else ""

        # Extraction calls — return sample data
        if "Extract structured data" in user_msg:
            return LLMResponse(
                content=json.dumps(
                    {
                        "title": "Sample Product",
                        "rating": 4.5,
                        "price": 99.99,
                    }
                ),
                input_tokens=200,
                output_tokens=100,
            )

        # Enrichment calls — return field descriptions
        return LLMResponse(
            content=json.dumps(
                {
                    "fields": {
                        "title": {
                            "description": "The product name or title",
                            "examples": ["Samsung Galaxy S25", "Sony WH-1000XM6"],
                            "synonyms": ["product_name", "name", "item"],
                            "constraints": "Non-empty string",
                        },
                        "rating": {
                            "description": "Numeric rating given by reviewer",
                            "examples": ["9/10", "5 stars", "4.5"],
                            "synonyms": ["score", "grade", "stars"],
                            "constraints": "Number between 0 and 10",
                        },
                        "price": {
                            "description": "Price paid for the product in USD",
                            "examples": ["$1,299", "$399", "$99"],
                            "synonyms": ["cost", "price_paid", "amount"],
                            "constraints": "Positive number",
                        },
                    }
                }
            ),
            input_tokens=300,
            output_tokens=200,
        )

    async def astructured_complete(
        self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kwargs: Any
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


def _make_docs(n: int = 5) -> list[Document]:
    return [
        Document(
            content=f"Product review {i}: Great product, rating {i + 3}/10, price ${i * 100 + 99}",
            id=f"doc{i}",
        )
        for i in range(n)
    ]


def _make_schema() -> Schema:
    return Schema(
        model=None,
        json_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "rating": {"type": "number"},
                "price": {"type": "number"},
            },
            "required": ["title"],
        },
        lineage=["SinglePassDiscovery"],
    )


class TestSchemaOptimizer:
    async def test_optimize_enriches_metadata(self) -> None:
        mock_llm = MockOptimizerLLM()
        optimizer = SchemaOptimizer(model="mock", num_iterations=2)

        import catchfly.discovery.optimizer as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            result = await optimizer.aoptimize(_make_schema(), _make_docs())
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        # field_metadata should be populated
        assert "title" in result.field_metadata
        assert "description" in result.field_metadata["title"]
        assert "examples" in result.field_metadata["title"]
        assert "synonyms" in result.field_metadata["title"]

    async def test_lineage_tracks_iterations(self) -> None:
        mock_llm = MockOptimizerLLM()
        optimizer = SchemaOptimizer(model="mock", num_iterations=3)

        import catchfly.discovery.optimizer as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            result = await optimizer.aoptimize(_make_schema(), _make_docs())
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        assert "SinglePassDiscovery" in result.lineage
        assert "SchemaOptimizer:iter1" in result.lineage
        assert "SchemaOptimizer:iter2" in result.lineage
        assert "SchemaOptimizer:iter3" in result.lineage

    async def test_accepts_pydantic_model(self) -> None:
        class ReviewModel(BaseModel):
            title: str
            rating: float

        mock_llm = MockOptimizerLLM()
        optimizer = SchemaOptimizer(model="mock", num_iterations=1)

        import catchfly.discovery.optimizer as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            result = await optimizer.aoptimize(ReviewModel, _make_docs(3))
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        assert result.model is ReviewModel
        assert "title" in result.field_metadata

    async def test_empty_docs_raises(self) -> None:
        optimizer = SchemaOptimizer()
        with pytest.raises(DiscoveryError, match="No test documents"):
            await optimizer.aoptimize(_make_schema(), [])

    def test_analyze_gaps_low_coverage(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string"},
                "rare_field": {"type": "string"},
            }
        }
        extracted = [
            {"name": "A", "rare_field": None},
            {"name": "B", "rare_field": None},
            {"name": "C", "rare_field": "value"},
        ]
        optimizer = SchemaOptimizer(model="mock")
        analysis = optimizer._analyze_gaps(schema, extracted)
        assert "rare_field" in analysis
        assert "low coverage" in analysis["rare_field"]

    def test_analyze_gaps_empty(self) -> None:
        optimizer = SchemaOptimizer(model="mock")
        assert optimizer._analyze_gaps({}, []) == {}

    def test_parse_enrichment_valid(self) -> None:
        content = json.dumps({"fields": {"name": {"description": "A name"}}})
        result = SchemaOptimizer._parse_enrichment(content)
        assert result["fields"]["name"]["description"] == "A name"

    def test_parse_enrichment_with_fences(self) -> None:
        content = '```json\n{"fields": {"x": {"description": "X"}}}\n```'
        result = SchemaOptimizer._parse_enrichment(content)
        assert "x" in result["fields"]

    def test_normalize_schema_from_schema(self) -> None:
        schema = _make_schema()
        result = SchemaOptimizer._normalize_schema(schema)
        assert result is schema

    def test_normalize_schema_from_model(self) -> None:
        class M(BaseModel):
            name: str

        result = SchemaOptimizer._normalize_schema(M)
        assert result.model is M
        assert "name" in result.json_schema["properties"]

    def test_normalize_schema_invalid(self) -> None:
        with pytest.raises(DiscoveryError):
            SchemaOptimizer._normalize_schema("not a schema")  # type: ignore[arg-type]

    def test_sync_wrapper(self) -> None:
        mock_llm = MockOptimizerLLM()
        optimizer = SchemaOptimizer(model="mock", num_iterations=1)

        import catchfly.discovery.optimizer as mod

        original = mod.OpenAICompatibleClient
        mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        try:
            result = optimizer.optimize(_make_schema(), _make_docs(2))
        finally:
            mod.OpenAICompatibleClient = original  # type: ignore[assignment]

        assert "title" in result.field_metadata
