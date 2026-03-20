"""Integration test: full pipeline end-to-end with mocks."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from catchfly import Document, Pipeline
from catchfly.demo import load_samples
from catchfly.providers.llm import LLMResponse


class MockPipelineLLM:
    """Mock LLM that returns appropriate responses for discovery and extraction."""

    def __init__(self) -> None:
        self._call_count = 0

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self._call_count += 1
        user_msg = messages[-1]["content"] if messages else ""

        # Discovery call — return a schema
        if "JSON Schema" in user_msg and "propose" in user_msg.lower():
            schema = {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "rating": {"type": "number"},
                    "pros": {"type": "array", "items": {"type": "string"}},
                    "price": {"type": "number"},
                },
                "required": ["product_name"],
            }
            return LLMResponse(content=json.dumps(schema), input_tokens=500, output_tokens=200)

        # Extraction call — return extracted data
        return LLMResponse(
            content=json.dumps(
                {
                    "product_name": "Test Product",
                    "rating": 8.5,
                    "pros": ["fast", "reliable"],
                    "price": 299.99,
                }
            ),
            input_tokens=300,
            output_tokens=150,
        )

    async def astructured_complete(
        self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kwargs: Any
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


class TestFullPipelineIntegration:
    async def test_end_to_end_with_demo_data(self) -> None:
        """Full pipeline: load demo → discover → extract → normalize."""
        docs = load_samples("product_reviews")
        assert len(docs) == 10

        mock_llm = MockPipelineLLM()

        # Patch both discovery and extraction to use our mock
        import catchfly.discovery.single_pass as sp_mod
        import catchfly.extraction.llm_direct as ext_mod

        original_sp_client = sp_mod.OpenAICompatibleClient
        original_ext_client = ext_mod.OpenAICompatibleClient

        sp_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        ext_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]

        try:
            pipeline = Pipeline.quick(model="mock-model")
            result = await pipeline.arun(
                docs[:3],  # Use subset for speed
                domain_hint="Electronics product reviews",
            )

            # Discovery should have found a schema
            assert result.schema is not None
            assert "product_name" in result.schema.json_schema.get("properties", {})

            # Extraction should have produced records
            assert len(result.records) == 3
            for record in result.records:
                assert hasattr(record, "product_name")

        finally:
            sp_mod.OpenAICompatibleClient = original_sp_client  # type: ignore[assignment]
            ext_mod.OpenAICompatibleClient = original_ext_client  # type: ignore[assignment]

    async def test_pipeline_with_user_schema(self) -> None:
        """Pipeline with user-provided schema, skipping discovery."""

        class ReviewSchema(BaseModel):
            product_name: str
            rating: float
            price: float | None = None

        mock_llm = MockPipelineLLM()

        import catchfly.extraction.llm_direct as ext_mod

        original_client = ext_mod.OpenAICompatibleClient
        ext_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]

        try:
            pipeline = Pipeline.quick(model="mock")
            docs = [Document(content="Great product, 9/10, $199", id="test1")]
            result = await pipeline.arun(docs, schema=ReviewSchema)

            assert result.schema is not None
            assert result.schema.lineage == ["user-provided"]
            assert len(result.records) == 1

        finally:
            ext_mod.OpenAICompatibleClient = original_client  # type: ignore[assignment]

    async def test_result_exports(self) -> None:
        """Verify PipelineResult export methods work."""
        mock_llm = MockPipelineLLM()

        import catchfly.discovery.single_pass as sp_mod
        import catchfly.extraction.llm_direct as ext_mod

        original_sp = sp_mod.OpenAICompatibleClient
        original_ext = ext_mod.OpenAICompatibleClient
        sp_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]
        ext_mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]

        try:
            pipeline = Pipeline.quick(model="mock")
            result = await pipeline.arun(
                load_samples("product_reviews")[:2],
                domain_hint="Reviews",
            )

            # to_dataframe should work
            df = result.to_dataframe()
            assert len(df) == 2
            assert "product_name" in df.columns

        finally:
            sp_mod.OpenAICompatibleClient = original_sp  # type: ignore[assignment]
            ext_mod.OpenAICompatibleClient = original_ext  # type: ignore[assignment]

    def test_imports_from_top_level(self) -> None:
        """Verify public API is accessible from top-level import."""
        import catchfly

        assert hasattr(catchfly, "Pipeline")
        assert hasattr(catchfly, "Document")
        assert hasattr(catchfly, "Schema")
        assert hasattr(catchfly, "ExtractionResult")
        assert hasattr(catchfly, "NormalizationResult")
        assert hasattr(catchfly, "PipelineResult")
        assert hasattr(catchfly, "RecordProvenance")
        assert hasattr(catchfly, "UsageReport")
        assert hasattr(catchfly, "__version__")
