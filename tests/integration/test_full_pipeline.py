"""Integration test: full pipeline end-to-end with mocks."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from catchfly import Document, Pipeline
from catchfly.demo import load_samples
from catchfly.providers.llm import LLMResponse


class MockPipelineLLM:
    """Mock LLM that returns appropriate responses for discovery, extraction,
    field selection, and normalization calls."""

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

        # Field selection call — return fields to normalize
        if "normalization" in user_msg.lower() and "fields" in user_msg.lower():
            return LLMResponse(
                content=json.dumps(["product_name"]),
                input_tokens=200,
                output_tokens=50,
            )

        # Normalization / canonicalization call — return grouping
        if "canonical" in user_msg.lower() or "group" in user_msg.lower():
            return LLMResponse(
                content=json.dumps(
                    [{"canonical": "Test Product", "members": ["Test Product"], "rationale": "ok"}]
                ),
                input_tokens=200,
                output_tokens=100,
            )

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


def _make_pipeline(mock_llm: MockPipelineLLM, model: str = "mock") -> Pipeline:
    """Create a Pipeline.quick() with all LLM clients replaced by mock.

    Uses dependency injection (client=mock_llm) for strategies that support it,
    and module-level patching for those that import OpenAICompatibleClient
    at module scope (discovery, extraction).
    """
    pipeline = Pipeline.quick(model=model)

    # Inject mock into strategies that use DI (_get_client pattern)
    if pipeline.field_selector is not None:
        pipeline.field_selector.client = mock_llm  # type: ignore[union-attr]
    if pipeline.normalization is not None:
        pipeline.normalization.client = mock_llm  # type: ignore[union-attr]

    return pipeline


class _PatchContext:
    """Context manager to monkey-patch module-level OpenAICompatibleClient imports."""

    def __init__(self, mock_llm: MockPipelineLLM) -> None:
        self._mock = mock_llm
        self._originals: dict[int, Any] = {}
        self._modules: list[Any] = []

    def __enter__(self) -> _PatchContext:
        import catchfly.discovery.single_pass as sp_mod
        import catchfly.extraction.llm_direct as ext_mod

        self._modules = [sp_mod, ext_mod]
        self._originals = {id(m): m.OpenAICompatibleClient for m in self._modules}
        for m in self._modules:
            m.OpenAICompatibleClient = lambda **kw: self._mock  # type: ignore[assignment,misc]
        return self

    def __exit__(self, *exc: Any) -> None:
        for m in self._modules:
            m.OpenAICompatibleClient = self._originals[id(m)]  # type: ignore[assignment]


class TestFullPipelineIntegration:
    async def test_end_to_end_with_demo_data(self) -> None:
        """Full pipeline: load demo → discover → extract → normalize."""
        docs = load_samples("product_reviews")
        assert len(docs) == 10

        mock_llm = MockPipelineLLM()
        with _PatchContext(mock_llm):
            pipeline = _make_pipeline(mock_llm, model="mock-model")
            result = await pipeline.arun(
                docs[:3],
                domain_hint="Electronics product reviews",
            )

        assert result.schema is not None
        assert "product_name" in result.schema.json_schema.get("properties", {})
        assert len(result.records) == 3
        for record in result.records:
            assert hasattr(record, "product_name")

    async def test_pipeline_with_user_schema(self) -> None:
        """Pipeline with user-provided schema, skipping discovery."""

        class ReviewSchema(BaseModel):
            product_name: str
            rating: float
            price: float | None = None

        mock_llm = MockPipelineLLM()
        with _PatchContext(mock_llm):
            pipeline = _make_pipeline(mock_llm)
            docs = [Document(content="Great product, 9/10, $199", id="test1")]
            result = await pipeline.arun(docs, schema=ReviewSchema)

        assert result.schema is not None
        assert result.schema.lineage == ["user-provided"]
        assert len(result.records) == 1

    async def test_result_exports(self) -> None:
        """Verify PipelineResult export methods work."""
        mock_llm = MockPipelineLLM()
        with _PatchContext(mock_llm):
            pipeline = _make_pipeline(mock_llm)
            result = await pipeline.arun(
                load_samples("product_reviews")[:2],
                domain_hint="Reviews",
            )

        df = result.to_dataframe()
        assert len(df) == 2
        assert "product_name" in df.columns

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
