"""Tests for LLMDirectExtraction."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from catchfly._types import Document
from catchfly.exceptions import ExtractionError
from catchfly.extraction.llm_direct import LLMDirectExtraction
from catchfly.providers.llm import LLMResponse


class ProductReview(BaseModel):
    title: str
    rating: int
    pros: list[str]


class MockExtractionLLM:
    """Mock LLM that returns extraction results."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self._call_index = 0

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        if self._responses:
            content = self._responses[self._call_index % len(self._responses)]
        else:
            content = json.dumps({"title": "Great product", "rating": 5, "pros": ["fast"]})
        self._call_index += 1
        return LLMResponse(content=content, input_tokens=200, output_tokens=100, model="mock")

    async def astructured_complete(
        self, messages: list[dict[str, str]], output_schema: dict[str, Any], **kwargs: Any
    ) -> LLMResponse:
        return await self.acomplete(messages, **kwargs)


def _patch_extraction(extractor: LLMDirectExtraction, mock_llm: MockExtractionLLM) -> None:
    """Monkey-patch the extractor to use a mock LLM client."""
    import catchfly.extraction.llm_direct as mod

    mod.OpenAICompatibleClient = lambda **kw: mock_llm  # type: ignore[assignment,misc]


def _unpatch_extraction() -> None:
    import catchfly.extraction.llm_direct as mod
    from catchfly.providers.llm import OpenAICompatibleClient

    mod.OpenAICompatibleClient = OpenAICompatibleClient  # type: ignore[assignment]


class TestLLMDirectExtraction:
    def _make_docs(self, n: int = 3) -> list[Document]:
        return [
            Document(
                content=f"This product is great. Rating: {i + 3}/5. Pros: fast, reliable.",
                id=f"doc{i}",
                source=f"review_{i}.txt",
            )
            for i in range(n)
        ]

    async def test_extract_basic(self) -> None:
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="mock", max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(2))
            assert len(result.records) == 2
            assert all(isinstance(r, ProductReview) for r in result.records)
            assert result.records[0].title == "Great product"
            assert len(result.provenance) == 2
        finally:
            _unpatch_extraction()

    async def test_extract_empty_docs(self) -> None:
        extractor = LLMDirectExtraction()
        result = await extractor.aextract(ProductReview, [])
        assert result.records == []

    async def test_provenance_tracking(self) -> None:
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="mock", max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            docs = self._make_docs(1)
            result = await extractor.aextract(ProductReview, docs)
            prov = result.provenance[0]
            assert prov.source_document == "review_0.txt"
        finally:
            _unpatch_extraction()

    async def test_on_error_collect(self) -> None:
        bad_response = '{"invalid": "missing required fields"}'
        mock_llm = MockExtractionLLM(responses=[bad_response])
        extractor = LLMDirectExtraction(model="mock", max_retries=0, on_error="collect")
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(2))
            assert len(result.errors) == 2
            assert len(result.records) == 0
        finally:
            _unpatch_extraction()

    async def test_on_error_skip(self) -> None:
        bad_response = '{"invalid": true}'
        mock_llm = MockExtractionLLM(responses=[bad_response])
        extractor = LLMDirectExtraction(model="mock", max_retries=0, on_error="skip")
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(1))
            assert len(result.records) == 0
            assert len(result.errors) == 0  # skip mode doesn't collect errors
        finally:
            _unpatch_extraction()

    async def test_on_error_raise(self) -> None:
        bad_response = '{"invalid": true}'
        mock_llm = MockExtractionLLM(responses=[bad_response])
        extractor = LLMDirectExtraction(model="mock", max_retries=0, on_error="raise")
        _patch_extraction(extractor, mock_llm)
        try:
            with pytest.raises(ExtractionError):
                await extractor.aextract(ProductReview, self._make_docs(1))
        finally:
            _unpatch_extraction()

    async def test_retry_on_validation_error(self) -> None:
        """First response is bad, second is good — retry should succeed."""
        good = json.dumps({"title": "Fixed", "rating": 4, "pros": ["nice"]})
        bad = '{"bad": "data"}'
        mock_llm = MockExtractionLLM(responses=[bad, good])
        extractor = LLMDirectExtraction(model="mock", max_retries=2)
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(1))
            assert len(result.records) == 1
            assert result.records[0].title == "Fixed"
        finally:
            _unpatch_extraction()

    async def test_chunking_long_doc(self) -> None:
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(
            model="mock", chunk_size=50, chunk_overlap=10, max_retries=0
        )
        _patch_extraction(extractor, mock_llm)
        try:
            long_doc = Document(content="x" * 200, id="long")
            result = await extractor.aextract(ProductReview, [long_doc])
            # Should have multiple records (one per chunk)
            assert len(result.records) > 1
        finally:
            _unpatch_extraction()

    def test_parse_json_basic(self) -> None:
        result = LLMDirectExtraction._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_fences(self) -> None:
        result = LLMDirectExtraction._parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_invalid(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            LLMDirectExtraction._parse_json("not json")

    async def test_confidence_first_attempt(self) -> None:
        """Confidence should be 1.0 when extraction succeeds on first attempt."""
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="mock", max_retries=1)
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(1))
            assert result.provenance[0].confidence == 1.0
        finally:
            _unpatch_extraction()

    async def test_confidence_after_retry(self) -> None:
        """Confidence should decrease after retries."""
        bad_then_good = [
            '{"bad": "json"}',
            json.dumps({"title": "OK", "rating": 4, "pros": ["fine"]}),
        ]
        mock_llm = MockExtractionLLM(responses=bad_then_good)
        extractor = LLMDirectExtraction(model="mock", max_retries=2)
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(1))
            assert len(result.records) == 1
            assert result.provenance[0].confidence is not None
            assert result.provenance[0].confidence < 1.0
        finally:
            _unpatch_extraction()
