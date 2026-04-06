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

    async def test_chunking_strategy_param(self) -> None:
        """LLMDirectExtraction uses chunking_strategy when provided."""
        from catchfly.extraction.chunking_fixed import FixedSizeChunking

        mock_llm = MockExtractionLLM()
        strategy = FixedSizeChunking(chunk_size=50, overlap=10)
        extractor = LLMDirectExtraction(
            model="mock",
            chunking_strategy=strategy,
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            long_doc = Document(content="x" * 200, id="long")
            result = await extractor.aextract(ProductReview, [long_doc])
            assert len(result.records) > 1
            for prov in result.provenance:
                assert prov.char_start is not None
                assert prov.char_end is not None
        finally:
            _unpatch_extraction()

    async def test_extract_with_dict_schema(self) -> None:
        """extract() / aextract() accept a raw JSON Schema dict."""
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="mock", max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            dict_schema = {
                "title": "Review",
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rating": {"type": "integer"},
                    "pros": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "rating", "pros"],
            }
            result = await extractor.aextract(dict_schema, self._make_docs(1))
            assert len(result.records) == 1
            assert result.records[0].title == "Great product"
        finally:
            _unpatch_extraction()

    def test_coerce_nulls_nested_object(self) -> None:
        """_coerce_nulls recurses into nested objects."""
        data = {"patient": {"name": "Jan", "tests": None, "notes": None}}
        schema = {
            "properties": {
                "patient": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tests": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "object"},
                    },
                },
            },
        }
        result = LLMDirectExtraction._coerce_nulls(data, schema)
        assert result["patient"]["tests"] == []
        assert result["patient"]["notes"] == {}

    def test_coerce_nulls_array_items(self) -> None:
        """_coerce_nulls recurses into array item objects."""
        data = {
            "symptoms": [
                {"name": "fever", "related_conditions": None},
                {"name": "cough", "related_conditions": None},
            ],
        }
        schema = {
            "properties": {
                "symptoms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "related_conditions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
        result = LLMDirectExtraction._coerce_nulls(data, schema)
        assert result["symptoms"][0]["related_conditions"] == []
        assert result["symptoms"][1]["related_conditions"] == []

    def test_coerce_nulls_top_level_unchanged(self) -> None:
        """Top-level coercion still works after refactor."""
        data = {"tags": None, "meta": None, "name": "test"}
        schema = {
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "meta": {"type": "object"},
                "name": {"type": "string"},
            },
        }
        result = LLMDirectExtraction._coerce_nulls(data, schema)
        assert result["tags"] == []
        assert result["meta"] == {}
        assert result["name"] == "test"

    async def test_extract_nested_pydantic_model(self) -> None:
        """Full extraction with nested Pydantic model (PatientFindings/Symptom)."""

        class Symptom(BaseModel):
            name: str
            present: bool

        class PatientFindings(BaseModel):
            patient_id: str
            symptoms: list[Symptom]

        response_json = json.dumps(
            {
                "patient_id": "P1",
                "symptoms": [
                    {"name": "splenomegaly", "present": True},
                    {"name": "fever", "present": False},
                ],
            }
        )
        mock_llm = MockExtractionLLM(responses=[response_json])
        extractor = LLMDirectExtraction(model="mock", max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content="Patient with splenomegaly, no fever.", id="cr1")]
            result = await extractor.aextract(PatientFindings, docs)
            assert len(result.records) == 1
            record = result.records[0]
            assert isinstance(record, PatientFindings)
            assert record.patient_id == "P1"
            assert len(record.symptoms) == 2
            assert record.symptoms[0].name == "splenomegaly"
            assert record.symptoms[0].present is True
            assert record.symptoms[1].present is False
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

    # --- multi_record tests ---

    async def test_multi_record_basic(self) -> None:
        """multi_record=True extracts multiple records from one document."""
        response = json.dumps([
            {"title": "Product A", "rating": 5, "pros": ["fast"]},
            {"title": "Product B", "rating": 3, "pros": ["cheap"]},
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(model="mock", multi_record=True, max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content="Two products reviewed.", id="doc0")]
            result = await extractor.aextract(ProductReview, docs)
            assert len(result.records) == 2
            assert result.records[0].title == "Product A"
            assert result.records[1].title == "Product B"
            assert len(result.provenance) == 2
        finally:
            _unpatch_extraction()

    async def test_multi_record_single_object_wrapped(self) -> None:
        """If LLM returns a single object in multi-record mode, wrap it."""
        response = json.dumps({"title": "Solo", "rating": 4, "pros": ["good"]})
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(model="mock", multi_record=True, max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content="One product.", id="doc0")]
            result = await extractor.aextract(ProductReview, docs)
            assert len(result.records) == 1
            assert result.records[0].title == "Solo"
        finally:
            _unpatch_extraction()

    async def test_multi_record_no_chunking_default(self) -> None:
        """multi_record=True defaults to no chunking (chunk_size=0)."""
        response = json.dumps([
            {"title": "A", "rating": 5, "pros": ["x"]},
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        # Long document that would normally be chunked at default 4000
        extractor = LLMDirectExtraction(model="mock", multi_record=True, max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            long_doc = Document(content="x" * 8000, id="long")
            result = await extractor.aextract(ProductReview, [long_doc])
            # Should NOT be chunked — only 1 LLM call, 1 record
            assert mock_llm._call_index == 1
            assert len(result.records) == 1
        finally:
            _unpatch_extraction()

    async def test_multi_record_explicit_chunk_size(self) -> None:
        """multi_record=True with explicit chunk_size still chunks."""
        response = json.dumps([
            {"title": "A", "rating": 5, "pros": ["x"]},
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock", multi_record=True, chunk_size=50, chunk_overlap=10, max_retries=0
        )
        _patch_extraction(extractor, mock_llm)
        try:
            long_doc = Document(content="x" * 200, id="long")
            result = await extractor.aextract(ProductReview, [long_doc])
            # Should be chunked — multiple LLM calls
            assert mock_llm._call_index > 1
        finally:
            _unpatch_extraction()

    async def test_multi_record_deduplication(self) -> None:
        """Duplicate records across chunks are removed when deduplicate=True."""
        # Both chunks return the same record
        response = json.dumps([
            {"title": "Duplicate", "rating": 5, "pros": ["fast"]},
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=True,
            deduplicate=True,
            chunk_size=50,
            chunk_overlap=10,
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            long_doc = Document(content="x" * 200, id="long")
            result = await extractor.aextract(ProductReview, [long_doc])
            # Multiple chunks but only 1 unique record
            assert mock_llm._call_index > 1
            assert len(result.records) == 1
            assert result.records[0].title == "Duplicate"
        finally:
            _unpatch_extraction()

    async def test_multi_record_no_deduplication(self) -> None:
        """Duplicates preserved when deduplicate=False."""
        response = json.dumps([
            {"title": "Duplicate", "rating": 5, "pros": ["fast"]},
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=True,
            deduplicate=False,
            chunk_size=50,
            chunk_overlap=10,
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            long_doc = Document(content="x" * 200, id="long")
            result = await extractor.aextract(ProductReview, [long_doc])
            # Multiple chunks, each returning same record — all kept
            assert len(result.records) > 1
        finally:
            _unpatch_extraction()

    async def test_multi_record_backward_compat(self) -> None:
        """Default multi_record=False preserves existing single-record behavior."""
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="mock", max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(2))
            assert len(result.records) == 2
            assert all(isinstance(r, ProductReview) for r in result.records)
        finally:
            _unpatch_extraction()

    async def test_multi_record_with_dict_schema(self) -> None:
        """multi_record works with a raw JSON Schema dict."""
        response = json.dumps([
            {"title": "A", "rating": 5, "pros": ["x"]},
            {"title": "B", "rating": 3, "pros": ["y"]},
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(model="mock", multi_record=True, max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            dict_schema = {
                "title": "Review",
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rating": {"type": "integer"},
                    "pros": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "rating", "pros"],
            }
            docs = [Document(content="Two products.", id="doc0")]
            result = await extractor.aextract(dict_schema, docs)
            assert len(result.records) == 2
        finally:
            _unpatch_extraction()

    def test_parse_json_array_basic(self) -> None:
        result = LLMDirectExtraction._parse_json_array('[{"a": 1}, {"a": 2}]')
        assert len(result) == 2
        assert result[0] == {"a": 1}

    def test_parse_json_array_single_object(self) -> None:
        """A single object is wrapped into a list."""
        result = LLMDirectExtraction._parse_json_array('{"a": 1}')
        assert result == [{"a": 1}]

    def test_parse_json_array_invalid(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            LLMDirectExtraction._parse_json_array('"just a string"')

    def test_parse_json_array_with_fences(self) -> None:
        result = LLMDirectExtraction._parse_json_array(
            '```json\n[{"a": 1}]\n```'
        )
        assert result == [{"a": 1}]

    async def test_multi_record_retry(self) -> None:
        """Multi-record extraction retries on validation error."""
        bad = '[{"bad": "data"}]'
        good = json.dumps([{"title": "OK", "rating": 4, "pros": ["fine"]}])
        mock_llm = MockExtractionLLM(responses=[bad, good])
        extractor = LLMDirectExtraction(model="mock", multi_record=True, max_retries=2)
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content="Product review.", id="doc0")]
            result = await extractor.aextract(ProductReview, docs)
            assert len(result.records) == 1
            assert result.records[0].title == "OK"
        finally:
            _unpatch_extraction()
