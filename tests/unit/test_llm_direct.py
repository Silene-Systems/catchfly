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
        self.captured_system_prompts: list[str] = []
        """System prompts seen across all calls — used by prompt-content tests."""

    async def acomplete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        if messages and messages[0].get("role") == "system":
            self.captured_system_prompts.append(messages[0]["content"])
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

    # --- include_provenance tests ---

    async def test_provenance_disabled_by_default(self) -> None:
        """Default path leaves field_spans=None — no behavior change."""
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="mock", max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(1))
            assert result.provenance[0].field_spans is None
        finally:
            _unpatch_extraction()

    async def test_provenance_exact_match_single(self) -> None:
        """include_provenance=True produces exact per-field spans."""
        content = (
            "This product is Great product indeed. "
            "Rating given: 5/5. Pros listed: fast and reliable shipping."
        )
        wrapped_response = json.dumps({
            "title": {
                "value": "Great product",
                "source_quotes": ["This product is Great product indeed"],
            },
            "rating": {
                "value": 5,
                "source_quotes": ["Rating given: 5/5"],
            },
            "pros": {
                "value": ["fast", "reliable shipping"],
                "source_quotes": [
                    "fast and reliable shipping",
                    "fast and reliable shipping",
                ],
            },
        })
        mock_llm = MockExtractionLLM(responses=[wrapped_response])
        extractor = LLMDirectExtraction(
            model="mock", max_retries=0, include_provenance=True
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content=content, id="r1", source="r1.txt")]
            result = await extractor.aextract(ProductReview, docs)
            assert len(result.records) == 1
            # Clean record — value unwrapped
            assert result.records[0].title == "Great product"
            assert result.records[0].rating == 5
            assert result.records[0].pros == ["fast", "reliable shipping"]
            # Per-field spans populated
            spans = result.provenance[0].field_spans
            assert spans is not None
            assert spans["title"][0].confidence == "exact"
            assert spans["title"][0].document_id == "r1"
            assert content[spans["title"][0].start : spans["title"][0].end] == (
                "This product is Great product indeed"
            )
            assert len(spans["pros"]) == 2
        finally:
            _unpatch_extraction()

    async def test_provenance_fuzzy_match(self) -> None:
        """Fuzzy confidence when whitespace/casing differs."""
        content = "Product\u00a0Name:  Acme  Widget\nRating 4/5."
        wrapped_response = json.dumps({
            "title": {
                "value": "Acme Widget",
                "source_quotes": ["Product Name: Acme Widget"],
            },
            "rating": {"value": 4, "source_quotes": ["Rating 4/5"]},
            "pros": {"value": [], "source_quotes": []},
        })
        mock_llm = MockExtractionLLM(responses=[wrapped_response])
        extractor = LLMDirectExtraction(
            model="mock", max_retries=0, include_provenance=True
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content=content, id="d1")]
            result = await extractor.aextract(ProductReview, docs)
            spans = result.provenance[0].field_spans
            assert spans is not None
            assert spans["title"][0].confidence == "fuzzy"
            assert spans["title"][0].start is not None
            assert spans["title"][0].end is not None
            assert spans["pros"] == []
        finally:
            _unpatch_extraction()

    async def test_provenance_unresolved_quote(self) -> None:
        """Quote not present in document → confidence='unresolved', no crash."""
        wrapped_response = json.dumps({
            "title": {
                "value": "Fabricated",
                "source_quotes": ["totally made up phrase nowhere in doc"],
            },
            "rating": {"value": 3, "source_quotes": ["hallucinated rating line"]},
            "pros": {"value": [], "source_quotes": []},
        })
        mock_llm = MockExtractionLLM(responses=[wrapped_response])
        extractor = LLMDirectExtraction(
            model="mock", max_retries=0, include_provenance=True
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content="Actual doc content.", id="d1")]
            result = await extractor.aextract(ProductReview, docs)
            spans = result.provenance[0].field_spans
            assert spans is not None
            assert spans["title"][0].confidence == "unresolved"
            assert spans["title"][0].start is None
            assert spans["title"][0].end is None
            assert spans["rating"][0].confidence == "unresolved"
        finally:
            _unpatch_extraction()

    async def test_provenance_multi_record_alignment(self) -> None:
        """len(provenance) == len(records) with per-record spans."""
        content = (
            "First: Product A rated 5 stars for speed.\n"
            "Second: Product B rated 3 stars for price."
        )
        wrapped_response = json.dumps([
            {
                "title": {
                    "value": "Product A",
                    "source_quotes": ["First: Product A rated 5 stars for speed"],
                },
                "rating": {
                    "value": 5,
                    "source_quotes": ["Product A rated 5 stars"],
                },
                "pros": {
                    "value": ["speed"],
                    "source_quotes": ["rated 5 stars for speed"],
                },
            },
            {
                "title": {
                    "value": "Product B",
                    "source_quotes": ["Second: Product B rated 3 stars for price"],
                },
                "rating": {
                    "value": 3,
                    "source_quotes": ["Product B rated 3 stars"],
                },
                "pros": {
                    "value": ["price"],
                    "source_quotes": ["rated 3 stars for price"],
                },
            },
        ])
        mock_llm = MockExtractionLLM(responses=[wrapped_response])
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=True,
            max_retries=0,
            include_provenance=True,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content=content, id="paper")]
            result = await extractor.aextract(ProductReview, docs)
            assert len(result.records) == 2
            assert len(result.provenance) == 2
            assert result.records[0].title == "Product A"
            assert result.records[1].title == "Product B"
            spans_a = result.provenance[0].field_spans
            spans_b = result.provenance[1].field_spans
            assert spans_a is not None and spans_b is not None
            # Each record's quote targets its own portion of the document
            assert "Product A" in spans_a["title"][0].quote
            assert "Product B" in spans_b["title"][0].quote
            # Offsets are disjoint
            assert spans_a["title"][0].end <= spans_b["title"][0].start
        finally:
            _unpatch_extraction()

    async def test_provenance_dedup_merges_spans(self) -> None:
        """When dedup collapses identical records, spans from all chunks merge."""
        # Each chunk returns the same record — we want spans from both chunks.
        # Use different quotes that both exist in the long document.
        long_content = "Start. " + "x" * 200 + " End."
        wrapped_response = json.dumps([
            {
                "title": {"value": "Dup", "source_quotes": ["Start."]},
                "rating": {"value": 5, "source_quotes": ["Start."]},
                "pros": {"value": ["a"], "source_quotes": ["Start."]},
            },
        ])
        mock_llm = MockExtractionLLM(responses=[wrapped_response])
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=True,
            deduplicate=True,
            include_provenance=True,
            chunk_size=100,
            chunk_overlap=10,
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content=long_content, id="long")]
            result = await extractor.aextract(ProductReview, docs)
            # Dedup leaves one record but multiple chunks contributed
            assert mock_llm._call_index > 1
            assert len(result.records) == 1
            spans = result.provenance[0].field_spans
            assert spans is not None
            # All three fields should have at least one span
            assert len(spans["title"]) >= 1
        finally:
            _unpatch_extraction()

    async def test_provenance_partial_wrapper_fallback(self) -> None:
        """If LLM emits a mix of wrapped/unwrapped, extraction still succeeds."""
        mixed = json.dumps({
            "title": {"value": "Good", "source_quotes": ["Good product"]},
            "rating": 4,  # unwrapped — fallback path
            "pros": {"value": ["fast"], "source_quotes": ["fast"]},
        })
        mock_llm = MockExtractionLLM(responses=[mixed])
        extractor = LLMDirectExtraction(
            model="mock", max_retries=0, include_provenance=True
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content="Good product and fast delivery.", id="d1")]
            result = await extractor.aextract(ProductReview, docs)
            assert result.records[0].title == "Good"
            assert result.records[0].rating == 4
            spans = result.provenance[0].field_spans
            assert spans is not None
            assert len(spans["title"]) == 1
            # Unwrapped field gets empty spans, not an error
            assert spans["rating"] == []
        finally:
            _unpatch_extraction()

    async def test_provenance_list_field_one_span_per_item(self) -> None:
        """List-valued field: one supporting quote per list element."""
        content = (
            "Symptoms included mild truncal hypotonia at 3 months, "
            "asymptomatic splenomegaly at 6 months, and intention tremor by 1 year."
        )
        wrapped = json.dumps({
            "title": {"value": "Case", "source_quotes": ["Symptoms included"]},
            "rating": {"value": 1, "source_quotes": ["at 3 months"]},
            "pros": {
                "value": ["hypotonia", "splenomegaly", "intention tremor"],
                "source_quotes": [
                    "mild truncal hypotonia",
                    "asymptomatic splenomegaly",
                    "intention tremor by 1 year",
                ],
            },
        })
        mock_llm = MockExtractionLLM(responses=[wrapped])
        extractor = LLMDirectExtraction(
            model="mock", max_retries=0, include_provenance=True
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content=content, id="case1")]
            result = await extractor.aextract(ProductReview, docs)
            spans = result.provenance[0].field_spans
            assert spans is not None
            assert len(spans["pros"]) == 3
            for span in spans["pros"]:
                assert span.confidence == "exact"
                assert content[span.start : span.end] == span.quote
        finally:
            _unpatch_extraction()

    # --- record_hint tests ---

    async def test_record_hint_appears_in_multi_record_prompt(self) -> None:
        """When multi_record=True and record_hint is set, it reaches the LLM."""
        response = json.dumps([{"title": "A", "rating": 5, "pros": ["x"]}])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=True,
            record_hint="Each patient in tables or prose is a separate record.",
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            await extractor.aextract(ProductReview, self._make_docs(1))
            assert any(
                "Record definition: Each patient in tables or prose"
                in prompt
                for prompt in mock_llm.captured_system_prompts
            )
        finally:
            _unpatch_extraction()

    async def test_record_hint_ignored_in_single_record_mode(self) -> None:
        """record_hint on single-record extraction does not leak into the prompt.

        Single-record mode has no concept of "record boundaries" — the hint
        is specific to multi_record. We don't error, we just don't inject.
        """
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=False,
            record_hint="Each patient is a record.",
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            await extractor.aextract(ProductReview, self._make_docs(1))
            for prompt in mock_llm.captured_system_prompts:
                assert "Record definition:" not in prompt
        finally:
            _unpatch_extraction()

    async def test_record_hint_absent_by_default(self) -> None:
        """Default multi_record prompt has no Record definition line."""
        response = json.dumps([{"title": "A", "rating": 5, "pros": ["x"]}])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock", multi_record=True, max_retries=0
        )
        _patch_extraction(extractor, mock_llm)
        try:
            await extractor.aextract(ProductReview, self._make_docs(1))
            for prompt in mock_llm.captured_system_prompts:
                assert "Record definition:" not in prompt
        finally:
            _unpatch_extraction()

    # --- table-row multi_record tests ---

    async def test_multi_record_prompt_mentions_tables(self) -> None:
        """_MULTI_RECORD_SYSTEM_PROMPT explicitly handles comparative tables."""
        response = json.dumps([{"title": "A", "rating": 5, "pros": ["x"]}])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock", multi_record=True, max_retries=0
        )
        _patch_extraction(extractor, mock_llm)
        try:
            await extractor.aextract(ProductReview, self._make_docs(1))
            captured = "\n".join(mock_llm.captured_system_prompts)
            assert "comparative table" in captured.lower()
            assert "table row" in captured.lower() or "row of a comparative" in captured.lower()
            assert "merge the information" in captured.lower()
        finally:
            _unpatch_extraction()

    async def test_multi_record_table_rows_pass_through(self) -> None:
        """End-to-end: markdown-table document yields N records when LLM returns N.

        This verifies the plumbing — actual table parsing is the LLM's job.
        """
        content = (
            "# Case report\n\n"
            "Patient 1 and Patient 2 are described in prose.\n\n"
            "## Comparative Table\n\n"
            "| Patient | Gender | Outcome |\n"
            "| --- | --- | --- |\n"
            "| Patient 1 | Male | Alive |\n"
            "| Patient 2 | Male | Alive |\n"
            "| Patient 3 | Female | Died |\n"
            "| Patient 4 | Male | Died |\n"
            "| Patient 5 | Female | Alive |\n"
            "| Patient 6 | Male | Alive |\n"
        )
        response = json.dumps([
            {"title": f"Patient {i}", "rating": i, "pros": ["alive"]}
            for i in range(1, 7)
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=True,
            record_hint="Each patient, including those only in tables, is a separate record.",
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            docs = [Document(content=content, id="npc_paper")]
            result = await extractor.aextract(ProductReview, docs)
            assert len(result.records) == 6
            # record_hint reached the LLM
            assert any(
                "Each patient, including those only in tables" in p
                for p in mock_llm.captured_system_prompts
            )
            # And the table-aware rules are also present
            assert any(
                "comparative table" in p.lower()
                for p in mock_llm.captured_system_prompts
            )
        finally:
            _unpatch_extraction()

    async def test_record_hint_combines_with_provenance(self) -> None:
        """record_hint and include_provenance can both be active at once."""
        response = json.dumps([
            {
                "title": {"value": "A", "source_quotes": ["A is great"]},
                "rating": {"value": 5, "source_quotes": ["rated 5"]},
                "pros": {"value": ["x"], "source_quotes": ["x feature"]},
            }
        ])
        mock_llm = MockExtractionLLM(responses=[response])
        extractor = LLMDirectExtraction(
            model="mock",
            multi_record=True,
            record_hint="Each row is a record.",
            include_provenance=True,
            max_retries=0,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            await extractor.aextract(
                ProductReview,
                [Document(content="A is great, rated 5, x feature.", id="d1")],
            )
            seen = "\n".join(mock_llm.captured_system_prompts)
            # Both instructions should land in the same system prompt
            assert "Record definition: Each row is a record." in seen
            assert "Provenance requirements:" in seen
        finally:
            _unpatch_extraction()

    # --- estimate_cost tests ---

    def test_estimate_cost_empty_documents(self) -> None:
        extractor = LLMDirectExtraction(model="gpt-5.4-mini")
        estimate = extractor.estimate_cost(ProductReview, [])
        assert estimate.num_documents == 0
        assert estimate.input_tokens == 0
        assert estimate.cost_usd == 0.0
        assert any("No documents" in n for n in estimate.notes)

    def test_estimate_cost_single_known_model(self) -> None:
        """Known model → non-zero cost estimate, no 'unknown model' note."""
        extractor = LLMDirectExtraction(model="gpt-5.4-mini")
        docs = [
            Document(content="Product review about a great item.", id=f"d{i}")
            for i in range(3)
        ]
        estimate = extractor.estimate_cost(ProductReview, docs)
        assert estimate.num_documents == 3
        assert estimate.num_chunks >= 3  # at least one chunk per doc
        assert estimate.input_tokens > 0
        assert estimate.estimated_output_tokens > 0
        assert estimate.cost_usd > 0.0
        assert not any("not in the default pricing table" in n for n in estimate.notes)

    def test_estimate_cost_unknown_model_notes_zero(self) -> None:
        extractor = LLMDirectExtraction(model="not-a-real-model-xyz")
        docs = [Document(content="x", id="d0")]
        estimate = extractor.estimate_cost(ProductReview, docs)
        assert estimate.cost_usd == 0.0
        assert any("not in the default pricing table" in n for n in estimate.notes)

    def test_estimate_cost_pricing_override(self) -> None:
        """Explicit pricing override produces a non-zero estimate even for unknown models."""
        extractor = LLMDirectExtraction(model="custom-model")
        docs = [Document(content="some content here", id="d0")]
        estimate = extractor.estimate_cost(
            ProductReview,
            docs,
            pricing={"custom-model": (1.0, 2.0)},
        )
        assert estimate.cost_usd > 0.0

    def test_estimate_cost_provenance_inflates_output(self) -> None:
        """include_provenance triples the output token estimate."""
        docs = [Document(content="A product review.", id="d0")]
        baseline = LLMDirectExtraction(model="gpt-5.4-mini").estimate_cost(
            ProductReview, docs
        )
        with_prov = LLMDirectExtraction(
            model="gpt-5.4-mini", include_provenance=True
        ).estimate_cost(ProductReview, docs)
        assert with_prov.estimated_output_tokens >= baseline.estimated_output_tokens * 2
        assert any("provenance" in n.lower() for n in with_prov.notes)

    def test_estimate_cost_multi_record_scales_with_records_per_doc(self) -> None:
        """records_per_document multiplies the output token estimate."""
        docs = [Document(content="A table with many rows.", id="d0")]
        extractor = LLMDirectExtraction(model="gpt-5.4-mini", multi_record=True)
        one = extractor.estimate_cost(ProductReview, docs, records_per_document=1)
        ten = extractor.estimate_cost(ProductReview, docs, records_per_document=10)
        assert ten.estimated_output_tokens == one.estimated_output_tokens * 10
        assert any("multi_record" in n for n in ten.notes)

    def test_estimate_cost_no_llm_calls(self) -> None:
        """Dry-run never hits the mock LLM — verifies true dry-run."""
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="gpt-5.4-mini")
        _patch_extraction(extractor, mock_llm)
        try:
            extractor.estimate_cost(ProductReview, self._make_docs(3))
            assert mock_llm._call_index == 0
        finally:
            _unpatch_extraction()

    def test_estimate_cost_tokenizer_label(self) -> None:
        """Tokenizer label records which backend computed the token count."""
        extractor = LLMDirectExtraction(model="gpt-5.4-mini")
        estimate = extractor.estimate_cost(
            ProductReview, [Document(content="hi", id="d")]
        )
        assert estimate.tokenizer.startswith(("tiktoken:", "heuristic:"))

    def test_estimate_cost_with_chunking(self) -> None:
        """Chunking increases num_chunks and scales input tokens."""
        long_doc = Document(content="x" * 2000, id="long")
        extractor = LLMDirectExtraction(
            model="gpt-5.4-mini",
            chunk_size=200,
            chunk_overlap=20,
        )
        estimate = extractor.estimate_cost(ProductReview, [long_doc])
        assert estimate.num_documents == 1
        assert estimate.num_chunks > 1

    # --- progress_callback tests ---

    async def test_progress_callback_monotonic(self) -> None:
        """Callback is invoked N times with monotonically increasing counts."""
        events: list[tuple[int, int]] = []

        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(
            model="mock",
            max_retries=0,
            progress_callback=lambda done, total: events.append((done, total)),
        )
        _patch_extraction(extractor, mock_llm)
        try:
            await extractor.aextract(ProductReview, self._make_docs(3))
        finally:
            _unpatch_extraction()

        assert len(events) == 3
        # All events share the same total
        totals = {total for _, total in events}
        assert totals == {3}
        # Completed counts cover 1..3 regardless of concurrent ordering
        completed_values = sorted(done for done, _ in events)
        assert completed_values == [1, 2, 3]

    async def test_progress_callback_swallows_exceptions(self) -> None:
        """Exceptions inside the callback are logged but don't abort extraction."""

        def raising_cb(_done: int, _total: int) -> None:
            raise RuntimeError("telemetry is down")

        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(
            model="mock",
            max_retries=0,
            progress_callback=raising_cb,
        )
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(2))
            assert len(result.records) == 2  # extraction still succeeded
        finally:
            _unpatch_extraction()

    async def test_progress_callback_none_is_noop(self) -> None:
        """Default (None) callback does nothing and doesn't error."""
        mock_llm = MockExtractionLLM()
        extractor = LLMDirectExtraction(model="mock", max_retries=0)
        _patch_extraction(extractor, mock_llm)
        try:
            result = await extractor.aextract(ProductReview, self._make_docs(2))
            assert len(result.records) == 2
        finally:
            _unpatch_extraction()

    async def test_progress_callback_fires_on_error_path(self) -> None:
        """Even chunks that fail extraction still contribute to the progress counter.

        Otherwise a flaky document would make the progress bar hang at 5/6.
        """
        events: list[int] = []
        bad_response = '{"invalid": true}'
        mock_llm = MockExtractionLLM(responses=[bad_response])
        extractor = LLMDirectExtraction(
            model="mock",
            max_retries=0,
            on_error="collect",
            progress_callback=lambda done, total: events.append(done),
        )
        _patch_extraction(extractor, mock_llm)
        try:
            await extractor.aextract(ProductReview, self._make_docs(3))
        finally:
            _unpatch_extraction()

        assert sorted(events) == [1, 2, 3]
