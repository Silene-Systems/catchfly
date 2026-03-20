"""Tests for core domain types."""

from __future__ import annotations

from pydantic import BaseModel

from catchfly._types import (
    Document,
    ExtractionResult,
    NormalizationResult,
    PipelineResult,
    RecordProvenance,
    Schema,
    UsageReport,
)


class TestDocument:
    def test_create_minimal(self) -> None:
        doc = Document(content="hello")
        assert doc.content == "hello"
        assert doc.id is None
        assert doc.source is None
        assert doc.metadata == {}

    def test_create_full(self) -> None:
        doc = Document(
            content="hello",
            id="doc1",
            source="/path/to/doc.txt",
            metadata={"lang": "en"},
        )
        assert doc.id == "doc1"
        assert doc.metadata["lang"] == "en"


class TestSchema:
    def test_create_with_model(self) -> None:
        class MyModel(BaseModel):
            name: str

        schema = Schema(
            model=MyModel,
            json_schema=MyModel.model_json_schema(),
            lineage=["SinglePassDiscovery"],
        )
        assert schema.model is MyModel
        assert "properties" in schema.json_schema
        assert schema.lineage == ["SinglePassDiscovery"]

    def test_create_without_model(self) -> None:
        schema = Schema(model=None, json_schema={"type": "object"})
        assert schema.model is None


class TestExtractionResult:
    def test_empty(self) -> None:
        result: ExtractionResult[BaseModel] = ExtractionResult(records=[])
        assert result.records == []
        assert result.errors == []
        assert result.provenance == []


class TestRecordProvenance:
    def test_defaults(self) -> None:
        prov = RecordProvenance(source_document="doc1.txt")
        assert prov.chunk_index is None
        assert prov.char_start is None
        assert prov.confidence is None

    def test_full(self) -> None:
        prov = RecordProvenance(
            source_document="doc1.txt",
            chunk_index=0,
            char_start=100,
            char_end=200,
            confidence=0.95,
        )
        assert prov.char_start == 100
        assert prov.confidence == 0.95


class TestNormalizationResult:
    def test_explain_found(self) -> None:
        result = NormalizationResult(
            mapping={"NYC": "New York", "NY": "New York"},
            clusters={"New York": ["NYC", "NY"]},
        )
        explanation = result.explain("NYC")
        assert "NYC" in explanation
        assert "New York" in explanation

    def test_explain_not_found(self) -> None:
        result = NormalizationResult(mapping={"NYC": "New York"})
        explanation = result.explain("LA")
        assert "not found" in explanation

    def test_explain_with_details(self) -> None:
        result = NormalizationResult(
            mapping={"NYC": "New York"},
            metadata={"explanations": {"NYC": "Common abbreviation"}},
        )
        explanation = result.explain("NYC")
        assert "Common abbreviation" in explanation


class TestPipelineResult:
    def test_empty(self) -> None:
        result = PipelineResult()
        assert result.schema is None
        assert result.records == []
        assert result.normalizations == {}

    def test_to_dataframe_no_pandas(self) -> None:
        """to_dataframe works when pandas is installed."""
        result = PipelineResult()
        # Should return empty DataFrame
        df = result.to_dataframe()
        assert len(df) == 0


class TestUsageReport:
    def test_defaults(self) -> None:
        report = UsageReport()
        assert report.total_cost_usd == 0.0
        assert report.total_input_tokens == 0
