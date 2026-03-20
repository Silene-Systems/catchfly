"""Tests for Pipeline orchestrator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from catchfly._types import Document, NormalizationResult
from catchfly.pipeline import Pipeline

# --- Mock strategies ---


class MockDiscovery:
    def __init__(self, schema_json: dict[str, Any] | None = None) -> None:
        self._schema_json = schema_json or {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "rating": {"type": "integer"},
            },
            "required": ["title"],
        }

    async def adiscover(
        self, documents: list[Document], *, domain_hint: str | None = None, **kwargs: Any
    ) -> Any:
        from catchfly._types import Schema
        from catchfly.schema.converters import json_schema_to_pydantic

        model = json_schema_to_pydantic(self._schema_json, "MockSchema")
        return Schema(model=model, json_schema=self._schema_json, lineage=["MockDiscovery"])

    def discover(self, documents: list[Document], **kwargs: Any) -> Any:
        from catchfly._compat import run_sync

        return run_sync(self.adiscover(documents, **kwargs))


class MockExtraction:
    async def aextract(
        self, schema: type[BaseModel], documents: list[Document], **kwargs: Any
    ) -> Any:
        from catchfly._types import ExtractionResult, RecordProvenance

        records = []
        provs = []
        for doc in documents:
            record = schema(title=f"Extracted from {doc.id}", rating=5)  # type: ignore[call-arg]
            records.append(record)
            provs.append(RecordProvenance(source_document=str(doc.id)))
        return ExtractionResult(records=records, provenance=provs)

    def extract(self, schema: type[BaseModel], documents: list[Document], **kwargs: Any) -> Any:
        from catchfly._compat import run_sync

        return run_sync(self.aextract(schema, documents, **kwargs))


class MockNormalization:
    async def anormalize(
        self, values: list[str], context_field: str = "", **kwargs: Any
    ) -> NormalizationResult:
        unique = list(set(values))
        canonical = unique[0] if unique else ""
        return NormalizationResult(
            mapping={v: canonical for v in unique},
            clusters={canonical: unique},
            metadata={"strategy": "mock"},
        )

    def normalize(self, values: list[str], **kwargs: Any) -> NormalizationResult:
        from catchfly._compat import run_sync

        return run_sync(self.anormalize(values, **kwargs))


# --- Tests ---


def _make_docs(n: int = 3) -> list[Document]:
    return [Document(content=f"Document {i} about topic", id=f"doc{i}") for i in range(n)]


class TestPipeline:
    async def test_full_pipeline(self) -> None:
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
        )
        result = await pipeline.arun(
            _make_docs(3),
            domain_hint="test",
            normalize_fields=["title"],
        )

        assert result.schema is not None
        assert len(result.records) == 3
        assert "title" in result.normalizations

    async def test_with_provided_schema(self) -> None:
        class MySchema(BaseModel):
            title: str
            rating: int

        pipeline = Pipeline(
            extraction=MockExtraction(),
        )
        result = await pipeline.arun(
            _make_docs(2),
            schema=MySchema,
        )

        assert result.schema is not None
        assert result.schema.lineage == ["user-provided"]
        assert len(result.records) == 2

    async def test_discovery_only(self) -> None:
        pipeline = Pipeline(discovery=MockDiscovery())
        result = await pipeline.arun(_make_docs(), domain_hint="test domain")

        assert result.schema is not None
        assert result.records == []

    async def test_empty_pipeline(self) -> None:
        pipeline = Pipeline()
        result = await pipeline.arun(_make_docs())
        assert result.schema is None
        assert result.records == []

    async def test_skip_normalization_no_fields(self) -> None:
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
        )
        result = await pipeline.arun(_make_docs())
        assert result.normalizations == {}

    def test_sync_run(self) -> None:
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
        )
        result = pipeline.run(_make_docs(2), domain_hint="test")
        assert len(result.records) == 2

    def test_estimate_cost(self) -> None:
        pipeline = Pipeline.quick(model="gpt-5.4-mini")
        docs = _make_docs(10)
        estimate = pipeline.estimate_cost(docs)
        assert "total" in estimate
        assert "discovery" in estimate
        assert "extraction" in estimate
        assert estimate["total"] >= 0


class TestPipelineQuick:
    def test_quick_creates_all_strategies(self) -> None:
        pipeline = Pipeline.quick(model="gpt-5.4-mini")
        assert pipeline.discovery is not None
        assert pipeline.extraction is not None
        assert pipeline.normalization is not None

    def test_quick_with_base_url(self) -> None:
        pipeline = Pipeline.quick(
            model="qwen3.5",
            base_url="http://localhost:11434/v1",
        )
        assert pipeline.discovery is not None
