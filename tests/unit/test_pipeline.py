"""Tests for Pipeline orchestrator."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from catchfly._types import Document, NormalizationResult
from catchfly.exceptions import SchemaError
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

    async def test_normalize_fields_all(self) -> None:
        """normalize_fields='all' auto-detects string fields from schema."""
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
        )
        result = await pipeline.arun(_make_docs(2), domain_hint="test", normalize_fields="all")
        # MockDiscovery schema has "title" (string) and "rating" (integer)
        # Only "title" should be normalized
        assert "title" in result.normalizations
        assert "rating" not in result.normalizations

    async def test_normalize_fields_all_no_schema(self) -> None:
        """normalize_fields='all' without schema should skip normalization."""
        pipeline = Pipeline(normalization=MockNormalization())
        result = await pipeline.arun(_make_docs(1), normalize_fields="all")
        assert result.normalizations == {}

    def test_estimate_cost(self) -> None:
        pipeline = Pipeline.quick(model="gpt-5.4-mini")
        docs = _make_docs(10)
        estimate = pipeline.estimate_cost(docs)
        assert "total" in estimate
        assert "discovery" in estimate
        assert "extraction" in estimate
        assert estimate["total"] >= 0


class TestDictSchema:
    async def test_arun_with_dict_schema(self) -> None:
        """Pipeline accepts a plain dict JSON Schema and converts it."""
        json_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "rating": {"type": "integer"},
            },
            "required": ["title"],
        }
        pipeline = Pipeline(extraction=MockExtraction())
        result = await pipeline.arun(_make_docs(2), schema=json_schema)

        assert result.schema is not None
        assert result.schema.lineage == ["user-provided"]
        assert result.schema.model is not None
        assert "title" in result.schema.json_schema["properties"]
        assert len(result.records) == 2

    async def test_arun_with_invalid_dict_schema_raises(self) -> None:
        """Pipeline raises SchemaError for a dict without properties."""
        bad_schema: dict[str, Any] = {"type": "object"}
        pipeline = Pipeline(extraction=MockExtraction())

        with pytest.raises(SchemaError, match="Could not convert dict schema"):
            await pipeline.arun(_make_docs(1), schema=bad_schema)


class TestOnSchemaReady:
    async def test_callback_modifies_schema(self) -> None:
        """on_schema_ready callback can modify the schema."""
        from catchfly._types import Schema

        def modify_schema(schema: Schema) -> Schema:
            return Schema(
                model=schema.model,
                json_schema=schema.json_schema,
                lineage=["modified"],
            )

        pipeline = Pipeline(discovery=MockDiscovery(), extraction=MockExtraction())
        result = await pipeline.arun(
            _make_docs(1), domain_hint="test", on_schema_ready=modify_schema
        )

        assert result.schema is not None
        assert result.schema.lineage == ["modified"]

    async def test_callback_returns_none(self) -> None:
        """Callback returning None does not modify schema."""
        calls: list[Any] = []

        def inspect_schema(schema: Any) -> None:
            calls.append(schema)
            return None

        pipeline = Pipeline(discovery=MockDiscovery(), extraction=MockExtraction())
        result = await pipeline.arun(
            _make_docs(1), domain_hint="test", on_schema_ready=inspect_schema
        )

        assert len(calls) == 1
        assert result.schema is not None
        assert result.schema.lineage == ["MockDiscovery"]

    async def test_no_callback(self) -> None:
        """No callback — schema unchanged."""
        pipeline = Pipeline(discovery=MockDiscovery(), extraction=MockExtraction())
        result = await pipeline.arun(_make_docs(1), domain_hint="test")
        assert result.schema is not None
        assert result.schema.lineage == ["MockDiscovery"]


class TestUsageTrackerWiring:
    async def test_tracker_records_usage(self) -> None:
        """Usage callback on strategies should populate result.report."""
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
        )
        result = await pipeline.arun(_make_docs(2), domain_hint="test")
        # Tracker is created but mock strategies don't call LLM,
        # so report is empty — this tests wiring doesn't crash
        assert result.report is not None
        assert result.report.total_cost_usd >= 0

    async def test_usage_callback_attribute_set(self) -> None:
        """Pipeline sets _usage_callback on strategies that support it."""
        disc = MockDiscovery()
        ext = MockExtraction()
        norm = MockNormalization()
        # Add _usage_callback attribute to simulate real strategies
        disc._usage_callback = None  # type: ignore[attr-defined]
        ext._usage_callback = None  # type: ignore[attr-defined]
        norm._usage_callback = None  # type: ignore[attr-defined]
        pipeline = Pipeline(discovery=disc, extraction=ext, normalization=norm)
        await pipeline.arun(_make_docs(1), domain_hint="test")
        assert disc._usage_callback is not None  # type: ignore[attr-defined]
        assert ext._usage_callback is not None  # type: ignore[attr-defined]
        assert norm._usage_callback is not None  # type: ignore[attr-defined]

    async def test_wire_tracker_skips_strategies_without_callback(self) -> None:
        """Pipeline doesn't crash on strategies without _usage_callback."""
        disc = MockDiscovery()  # no _usage_callback
        pipeline = Pipeline(discovery=disc)
        # Should not raise
        await pipeline.arun(_make_docs(1), domain_hint="test")


class TestVerbose:
    async def test_verbose_does_not_crash(self) -> None:
        """verbose=True should work even without tqdm installed."""
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
            verbose=True,
        )
        result = await pipeline.arun(
            _make_docs(2),
            domain_hint="test",
            normalize_fields=["title"],
        )
        assert len(result.records) == 2


class TestGlobDocuments:
    async def test_glob_strings_resolved(self, tmp_path: Any) -> None:
        """Pipeline accepts glob pattern strings as documents."""
        from pathlib import Path

        p = Path(str(tmp_path))
        (p / "a.txt").write_text("Document about topic A")
        (p / "b.txt").write_text("Document about topic B")

        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
        )
        result = await pipeline.arun([str(p / "*.txt")], domain_hint="test")
        assert len(result.records) == 2


class TestFieldSelector:
    async def test_field_selector_auto_selects(self) -> None:
        """When normalize_fields is None and field_selector is set, selector decides."""

        class MockFieldSelector:
            async def aselect(self, schema: Any, records: list[Any], **kwargs: Any) -> list[str]:
                return ["title"]

            def select(self, schema: Any, records: list[Any], **kwargs: Any) -> list[str]:
                return ["title"]

        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
            field_selector=MockFieldSelector(),
        )
        result = await pipeline.arun(_make_docs(3), domain_hint="test")

        # Selector should have picked "title"
        assert "title" in result.normalizations

    async def test_explicit_normalize_fields_overrides_selector(self) -> None:
        """Explicit normalize_fields should bypass field_selector."""

        class MockFieldSelector:
            def __init__(self) -> None:
                self.called = False

            async def aselect(self, schema: Any, records: list[Any], **kwargs: Any) -> list[str]:
                self.called = True
                return ["title"]

            def select(self, schema: Any, records: list[Any], **kwargs: Any) -> list[str]:
                self.called = True
                return ["title"]

        selector = MockFieldSelector()
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
            field_selector=selector,
        )
        result = await pipeline.arun(_make_docs(2), domain_hint="test", normalize_fields=["title"])

        # Selector should NOT have been called
        assert not selector.called
        assert "title" in result.normalizations

    async def test_no_selector_no_normalize_fields_skips(self) -> None:
        """No selector + no normalize_fields = no normalization (backwards-compat)."""
        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
        )
        result = await pipeline.arun(_make_docs(2), domain_hint="test")
        assert result.normalizations == {}

    async def test_selector_returns_empty_list(self) -> None:
        """Selector returning [] means nothing to normalize."""

        class EmptySelector:
            async def aselect(self, schema: Any, records: list[Any], **kwargs: Any) -> list[str]:
                return []

            def select(self, schema: Any, records: list[Any], **kwargs: Any) -> list[str]:
                return []

        pipeline = Pipeline(
            discovery=MockDiscovery(),
            extraction=MockExtraction(),
            normalization=MockNormalization(),
            field_selector=EmptySelector(),
        )
        result = await pipeline.arun(_make_docs(2), domain_hint="test")
        assert result.normalizations == {}


class TestPipelineQuick:
    def test_quick_creates_all_strategies(self) -> None:
        pipeline = Pipeline.quick(model="gpt-5.4-mini")
        assert pipeline.discovery is not None
        assert pipeline.extraction is not None
        assert pipeline.normalization is not None
        assert pipeline.field_selector is not None

    def test_quick_with_base_url(self) -> None:
        pipeline = Pipeline.quick(
            model="qwen3.5",
            base_url="http://localhost:11434/v1",
        )
        assert pipeline.discovery is not None


class TestNormalizeFieldsGuard:
    @pytest.mark.asyncio
    async def test_normalize_fields_raises_without_strategy(self) -> None:
        """_normalize_fields raises RuntimeError if normalization is None."""
        pipeline = Pipeline()
        with pytest.raises(RuntimeError, match="normalization strategy"):
            await pipeline._normalize_fields([], ["field_a"])
