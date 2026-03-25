"""Tests for core domain types — logic and behavior, not defaults."""

from __future__ import annotations

from pydantic import BaseModel

from catchfly._types import (
    NormalizationResult,
    PipelineResult,
    Schema,
    UsageReport,
)


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
    def test_to_dataframe(self) -> None:
        """to_dataframe works when pandas is installed."""
        result = PipelineResult()
        df = result.to_dataframe()
        assert len(df) == 0


class TestUsageReport:
    def test_cost_usd_alias(self) -> None:
        report = UsageReport(total_cost_usd=1.23)
        assert report.cost_usd == 1.23
        assert report.cost_usd == report.total_cost_usd
