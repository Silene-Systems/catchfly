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

    def test_apply_normalizations_with_pydantic_records(self) -> None:
        class Product(BaseModel):
            brand: str
            color: str | None = None

        result = PipelineResult(
            records=[
                Product(brand="NIKE", color="red"),
                Product(brand="nike", color="blue"),
            ],
            normalizations={
                "brand": NormalizationResult(
                    mapping={"NIKE": "Nike", "nike": "Nike"},
                ),
            },
        )
        normalized = result.apply_normalizations()
        assert len(normalized) == 2
        assert normalized[0]["brand"] == "Nike"
        assert normalized[1]["brand"] == "Nike"
        # Non-normalized field unchanged
        assert normalized[0]["color"] == "red"

    def test_apply_normalizations_with_dict_records(self) -> None:
        result = PipelineResult(
            records=[
                {"city": "NYC", "name": "Alice"},
                {"city": "New York City", "name": "Bob"},
            ],
            normalizations={
                "city": NormalizationResult(
                    mapping={"NYC": "New York", "New York City": "New York"},
                ),
            },
        )
        normalized = result.apply_normalizations()
        assert normalized[0]["city"] == "New York"
        assert normalized[1]["city"] == "New York"
        assert normalized[0]["name"] == "Alice"

    def test_apply_normalizations_with_list_field(self) -> None:
        class Patient(BaseModel):
            symptoms: list[str]

        result = PipelineResult(
            records=[
                Patient(symptoms=["heart attack", "chest pain", "MI"]),
            ],
            normalizations={
                "symptoms": NormalizationResult(
                    mapping={
                        "heart attack": "myocardial infarction",
                        "MI": "myocardial infarction",
                        "chest pain": "chest pain",
                    },
                ),
            },
        )
        normalized = result.apply_normalizations()
        assert normalized[0]["symptoms"] == [
            "myocardial infarction",
            "chest pain",
            "myocardial infarction",
        ]

    def test_apply_normalizations_no_normalizations(self) -> None:
        result = PipelineResult(
            records=[{"a": "1"}, {"a": "2"}],
        )
        normalized = result.apply_normalizations()
        assert len(normalized) == 2
        assert normalized[0] == {"a": "1"}

    def test_apply_normalizations_empty_records(self) -> None:
        result = PipelineResult()
        assert result.apply_normalizations() == []

    def test_apply_normalizations_does_not_mutate_records(self) -> None:
        original = {"brand": "NIKE"}
        result = PipelineResult(
            records=[original],
            normalizations={
                "brand": NormalizationResult(mapping={"NIKE": "Nike"}),
            },
        )
        result.apply_normalizations()
        assert original["brand"] == "NIKE"

    def test_normalized_records_property(self) -> None:
        result = PipelineResult(
            records=[{"x": "a"}],
            normalizations={
                "x": NormalizationResult(mapping={"a": "A"}),
            },
        )
        assert result.normalized_records == result.apply_normalizations()

    def test_apply_normalizations_none_values_skipped(self) -> None:
        result = PipelineResult(
            records=[{"brand": None, "color": "red"}],
            normalizations={
                "brand": NormalizationResult(mapping={"NIKE": "Nike"}),
            },
        )
        normalized = result.apply_normalizations()
        assert normalized[0]["brand"] is None

    def test_apply_normalizations_unmapped_values_pass_through(self) -> None:
        result = PipelineResult(
            records=[{"brand": "Adidas"}],
            normalizations={
                "brand": NormalizationResult(mapping={"NIKE": "Nike"}),
            },
        )
        normalized = result.apply_normalizations()
        assert normalized[0]["brand"] == "Adidas"


class TestUsageReport:
    def test_cost_usd_alias(self) -> None:
        report = UsageReport(total_cost_usd=1.23)
        assert report.cost_usd == 1.23
        assert report.cost_usd == report.total_cost_usd
