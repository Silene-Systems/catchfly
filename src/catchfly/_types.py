"""Core domain types used across all catchfly modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """A single document for processing."""

    content: str
    id: str | None = None
    source: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass
class Schema:
    """A discovered or user-provided schema."""

    model: type[BaseModel] | None
    json_schema: dict[str, Any]
    field_metadata: dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)


SchemaT = TypeVar("SchemaT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Extraction results
# ---------------------------------------------------------------------------
@dataclass
class RecordProvenance:
    """Provenance information for a single extracted record."""

    source_document: str
    chunk_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    confidence: float | None = None


@dataclass
class ExtractionResult(Generic[SchemaT]):
    """Result of running an extraction strategy."""

    records: list[SchemaT]
    errors: list[tuple[Document, Exception]] = field(default_factory=list)
    provenance: list[RecordProvenance] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalization results
# ---------------------------------------------------------------------------
@dataclass
class NormalizationResult:
    """Result of running a normalization strategy."""

    mapping: dict[str, str]
    clusters: dict[str, list[str]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def explain(self, value: str) -> str:
        """Return human-readable explanation of why value was mapped."""
        canonical = self.mapping.get(value)
        if canonical is None:
            return f"'{value}' was not found in the normalization mapping."

        details = self.metadata.get("explanations", {}).get(value, "")
        if details:
            return f"'{value}' → '{canonical}': {details}"

        if self.clusters:
            for cluster_name, members in self.clusters.items():
                if value in members:
                    return (
                        f"'{value}' → '{canonical}' "
                        f"(cluster '{cluster_name}', {len(members)} members)"
                    )

        return f"'{value}' → '{canonical}'"


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------
@dataclass
class UsageReport:
    """Cost and usage statistics for a pipeline run."""

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    breakdown: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        """Alias for total_cost_usd (PRD compatibility)."""
        return self.total_cost_usd


@dataclass
class PipelineResult:
    """Full result of a pipeline run."""

    schema: Schema | None = None
    records: list[Any] = field(default_factory=list)
    normalizations: dict[str, NormalizationResult] = field(default_factory=dict)
    errors: list[tuple[Document, Exception]] = field(default_factory=list)
    report: UsageReport = field(default_factory=UsageReport)

    def to_dataframe(self) -> Any:
        """Convert records to a pandas DataFrame."""
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install catchfly[export]"
            ) from e

        if not self.records:
            return pd.DataFrame()

        rows = [r.model_dump() if hasattr(r, "model_dump") else r for r in self.records]
        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path) -> None:
        """Export records to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def to_parquet(self, path: str | Path) -> None:
        """Export records to Parquet."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    def to_json(self, path: str | Path) -> None:
        """Export records to JSON."""
        df = self.to_dataframe()
        df.to_json(path, orient="records", indent=2)
