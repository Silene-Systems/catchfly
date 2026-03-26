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

    def to_dictionary(self, *, min_confidence: float = 0.0) -> dict[str, str]:
        """Export non-identity mappings as a plain dictionary.

        Returns only entries where ``canonical != raw`` and the per-value
        confidence (if available) meets *min_confidence*.  Useful for
        building a :class:`DictionaryNormalization` from results.
        """
        per_value = self.metadata.get("per_value", {})
        result: dict[str, str] = {}
        for raw, canonical in self.mapping.items():
            if raw == canonical:
                continue
            conf = per_value.get(raw, {}).get("confidence", 1.0)
            if conf >= min_confidence:
                result[raw] = canonical
        return result

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

    def apply_normalizations(self) -> list[dict[str, Any]]:
        """Return records with normalized values applied.

        For each record, replaces raw field values with their canonical forms
        from ``self.normalizations``. Handles both scalar string fields and
        list-of-string fields.

        Returns:
            List of dicts with normalized values. Pydantic model records are
            converted to dicts first since normalization may change value types.
        """
        rows = self._records_to_dicts()

        if not self.normalizations:
            return rows

        result = []
        for row in rows:
            normalized = dict(row)
            for field_name, norm_result in self.normalizations.items():
                if field_name not in normalized:
                    continue
                value = normalized[field_name]
                if isinstance(value, list):
                    normalized[field_name] = [
                        norm_result.mapping.get(str(v), str(v)) for v in value
                    ]
                elif value is not None:
                    str_val = str(value)
                    normalized[field_name] = norm_result.mapping.get(str_val, str_val)
            result.append(normalized)
        return result

    @property
    def normalized_records(self) -> list[dict[str, Any]]:
        """Alias for :meth:`apply_normalizations`."""
        return self.apply_normalizations()

    def _records_to_dicts(self) -> list[dict[str, Any]]:
        """Convert records to list of dicts regardless of their type."""
        rows: list[dict[str, Any]] = []
        for r in self.records:
            if hasattr(r, "model_dump"):
                rows.append(r.model_dump())
            elif isinstance(r, dict):
                rows.append(dict(r))
            else:
                rows.append({"_raw": str(r)})
        return rows

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

        return pd.DataFrame(self._records_to_dicts())

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
