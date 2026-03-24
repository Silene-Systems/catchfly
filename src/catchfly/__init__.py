"""
catchfly — Catch the structured data.

Schema discovery, structured extraction, and normalization
for unstructured text data.

https://github.com/silene-systems/catchfly
"""

from catchfly._types import (
    Document,
    ExtractionResult,
    NormalizationResult,
    PipelineResult,
    RecordProvenance,
    Schema,
    UsageReport,
)
from catchfly.pipeline import Pipeline

__version__ = "1.0.0"

__all__ = [
    "Document",
    "ExtractionResult",
    "NormalizationResult",
    "Pipeline",
    "PipelineResult",
    "RecordProvenance",
    "Schema",
    "UsageReport",
    "__version__",
]
