"""
catchfly — Catch the structured data.

Schema discovery, structured extraction, and normalization
for unstructured text data.

https://github.com/silene-systems/catchfly
"""

import logging

from catchfly._defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL
from catchfly._types import (
    Document,
    ExtractionResult,
    NormalizationResult,
    PipelineResult,
    RecordProvenance,
    Schema,
    UsageReport,
)
from catchfly.exceptions import (
    BudgetExceededError,
    CatchflyError,
    DiscoveryError,
    ExtractionError,
    NormalizationError,
    ProviderError,
    SchemaError,
)
from catchfly.pipeline import Pipeline

__version__ = "1.1.1"

__all__ = [
    "BudgetExceededError",
    "CatchflyError",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL",
    "DiscoveryError",
    "Document",
    "ExtractionError",
    "ExtractionResult",
    "NormalizationError",
    "NormalizationResult",
    "Pipeline",
    "PipelineResult",
    "ProviderError",
    "RecordProvenance",
    "Schema",
    "SchemaError",
    "UsageReport",
    "__version__",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())


def __dir__() -> list[str]:
    return __all__
