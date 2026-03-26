"""Dictionary-based normalization — exact or case-insensitive lookup."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from catchfly._compat import run_sync
from catchfly._types import NormalizationResult
from catchfly.exceptions import NormalizationError

logger = logging.getLogger(__name__)


class DictionaryNormalization(BaseModel):
    """Normalize values using a static dictionary mapping.

    Supports exact match and optional case-insensitive matching.
    Values not found in the dictionary pass through unchanged
    (or raise if passthrough_unmapped=False).
    """

    mapping: dict[str, str]
    case_insensitive: bool = False
    passthrough_unmapped: bool = True

    model_config = {"arbitrary_types_allowed": True}

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values using dictionary lookup."""
        if not values:
            return NormalizationResult(mapping={}, clusters=None, metadata={})

        lookup = self.mapping
        if self.case_insensitive:
            lookup = {k.lower(): v for k, v in self.mapping.items()}

        result_mapping: dict[str, str] = {}
        per_value: dict[str, dict[str, Any]] = {}
        for v in set(values):
            key = v.lower() if self.case_insensitive else v
            if key in lookup:
                result_mapping[v] = lookup[key]
                per_value[v] = {"confidence": 1.0}
            elif self.passthrough_unmapped:
                result_mapping[v] = v
                per_value[v] = {"confidence": 0.0}
            else:
                raise NormalizationError(
                    f"Value '{v}' not found in dictionary for field '{context_field}'"
                )

        clusters: dict[str, list[str]] = {}
        for raw, canonical in result_mapping.items():
            clusters.setdefault(canonical, []).append(raw)

        logger.info(
            "DictionaryNormalization: mapped %d unique values to %d canonicals for field '%s'",
            len(result_mapping),
            len(clusters),
            context_field,
        )

        return NormalizationResult(
            mapping=result_mapping,
            clusters=clusters,
            metadata={
                "strategy": "dictionary",
                "field": context_field,
                "per_value": per_value,
            },
        )

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Synchronous wrapper."""
        return run_sync(self.anormalize(values, context_field=context_field, **kwargs))
