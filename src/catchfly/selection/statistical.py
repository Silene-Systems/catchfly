"""StatisticalFieldSelector — heuristic-based field selection with zero LLM cost."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from catchfly._compat import run_sync

if TYPE_CHECKING:
    from catchfly._types import Schema

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "description",
    "text",
    "content",
    "summary",
    "note",
    "comment",
    "id",
    "url",
    "email",
    "name",
    "title",
]


class StatisticalFieldSelector(BaseModel):
    """Select normalization-worthy fields using value statistics.

    Analyzes extracted values per field: cardinality ratio, average string
    length, and name patterns. Zero LLM cost — pure Python analysis.

    Best for: cost-sensitive pipelines or as a fast pre-filter.
    For more accurate selection, use ``LLMFieldSelector``.
    """

    max_cardinality_ratio: float = 0.5
    """Fields with unique_values / total_records above this are skipped."""
    max_avg_length: int = 50
    """Fields with average value length above this are skipped (likely free-text)."""
    min_unique_values: int = 3
    """Fields with fewer unique values than this are skipped (nothing to normalize)."""
    exclude_patterns: list[str] = _DEFAULT_EXCLUDE_PATTERNS
    """Field names containing any of these substrings are skipped."""

    async def aselect(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]:
        """Select fields for normalization asynchronously."""
        props = schema.json_schema.get("properties", {})
        selected: list[str] = []

        for field_name, spec in props.items():
            # Type filter: only string or array-of-string fields
            field_type = spec.get("type", "")
            is_string = field_type == "string"
            is_string_array = (
                field_type == "array"
                and spec.get("items", {}).get("type") == "string"
            )
            if not (is_string or is_string_array):
                logger.debug(
                    "StatisticalFieldSelector: skipping '%s' — type '%s'",
                    field_name,
                    field_type,
                )
                continue

            # Name exclusion
            if any(pat in field_name.lower() for pat in self.exclude_patterns):
                logger.debug(
                    "StatisticalFieldSelector: skipping '%s' — name matches exclude pattern",
                    field_name,
                )
                continue

            # Collect values
            values: list[str] = []
            for record in records:
                val = (
                    getattr(record, field_name, None)
                    if hasattr(record, field_name)
                    else record.get(field_name) if isinstance(record, dict) else None
                )
                if val is None:
                    continue
                if isinstance(val, list):
                    values.extend(str(v) for v in val)
                else:
                    values.append(str(val))

            if not values:
                continue

            unique = set(values)

            # Minimum unique values
            if len(unique) < self.min_unique_values:
                logger.debug(
                    "StatisticalFieldSelector: skipping '%s' — %d unique < %d minimum",
                    field_name,
                    len(unique),
                    self.min_unique_values,
                )
                continue

            # Cardinality ratio
            cardinality = len(unique) / max(len(records), 1)
            if cardinality > self.max_cardinality_ratio:
                logger.debug(
                    "StatisticalFieldSelector: skipping '%s' — cardinality %.2f > %.2f",
                    field_name,
                    cardinality,
                    self.max_cardinality_ratio,
                )
                continue

            # Average length
            avg_len = sum(len(v) for v in values) / len(values)
            if avg_len > self.max_avg_length:
                logger.debug(
                    "StatisticalFieldSelector: skipping '%s' — avg length %.1f > %d",
                    field_name,
                    avg_len,
                    self.max_avg_length,
                )
                continue

            selected.append(field_name)

        logger.info(
            "StatisticalFieldSelector: selected %d fields: %s",
            len(selected),
            selected,
        )
        return selected

    def select(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]:
        """Select fields for normalization synchronously."""
        return run_sync(self.aselect(schema, records, **kwargs))
