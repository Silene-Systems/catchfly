"""Field selector protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from catchfly._types import Schema


@runtime_checkable
class FieldSelector(Protocol):
    """Protocol for selecting which fields should be normalized.

    Given a discovered schema and extracted records, decides which
    fields are good candidates for normalization (categorical, repetitive)
    vs which should be skipped (free-text, identifiers, numerics).
    """

    def select(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]: ...

    async def aselect(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]: ...
