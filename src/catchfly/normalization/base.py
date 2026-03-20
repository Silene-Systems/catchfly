"""Normalization strategy protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from catchfly._types import NormalizationResult


@runtime_checkable
class NormalizationStrategy(Protocol):
    """Protocol for value normalization strategies."""

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult: ...

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult: ...
