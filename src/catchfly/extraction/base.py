"""Extraction strategy protocol and shared types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from catchfly._types import Document, ExtractionResult

ErrorMode = Literal["raise", "skip", "collect"]


@runtime_checkable
class ExtractionStrategy(Protocol):
    """Protocol for data extraction strategies."""

    def extract(
        self,
        schema: type[BaseModel],
        documents: list[Document],
        **kwargs: Any,
    ) -> ExtractionResult[Any]: ...

    async def aextract(
        self,
        schema: type[BaseModel],
        documents: list[Document],
        **kwargs: Any,
    ) -> ExtractionResult[Any]: ...
