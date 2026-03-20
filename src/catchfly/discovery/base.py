"""Discovery strategy protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from catchfly._types import Document, Schema


@dataclass
class DiscoveryReport:
    """Report from a discovery run."""

    num_documents_sampled: int = 0
    field_coverage: dict[str, float] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DiscoveryStrategy(Protocol):
    """Protocol for schema discovery strategies."""

    def discover(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Schema: ...

    async def adiscover(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Schema: ...
