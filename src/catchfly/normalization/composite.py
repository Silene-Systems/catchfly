"""Composite normalization — route different fields to different strategies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from catchfly._compat import run_sync
from catchfly.exceptions import NormalizationError

if TYPE_CHECKING:
    from catchfly._types import NormalizationResult

logger = logging.getLogger(__name__)


class CompositeNormalization:
    """Routes normalization to field-specific strategies with an optional fallback.

    Example::

        composite = CompositeNormalization(
            field_strategies={
                "symptoms": OntologyMapping(ontology="hpo"),
                "brand": LLMCanonicalization(model="gpt-5.4-mini"),
            },
            default=DictionaryNormalization(mapping={...}),
        )
    """

    def __init__(
        self,
        field_strategies: dict[str, Any],
        default: Any | None = None,
    ) -> None:
        self._field_strategies = field_strategies
        self._default = default

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values using the strategy assigned to context_field."""
        strategy = self._field_strategies.get(context_field, self._default)
        if strategy is None:
            raise NormalizationError(
                f"No normalization strategy configured for field "
                f"'{context_field}' and no default strategy set."
            )

        # Propagate usage callback if set on composite
        usage_cb = getattr(self, "_usage_callback", None)
        if usage_cb is not None and not hasattr(strategy, "_usage_callback"):
            strategy._usage_callback = usage_cb  # type: ignore[attr-defined]

        logger.debug(
            "CompositeNormalization: routing field '%s' to %s",
            context_field,
            type(strategy).__name__,
        )
        return await strategy.anormalize(
            values, context_field=context_field, **kwargs
        )

    def normalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Synchronous wrapper."""
        return run_sync(
            self.anormalize(values, context_field=context_field, **kwargs)
        )
