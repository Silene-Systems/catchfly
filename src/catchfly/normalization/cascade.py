"""Cascade normalization — chain strategies with fallback for unmapped values."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, PrivateAttr

from catchfly._compat import run_sync
from catchfly._types import NormalizationResult

logger = logging.getLogger(__name__)


class CascadeNormalization(BaseModel):
    """Chain multiple normalization strategies with fallback.

    Each step receives only the values that previous steps could not
    normalize (identity-mapped values). Final result merges all mappings.

    Recommended cascade: Dictionary → LLM → Ontology.

    Example::

        cascade = CascadeNormalization(steps=[
            DictionaryNormalization(mapping={"NYC": "New York"}),
            LLMCanonicalization(model="gpt-5.4-mini"),
            OntologyMapping(ontology="hpo"),
        ])
    """

    steps: list[Any]  # list[NormalizationStrategy] — Any for Pydantic duck-typing compat
    _usage_callback: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    async def anormalize(
        self,
        values: list[str],
        context_field: str = "",
        **kwargs: Any,
    ) -> NormalizationResult:
        """Normalize values by cascading through strategies."""
        if not values:
            return NormalizationResult(mapping={}, clusters=None, metadata={})

        unique_values = list(set(values))
        merged_mapping: dict[str, str] = {}
        step_metadata: list[dict[str, Any]] = []
        remaining = unique_values

        for i, step in enumerate(self.steps):
            if not remaining:
                break

            # Propagate usage callback
            usage_cb = self._usage_callback
            if usage_cb is not None:
                step._usage_callback = usage_cb

            logger.debug(
                "CascadeNormalization step %d (%s): %d values",
                i,
                type(step).__name__,
                len(remaining),
            )

            result = await step.anormalize(
                remaining, context_field=context_field, **kwargs
            )

            # Collect mapped values (canonical != raw)
            newly_mapped: dict[str, str] = {}
            still_unmapped: list[str] = []
            for v in remaining:
                canonical = result.mapping.get(v, v)
                if canonical != v:
                    newly_mapped[v] = canonical
                else:
                    still_unmapped.append(v)

            merged_mapping.update(newly_mapped)
            step_metadata.append(
                {
                    "step": i,
                    "strategy": type(step).__name__,
                    "mapped_count": len(newly_mapped),
                    "remaining_count": len(still_unmapped),
                }
            )

            logger.debug(
                "CascadeNormalization step %d: mapped %d, remaining %d",
                i,
                len(newly_mapped),
                len(still_unmapped),
            )

            remaining = still_unmapped

        # Passthrough any values still unmapped after all steps
        for v in remaining:
            merged_mapping[v] = v

        # Build clusters from merged mapping
        clusters: dict[str, list[str]] = {}
        for raw, canonical in merged_mapping.items():
            clusters.setdefault(canonical, []).append(raw)

        logger.info(
            "CascadeNormalization: %d unique values → %d groups "
            "across %d steps for field '%s'",
            len(unique_values),
            len(clusters),
            len(self.steps),
            context_field,
        )

        return NormalizationResult(
            mapping=merged_mapping,
            clusters=clusters,
            metadata={
                "strategy": "cascade",
                "field": context_field,
                "steps": step_metadata,
                "n_groups": len(clusters),
            },
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

    @classmethod
    def default(
        cls,
        *,
        dictionary: dict[str, str] | None = None,
        model: str = "gpt-5.4-mini",
        ontology: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> CascadeNormalization:
        """Create the recommended cascade: Dictionary → LLM → Ontology.

        Args:
            dictionary: Optional static mapping for the dictionary step.
                If None, the dictionary step is skipped.
            model: LLM model name for canonicalization.
            ontology: Ontology source for the final step (e.g. "hpo",
                path to .obo/.csv/.json). If None, ontology step is skipped.
            base_url: API base URL for LLM/embedding providers.
            api_key: API key for LLM/embedding providers.
        """
        from catchfly.normalization.dictionary import DictionaryNormalization
        from catchfly.normalization.llm_canonical import LLMCanonicalization

        steps: list[Any] = []

        if dictionary:
            steps.append(
                DictionaryNormalization(
                    mapping=dictionary,
                    case_insensitive=True,
                    passthrough_unmapped=True,
                )
            )

        steps.append(
            LLMCanonicalization(
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        )

        if ontology:
            from catchfly.normalization.ontology_mapping import OntologyMapping

            steps.append(
                OntologyMapping(
                    ontology=ontology,
                    base_url=base_url,
                    api_key=api_key,
                )
            )

        return cls(steps=steps)
