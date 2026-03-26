"""Cascade normalization — chain strategies with fallback for unmapped values."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, PrivateAttr

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._types import NormalizationResult

logger = logging.getLogger(__name__)


class CascadeNormalization(BaseModel):
    """Chain multiple normalization strategies with fallback.

    Each step receives only the values that previous steps could not
    normalize (identity-mapped values). Final result merges all mappings.

    When ``confidence_thresholds`` is provided, routing uses per-value
    confidence scores instead of identity checks: a value is considered
    "resolved" only when its confidence meets the step's threshold.
    This prevents low-confidence mappings from short-circuiting the cascade.

    Recommended cascade: Dictionary → LLM → Ontology.

    Example::

        cascade = CascadeNormalization(steps=[
            DictionaryNormalization(mapping={"NYC": "New York"}),
            LLMCanonicalization(model="gpt-5.4-mini"),
            OntologyMapping(ontology="hpo"),
        ])
    """

    steps: list[Any]  # list[NormalizationStrategy] — Any for Pydantic duck-typing compat

    confidence_thresholds: list[float] | None = None
    """Per-step confidence thresholds.  Values mapped with confidence
    ``>= threshold`` are considered resolved and skip remaining steps.
    Must have the same length as ``steps``.  If ``None`` (default),
    falls back to identity-check routing (backward compatible)."""

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

        if self.confidence_thresholds is not None and len(self.confidence_thresholds) != len(
            self.steps
        ):
            raise ValueError(
                f"confidence_thresholds length ({len(self.confidence_thresholds)}) "
                f"must match steps length ({len(self.steps)})"
            )

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

            result = await step.anormalize(remaining, context_field=context_field, **kwargs)

            # Route: confidence-based or identity-check
            newly_mapped: dict[str, str] = {}
            still_unmapped: list[str] = []

            if self.confidence_thresholds is not None:
                threshold = self.confidence_thresholds[i]
                per_value = result.metadata.get("per_value", {})
                for v in remaining:
                    canonical = result.mapping.get(v, v)
                    conf = per_value.get(v, {}).get("confidence", 0.0)
                    if canonical != v and conf >= threshold:
                        newly_mapped[v] = canonical
                    else:
                        still_unmapped.append(v)
            else:
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
            "CascadeNormalization: %d unique values → %d groups across %d steps for field '%s'",
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
        return run_sync(self.anormalize(values, context_field=context_field, **kwargs))

    @classmethod
    def default(
        cls,
        *,
        dictionary: dict[str, str] | None = None,
        model: str = DEFAULT_MODEL,
        ontology: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        use_confidence: bool = False,
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
            use_confidence: Enable confidence-based routing with
                sensible default thresholds (Dictionary=1.0, LLM=0.80,
                Ontology=0.90).
        """
        from catchfly.normalization.dictionary import DictionaryNormalization
        from catchfly.normalization.llm_canonical import LLMCanonicalization

        steps: list[Any] = []
        thresholds: list[float] = []

        if dictionary:
            steps.append(
                DictionaryNormalization(
                    mapping=dictionary,
                    case_insensitive=True,
                    passthrough_unmapped=True,
                )
            )
            thresholds.append(1.0)

        steps.append(
            LLMCanonicalization(
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        )
        thresholds.append(0.80)

        if ontology:
            from catchfly.normalization.ontology_mapping import OntologyMapping

            steps.append(
                OntologyMapping(
                    ontology=ontology,
                    base_url=base_url,
                    api_key=api_key,
                )
            )
            thresholds.append(0.90)

        return cls(
            steps=steps,
            confidence_thresholds=thresholds if use_confidence else None,
        )

    def learn(self, result: NormalizationResult, *, min_confidence: float = 0.80) -> None:
        """Prepend a learned dictionary step from normalization result.

        High-confidence mappings from *result* become a
        :class:`DictionaryNormalization` at the front of the cascade,
        so subsequent runs resolve known values instantly ($0).

        If the first step is already a ``DictionaryNormalization``, the
        learned mappings are merged into it instead of creating a new step.
        """
        from catchfly.normalization.dictionary import DictionaryNormalization

        learned = result.to_dictionary(min_confidence=min_confidence)
        if not learned:
            logger.debug("CascadeNormalization.learn: no mappings above threshold, skipping")
            return

        if self.steps and isinstance(self.steps[0], DictionaryNormalization):
            # Merge into existing dictionary step
            merged = {**self.steps[0].mapping, **learned}
            self.steps[0] = DictionaryNormalization(
                mapping=merged,
                case_insensitive=self.steps[0].case_insensitive,
                passthrough_unmapped=True,
            )
            logger.info(
                "CascadeNormalization.learn: merged %d entries "
                "into existing dictionary (%d total)",
                len(learned),
                len(merged),
            )
        else:
            dict_step = DictionaryNormalization(
                mapping=learned,
                case_insensitive=True,
                passthrough_unmapped=True,
            )
            self.steps.insert(0, dict_step)

            # Keep confidence_thresholds in sync
            if self.confidence_thresholds is not None:
                self.confidence_thresholds = [1.0, *self.confidence_thresholds]

            logger.info(
                "CascadeNormalization.learn: prepended dictionary with %d entries",
                len(learned),
            )
