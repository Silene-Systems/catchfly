"""Pipeline orchestrator — connects discovery, extraction, and normalization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from catchfly._compat import run_sync
from catchfly._types import (
    Document,
    NormalizationResult,
    PipelineResult,
    Schema,
)
from catchfly.exceptions import BudgetExceededError
from catchfly.telemetry.tracker import UsageTracker

if TYPE_CHECKING:
    from pydantic import BaseModel

    from catchfly.discovery.base import DiscoveryStrategy
    from catchfly.extraction.base import ExtractionStrategy
    from catchfly.normalization.base import NormalizationStrategy

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates discovery → extraction → normalization.

    Each stage is optional — users can provide a pre-existing schema
    to skip discovery, or skip normalization entirely.
    """

    def __init__(
        self,
        discovery: DiscoveryStrategy | None = None,
        extraction: ExtractionStrategy | None = None,
        normalization: NormalizationStrategy | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize pipeline with optional strategy instances.

        Args:
            discovery: Schema discovery strategy (or None to skip).
            extraction: Data extraction strategy (or None to skip).
            normalization: Value normalization strategy (or None to skip).
            verbose: Enable progress logging.
        """
        self.discovery = discovery
        self.extraction = extraction
        self.normalization = normalization
        self.verbose = verbose

    @classmethod
    def quick(
        cls,
        model: str = "gpt-5.4-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        verbose: bool = True,
        embedding_model: str = "text-embedding-3-small",
    ) -> Pipeline:
        """Create a pipeline with sensible defaults.

        Uses SinglePassDiscovery + LLMDirectExtraction + EmbeddingClustering.
        """
        from catchfly.discovery.single_pass import SinglePassDiscovery
        from catchfly.extraction.llm_direct import LLMDirectExtraction
        from catchfly.normalization.embedding_cluster import EmbeddingClustering

        return cls(
            discovery=SinglePassDiscovery(
                model=model,
                base_url=base_url,
                api_key=api_key,
            ),
            extraction=LLMDirectExtraction(
                model=model,
                base_url=base_url,
                api_key=api_key,
            ),
            normalization=EmbeddingClustering(
                embedding_model=embedding_model,
                base_url=base_url,
                api_key=api_key,
            ),
            verbose=verbose,
        )

    async def arun(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        normalize_fields: list[str] | None = None,
        max_cost_usd: float | None = None,
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Run the full pipeline asynchronously.

        Args:
            documents: Documents to process.
            domain_hint: Optional hint about the document domain.
            normalize_fields: Fields to normalize (None = skip normalization).
            max_cost_usd: Budget limit — pipeline halts if exceeded.
            schema: Pre-existing Pydantic model — skips discovery if provided.
        """
        tracker = UsageTracker(max_cost_usd=max_cost_usd)
        result = PipelineResult()

        try:
            # --- Discovery ---
            discovered_schema: Schema | None = None
            extraction_model: type[BaseModel] | None = schema

            if schema is None and self.discovery is not None:
                logger.info("Pipeline: starting discovery")
                discovered_schema = await self.discovery.adiscover(
                    documents, domain_hint=domain_hint
                )
                result.schema = discovered_schema
                extraction_model = discovered_schema.model
                logger.info(
                    "Pipeline: discovery complete — %d fields",
                    len(discovered_schema.json_schema.get("properties", {})),
                )
            elif schema is not None:
                result.schema = Schema(
                    model=schema,
                    json_schema=schema.model_json_schema(),
                    lineage=["user-provided"],
                )

            # --- Extraction ---
            if extraction_model is not None and self.extraction is not None:
                logger.info("Pipeline: starting extraction on %d documents", len(documents))
                extraction_result = await self.extraction.aextract(extraction_model, documents)
                result.records = extraction_result.records
                result.errors = extraction_result.errors
                logger.info(
                    "Pipeline: extraction complete — %d records, %d errors",
                    len(result.records),
                    len(result.errors),
                )
            elif extraction_model is None and self.extraction is not None:
                logger.warning(
                    "Pipeline: skipping extraction — no schema available "
                    "(discovery failed or not configured)"
                )

            # --- Normalization ---
            if normalize_fields and self.normalization is not None and result.records:
                logger.info(
                    "Pipeline: starting normalization for fields: %s",
                    normalize_fields,
                )
                result.normalizations = await self._normalize_fields(
                    result.records, normalize_fields
                )
                logger.info(
                    "Pipeline: normalization complete — %d fields normalized",
                    len(result.normalizations),
                )

        except BudgetExceededError:
            logger.warning("Pipeline: budget exceeded, returning partial results")
            raise

        result.report = tracker.report()
        return result

    def run(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        normalize_fields: list[str] | None = None,
        max_cost_usd: float | None = None,
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Run the full pipeline synchronously."""
        return run_sync(
            self.arun(
                documents,
                domain_hint=domain_hint,
                normalize_fields=normalize_fields,
                max_cost_usd=max_cost_usd,
                schema=schema,
                **kwargs,
            )
        )

    def estimate_cost(self, documents: list[Document]) -> dict[str, float]:
        """Estimate pipeline cost before running.

        Returns approximate cost breakdown by stage (in USD).
        """
        from catchfly.extraction.chunking import estimate_chunks

        n_docs = len(documents)
        avg_tokens = sum(len(d.content) // 4 for d in documents) / max(n_docs, 1)
        n_chunks = estimate_chunks(documents)

        # Very rough estimates based on typical token counts
        discovery_cost = avg_tokens * 6 / 1_000_000 * 0.15  # ~6 docs, mini pricing
        extraction_cost = n_chunks * avg_tokens * 2 / 1_000_000 * 0.15
        normalization_cost = n_docs * 0.001  # embedding costs are low

        return {
            "discovery": round(discovery_cost, 4),
            "extraction": round(extraction_cost, 4),
            "normalization": round(normalization_cost, 4),
            "total": round(discovery_cost + extraction_cost + normalization_cost, 4),
        }

    async def _normalize_fields(
        self,
        records: list[Any],
        fields: list[str],
    ) -> dict[str, NormalizationResult]:
        """Normalize specified fields from extracted records."""
        assert self.normalization is not None
        normalizations: dict[str, NormalizationResult] = {}

        for field_name in fields:
            # Collect unique values for this field from all records
            values: list[str] = []
            for record in records:
                val = None
                if hasattr(record, field_name):
                    val = getattr(record, field_name)
                elif isinstance(record, dict):
                    val = record.get(field_name)

                if val is None:
                    continue
                if isinstance(val, list):
                    values.extend(str(v) for v in val)
                else:
                    values.append(str(val))

            if not values:
                logger.debug("Pipeline: no values found for field '%s', skipping", field_name)
                continue

            logger.info(
                "Pipeline: normalizing field '%s' (%d values, %d unique)",
                field_name,
                len(values),
                len(set(values)),
            )
            normalizations[field_name] = await self.normalization.anormalize(
                values, context_field=field_name
            )

        return normalizations
