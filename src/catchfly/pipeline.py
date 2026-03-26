"""Pipeline orchestrator — connects discovery, extraction, and normalization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._types import (
    Document,
    NormalizationResult,
    PipelineResult,
    Schema,
)
from catchfly.checkpoint import _Checkpoint
from catchfly.exceptions import BudgetExceededError, SchemaError
from catchfly.telemetry.tracker import UsageTracker

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pydantic import BaseModel

    from catchfly.discovery.base import DiscoveryStrategy
    from catchfly.extraction.base import ExtractionStrategy
    from catchfly.normalization.base import NormalizationStrategy
    from catchfly.selection.base import FieldSelector

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
        normalization: NormalizationStrategy | dict[str, NormalizationStrategy] | None = None,
        field_selector: FieldSelector | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialize pipeline with optional strategy instances.

        Args:
            discovery: Schema discovery strategy (or None to skip).
            extraction: Data extraction strategy (or None to skip).
            normalization: Value normalization strategy, or a dict mapping
                field names to strategies (auto-wrapped in CompositeNormalization),
                or None to skip.
            field_selector: Strategy for auto-selecting which fields to normalize.
                When set, ``normalize_fields`` in ``run()`` can be omitted — the
                selector decides. Explicit ``normalize_fields`` overrides the selector.
            verbose: Show tqdm progress bars during extraction/normalization.
        """
        self.discovery = discovery
        self.extraction = extraction
        self.field_selector = field_selector
        self.verbose = verbose
        if isinstance(normalization, dict):
            from catchfly.normalization.composite import CompositeNormalization

            self.normalization = CompositeNormalization(field_strategies=normalization)
        else:
            self.normalization = normalization

    @classmethod
    def quick(
        cls,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        on_error: Literal["raise", "skip", "collect"] = "raise",
        *,
        verbose: bool = False,
    ) -> Pipeline:
        """Create a pipeline with sensible defaults.

        Uses SinglePassDiscovery + LLMDirectExtraction + LLMFieldSelector
        + LLMCanonicalization.  The field selector automatically identifies
        which discovered fields are categorical and worth normalizing — no
        ``normalize_fields`` parameter needed in ``run()``.

        Args:
            model: LLM model name to use across all stages.
            base_url: Optional custom API base URL.
            api_key: Optional API key.
            on_error: Error handling mode for extraction
                ("raise", "skip", or "collect").
            verbose: Show tqdm progress bars during processing.
        """
        from catchfly.discovery.single_pass import SinglePassDiscovery
        from catchfly.extraction.llm_direct import LLMDirectExtraction
        from catchfly.normalization.llm_canonical import LLMCanonicalization
        from catchfly.selection.llm import LLMFieldSelector

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
                on_error=on_error,
            ),
            normalization=LLMCanonicalization(
                model=model,
                base_url=base_url,
                api_key=api_key,
            ),
            field_selector=LLMFieldSelector(
                model=model,
                base_url=base_url,
                api_key=api_key,
            ),
            verbose=verbose,
        )

    async def arun(
        self,
        documents: list[Document] | list[str],
        *,
        domain_hint: str | None = None,
        normalize_fields: list[str] | Literal["all"] | None = None,
        max_cost_usd: float | None = None,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        checkpoint_dir: str | Path | None = None,
        on_schema_ready: Callable[[Schema], Schema | None] | None = None,
    ) -> PipelineResult:
        """Run the full pipeline asynchronously.

        Args:
            documents: Documents to process, or glob pattern strings
                (e.g. ``["data/*.txt"]``) that will be auto-loaded.
            domain_hint: Optional hint about the document domain.
            normalize_fields: Fields to normalize. Use ``"all"`` to auto-detect
                string fields from the schema. ``None`` skips normalization.
            max_cost_usd: Budget limit — pipeline halts if exceeded.
            schema: Pre-existing Pydantic model or JSON Schema dict —
                skips discovery if provided.
            checkpoint_dir: Directory for checkpoint state (enables resume).
        """
        # Resolve glob patterns to Document objects
        if documents and isinstance(documents[0], str):
            from catchfly.loaders import resolve_documents

            documents = resolve_documents(documents)  # type: ignore[arg-type]

        tracker = UsageTracker(max_cost_usd=max_cost_usd)
        result = PipelineResult()
        checkpoint = _Checkpoint(checkpoint_dir) if checkpoint_dir else None

        # Wire usage tracking into strategies
        self._wire_tracker(tracker)

        try:
            # --- Discovery ---
            extraction_model: type[BaseModel] | None = (
                schema if not isinstance(schema, dict) else None
            )

            if schema is None and self.discovery is not None:
                # Check checkpoint for saved schema
                saved_schema = checkpoint.load_schema() if checkpoint else None
                if saved_schema is not None:
                    logger.info("Pipeline: loaded schema from checkpoint")
                    result.schema = saved_schema
                    extraction_model = saved_schema.model
                else:
                    logger.info("Pipeline: starting discovery")
                    discovered_schema = await self.discovery.adiscover(
                        documents, domain_hint=domain_hint
                    )
                    result.schema = discovered_schema
                    extraction_model = discovered_schema.model
                    if checkpoint:
                        checkpoint.save_schema(discovered_schema)
                    logger.info(
                        "Pipeline: discovery complete — %d fields",
                        len(discovered_schema.json_schema.get("properties", {})),
                    )
            elif schema is not None:
                if isinstance(schema, dict):
                    from catchfly.schema.converters import json_schema_to_pydantic

                    try:
                        pydantic_model = json_schema_to_pydantic(schema, "UserSchema")
                    except SchemaError as e:
                        raise SchemaError(
                            f"Could not convert dict schema to Pydantic model. "
                            f"Ensure the dict has 'properties' with valid "
                            f"JSON Schema types: {e}"
                        ) from e
                    result.schema = Schema(
                        model=pydantic_model,
                        json_schema=schema,
                        lineage=["user-provided"],
                    )
                    extraction_model = pydantic_model
                else:
                    result.schema = Schema(
                        model=schema,
                        json_schema=schema.model_json_schema(),
                        lineage=["user-provided"],
                    )

            # --- Schema callback ---
            if on_schema_ready is not None and result.schema is not None:
                modified = on_schema_ready(result.schema)
                if modified is not None:
                    result.schema = modified
                    extraction_model = modified.model
                logger.info("Pipeline: on_schema_ready callback executed")

            # --- Extraction ---
            if extraction_model is not None and self.extraction is not None:
                # Check checkpoint for already-processed docs
                processed_ids = checkpoint.load_processed_ids() if checkpoint else set()
                remaining_docs = [d for d in documents if (d.id or "") not in processed_ids]

                if remaining_docs:
                    logger.info(
                        "Pipeline: extracting %d documents (%d already processed)",
                        len(remaining_docs),
                        len(processed_ids),
                    )
                    extraction_result = await self.extraction.aextract(
                        extraction_model, remaining_docs
                    )

                    # Save new records to checkpoint
                    if checkpoint:
                        for record in extraction_result.records:
                            checkpoint.append_record(record)
                        for doc in remaining_docs:
                            if doc.id:
                                checkpoint.mark_processed(doc.id)

                    result.records = extraction_result.records
                    result.errors = extraction_result.errors
                else:
                    logger.info("Pipeline: all documents already processed (checkpoint)")

                # Load previously checkpointed records
                if checkpoint and processed_ids:
                    prev_records = checkpoint.load_records()
                    result.records = prev_records + result.records

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

            # --- Resolve normalize_fields ---
            if normalize_fields == "all" and result.schema is not None:
                props = result.schema.json_schema.get("properties", {})
                normalize_fields = [
                    name
                    for name, spec in props.items()
                    if spec.get("type") == "string"
                    or (
                        spec.get("type") == "array"
                        and spec.get("items", {}).get("type") == "string"
                    )
                ]
                logger.info(
                    "Pipeline: normalize_fields='all' resolved to %s",
                    normalize_fields,
                )
            elif (
                normalize_fields is None
                and self.field_selector is not None
                and self.normalization is not None
                and result.schema is not None
                and result.records
            ):
                # --- Field Selection (auto) ---
                logger.info("Pipeline: running field selector to choose normalization targets")
                if self.field_selector is not None:
                    self._inject_callback(
                        self.field_selector, tracker.make_callback("field_selection")
                    )
                normalize_fields = await self.field_selector.aselect(result.schema, result.records)
                logger.info(
                    "Pipeline: field selector chose %d fields: %s",
                    len(normalize_fields),
                    normalize_fields,
                )

            # --- Normalization ---
            if normalize_fields and self.normalization is not None and result.records:
                logger.info(
                    "Pipeline: starting normalization for fields: %s",
                    normalize_fields,
                )
                # Pass field_metadata from schema to normalization strategies
                schema_metadata = result.schema.field_metadata if result.schema else {}
                result.normalizations = await self._normalize_fields(
                    result.records, normalize_fields, schema_metadata
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
        documents: list[Document] | list[str],
        *,
        domain_hint: str | None = None,
        normalize_fields: list[str] | Literal["all"] | None = None,
        max_cost_usd: float | None = None,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        checkpoint_dir: str | Path | None = None,
        on_schema_ready: Callable[[Schema], Schema | None] | None = None,
    ) -> PipelineResult:
        """Run the full pipeline synchronously."""
        return run_sync(
            self.arun(
                documents,
                domain_hint=domain_hint,
                normalize_fields=normalize_fields,
                max_cost_usd=max_cost_usd,
                schema=schema,
                checkpoint_dir=checkpoint_dir,
                on_schema_ready=on_schema_ready,
            )
        )

    def estimate_cost(self, documents: list[Document]) -> dict[str, float]:
        """Estimate pipeline cost before running.

        Returns approximate cost breakdown by stage (in USD).
        """
        from catchfly.extraction.chunking import estimate_chunks

        n_docs = len(documents)
        avg_tokens = sum(len(d.content) // 4 for d in documents) / max(n_docs, 1)

        # Use strategy-aware estimation if available
        if (
            self.extraction is not None
            and hasattr(self.extraction, "chunking_strategy")
            and self.extraction.chunking_strategy is not None
        ):
            n_chunks = self.extraction.chunking_strategy.estimate_chunks(documents)
        else:
            n_chunks = estimate_chunks(documents)

        # Multipliers assume gpt-5.4-mini pricing ($0.15/1M input tokens).
        # Discovery sends ~6x avg_tokens (samples + schema prompt overhead).
        # Extraction sends ~2x avg_tokens per chunk (schema + doc content).
        # Normalization is ~$0.001 per document (one short LLM call each).
        _price_per_1m = 0.15
        _discovery_token_multiplier = 6
        _extraction_token_multiplier = 2
        _norm_cost_per_doc = 0.001

        discovery_cost = avg_tokens * _discovery_token_multiplier / 1_000_000 * _price_per_1m
        extraction_cost = (
            n_chunks * avg_tokens * _extraction_token_multiplier / 1_000_000 * _price_per_1m
        )
        normalization_cost = n_docs * _norm_cost_per_doc

        return {
            "discovery": round(discovery_cost, 4),
            "extraction": round(extraction_cost, 4),
            "normalization": round(normalization_cost, 4),
            "total": round(discovery_cost + extraction_cost + normalization_cost, 4),
        }

    @staticmethod
    def _iter_progress(
        iterable: Any, *, verbose: bool, desc: str, total: int | None = None
    ) -> Any:
        """Wrap an iterable with tqdm if verbose=True and tqdm is available."""
        if not verbose:
            return iterable
        try:
            from tqdm.auto import tqdm  # type: ignore[import-untyped]

            return tqdm(iterable, desc=desc, total=total)
        except ImportError:
            logger.debug("tqdm not installed, progress bars disabled")
            return iterable

    def _wire_tracker(self, tracker: UsageTracker) -> None:
        """Wire usage tracking into all configured strategies."""
        stages: list[tuple[str, Any]] = [
            ("discovery", self.discovery),
            ("extraction", self.extraction),
            ("normalization", self.normalization),
        ]
        for stage_name, strategy in stages:
            if strategy is not None:
                self._inject_callback(strategy, tracker.make_callback(stage_name))

    @staticmethod
    def _inject_callback(strategy: Any, callback: Any) -> None:
        """Inject a usage callback into a strategy.

        Sets ``_usage_callback`` on the strategy so that ``_get_client()``
        picks it up when creating a default LLM client. Falls back silently
        if the strategy doesn't support usage tracking.
        """
        if hasattr(strategy, "_usage_callback"):
            strategy._usage_callback = callback

    async def _normalize_fields(
        self,
        records: list[Any],
        fields: list[str],
        schema_metadata: dict[str, Any] | None = None,
    ) -> dict[str, NormalizationResult]:
        """Normalize specified fields from extracted records.

        If schema_metadata is provided, it will be passed as field_metadata
        to the normalization strategy (e.g. LLMCanonicalization uses it
        for schema-aware prompting).
        """
        if self.normalization is None:
            msg = (
                "Pipeline._normalize_fields() called without a normalization strategy. "
                "This is an internal error — please report it."
            )
            raise RuntimeError(msg)
        normalizations: dict[str, NormalizationResult] = {}

        for field_name in self._iter_progress(
            fields, verbose=self.verbose, desc="Normalizing fields"
        ):
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
                logger.debug(
                    "Pipeline: no values found for field '%s', skipping",
                    field_name,
                )
                continue

            logger.info(
                "Pipeline: normalizing field '%s' (%d values, %d unique)",
                field_name,
                len(values),
                len(set(values)),
            )

            # Pass field-specific metadata if available
            normalize_kwargs: dict[str, Any] = {}
            if schema_metadata and field_name in schema_metadata:
                normalize_kwargs["field_metadata"] = schema_metadata[field_name]

            normalizations[field_name] = await self.normalization.anormalize(
                values, context_field=field_name, **normalize_kwargs
            )

        return normalizations
