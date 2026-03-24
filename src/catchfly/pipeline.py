"""Pipeline orchestrator — connects discovery, extraction, and normalization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
    from collections.abc import Callable

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
        normalization: NormalizationStrategy | dict[str, NormalizationStrategy] | None = None,
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
            verbose: Show tqdm progress bars during extraction/normalization.
        """
        self.discovery = discovery
        self.extraction = extraction
        self.verbose = verbose
        if isinstance(normalization, dict):
            from catchfly.normalization.composite import CompositeNormalization

            self.normalization = CompositeNormalization(field_strategies=normalization)
        else:
            self.normalization = normalization

    @classmethod
    def quick(
        cls,
        model: str = "gpt-5.4-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        on_error: Literal["raise", "skip", "collect"] = "raise",
        *,
        verbose: bool = False,
    ) -> Pipeline:
        """Create a pipeline with sensible defaults.

        Uses SinglePassDiscovery + LLMDirectExtraction + LLMCanonicalization.

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
            verbose=verbose,
        )

    async def arun(
        self,
        documents: list[Document] | list[str],
        *,
        domain_hint: str | None = None,
        normalize_fields: list[str] | Literal["all"] | None = None,
        max_cost_usd: float | None = None,
        schema: type[BaseModel] | None = None,
        checkpoint_dir: str | Path | None = None,
        on_schema_ready: Callable[[Schema], Schema | None] | None = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """Run the full pipeline asynchronously.

        Args:
            documents: Documents to process, or glob pattern strings
                (e.g. ``["data/*.txt"]``) that will be auto-loaded.
            domain_hint: Optional hint about the document domain.
            normalize_fields: Fields to normalize. Use ``"all"`` to auto-detect
                string fields from the schema. ``None`` skips normalization.
            max_cost_usd: Budget limit — pipeline halts if exceeded.
            schema: Pre-existing Pydantic model — skips discovery if provided.
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
        if self.discovery is not None:
            self.discovery._usage_callback = tracker.make_callback("discovery")  # type: ignore[attr-defined]
        if self.extraction is not None:
            self.extraction._usage_callback = tracker.make_callback("extraction")  # type: ignore[attr-defined]
        if self.normalization is not None:
            self.normalization._usage_callback = tracker.make_callback("normalization")  # type: ignore[attr-defined]

        try:
            # --- Discovery ---
            extraction_model: type[BaseModel] | None = schema

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

            # --- Resolve normalize_fields="all" ---
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
        schema: type[BaseModel] | None = None,
        checkpoint_dir: str | Path | None = None,
        on_schema_ready: Callable[[Schema], Schema | None] | None = None,
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
                checkpoint_dir=checkpoint_dir,
                on_schema_ready=on_schema_ready,
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

        # Use strategy-aware estimation if available
        if (
            self.extraction is not None
            and hasattr(self.extraction, "chunking_strategy")
            and self.extraction.chunking_strategy is not None
        ):
            n_chunks = self.extraction.chunking_strategy.estimate_chunks(documents)
        else:
            n_chunks = estimate_chunks(documents)

        discovery_cost = avg_tokens * 6 / 1_000_000 * 0.15
        extraction_cost = n_chunks * avg_tokens * 2 / 1_000_000 * 0.15
        normalization_cost = n_docs * 0.001

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
        assert self.normalization is not None
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


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------


class _Checkpoint:
    """Manages pipeline checkpoint state on disk.

    Files:
    - schema.json — discovered schema
    - records.jsonl — extracted records (one per line, append-safe)
    - state.json — set of processed document IDs
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def _schema_path(self) -> Path:
        return self._dir / "schema.json"

    @property
    def _records_path(self) -> Path:
        return self._dir / "records.jsonl"

    @property
    def _state_path(self) -> Path:
        return self._dir / "state.json"

    def save_schema(self, schema: Schema) -> None:
        """Persist discovered schema."""
        data = {
            "json_schema": schema.json_schema,
            "field_metadata": schema.field_metadata,
            "lineage": schema.lineage,
        }
        self._schema_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Checkpoint: saved schema to %s", self._schema_path)

    def load_schema(self) -> Schema | None:
        """Load schema from checkpoint, or None if not found."""
        if not self._schema_path.exists():
            return None

        try:
            data = json.loads(self._schema_path.read_text(encoding="utf-8"))
            import contextlib

            from catchfly.schema.converters import json_schema_to_pydantic

            json_schema = data["json_schema"]
            model = None
            with contextlib.suppress(Exception):
                model = json_schema_to_pydantic(json_schema, "CheckpointSchema")

            return Schema(
                model=model,
                json_schema=json_schema,
                field_metadata=data.get("field_metadata", {}),
                lineage=data.get("lineage", []),
            )
        except Exception:
            logger.warning("Checkpoint: failed to load schema", exc_info=True)
            return None

    def append_record(self, record: Any) -> None:
        """Append a single record to the JSONL file."""
        if hasattr(record, "model_dump"):
            data = record.model_dump()
        elif isinstance(record, dict):
            data = record
        else:
            data = {"_raw": str(record)}

        with open(self._records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def load_records(self) -> list[dict[str, Any]]:
        """Load all records from checkpoint JSONL."""
        if not self._records_path.exists():
            return []

        records: list[dict[str, Any]] = []
        for line in self._records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def mark_processed(self, doc_id: str) -> None:
        """Mark a document ID as processed."""
        ids = self.load_processed_ids()
        ids.add(doc_id)
        self._state_path.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")

    def load_processed_ids(self) -> set[str]:
        """Load the set of already-processed document IDs."""
        if not self._state_path.exists():
            return set()

        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return set(data) if isinstance(data, list) else set()
        except Exception:
            return set()
