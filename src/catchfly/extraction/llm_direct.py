"""LLMDirectExtraction — extract structured data using LLM with structured output."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, PrivateAttr, ValidationError

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._parsing import strip_markdown_fences
from catchfly._types import (
    CostEstimate,
    Document,
    ExtractionResult,
    RecordProvenance,
    SourceSpan,
)
from catchfly.exceptions import ExtractionError
from catchfly.extraction._provenance import (
    build_field_spans,
    merge_field_spans,
    unwrap_record,
    wrap_schema_with_provenance,
)
from catchfly.extraction._tokenize import count_messages_tokens
from catchfly.extraction.chunking import chunk_document
from catchfly.providers.llm import LLMResponse, OpenAICompatibleClient
from catchfly.schema.converters import resolve_refs
from catchfly.telemetry.tracker import estimate_llm_cost, is_model_priced

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a structured data extraction assistant. Given a document and a JSON Schema, \
extract the structured data that matches the schema.

Rules:
- Output ONLY valid JSON matching the provided schema.
- Extract information directly stated in the document.
- If a required field is not found, use null.
- For array fields, include all instances found.
- Do not invent or hallucinate data not present in the document.\
"""

_MULTI_RECORD_SYSTEM_PROMPT = """\
You are a structured data extraction assistant. Given a document and a JSON Schema, \
extract ALL matching records from the document as a JSON array.

Rules:
- Each element should be a separate entity (e.g., patient, sample, experiment).
- If only one entity exists, return an array with one element.
- Output ONLY a valid JSON array matching the provided schema.
- Extract information directly stated in the document.
- If a required field is not found, use null.
- For array fields within each record, include all instances found.
- Do not invent or hallucinate data not present in the document.

Comparative tables are a primary source of records, not metadata:
- Each row of a comparative table that describes a distinct entity counts as a \
separate record. Do not skip table rows.
- Include rows from tables even when the entity is not also described in prose \
elsewhere in the document.
- When the same entity (matched by identifier such as "Patient 1") appears in \
both prose and a table row, merge the information from both sources into a \
single record rather than emitting duplicates.

Output: [{"field": "value", ...}, ...]\
"""

_PROVENANCE_INSTRUCTIONS = """\

Provenance requirements:
- For every extracted field, populate BOTH "value" and "source_quotes".
- "source_quotes" must contain verbatim excerpt(s) copied character-for-character \
from the source document — do not paraphrase, summarize, or reformat.
- Each excerpt should include roughly 10-30 words of surrounding context so that \
it can be uniquely located within the document.
- For list-valued fields, return one excerpt per list item in the same order.
- For fields synthesized from multiple non-adjacent passages, return every \
supporting excerpt.
- If a value was inferred without any directly supporting passage, return an \
empty "source_quotes" array (not null) — never fabricate a quote.\
"""


def _build_extraction_prompt(
    doc: Document,
    schema: dict[str, Any],
    *,
    multi_record: bool = False,
) -> str:
    schema_str = json.dumps(schema, indent=2)
    if multi_record:
        instruction = "Extract ALL matching records as a JSON array."
    else:
        instruction = "Extract the structured data as JSON."
    return (
        f"JSON Schema to extract:\n```json\n{schema_str}\n```\n\n"
        f"Document to extract from:\n---\n{doc.content}\n---\n\n"
        f"{instruction}"
    )


def _build_retry_prompt(
    doc: Document,
    schema: dict[str, Any],
    previous_output: str,
    error_msg: str,
    *,
    multi_record: bool = False,
) -> str:
    schema_str = json.dumps(schema, indent=2)
    if multi_record:
        instruction = (
            "Please fix the extraction and output a valid JSON array matching the schema."
        )
    else:
        instruction = "Please fix the extraction and output valid JSON matching the schema."
    return (
        f"JSON Schema:\n```json\n{schema_str}\n```\n\n"
        f"Document:\n---\n{doc.content}\n---\n\n"
        f"Your previous extraction had an error:\n{error_msg}\n\n"
        f"Previous output:\n{previous_output}\n\n"
        f"{instruction}"
    )


_CHUNK_SIZE_SENTINEL = -1


def _count_leaf_fields(json_schema: dict[str, Any]) -> int:
    """Count scalar + list leaves in a JSON Schema for cost estimation.

    Walks ``properties`` recursively. Nested objects contribute the
    count of their leaves; arrays of scalars count as a single leaf;
    arrays of objects descend into the item schema.
    """

    def _walk(node: dict[str, Any]) -> int:
        if not isinstance(node, dict):
            return 0
        node_type = node.get("type")
        if node_type == "object":
            props = node.get("properties") or {}
            if not props:
                return 1  # free-form object counts as one leaf
            return sum(_walk(child) for child in props.values())
        if node_type == "array":
            items = node.get("items") or {}
            if isinstance(items, dict) and items.get("type") == "object":
                return _walk(items)
            return 1
        return 1

    props = json_schema.get("properties") or {}
    if not props:
        return 1
    return sum(_walk(child) for child in props.values())


class LLMDirectExtraction(BaseModel):
    """Extract structured data from documents using LLM with structured output.

    For each document (or chunk), sends the schema and text to the LLM,
    parses the response into Pydantic model instances, and retries on
    validation errors with feedback.

    When ``multi_record=True``, the LLM is instructed to extract **all**
    matching entities from each chunk as a JSON array.  Combined with
    ``chunk_size=0`` (no chunking), this enables correct multi-entity
    extraction from a single document (e.g. 6 patients from one paper).
    """

    model: str = DEFAULT_MODEL
    chunk_size: int = _CHUNK_SIZE_SENTINEL
    """4000 chars (~1000 tokens) by default.  Set to ``0`` to disable chunking.
    When ``multi_record=True`` and chunk_size is not explicitly set,
    defaults to ``0`` (no chunking)."""
    chunk_overlap: int = 200
    """200 chars overlap prevents splitting entities at chunk boundaries."""
    chunking_strategy: Any | None = None
    max_retries: int = 3
    """3 retries with validation feedback is sufficient for most extraction errors."""
    batch_size: int = 10
    """10 concurrent requests balances throughput vs rate-limit pressure."""
    on_error: Literal["raise", "skip", "collect"] = "raise"
    multi_record: bool = False
    """When ``True``, extract all matching entities from each chunk as a
    JSON array instead of a single record."""
    record_hint: str | None = None
    """Free-text definition of what constitutes a single record.

    Only meaningful when ``multi_record=True``. The LLM normally guesses
    record boundaries from the schema, which is ambiguous for clinical
    corpora ("patient" vs "visit" vs "dose") or comparative studies. Set
    this to an explicit sentence such as ``"Each patient described in
    case reports or comparative tables is a separate record, even if
    only mentioned in a table row."`` It is appended to the multi-record
    system prompt."""
    deduplicate: bool = True
    """When ``True`` and ``multi_record=True``, deduplicate records across
    chunks by exact JSON match."""
    include_provenance: bool = False
    """When ``True``, request per-field supporting quotes from the LLM and
    attach them as :class:`SourceSpan` objects on
    :attr:`RecordProvenance.field_spans`. This transforms the extraction
    schema so every top-level field is wrapped in
    ``{"value": ..., "source_quotes": [...]}``, roughly 2-3x'ing output
    tokens. Disabled by default to keep the zero-config path cheap. See
    :mod:`catchfly.extraction._provenance` for the matching algorithm."""
    progress_callback: Any = None
    """Optional ``Callable[[int, int], None]`` invoked after each chunk
    completes, with ``(completed, total)`` counts. Useful for driving
    tqdm or a UI progress bar on long-running extractions. The callback
    is invoked from whichever asyncio task finishes first — user code
    must be re-entrant if it mutates shared state. Exceptions raised
    inside the callback are logged and swallowed so that telemetry
    failures never derail extraction."""
    base_url: str | None = None
    api_key: str | None = None
    client: Any | None = None
    """Pre-configured LLM client. If ``None``, a default client is created
    from ``model``, ``base_url``, and ``api_key``."""

    _usage_callback: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _effective_chunk_size(self) -> int:
        """Resolve the actual chunk_size, applying multi_record default."""
        if self.chunk_size != _CHUNK_SIZE_SENTINEL:
            return self.chunk_size
        return 0 if self.multi_record else 4000

    def _get_client(self) -> Any:
        """Return the injected client or create a default one."""
        if self.client is not None:
            return self.client
        return OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            usage_callback=self._usage_callback,
        )

    async def aextract(
        self,
        schema: type[BaseModel] | dict[str, Any],
        documents: list[Document],
        **kwargs: Any,
    ) -> ExtractionResult[Any]:
        """Extract structured data asynchronously."""
        if not documents:
            return ExtractionResult(records=[])

        if isinstance(schema, dict):
            from catchfly.schema.converters import json_schema_to_pydantic

            schema = json_schema_to_pydantic(schema, name=schema.get("title", "Extraction"))

        json_schema = resolve_refs(schema.model_json_schema())

        # Schema seen by the LLM: optionally wrapped so every top-level
        # field becomes {value, source_quotes}. The validation schema
        # (passed to Pydantic) is always the clean, unwrapped form.
        if self.include_provenance:
            prompt_json_schema = wrap_schema_with_provenance(json_schema)
        else:
            prompt_json_schema = json_schema

        # In multi-record mode, wrap schema in array for the LLM prompt
        if self.multi_record:
            array_schema = {"type": "array", "items": prompt_json_schema}
        else:
            array_schema = None

        client = self._get_client()
        effective_chunk_size = self._effective_chunk_size

        # Chunk documents (chunk_size=0 means no chunking)
        all_chunks: list[Document] = []
        if self.chunking_strategy is not None:
            for doc in documents:
                all_chunks.extend(self.chunking_strategy.chunk(doc))
        elif effective_chunk_size <= 0:
            all_chunks = list(documents)
        else:
            for doc in documents:
                all_chunks.extend(
                    chunk_document(doc, effective_chunk_size, self.chunk_overlap)
                )

        logger.info(
            "LLMDirectExtraction: processing %d documents (%d chunks), "
            "model=%s, multi_record=%s",
            len(documents),
            len(all_chunks),
            self.model,
            self.multi_record,
        )

        # Process with concurrency control
        semaphore = asyncio.Semaphore(self.batch_size)
        records: list[Any] = []
        errors: list[tuple[Document, Exception]] = []
        provenances: list[RecordProvenance] = []
        total_chunks = len(all_chunks)
        completed_chunks = 0

        def _notify_progress() -> None:
            """Invoke progress_callback, swallowing any errors."""
            if self.progress_callback is None:
                return
            try:
                self.progress_callback(completed_chunks, total_chunks)
            except Exception as cb_err:  # noqa: BLE001 — user code
                logger.warning(
                    "LLMDirectExtraction.progress_callback raised: %s",
                    cb_err,
                )

        async def process_chunk(chunk: Document) -> None:
            nonlocal completed_chunks
            async with semaphore:
                try:
                    if self.multi_record:
                        chunk_records, chunk_field_spans = await self._extract_multi(
                            client, schema, json_schema, prompt_json_schema, array_schema, chunk
                        )
                        records.extend(chunk_records)
                        for rec_spans in chunk_field_spans:
                            provenances.append(
                                RecordProvenance(
                                    source_document=str(
                                        chunk.source or chunk.id or "unknown"
                                    ),
                                    chunk_index=chunk.metadata.get("chunk_index"),
                                    char_start=chunk.metadata.get("char_start"),
                                    char_end=chunk.metadata.get("char_end"),
                                    confidence=1.0,
                                    field_spans=rec_spans,
                                )
                            )
                    else:
                        record, provenance = await self._extract_single(
                            client, schema, json_schema, prompt_json_schema, chunk
                        )
                        records.append(record)
                        provenances.append(provenance)
                except Exception as e:
                    if self.on_error == "raise":
                        raise
                    elif self.on_error == "collect":
                        errors.append((chunk, e))
                        logger.error(
                            "Extraction failed for document '%s': %s",
                            chunk.id,
                            e,
                        )
                    else:  # skip
                        logger.warning(
                            "Skipping document '%s' due to error: %s",
                            chunk.id,
                            e,
                        )
            # Progress callback fires after the semaphore is released so
            # slow callbacks don't monopolise concurrency slots. Under
            # asyncio the read-modify-write on completed_chunks is
            # atomic (no await between the two lines).
            completed_chunks += 1
            _notify_progress()

        tasks = [process_chunk(chunk) for chunk in all_chunks]
        await asyncio.gather(*tasks)

        # Deduplicate multi-record results across chunks
        if self.multi_record and self.deduplicate and len(records) > 0:
            seen: dict[str, int] = {}
            unique_records: list[Any] = []
            unique_provenances: list[RecordProvenance] = []
            for rec, prov in zip(records, provenances):
                key = json.dumps(rec.model_dump(), sort_keys=True, default=str)
                existing_idx = seen.get(key)
                if existing_idx is None:
                    seen[key] = len(unique_records)
                    unique_records.append(rec)
                    unique_provenances.append(prov)
                elif self.include_provenance:
                    # Preserve supporting quotes from every chunk that found
                    # this record — same value, multiple evidence passages.
                    existing = unique_provenances[existing_idx]
                    existing.field_spans = merge_field_spans(
                        existing.field_spans, prov.field_spans
                    )
            records = unique_records
            provenances = unique_provenances

        logger.info(
            "LLMDirectExtraction: extracted %d records, %d errors",
            len(records),
            len(errors),
        )

        return ExtractionResult(
            records=records,
            errors=errors,
            provenance=provenances,
        )

    def extract(
        self,
        schema: type[BaseModel] | dict[str, Any],
        documents: list[Document],
        **kwargs: Any,
    ) -> ExtractionResult[Any]:
        """Extract structured data synchronously."""
        return run_sync(self.aextract(schema, documents, **kwargs))

    def estimate_cost(
        self,
        schema: type[BaseModel] | dict[str, Any],
        documents: list[Document],
        *,
        records_per_document: int = 1,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> CostEstimate:
        """Estimate the cost of extracting *documents* without calling the LLM.

        Tokenizes the exact prompts that :meth:`aextract` would send —
        same chunking, same schema wrapping for provenance, same system /
        user prompt composition — then looks up per-1M-token pricing from
        :mod:`catchfly.telemetry.tracker`. Output tokens are estimated
        conservatively: one JSON record per document (or
        ``records_per_document`` when ``multi_record=True``), scaled up
        by ~3x when ``include_provenance=True`` to account for the
        per-field quote payload.

        Args:
            schema: Pydantic model class or JSON Schema dict — same as
                :meth:`aextract`.
            documents: Documents that would be extracted.
            records_per_document: Expected number of records the LLM
                will emit per document. Only used when
                ``multi_record=True``. Defaults to ``1`` — set higher
                for corpora like comparative-table studies where each
                document yields many records.
            pricing: Optional override mapping ``{model: (input_rate,
                output_rate)}`` in USD per 1M tokens. Merged on top of
                the default table.

        Returns:
            :class:`CostEstimate` with input/output token counts,
            estimated USD cost, and notes describing any caveats
            (unknown model, provenance multiplier, etc.).
        """
        if not documents:
            return CostEstimate(
                model=self.model,
                num_documents=0,
                num_chunks=0,
                input_tokens=0,
                estimated_output_tokens=0,
                cost_usd=0.0,
                tokenizer="heuristic:chars-over-4",
                notes=["No documents provided."],
            )

        if isinstance(schema, dict):
            from catchfly.schema.converters import json_schema_to_pydantic

            schema = json_schema_to_pydantic(
                schema, name=schema.get("title", "Extraction")
            )

        json_schema = resolve_refs(schema.model_json_schema())
        prompt_json_schema = (
            wrap_schema_with_provenance(json_schema)
            if self.include_provenance
            else json_schema
        )
        if self.multi_record:
            outer_prompt_schema: dict[str, Any] = {
                "type": "array",
                "items": prompt_json_schema,
            }
        else:
            outer_prompt_schema = prompt_json_schema

        # Chunk documents using the same path as aextract().
        effective_chunk_size = self._effective_chunk_size
        chunks: list[Document] = []
        if self.chunking_strategy is not None:
            for doc in documents:
                chunks.extend(self.chunking_strategy.chunk(doc))
        elif effective_chunk_size <= 0:
            chunks = list(documents)
        else:
            for doc in documents:
                chunks.extend(
                    chunk_document(doc, effective_chunk_size, self.chunk_overlap)
                )

        # Compose the system prompt exactly like _extract_single /
        # _extract_multi would — this keeps the estimate honest about
        # the record_hint + provenance flag contribution.
        system_prompt = (
            _MULTI_RECORD_SYSTEM_PROMPT if self.multi_record else _SYSTEM_PROMPT
        )
        if self.multi_record and self.record_hint:
            system_prompt = (
                f"{system_prompt}\n\nRecord definition: {self.record_hint}"
            )
        if self.include_provenance:
            system_prompt = system_prompt + _PROVENANCE_INSTRUCTIONS

        total_input_tokens = 0
        tokenizer_label = "heuristic:chars-over-4"
        for chunk in chunks:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_extraction_prompt(
                        chunk, outer_prompt_schema, multi_record=self.multi_record
                    ),
                },
            ]
            tokens, tokenizer_label = count_messages_tokens(messages, self.model)
            total_input_tokens += tokens

        # Output-token heuristic: rough JSON size per record × record
        # multiplier × provenance multiplier. Tokens-per-record uses a
        # per-leaf baseline so the estimate scales with schema width.
        num_leaf_fields = _count_leaf_fields(json_schema)
        tokens_per_record = max(30, num_leaf_fields * 30)
        records_per_chunk = records_per_document if self.multi_record else 1
        output_tokens = tokens_per_record * records_per_chunk * len(chunks)
        if self.include_provenance:
            # Each field gains a ~10-30 word quote — roughly triples
            # output size in practice.
            output_tokens *= 3

        cost = estimate_llm_cost(
            self.model,
            total_input_tokens,
            output_tokens,
            pricing=pricing,
        )

        notes: list[str] = []
        if not is_model_priced(self.model) and pricing is None:
            notes.append(
                f"Model '{self.model}' is not in the default pricing table; "
                "cost_usd returned as 0.0. Pass a `pricing=` override to "
                "get a meaningful estimate."
            )
        if self.include_provenance:
            notes.append(
                "include_provenance=True — output token estimate inflated 3x "
                "to account for per-field source quotes."
            )
        if self.multi_record:
            notes.append(
                f"multi_record=True — assumed {records_per_document} record(s) "
                "per document. Pass records_per_document= to refine."
            )

        return CostEstimate(
            model=self.model,
            num_documents=len(documents),
            num_chunks=len(chunks),
            input_tokens=total_input_tokens,
            estimated_output_tokens=output_tokens,
            cost_usd=cost,
            tokenizer=tokenizer_label,
            notes=notes,
        )

    async def _extract_single(
        self,
        client: OpenAICompatibleClient,
        schema: type[BaseModel],
        json_schema: dict[str, Any],
        prompt_schema: dict[str, Any],
        doc: Document,
    ) -> tuple[Any, RecordProvenance]:
        """Extract a single record from a document/chunk with retries."""
        system_prompt = (
            _SYSTEM_PROMPT + _PROVENANCE_INSTRUCTIONS
            if self.include_provenance
            else _SYSTEM_PROMPT
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_extraction_prompt(doc, prompt_schema)},
        ]

        last_error: Exception | None = None
        last_output = ""

        for attempt in range(self.max_retries + 1):
            response: LLMResponse = await client.astructured_complete(
                messages,
                output_schema=prompt_schema,
                schema_name="extraction",
            )

            try:
                raw_data = self._parse_json(response.content)
                if self.include_provenance:
                    clean_data, quotes_by_field = unwrap_record(raw_data)
                else:
                    clean_data, quotes_by_field = raw_data, {}
                clean_data = self._coerce_nulls(clean_data, json_schema)
                record = schema.model_validate(clean_data)

                field_spans: dict[str, list[SourceSpan]] | None = None
                if self.include_provenance:
                    field_spans = build_field_spans(quotes_by_field, [doc])

                provenance = RecordProvenance(
                    source_document=str(doc.source or doc.id or "unknown"),
                    chunk_index=doc.metadata.get("chunk_index"),
                    char_start=doc.metadata.get("char_start"),
                    char_end=doc.metadata.get("char_end"),
                    confidence=round(max(0.1, 1.0 - attempt * 0.3), 2),
                    field_spans=field_spans,
                )

                return record, provenance

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                last_output = response.content
                if attempt < self.max_retries:
                    logger.warning(
                        "Extraction validation failed (attempt %d/%d) for '%s': %s",
                        attempt + 1,
                        self.max_retries + 1,
                        doc.id,
                        e,
                    )
                    # Build retry prompt with error feedback
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": _build_retry_prompt(
                                doc, prompt_schema, last_output, str(e)
                            ),
                        },
                    ]

        raise ExtractionError(
            f"Extraction failed after {self.max_retries + 1} attempts "
            f"for document '{doc.id}': {last_error}"
        ) from last_error

    async def _extract_multi(
        self,
        client: OpenAICompatibleClient,
        schema: type[BaseModel],
        json_schema: dict[str, Any],
        prompt_item_schema: dict[str, Any],
        array_schema: dict[str, Any] | None,
        doc: Document,
    ) -> tuple[list[Any], list[dict[str, list[SourceSpan]] | None]]:
        """Extract multiple records from a document/chunk with retries.

        Returns ``(records, field_spans_per_record)`` where the second list
        is aligned with the first and contains ``None`` entries when
        provenance tracking is disabled.
        """
        outer_schema = array_schema or prompt_item_schema
        system_prompt = _MULTI_RECORD_SYSTEM_PROMPT
        if self.record_hint:
            system_prompt = (
                f"{system_prompt}\n\nRecord definition: {self.record_hint}"
            )
        if self.include_provenance:
            system_prompt = system_prompt + _PROVENANCE_INSTRUCTIONS
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_extraction_prompt(
                    doc, outer_schema, multi_record=True
                ),
            },
        ]

        last_error: Exception | None = None
        last_output = ""

        for attempt in range(self.max_retries + 1):
            response: LLMResponse = await client.astructured_complete(
                messages,
                output_schema=outer_schema,
                schema_name="extraction",
            )

            try:
                raw_list = self._parse_json_array(response.content)
                validated: list[Any] = []
                spans_per_record: list[dict[str, list[SourceSpan]] | None] = []
                for item in raw_list:
                    if self.include_provenance:
                        clean_item, quotes_by_field = unwrap_record(item)
                    else:
                        clean_item, quotes_by_field = item, {}
                    clean_item = self._coerce_nulls(clean_item, json_schema)
                    validated.append(schema.model_validate(clean_item))
                    if self.include_provenance:
                        spans_per_record.append(
                            build_field_spans(quotes_by_field, [doc])
                        )
                    else:
                        spans_per_record.append(None)
                return validated, spans_per_record

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                last_output = response.content
                if attempt < self.max_retries:
                    logger.warning(
                        "Multi-record extraction failed (attempt %d/%d) for '%s': %s",
                        attempt + 1,
                        self.max_retries + 1,
                        doc.id,
                        e,
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": _build_retry_prompt(
                                doc,
                                outer_schema,
                                last_output,
                                str(e),
                                multi_record=True,
                            ),
                        },
                    ]

        raise ExtractionError(
            f"Multi-record extraction failed after {self.max_retries + 1} attempts "
            f"for document '{doc.id}': {last_error}"
        ) from last_error

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        text = strip_markdown_fences(content)

        data = json.loads(text)
        if not isinstance(data, dict):
            msg = f"Expected JSON object, got {type(data).__name__}"
            raise json.JSONDecodeError(msg, text, 0)
        return data

    @staticmethod
    def _parse_json_array(content: str) -> list[dict[str, Any]]:
        """Parse a JSON array from LLM response, handling markdown fences."""
        text = strip_markdown_fences(content)

        data = json.loads(text)
        if isinstance(data, dict):
            # LLM returned a single object — wrap it in a list
            return [data]
        if not isinstance(data, list):
            msg = f"Expected JSON array, got {type(data).__name__}"
            raise json.JSONDecodeError(msg, text, 0)
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                msg = f"Expected JSON object at index {i}, got {type(item).__name__}"
                raise json.JSONDecodeError(msg, text, 0)
        return data

    @staticmethod
    def _coerce_nulls(data: dict[str, Any], json_schema: dict[str, Any]) -> dict[str, Any]:
        """Replace null values with schema-appropriate defaults, recursively.

        LLMs commonly return ``null`` for empty arrays/objects instead of
        ``[]``/``{}``.  This coercion prevents unnecessary validation retries.
        Recurses into nested objects and array items.
        """
        properties = json_schema.get("properties", {})
        for key, value in data.items():
            prop = properties.get(key, {})
            prop_type = prop.get("type")

            if value is None:
                if prop_type == "array":
                    data[key] = []
                elif prop_type == "object":
                    data[key] = {}
                continue

            # Recurse into nested objects
            if isinstance(value, dict) and prop_type == "object" and "properties" in prop:
                LLMDirectExtraction._coerce_nulls(value, prop)

            # Recurse into array items
            elif isinstance(value, list) and prop_type == "array":
                items_schema = prop.get("items", {})
                if items_schema.get("type") == "object" and "properties" in items_schema:
                    for item in value:
                        if isinstance(item, dict):
                            LLMDirectExtraction._coerce_nulls(item, items_schema)

        return data
