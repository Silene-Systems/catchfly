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
from catchfly._types import Document, ExtractionResult, RecordProvenance
from catchfly.exceptions import ExtractionError
from catchfly.extraction.chunking import chunk_document
from catchfly.providers.llm import LLMResponse, OpenAICompatibleClient
from catchfly.schema.converters import resolve_refs

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
- Output: [{"field": "value", ...}, ...]\
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
    deduplicate: bool = True
    """When ``True`` and ``multi_record=True``, deduplicate records across
    chunks by exact JSON match."""
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

        # In multi-record mode, wrap schema in array for the LLM prompt
        if self.multi_record:
            array_schema = {"type": "array", "items": json_schema}
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

        async def process_chunk(chunk: Document) -> None:
            async with semaphore:
                try:
                    if self.multi_record:
                        chunk_records = await self._extract_multi(
                            client, schema, json_schema, array_schema, chunk
                        )
                        records.extend(chunk_records)
                        for _ in chunk_records:
                            provenances.append(
                                RecordProvenance(
                                    source_document=str(
                                        chunk.source or chunk.id or "unknown"
                                    ),
                                    chunk_index=chunk.metadata.get("chunk_index"),
                                    char_start=chunk.metadata.get("char_start"),
                                    char_end=chunk.metadata.get("char_end"),
                                    confidence=1.0,
                                )
                            )
                    else:
                        record, provenance = await self._extract_single(
                            client, schema, json_schema, chunk
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

        tasks = [process_chunk(chunk) for chunk in all_chunks]
        await asyncio.gather(*tasks)

        # Deduplicate multi-record results across chunks
        if self.multi_record and self.deduplicate and len(records) > 0:
            seen: set[str] = set()
            unique_records: list[Any] = []
            unique_provenances: list[RecordProvenance] = []
            for rec, prov in zip(records, provenances):
                key = json.dumps(rec.model_dump(), sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    unique_records.append(rec)
                    unique_provenances.append(prov)
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

    async def _extract_single(
        self,
        client: OpenAICompatibleClient,
        schema: type[BaseModel],
        json_schema: dict[str, Any],
        doc: Document,
    ) -> tuple[Any, RecordProvenance]:
        """Extract a single record from a document/chunk with retries."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_extraction_prompt(doc, json_schema)},
        ]

        last_error: Exception | None = None
        last_output = ""

        for attempt in range(self.max_retries + 1):
            response: LLMResponse = await client.astructured_complete(
                messages,
                output_schema=json_schema,
                schema_name="extraction",
            )

            try:
                raw_data = self._parse_json(response.content)
                raw_data = self._coerce_nulls(raw_data, json_schema)
                record = schema.model_validate(raw_data)

                provenance = RecordProvenance(
                    source_document=str(doc.source or doc.id or "unknown"),
                    chunk_index=doc.metadata.get("chunk_index"),
                    char_start=doc.metadata.get("char_start"),
                    char_end=doc.metadata.get("char_end"),
                    confidence=round(max(0.1, 1.0 - attempt * 0.3), 2),
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
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _build_retry_prompt(doc, json_schema, last_output, str(e)),
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
        array_schema: dict[str, Any] | None,
        doc: Document,
    ) -> list[Any]:
        """Extract multiple records from a document/chunk with retries."""
        prompt_schema = array_schema or json_schema
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _MULTI_RECORD_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_extraction_prompt(
                    doc, prompt_schema, multi_record=True
                ),
            },
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
                raw_list = self._parse_json_array(response.content)
                validated: list[Any] = []
                for item in raw_list:
                    item = self._coerce_nulls(item, json_schema)
                    validated.append(schema.model_validate(item))
                return validated

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
                        {"role": "system", "content": _MULTI_RECORD_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _build_retry_prompt(
                                doc,
                                prompt_schema,
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
