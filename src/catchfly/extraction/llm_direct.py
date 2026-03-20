"""LLMDirectExtraction — extract structured data using LLM with structured output."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from catchfly._compat import run_sync
from catchfly._types import Document, ExtractionResult, RecordProvenance
from catchfly.exceptions import ExtractionError
from catchfly.extraction.chunking import chunk_document
from catchfly.providers.llm import LLMResponse, OpenAICompatibleClient

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


def _build_extraction_prompt(
    doc: Document,
    schema: dict[str, Any],
) -> str:
    schema_str = json.dumps(schema, indent=2)
    return (
        f"JSON Schema to extract:\n```json\n{schema_str}\n```\n\n"
        f"Document to extract from:\n---\n{doc.content}\n---\n\n"
        "Extract the structured data as JSON."
    )


def _build_retry_prompt(
    doc: Document,
    schema: dict[str, Any],
    previous_output: str,
    error_msg: str,
) -> str:
    schema_str = json.dumps(schema, indent=2)
    return (
        f"JSON Schema:\n```json\n{schema_str}\n```\n\n"
        f"Document:\n---\n{doc.content}\n---\n\n"
        f"Your previous extraction had an error:\n{error_msg}\n\n"
        f"Previous output:\n{previous_output}\n\n"
        "Please fix the extraction and output valid JSON matching the schema."
    )


class LLMDirectExtraction(BaseModel):
    """Extract structured data from documents using LLM with structured output.

    For each document (or chunk), sends the schema and text to the LLM,
    parses the response into Pydantic model instances, and retries on
    validation errors with feedback.
    """

    model: str = "gpt-5.4-mini"
    chunk_size: int = 4000
    chunk_overlap: int = 200
    max_retries: int = 3
    batch_size: int = 10
    on_error: Literal["raise", "skip", "collect"] = "raise"
    base_url: str | None = None
    api_key: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    async def aextract(
        self,
        schema: type[BaseModel],
        documents: list[Document],
        **kwargs: Any,
    ) -> ExtractionResult[Any]:
        """Extract structured data asynchronously."""
        if not documents:
            return ExtractionResult(records=[])

        json_schema = schema.model_json_schema()
        client = OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Chunk documents
        all_chunks: list[Document] = []
        for doc in documents:
            chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)

        logger.info(
            "LLMDirectExtraction: processing %d documents (%d chunks), model=%s",
            len(documents),
            len(all_chunks),
            self.model,
        )

        # Process with concurrency control
        semaphore = asyncio.Semaphore(self.batch_size)
        records: list[Any] = []
        errors: list[tuple[Document, Exception]] = []
        provenances: list[RecordProvenance] = []

        async def process_chunk(chunk: Document) -> None:
            async with semaphore:
                try:
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
        schema: type[BaseModel],
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
                record = schema.model_validate(raw_data)

                provenance = RecordProvenance(
                    source_document=str(doc.source or doc.id or "unknown"),
                    chunk_index=doc.metadata.get("chunk_index"),
                    char_start=doc.metadata.get("char_start"),
                    char_end=doc.metadata.get("char_end"),
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

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)
        if not isinstance(data, dict):
            msg = f"Expected JSON object, got {type(data).__name__}"
            raise json.JSONDecodeError(msg, text, 0)
        return data
