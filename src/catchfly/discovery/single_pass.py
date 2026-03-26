"""SinglePassDiscovery — discover schema from sample documents in one LLM call."""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from pydantic import BaseModel, PrivateAttr

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._parsing import strip_markdown_fences
from catchfly._types import Document, Schema
from catchfly.exceptions import DiscoveryError, SchemaError
from catchfly.providers.llm import LLMResponse, OpenAICompatibleClient
from catchfly.schema.converters import json_schema_to_pydantic

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a schema discovery assistant. Given sample documents, propose a JSON Schema \
that captures the key structured fields present across the documents.

Rules:
- Output ONLY valid JSON Schema (no markdown, no explanation).
- The schema must have "type": "object" and "properties".
- Use descriptive field names in snake_case.
- Include "required" array for fields present in most documents.
- Use appropriate types: string, integer, number, boolean, array.
- For array fields, include "items" with the element type.
- Keep the schema flat where possible; nest only when semantically necessary.
- Prefer fewer, high-value fields over many granular ones.\
"""


def _build_user_prompt(
    documents: list[Document],
    domain_hint: str | None = None,
    *,
    max_doc_chars: int = 3000,
    max_fields: int | None = None,
    suggested_fields: list[str] | None = None,
) -> str:
    parts: list[str] = []
    if domain_hint:
        parts.append(f"Domain context: {domain_hint}\n")

    if suggested_fields:
        fields_str = ", ".join(suggested_fields)
        parts.append(
            f"The schema should include these fields: {fields_str}. "
            "You may add additional fields if clearly present in the documents.\n"
        )

    if max_fields is not None:
        parts.append(
            f"Limit the schema to at most {max_fields} fields. "
            "Focus on the most important, high-value fields.\n"
        )

    parts.append(f"Here are {len(documents)} sample documents:\n")
    for i, doc in enumerate(documents):
        # Truncate very long documents in the prompt
        content = doc.content[:max_doc_chars]
        if len(doc.content) > max_doc_chars:
            content += "\n... [truncated]"
        parts.append(f"--- Document {i + 1} ---\n{content}\n")

    parts.append(
        "\nPropose a JSON Schema that captures the structured fields "
        "present across these documents. Output ONLY the JSON Schema."
    )
    return "\n".join(parts)


class SinglePassDiscovery(BaseModel):
    """Discover a schema from sample documents using a single LLM call.

    Sends a representative sample of documents to the LLM and asks it
    to propose a JSON Schema for structured extraction.
    """

    model: str = DEFAULT_MODEL
    num_samples: int = 5
    """5 samples provides good schema coverage without excessive prompt length."""
    max_doc_chars: int = 3000
    """Truncation limit per document — fits ~750 tokens, balancing context vs cost."""
    domain_hint: str | None = None
    max_fields: int | None = None
    """Maximum number of fields in the discovered schema. None = no limit."""
    suggested_fields: list[str] | None = None
    """Field names to include in the schema. LLM may add others if found in documents."""
    temperature: float = 0.3
    """Low temperature for deterministic schema proposals."""
    base_url: str | None = None
    api_key: str | None = None
    client: Any | None = None
    """Pre-configured LLM client. If ``None``, a default client is created
    from ``model``, ``base_url``, and ``api_key``."""

    _usage_callback: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

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

    async def adiscover(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Schema:
        """Discover schema asynchronously."""
        if not documents:
            raise DiscoveryError("No documents provided for schema discovery.")

        hint = domain_hint or self.domain_hint

        # Sample documents
        sample = self._sample_documents(documents)
        logger.info(
            "SinglePassDiscovery: sampling %d/%d documents, model=%s",
            len(sample),
            len(documents),
            self.model,
        )

        # Build prompt and call LLM
        client = self._get_client()

        user_content = _build_user_prompt(
            sample,
            hint,
            max_doc_chars=self.max_doc_chars,
            max_fields=self.max_fields,
            suggested_fields=self.suggested_fields,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Discovery output is itself a dynamic JSON Schema — tool calling
        # with a fixed output schema can't represent arbitrary nested schemas.
        # Use plain completion and rely on the system prompt to get JSON.
        response: LLMResponse = await client.acomplete(
            messages,
            temperature=self.temperature,
            **kwargs,
        )

        # Parse JSON Schema from response
        json_schema = self._parse_schema(response.content)

        # Try to create Pydantic model
        pydantic_model = self._build_pydantic_model(json_schema)

        schema = Schema(
            model=pydantic_model,
            json_schema=json_schema,
            lineage=["SinglePassDiscovery"],
        )

        logger.info(
            "SinglePassDiscovery: discovered schema with %d fields",
            len(json_schema.get("properties", {})),
        )
        return schema

    def discover(
        self,
        documents: list[Document],
        *,
        domain_hint: str | None = None,
        **kwargs: Any,
    ) -> Schema:
        """Discover schema synchronously."""
        return run_sync(self.adiscover(documents, domain_hint=domain_hint, **kwargs))

    def _sample_documents(self, documents: list[Document]) -> list[Document]:
        """Sample up to num_samples documents."""
        if len(documents) <= self.num_samples:
            return documents
        return random.sample(documents, self.num_samples)

    @staticmethod
    def _parse_schema(content: str) -> dict[str, Any]:
        """Parse JSON Schema from LLM response content."""
        # Strip markdown code fences if present
        text = strip_markdown_fences(content)

        try:
            schema = json.loads(text)
        except json.JSONDecodeError as e:
            raise DiscoveryError(
                f"LLM response is not valid JSON: {e}\nResponse: {text[:500]}"
            ) from e

        if not isinstance(schema, dict):
            raise DiscoveryError(f"Expected JSON object, got {type(schema).__name__}")

        # Ensure it has properties
        if "properties" not in schema:
            # Maybe the LLM wrapped it in a top-level key
            for _key, value in schema.items():
                if isinstance(value, dict) and "properties" in value:
                    schema = value
                    break

        if "properties" not in schema:
            raise DiscoveryError(
                "LLM response does not contain a valid JSON Schema with 'properties'."
            )

        # Ensure type is set
        schema.setdefault("type", "object")
        return schema

    @staticmethod
    def _build_pydantic_model(
        json_schema: dict[str, Any],
    ) -> type[BaseModel] | None:
        """Try to build a Pydantic model from JSON Schema, return None on failure."""
        try:
            return json_schema_to_pydantic(json_schema, "DiscoveredSchema")
        except (SchemaError, ValueError, TypeError) as e:
            logger.warning(
                "Could not create Pydantic model from discovered schema. "
                "Falling back to JSON Schema only: %s",
                e,
                exc_info=True,
            )
            return None
