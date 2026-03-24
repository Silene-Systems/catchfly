"""SinglePassDiscovery — discover schema from sample documents in one LLM call."""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from pydantic import BaseModel

from catchfly._compat import run_sync
from catchfly._types import Document, Schema
from catchfly.exceptions import DiscoveryError
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
- Keep the schema flat where possible; nest only when semantically necessary.\
"""


def _build_user_prompt(
    documents: list[Document],
    domain_hint: str | None = None,
    *,
    max_doc_chars: int = 3000,
) -> str:
    parts: list[str] = []
    if domain_hint:
        parts.append(f"Domain context: {domain_hint}\n")

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

    model: str = "gpt-5.4-mini"
    num_samples: int = 5
    max_doc_chars: int = 3000
    domain_hint: str | None = None
    temperature: float = 0.7
    base_url: str | None = None
    api_key: str | None = None

    model_config = {"arbitrary_types_allowed": True}

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
        client = OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            usage_callback=getattr(self, "_usage_callback", None),
        )

        user_content = _build_user_prompt(
            sample, hint, max_doc_chars=self.max_doc_chars
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
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (code fence markers)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

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
        except Exception:
            logger.warning(
                "Could not create Pydantic model from discovered schema. "
                "Falling back to JSON Schema only.",
                exc_info=True,
            )
            return None
