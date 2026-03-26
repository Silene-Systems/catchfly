"""SchemaOptimizer — PARSE-style iterative schema enrichment.

Iteratively enriches field descriptions by running extraction on test
documents, analyzing errors/gaps, and asking the LLM to improve field
definitions. The enriched field_metadata serves as semantic anchors
for downstream normalization (schema-aware prompting in LLMCanonicalization).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, PrivateAttr

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._parsing import strip_markdown_fences
from catchfly._types import Document, Schema
from catchfly.exceptions import DiscoveryError, ProviderError
from catchfly.providers.llm import OpenAICompatibleClient

logger = logging.getLogger(__name__)

_ENRICHMENT_SYSTEM_PROMPT = """\
You are a schema optimization assistant. Given a JSON Schema, sample extracted \
data, and extraction error analysis, your job is to enrich each field's definition \
to improve extraction quality.

For each field, provide:
- "description": A clear, detailed description of what this field captures
- "examples": 2-4 concrete example values from the documents
- "synonyms": Alternative names or phrasings for this concept
- "constraints": Any constraints on valid values (type, range, format)

Output ONLY valid JSON with the structure:
{
  "fields": {
    "field_name": {
      "description": "...",
      "examples": ["...", "..."],
      "synonyms": ["...", "..."],
      "constraints": "..."
    }
  }
}\
"""


def _build_enrichment_prompt(
    schema: dict[str, Any],
    extracted_samples: list[dict[str, Any]],
    error_analysis: dict[str, Any],
    iteration: int,
) -> str:
    parts: list[str] = []
    parts.append(f"Optimization iteration {iteration}.\n")
    parts.append(f"Current schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n")

    if extracted_samples:
        parts.append(f"Sample extracted records ({len(extracted_samples)} docs):")
        for i, sample in enumerate(extracted_samples[:5]):
            parts.append(f"  Record {i + 1}: {json.dumps(sample, default=str)}")
        parts.append("")

    if error_analysis:
        parts.append("Error analysis:")
        for field_name, info in error_analysis.items():
            parts.append(f"  - {field_name}: {info}")
        parts.append("")

    parts.append(
        "Enrich each field's definition to improve extraction accuracy. "
        "Focus on fields with high error rates or missing values."
    )
    return "\n".join(parts)


class SchemaOptimizer(BaseModel):
    """Iteratively enrich schema field descriptions (PARSE-style).

    Runs N optimization iterations, each consisting of:
    1. Extract data from test documents using current schema
    2. Analyze extraction errors and gaps
    3. Ask LLM to enrich field descriptions based on analysis
    4. Update schema field_metadata

    The enriched field_metadata (descriptions, examples, synonyms) can be
    used as prompt context for LLMCanonicalization normalization.
    """

    model: str = DEFAULT_MODEL
    num_iterations: int = 5
    """5 iterations balances enrichment quality vs LLM cost (PARSE paper default)."""
    max_docs_per_iteration: int = 10
    """10 docs per iteration gives sufficient signal without excessive API calls."""
    low_coverage_threshold: float = 0.8
    """Fields below 80% coverage are flagged as needing better descriptions."""
    max_doc_chars: int = 3000
    """Truncation limit per document — fits ~750 tokens, balancing context vs cost."""
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.3
    """Low temperature for deterministic enrichment suggestions."""
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

    async def aoptimize(
        self,
        schema: Schema | type[BaseModel],
        test_documents: list[Document],
        **kwargs: Any,
    ) -> Schema:
        """Optimize schema asynchronously.

        Args:
            schema: Input schema (Schema object or Pydantic model class).
            test_documents: Documents to test extraction against (10-20 recommended).

        Returns:
            Schema with enriched field_metadata.
        """
        if not test_documents:
            raise DiscoveryError("No test documents provided for schema optimization.")

        # Normalize input to Schema
        working_schema = self._normalize_schema(schema)
        json_schema = working_schema.json_schema

        client = self._get_client()

        field_metadata: dict[str, dict[str, Any]] = dict(working_schema.field_metadata)
        lineage = list(working_schema.lineage)

        logger.info(
            "SchemaOptimizer: starting %d iterations on %d test documents, model=%s",
            self.num_iterations,
            len(test_documents),
            self.model,
        )

        for iteration in range(1, self.num_iterations + 1):
            logger.info("SchemaOptimizer: iteration %d/%d", iteration, self.num_iterations)

            # Step 1: Try extraction on test docs
            extracted_samples = await self._try_extraction(client, json_schema, test_documents)

            # Step 2: Analyze errors/gaps
            error_analysis = self._analyze_gaps(json_schema, extracted_samples)

            # Step 3: Ask LLM to enrich
            enrichment = await self._get_enrichment(
                client, json_schema, extracted_samples, error_analysis, iteration
            )

            # Step 4: Merge enrichment into field_metadata
            if enrichment:
                for field_name, field_info in enrichment.items():
                    if field_name not in field_metadata:
                        field_metadata[field_name] = {}
                    field_metadata[field_name].update(field_info)

                logger.info(
                    "SchemaOptimizer: iteration %d enriched %d fields",
                    iteration,
                    len(enrichment),
                )
            else:
                logger.warning("SchemaOptimizer: iteration %d produced no enrichments", iteration)

            lineage.append(f"SchemaOptimizer:iter{iteration}")

        # Build output schema
        result = Schema(
            model=working_schema.model,
            json_schema=json_schema,
            field_metadata=field_metadata,
            lineage=lineage,
        )

        logger.info(
            "SchemaOptimizer: completed — %d fields enriched",
            len(field_metadata),
        )
        return result

    def optimize(
        self,
        schema: Schema | type[BaseModel],
        test_documents: list[Document],
        **kwargs: Any,
    ) -> Schema:
        """Optimize schema synchronously."""
        return run_sync(self.aoptimize(schema, test_documents, **kwargs))

    async def _try_extraction(
        self,
        client: OpenAICompatibleClient,
        json_schema: dict[str, Any],
        documents: list[Document],
    ) -> list[dict[str, Any]]:
        """Try extracting data from documents using current schema."""
        schema_str = json.dumps(json_schema, indent=2)
        results: list[dict[str, Any]] = []

        for doc in documents[: self.max_docs_per_iteration]:
            prompt = (
                f"Extract structured data from this document according to "
                f"the JSON Schema below.\n\n"
                f"Schema:\n```json\n{schema_str}\n```\n\n"
                f"Document:\n---\n{doc.content[: self.max_doc_chars]}\n---\n\n"
                f"Output ONLY the extracted JSON."
            )

            try:
                sys_msg = "You are a data extraction assistant. Output only valid JSON."
                response = await client.acomplete(
                    [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                data = json.loads(response.content)
                if isinstance(data, dict):
                    results.append(data)
            except json.JSONDecodeError:
                logger.debug(
                    "SchemaOptimizer: JSON parse error for doc '%s'",
                    doc.id or "(no id)",
                )
                results.append({})
            except (ProviderError, ValueError, KeyError, TypeError) as e:
                logger.debug(
                    "SchemaOptimizer: extraction failed for doc '%s': %s",
                    doc.id or "(no id)",
                    e,
                    exc_info=True,
                )
                results.append({})

        return results

    def _analyze_gaps(
        self,
        json_schema: dict[str, Any],
        extracted: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze which fields are frequently missing or empty."""
        properties = json_schema.get("properties", {})
        if not properties or not extracted:
            return {}

        analysis: dict[str, Any] = {}
        total = len(extracted)

        for field_name in properties:
            present = sum(1 for rec in extracted if rec.get(field_name) is not None)
            coverage = present / total if total > 0 else 0

            if coverage < self.low_coverage_threshold:
                analysis[field_name] = (
                    f"low coverage ({present}/{total} = {coverage:.0%}) — "
                    f"description may be unclear"
                )

            # Collect sample values for enrichment
            sample_values = [
                rec[field_name] for rec in extracted if rec.get(field_name) is not None
            ]
            if sample_values and field_name not in analysis:
                # Even well-covered fields benefit from examples
                analysis[field_name] = f"coverage OK ({coverage:.0%}), sample values available"

        return analysis

    async def _get_enrichment(
        self,
        client: OpenAICompatibleClient,
        json_schema: dict[str, Any],
        extracted: list[dict[str, Any]],
        error_analysis: dict[str, Any],
        iteration: int,
    ) -> dict[str, dict[str, Any]]:
        """Ask LLM to enrich field descriptions."""
        prompt = _build_enrichment_prompt(json_schema, extracted, error_analysis, iteration)

        try:
            response = await client.acomplete(
                [
                    {"role": "system", "content": _ENRICHMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            data = self._parse_enrichment(response.content)
            fields: dict[str, dict[str, Any]] = data.get("fields", {})
            return fields
        except (ProviderError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(
                "SchemaOptimizer: failed to parse enrichment response in iteration %d: %s",
                iteration,
                e,
                exc_info=True,
            )
            return {}

    @staticmethod
    def _parse_enrichment(content: str) -> dict[str, Any]:
        """Parse enrichment JSON from LLM response."""
        text = strip_markdown_fences(content)

        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        return data

    @staticmethod
    def _normalize_schema(schema: Schema | type[BaseModel]) -> Schema:
        """Convert input to a Schema object."""
        if isinstance(schema, Schema):
            return schema

        # It's a Pydantic model class
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return Schema(
                model=schema,
                json_schema=schema.model_json_schema(),
                lineage=["user-provided"],
            )

        msg = f"Expected Schema or Pydantic model, got {type(schema)}"
        raise DiscoveryError(msg)
