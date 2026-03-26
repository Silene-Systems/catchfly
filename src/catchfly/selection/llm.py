"""LLMFieldSelector — LLM-based field selection for normalization."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, PrivateAttr

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._parsing import strip_markdown_fences

if TYPE_CHECKING:
    from catchfly._types import Schema

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a data normalization expert. Given a schema and sample extracted values, \
decide which fields would benefit from value normalization (grouping synonyms, \
fixing inconsistent naming, canonicalizing variants).

Good candidates for normalization:
- Categorical fields with inconsistent values ("NYC" vs "New York")
- Fields where the same concept appears in different surface forms
- Tags, labels, categories, attributes with semantic overlap

Bad candidates (skip these):
- Free-text fields (descriptions, summaries, notes, comments)
- Unique identifiers (names, emails, URLs, IDs)
- Numeric fields already in consistent format
- Boolean fields
- Fields with very few unique values (< 3) where there's nothing to group

Output ONLY a JSON array of field names to normalize. Example: ["category", "brand", "pros"]
If no fields need normalization, output: []\
"""


def _build_user_prompt(
    schema: Schema,
    field_samples: dict[str, list[str]],
) -> str:
    parts: list[str] = []
    parts.append("Schema fields:\n")

    props = schema.json_schema.get("properties", {})
    for name, spec in props.items():
        field_type = spec.get("type", "unknown")
        desc = spec.get("description", "")
        if field_type == "array":
            item_type = spec.get("items", {}).get("type", "unknown")
            field_type = f"array of {item_type}"

        line = f"- {name} ({field_type})"
        if desc:
            line += f": {desc}"
        parts.append(line)

        # Add sample values if available
        samples = field_samples.get(name, [])
        if samples:
            sample_str = ", ".join(f'"{v}"' for v in samples[:10])
            parts.append(f"  Sample values: [{sample_str}]")

    parts.append(
        "\nWhich fields would benefit from normalization? Output ONLY a JSON array of field names."
    )
    return "\n".join(parts)


class LLMFieldSelector(BaseModel):
    """Select normalization-worthy fields using an LLM.

    Sends the schema and sample extracted values to the LLM, which
    classifies each field as normalizable or not. More accurate than
    ``StatisticalFieldSelector`` — understands semantic context.

    Cost: ~$0.001 (one short LLM call).
    """

    model: str = DEFAULT_MODEL
    num_sample_values: int = 20
    """Number of sample values to show per field in the prompt."""
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    """Low temperature for deterministic classification."""
    client: Any | None = None
    """Pre-configured LLM client. If ``None``, a default client is created
    from ``model``, ``base_url``, and ``api_key``."""

    _usage_callback: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> Any:
        """Return the injected client or create a default one."""
        if self.client is not None:
            return self.client
        from catchfly.providers.llm import OpenAICompatibleClient

        return OpenAICompatibleClient(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            usage_callback=self._usage_callback,
        )

    async def aselect(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]:
        """Select fields for normalization asynchronously."""
        props = schema.json_schema.get("properties", {})

        # Collect sample values per string/array-of-string field
        field_samples: dict[str, list[str]] = {}
        for field_name, spec in props.items():
            field_type = spec.get("type", "")
            is_string = field_type == "string"
            is_string_array = (
                field_type == "array" and spec.get("items", {}).get("type") == "string"
            )
            if not (is_string or is_string_array):
                continue

            values: list[str] = []
            for record in records:
                val = (
                    getattr(record, field_name, None)
                    if hasattr(record, field_name)
                    else record.get(field_name)
                    if isinstance(record, dict)
                    else None
                )
                if val is None:
                    continue
                if isinstance(val, list):
                    values.extend(str(v) for v in val)
                else:
                    values.append(str(val))

            # Deduplicate and sample
            unique = list(set(values))
            field_samples[field_name] = unique[: self.num_sample_values]

        if not field_samples:
            logger.info("LLMFieldSelector: no string fields found, returning empty")
            return []

        # Build prompt and call LLM
        client = self._get_client()

        user_content = _build_user_prompt(schema, field_samples)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = await client.acomplete(
            messages,
            temperature=self.temperature,
        )

        # Parse response
        selected = self._parse_response(response.content, props)
        logger.info(
            "LLMFieldSelector: selected %d fields: %s",
            len(selected),
            selected,
        )
        return selected

    def select(
        self,
        schema: Schema,
        records: list[Any],
        **kwargs: Any,
    ) -> list[str]:
        """Select fields for normalization synchronously."""
        return run_sync(self.aselect(schema, records, **kwargs))

    @staticmethod
    def _parse_response(
        content: str,
        props: dict[str, Any],
    ) -> list[str]:
        """Parse LLM response into a list of valid field names."""
        text = strip_markdown_fences(content)

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "LLMFieldSelector: could not parse response as JSON: %s",
                text[:200],
            )
            return []

        if not isinstance(result, list):
            logger.warning(
                "LLMFieldSelector: expected JSON array, got %s",
                type(result).__name__,
            )
            return []

        # Validate against actual schema field names
        valid = [name for name in result if isinstance(name, str) and name in props]
        if len(valid) != len(result):
            logger.debug(
                "LLMFieldSelector: filtered %d invalid field names from response",
                len(result) - len(valid),
            )
        return valid
