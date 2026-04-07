"""Schema refinement — apply user feedback to a discovered schema.

A discovered schema is rarely perfect on the first shot. This module
gives users a first-class API to iterate on it without hand-editing the
underlying ``json_schema`` dict.

Two modes are supported:

1. **Explicit** — populate :attr:`SchemaFeedback.add_fields`,
   :attr:`remove_fields`, :attr:`modify_fields` with deterministic
   changes. No LLM call is made.
2. **Instruction** — populate :attr:`SchemaFeedback.instruction` with a
   natural-language command (``"drop all administrative fields"``); the
   LLM proposes concrete changes that are then applied with the same
   invariants as :class:`ThreeStageDiscovery`.

The two modes compose: instructions run first (if present), then
explicit ops are applied on top. Both paths go through
``ThreeStageDiscovery._apply_changes`` so the empty-properties invariant
and deep-copy semantics stay consistent across the discovery surface.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from catchfly._compat import run_sync
from catchfly._defaults import DEFAULT_MODEL
from catchfly._types import Schema
from catchfly.discovery.single_pass import SinglePassDiscovery
from catchfly.discovery.three_stage import _CHANGES_OUTPUT_SCHEMA, ThreeStageDiscovery
from catchfly.exceptions import DiscoveryError, ProviderError
from catchfly.providers.llm import OpenAICompatibleClient

logger = logging.getLogger(__name__)


_REFINE_INSTRUCTION_SYSTEM = """\
You are a schema refinement assistant. The user provides a JSON Schema \
and a free-text instruction describing how they want it changed. Your \
job is to translate the instruction into explicit schema operations.

Output ONLY valid JSON with this structure:
{
  "add_fields": {"field_name": {"type": "...", "description": "..."}},
  "remove_fields": ["field_to_remove"],
  "modify_fields": {"field_name": {"type": "new_type", "description": "..."}},
  "rationale": "Brief explanation of how your changes honour the instruction"
}

Rules:
- Respect the user's intent literally — do not second-guess obvious requests.
- Preserve fields that the instruction does not explicitly touch.
- Use valid JSON Schema types: string, integer, number, boolean, array, object.
- If the user asks to change a type to an enum, use {"type": "string", "enum": [...]}.\
"""


@dataclass
class SchemaFeedback:
    """User feedback on a discovered schema.

    Can hold explicit operations, a free-text instruction, or both.
    An empty :class:`SchemaFeedback` is a no-op that returns the input
    schema unchanged (after re-deriving the Pydantic model).
    """

    add_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Fields to add. Keys are field names, values are JSON Schema
    field definitions ``{"type": "...", "description": "..."}``. Fields
    already present in the schema are left untouched."""

    remove_fields: list[str] = field(default_factory=list)
    """Field names to delete. Fields not present in the schema are
    silently ignored. Cannot reduce the schema to zero properties — the
    empty-properties invariant reverts the change with a warning."""

    modify_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-field partial updates. Values are merged into the existing
    field definition (e.g. ``{"type": "number"}`` changes only the type
    but preserves the description)."""

    instruction: str | None = None
    """Free-text instruction for LLM-guided refinement. When set,
    :func:`arefine_schema` calls the LLM with
    :data:`_REFINE_INSTRUCTION_SYSTEM` to translate the instruction into
    explicit operations, which are then applied before ``add_fields`` /
    ``remove_fields`` / ``modify_fields``."""

    def is_empty(self) -> bool:
        """True when no changes were requested."""
        return (
            not self.add_fields
            and not self.remove_fields
            and not self.modify_fields
            and not self.instruction
        )


async def arefine_schema(
    schema: Schema,
    feedback: SchemaFeedback,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
) -> Schema:
    """Apply :class:`SchemaFeedback` to *schema* and return a new Schema.

    Args:
        schema: The input schema to refine. Not mutated.
        feedback: Changes to apply. An empty feedback returns an
            equivalent Schema with lineage extended to record the
            refinement attempt.
        client: Pre-configured LLM client. Required only when
            ``feedback.instruction`` is set. If ``None``, a default
            client is created from ``model`` / ``base_url`` / ``api_key``.
        model: Model name for the default LLM client.
        base_url: Base URL for the default LLM client.
        api_key: API key for the default LLM client.
        temperature: Sampling temperature for instruction-mode calls.

    Returns:
        A new :class:`Schema` with the refined ``json_schema``, a freshly
        built Pydantic ``model``, and ``lineage`` extended to record the
        refinement step.

    Raises:
        DiscoveryError: If the refined schema cannot be converted back
            to a Pydantic model, or if instruction-mode LLM parsing fails.
    """
    if feedback.is_empty():
        logger.debug("arefine_schema: empty feedback, returning schema unchanged")
        return _rebuild_schema(schema.json_schema, schema.lineage + ["refine:noop"])

    json_schema = json.loads(json.dumps(schema.json_schema))  # deep copy
    lineage_steps: list[str] = []

    # --- Instruction mode (LLM-guided) ---
    if feedback.instruction:
        llm_changes = await _fetch_llm_changes(
            json_schema,
            feedback.instruction,
            client=client,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        json_schema = ThreeStageDiscovery._apply_changes(json_schema, llm_changes)
        lineage_steps.append("refine:instruction")

    # --- Explicit mode (deterministic ops) ---
    explicit_changes: dict[str, Any] = {
        "add_fields": feedback.add_fields,
        "remove_fields": feedback.remove_fields,
        "modify_fields": feedback.modify_fields,
    }
    if (
        feedback.add_fields
        or feedback.remove_fields
        or feedback.modify_fields
    ):
        json_schema = ThreeStageDiscovery._apply_changes(json_schema, explicit_changes)
        lineage_steps.append("refine:explicit")

    return _rebuild_schema(json_schema, schema.lineage + lineage_steps)


def refine_schema(
    schema: Schema,
    feedback: SchemaFeedback,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
) -> Schema:
    """Synchronous wrapper around :func:`arefine_schema`."""
    return run_sync(
        arefine_schema(
            schema,
            feedback,
            client=client,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _fetch_llm_changes(
    json_schema: dict[str, Any],
    instruction: str,
    *,
    client: Any | None,
    model: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
) -> dict[str, Any]:
    """Ask the LLM to translate *instruction* into explicit schema ops."""
    effective_client = client or OpenAICompatibleClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    user_prompt = (
        f"Current schema:\n```json\n{json.dumps(json_schema, indent=2)}\n```\n\n"
        f"User instruction: {instruction}\n\n"
        "Produce the schema operations that honour this instruction."
    )
    messages = [
        {"role": "system", "content": _REFINE_INSTRUCTION_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = await effective_client.astructured_complete(
            messages,
            output_schema=_CHANGES_OUTPUT_SCHEMA,
            schema_name="schema_changes",
            temperature=temperature,
        )
    except (ProviderError, ValueError, KeyError, TypeError) as e:
        raise DiscoveryError(
            f"Schema refinement: LLM call failed while processing "
            f"instruction '{instruction}': {e}"
        ) from e

    return ThreeStageDiscovery._parse_changes(response.content)


def _rebuild_schema(json_schema: dict[str, Any], lineage: list[str]) -> Schema:
    """Derive a fresh Pydantic model and return a new Schema.

    Uses the same hard-fail semantics as
    :meth:`SinglePassDiscovery._build_pydantic_model` — a refined schema
    that cannot be realised is a user error we surface loudly.
    """
    if not json_schema.get("properties"):
        raise DiscoveryError(
            "Schema refinement produced a schema with no properties. "
            "Check that the requested changes don't remove every field."
        )
    try:
        model = SinglePassDiscovery._build_pydantic_model(json_schema)
    except DiscoveryError:
        raise
    return Schema(
        model=model,
        json_schema=json_schema,
        lineage=lineage,
    )
