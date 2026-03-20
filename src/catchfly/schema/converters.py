"""Convert between JSON Schema, Pydantic models, and TypedDict."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, create_model

from catchfly.exceptions import SchemaError

logger = logging.getLogger(__name__)

# JSON Schema type → Python type mapping
_JSON_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model class to JSON Schema dict."""
    return model.model_json_schema()


def json_schema_to_pydantic(
    schema: dict[str, Any],
    name: str = "DynamicModel",
) -> type[BaseModel]:
    """Create a Pydantic model class from a JSON Schema dict.

    Handles flat and nested object schemas with basic types,
    arrays, optionals, and enums.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        raise SchemaError("JSON Schema has no 'properties' — cannot create model.")

    field_definitions: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        python_type = _resolve_type(field_schema, field_name, name)
        is_required = field_name in required

        if is_required:
            field_definitions[field_name] = (python_type, ...)
        else:
            field_definitions[field_name] = (python_type | None, None)

    try:
        model: type[BaseModel] = create_model(name, **field_definitions)
    except Exception as e:
        raise SchemaError(f"Failed to create Pydantic model '{name}': {e}") from e

    logger.debug("Created Pydantic model '%s' with %d fields", name, len(field_definitions))
    return model


def _resolve_type(
    field_schema: dict[str, Any],
    field_name: str,
    parent_name: str,
) -> type[Any]:
    """Resolve a JSON Schema field to a Python type.

    Handles standard JSON Schema types, union types (e.g. ["string", "null"]
    from Mistral/OpenAI), anyOf/oneOf constructs, and nested objects.
    """
    # Handle enum
    if "enum" in field_schema:
        return str

    # Handle anyOf / oneOf (e.g. {"anyOf": [{"type": "string"}, {"type": "null"}]})
    for union_key in ("anyOf", "oneOf"):
        if union_key in field_schema:
            types = field_schema[union_key]
            non_null = [t for t in types if t.get("type") != "null"]
            if non_null:
                return _resolve_type(non_null[0], field_name, parent_name)
            return str

    json_type = field_schema.get("type", "string")

    # Handle union types as list (e.g. ["string", "null"] from Mistral)
    if isinstance(json_type, list):
        non_null_types = [t for t in json_type if t != "null"]
        json_type = non_null_types[0] if non_null_types else "string"

    # Handle array
    if json_type == "array":
        items = field_schema.get("items", {})
        item_type = _resolve_type(items, f"{field_name}_item", parent_name)
        return list[item_type]  # type: ignore[valid-type]

    # Handle nested object
    if json_type == "object":
        if "properties" in field_schema:
            nested_name = f"{parent_name}_{field_name.title().replace('_', '')}"
            return json_schema_to_pydantic(field_schema, name=nested_name)
        # Object without properties = free-form dict
        return dict

    # Handle basic types
    return _JSON_TYPE_MAP.get(json_type, str)
