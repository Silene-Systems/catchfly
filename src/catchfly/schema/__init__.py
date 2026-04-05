"""Schema converters and registry."""

from catchfly.schema.converters import json_schema_to_pydantic, pydantic_to_json_schema, resolve_refs
from catchfly.schema.registry import SchemaRegistry

__all__ = [
    "SchemaRegistry",
    "json_schema_to_pydantic",
    "pydantic_to_json_schema",
    "resolve_refs",
]


def __dir__() -> list[str]:
    return __all__
