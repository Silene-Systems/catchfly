"""Property-based tests for schema converters using Hypothesis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catchfly.exceptions import SchemaError
from catchfly.schema.converters import json_schema_to_pydantic, pydantic_to_json_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

field_names = st.from_regex(r"[a-z][a-z0-9_]{1,15}", fullmatch=True)
field_types = st.sampled_from([str, int, float, bool])

# Strategy: generate a dict of {field_name: field_type} with 1-8 fields
field_specs = st.dictionaries(
    keys=field_names,
    values=field_types,
    min_size=1,
    max_size=8,
)


def _build_model(fields: dict[str, type]) -> type[BaseModel]:
    """Dynamically create a Pydantic model from a name->type mapping."""
    from pydantic import create_model

    field_defs = {name: (typ, ...) for name, typ in fields.items()}
    return create_model("TestModel", **field_defs)


def _python_type_to_json_type(t: type) -> str:
    return {str: "string", int: "integer", float: "number", bool: "boolean"}[t]


def _sample_value(t: type) -> Any:
    """Return a sample value for a given Python type."""
    return {str: "hello", int: 42, float: 3.14, bool: True}[t]


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestPydanticToJsonSchemaProperties:
    """pydantic_to_json_schema always returns a dict with 'properties'."""

    @given(fields=field_specs)
    @settings(max_examples=50)
    def test_always_has_properties(self, fields: dict[str, type]) -> None:
        model = _build_model(fields)
        schema = pydantic_to_json_schema(model)
        assert isinstance(schema, dict)
        assert "properties" in schema
        for name in fields:
            assert name in schema["properties"]


class TestRoundtripFieldNames:
    """Roundtrip: json_schema_to_pydantic(pydantic_to_json_schema(model))
    creates a model with the same field names."""

    @given(fields=field_specs)
    @settings(max_examples=50)
    def test_field_names_preserved(self, fields: dict[str, type]) -> None:
        original = _build_model(fields)
        json_schema = pydantic_to_json_schema(original)
        rebuilt = json_schema_to_pydantic(json_schema, "Rebuilt")

        original_fields = set(original.model_fields.keys())
        rebuilt_fields = set(rebuilt.model_fields.keys())
        assert original_fields == rebuilt_fields


class TestRoundtripValidation:
    """Roundtrip: the recreated model validates the same data as the original."""

    @given(fields=field_specs)
    @settings(max_examples=50)
    def test_same_data_validates(self, fields: dict[str, type]) -> None:
        original = _build_model(fields)
        json_schema = pydantic_to_json_schema(original)
        rebuilt = json_schema_to_pydantic(json_schema, "Rebuilt")

        # Build sample data using the field types
        sample_data = {name: _sample_value(typ) for name, typ in fields.items()}

        # Both models should accept the same data
        orig_instance = original(**sample_data)
        rebuilt_instance = rebuilt(**sample_data)

        for name in fields:
            assert getattr(orig_instance, name) == getattr(rebuilt_instance, name)


class TestJsonSchemaEmptyProperties:
    """json_schema_to_pydantic raises SchemaError for empty properties."""

    def test_empty_properties_raises(self) -> None:
        with pytest.raises(SchemaError):
            json_schema_to_pydantic({"type": "object", "properties": {}})

    def test_no_properties_key_raises(self) -> None:
        with pytest.raises(SchemaError):
            json_schema_to_pydantic({"type": "object"})


class TestRoundtripWithOptionalFields:
    """Roundtrip preserves optional/required distinction."""

    @given(
        required_fields=st.dictionaries(field_names, field_types, min_size=1, max_size=4),
        optional_fields=st.dictionaries(field_names, field_types, min_size=0, max_size=4),
    )
    @settings(max_examples=30)
    def test_optional_fields_roundtrip(
        self,
        required_fields: dict[str, type],
        optional_fields: dict[str, type],
    ) -> None:
        # Remove overlapping keys (optional yields to required)
        optional_fields = {
            k: v for k, v in optional_fields.items() if k not in required_fields
        }
        if not required_fields:
            return  # need at least one field

        # Build JSON Schema directly with required/optional distinction
        properties: dict[str, dict[str, str]] = {}
        for name, typ in required_fields.items():
            properties[name] = {"type": _python_type_to_json_type(typ)}
        for name, typ in optional_fields.items():
            properties[name] = {"type": _python_type_to_json_type(typ)}

        json_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": list(required_fields.keys()),
        }

        model = json_schema_to_pydantic(json_schema, "MixedModel")

        # Required fields only should be enough to create instance
        required_data = {name: _sample_value(typ) for name, typ in required_fields.items()}
        instance = model(**required_data)
        for name in required_fields:
            assert getattr(instance, name) == required_data[name]

        # Optional fields should default to None
        for name in optional_fields:
            assert getattr(instance, name) is None
